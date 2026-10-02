/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "trial.h"

#include "cold.h"
#include "workload.h"

#include <carta-zarr/carta_zarr.h>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace carta::zarr::bench {

namespace {

using Clock = std::chrono::steady_clock;

// How long past the trial's deadline the parent waits before killing a process. The library stops
// starting storage operations at the deadline but does not interrupt one in flight, so a process
// is given that long to finish the one it is in.
constexpr std::chrono::seconds kGrace{30};

constexpr const char* kReady = "READY";
constexpr const char* kRow = "ROW ";

bool WriteAll(int fd, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const char*>(data);
    while (size > 0) {
        const auto written = write(fd, bytes, size);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            return false;
        }
        bytes += written;
        size -= static_cast<std::size_t>(written);
    }
    return true;
}

// Whether `size` bytes arrived before the other end closed.
bool ReadAll(int fd, void* data, std::size_t size) {
    auto* bytes = static_cast<char*>(data);
    while (size > 0) {
        const auto got = read(fd, bytes, size);
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            return false;
        }
        bytes += got;
        size -= static_cast<std::size_t>(got);
    }
    return true;
}

struct ResourceUse {
    double user_s = 0.0;
    double sys_s = 0.0;
    std::uint64_t peak_rss_bytes = 0;
};

ResourceUse Resources() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    ResourceUse use;
    use.user_s = static_cast<double>(usage.ru_utime.tv_sec) + static_cast<double>(usage.ru_utime.tv_usec) * 1e-6;
    use.sys_s = static_cast<double>(usage.ru_stime.tv_sec) + static_cast<double>(usage.ru_stime.tv_usec) * 1e-6;
#if defined(__APPLE__)
    use.peak_rss_bytes = static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    use.peak_rss_bytes = static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
    return use;
}

// What this process has caused to be fetched from storage, as /proc/self/io counts it: reads the page
// cache answered are not in it. Linux only.
std::optional<std::uint64_t> StorageReadBytes() {
    std::ifstream io("/proc/self/io");
    std::string key;
    std::uint64_t value = 0;
    while (io >> key >> value) {
        if (key == "read_bytes:") {
            return value;
        }
    }
    return std::nullopt;
}

std::string UtcNow() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    const auto millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm utc{};
    gmtime_r(&seconds, &utc);
    std::array<char, 40> buffer{};
    const auto length = std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%S", &utc);
    std::snprintf(buffer.data() + length, buffer.size() - length, ".%03dZ", static_cast<int>(millis));
    return buffer.data();
}

std::size_t ItemSize(DataType type) {
    switch (type) {
        case DataType::boolean:
        case DataType::int8:
        case DataType::uint8:
            return 1;
        case DataType::int16:
        case DataType::uint16:
        case DataType::float16:
            return 2;
        case DataType::int32:
        case DataType::uint32:
        case DataType::float32:
            return 4;
        case DataType::int64:
        case DataType::uint64:
        case DataType::float64:
        case DataType::complex64:
            return 8;
        case DataType::complex128:
            return 16;
        case DataType::unknown:
            break;
    }
    return 0;
}

std::string Shape(const std::vector<AxisDescriptor>& axes, const std::vector<std::uint64_t>& lengths) {
    std::string text;
    for (std::size_t index = 0; index < axes.size() && index < lengths.size(); ++index) {
        if (!text.empty()) {
            text += ';';
        }
        text += axes[index].name + "=" + std::to_string(lengths[index]);
    }
    return text;
}

// The columns only an opened image can answer.
void DescribeImage(Row& row, const Image& image) {
    const auto& descriptor = image.descriptor();
    const auto& geometry = image.chunk_geometry();
    std::vector<std::uint64_t> lengths;
    for (const auto& axis : descriptor.axes) {
        lengths.push_back(axis.length);
    }
    row.image_id = descriptor.id;
    row.shape = Shape(descriptor.axes, lengths);
    row.chunk_shape = Shape(descriptor.axes, geometry.chunk_shape);
    row.shard_shape = geometry.sharded ? Shape(descriptor.axes, geometry.shard_shape) : "";
    if (row.codec.empty()) {
        row.codec = geometry.compressor;
    }
}

std::string Describe(const Error& error) {
    std::string text = std::string(ErrorCodeName(error.code)) + ": " + error.message;
    if (!error.node_path.empty()) {
        text += " (" + error.node_path + ")";
    }
    return text;
}

// The operations of one process: open what it reads, say it is ready, wait to be released, read, and
// report each operation as it ends. Never returns.
[[noreturn]] void Child(const RunOptions& options, Mode mode, unsigned trial, unsigned process_index, Row row,
                        int out, int go) {
    const auto send = [out](const std::string& line) {
        const auto text = line + "\n";
        WriteAll(out, text.data(), text.size());
    };
    const auto fail = [&](const std::string& message) {
        row.status = "error";
        row.error = message;
        send(kRow + FormatRow(row));
        _exit(1);
    };
    row.process_index = process_index;
    const unsigned ops = options.OpsFor(mode);

    std::optional<Runner> runner;
    std::vector<Operation> plan;
    std::size_t item_size = 4;
    if (mode == Mode::open) {
        runner.emplace(options.context, options.dataset, options.image_id);
        plan = PlanOperations(mode, CubeAxes{}, options.seed, trial, options.processes, process_index, ops);
    } else {
        auto context = Context::Create(options.context);
        if (!context) {
            fail("Context::Create: " + Describe(context.error()));
        }
        auto dataset = Dataset::Open(*context, options.dataset);
        if (!dataset) {
            fail("Dataset::Open: " + Describe(dataset.error()));
        }
        auto id = options.image_id;
        if (id.empty()) {
            if (!dataset->descriptor().default_image_id) {
                fail("the dataset lists no image that opens");
            }
            id = *dataset->descriptor().default_image_id;
        }
        auto image = dataset->OpenImage(id);
        if (!image) {
            fail("OpenImage: " + Describe(image.error()));
        }
        const auto axes = CubeAxes::Of(image->descriptor());
        if (!axes) {
            fail(Describe(axes.error()));
        }
        DescribeImage(row, *image);
        item_size = ItemSize(image->descriptor().stored_type);
        // Every process's plan, to tell which of this one's operations share a chunk with another's.
        std::vector<std::vector<Operation>> plans;
        for (unsigned process = 0; process < options.processes; ++process) {
            plans.push_back(PlanOperations(mode, *axes, options.seed, trial, options.processes, process, ops,
                                           options.region_fraction, options.animation_frames));
        }
        MarkSharedChunks(plans, *axes, image->chunk_geometry().chunk_shape);
        plan = std::move(plans[process_index]);
        runner.emplace(*context, std::move(image).value(), options.histogram, options.FirstTouchCacheBytes());
        runner->SetAnimation(options.animation_fps, options.animation_prefetch);
    }

    send(kReady);
    std::int64_t released_ns = 0;
    if (!ReadAll(go, &released_ns, sizeof(released_ns))) {
        // The parent called the trial off: another process failed to get ready.
        _exit(0);
    }
    const Clock::time_point released{Clock::duration(released_ns)};
    const auto deadline = released + options.trial_timeout;
    ReadOptions read_options;
    read_options.control.deadline = deadline;
    read_options.read_budget_bytes = options.read_budget_bytes;

    const auto since_release = [released](Clock::time_point at) {
        return std::chrono::duration<double>(at - released).count();
    };

    for (unsigned index = 0; index < plan.size(); ++index) {
        const auto& operation = plan[index];
        Row result = row;
        result.op_index = index;
        result.position = operation.Describe();
        result.overlap = operation.overlap;
        result.shares_chunks = operation.shares_chunks;
        if (Clock::now() >= deadline) {
            result.status = "timeout";
            result.error = "not started: the trial deadline had passed";
            send(kRow + FormatRow(result));
            continue;
        }
        if (auto prepared = runner->Prepare(operation); !prepared) {
            result.status = "error";
            result.error = "preparing: " + Describe(prepared.error());
            send(kRow + FormatRow(result));
            continue;
        }

        const auto storage_before = StorageReadBytes();
        const auto use_before = Resources();
        result.timestamp_utc = UtcNow();
        const auto start = Clock::now();
        auto elements = runner->Run(operation, read_options);
        const auto end = Clock::now();
        const auto use_after = Resources();
        const auto storage_after = StorageReadBytes();

        result.seconds = std::chrono::duration<double>(end - start).count();
        result.t_start_s = since_release(start);
        result.t_end_s = since_release(end);
        result.user_cpu_s = use_after.user_s - use_before.user_s;
        result.sys_cpu_s = use_after.sys_s - use_before.sys_s;
        result.peak_rss_bytes = use_after.peak_rss_bytes;
        if (storage_before && storage_after) {
            result.storage_read_bytes = *storage_after - *storage_before;
        }
        if (mode == Mode::open && runner->image()) {
            DescribeImage(result, *runner->image());
            item_size = ItemSize(runner->image()->descriptor().stored_type);
        }
        if (elements) {
            result.status = "ok";
            result.elements = *elements;
            result.logical_bytes = *elements * item_size;
            result.checksum = runner->Fingerprint();
            if (const auto& frames = runner->frame_stats()) {
                result.frame_first_s = frames->first_s;
                result.frame_median_s = frames->median_s;
                result.frame_max_s = frames->max_s;
                result.late_frames = frames->late;
                result.late_max_s = frames->late_max_s;
            }
        } else {
            const bool expired = elements.error().code == ErrorCode::cancelled && end >= deadline;
            result.status = expired ? "timeout" : "error";
            result.error = Describe(elements.error());
        }
        send(kRow + FormatRow(result));
    }
    _exit(0);
}

struct Process {
    pid_t pid = -1;
    int out = -1;  // what it reports, read here
    int go = -1;   // its release, written here
    std::string pending;
    std::vector<std::string> rows;
    bool ready = false;
    bool closed = false;
    bool killed = false;
};

void CloseFd(int& fd) {
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
}

// Reads whatever the processes have said until each has `until` or the time is up. Returns whether
// they all got there.
template <typename Until>
bool Listen(std::vector<Process>& processes, Clock::time_point give_up, Until until) {
    while (true) {
        std::vector<pollfd> waiting;
        std::vector<Process*> owners;
        for (auto& process : processes) {
            if (!process.closed && !until(process)) {
                waiting.push_back({process.out, POLLIN, 0});
                owners.push_back(&process);
            }
        }
        if (waiting.empty()) {
            return std::all_of(processes.begin(), processes.end(), until);
        }
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(give_up - Clock::now()).count();
        if (left <= 0) {
            return false;
        }
        const int ready = poll(waiting.data(), waiting.size(), static_cast<int>(std::min<long long>(left, 1000)));
        if (ready < 0 && errno != EINTR) {
            return false;
        }
        for (std::size_t index = 0; index < waiting.size(); ++index) {
            if (waiting[index].revents == 0) {
                continue;
            }
            auto& process = *owners[index];
            std::array<char, 65536> buffer{};
            const auto got = read(process.out, buffer.data(), buffer.size());
            if (got < 0 && errno == EINTR) {
                continue;
            }
            if (got <= 0) {
                process.closed = true;
                continue;
            }
            process.pending.append(buffer.data(), static_cast<std::size_t>(got));
            std::size_t newline = 0;
            while ((newline = process.pending.find('\n')) != std::string::npos) {
                const auto line = process.pending.substr(0, newline);
                process.pending.erase(0, newline + 1);
                if (line == kReady) {
                    process.ready = true;
                } else if (line.rfind(kRow, 0) == 0) {
                    process.rows.push_back(line.substr(std::strlen(kRow)));
                }
            }
        }
    }
}

std::string ExitDescription(int status) {
    if (WIFSIGNALED(status)) {
        return "the process was killed by signal " + std::to_string(WTERMSIG(status));
    }
    if (WIFEXITED(status)) {
        return "the process exited with status " + std::to_string(WEXITSTATUS(status));
    }
    return "the process ended";
}

}  // namespace

TrialOutcome RunTrial(const RunOptions& options, Mode mode, unsigned trial, ColdMethod cold, const Row& base) {
    Row row = base;
    row.trial = trial;
    const auto dropped = DropCaches(cold, options.drop_cache_command, options.dataset);
    row.cold_ok = cold != ColdMethod::off && dropped.empty();
    if (!dropped.empty()) {
        std::fprintf(stderr, "warning: trial %u is not cold: %s\n", trial, dropped.c_str());
    }

    // Nothing buffered may be copied into a child, or both would write it.
    std::fflush(stdout);
    std::fflush(stderr);

    std::vector<Process> processes(options.processes);
    for (unsigned index = 0; index < options.processes; ++index) {
        int out[2] = {-1, -1};
        int go[2] = {-1, -1};
        if (pipe(out) != 0 || pipe(go) != 0) {
            std::fprintf(stderr, "error: pipe: %s\n", std::strerror(errno));
            std::exit(1);
        }
        const pid_t pid = fork();
        if (pid < 0) {
            std::fprintf(stderr, "error: fork: %s\n", std::strerror(errno));
            std::exit(1);
        }
        if (pid == 0) {
            // Nothing of the processes forked before this one, or a release meant for another would
            // never reach end of file when the parent calls the trial off.
            for (auto& earlier : processes) {
                CloseFd(earlier.out);
                CloseFd(earlier.go);
            }
            close(out[0]);
            close(go[1]);
            Child(options, mode, trial, index, row, out[1], go[0]);
        }
        close(out[1]);
        close(go[0]);
        fcntl(out[0], F_SETFD, FD_CLOEXEC);
        fcntl(go[1], F_SETFD, FD_CLOEXEC);
        processes[index].pid = pid;
        processes[index].out = out[0];
        processes[index].go = go[1];
    }

    // Opening is given the trial's own timeout: a store that takes longer than that to open will not
    // read in time either.
    const bool ready = Listen(processes, Clock::now() + options.trial_timeout,
                              [](const Process& process) { return process.ready; });
    if (ready) {
        const auto released = Clock::now();
        const std::int64_t released_ns = released.time_since_epoch().count();
        for (auto& process : processes) {
            WriteAll(process.go, &released_ns, sizeof(released_ns));
        }
        for (auto& process : processes) {
            CloseFd(process.go);
        }
        Listen(processes, released + options.trial_timeout + kGrace,
               [](const Process& process) { return process.closed; });
    }
    for (auto& process : processes) {
        CloseFd(process.go);
        if (!process.closed) {
            kill(process.pid, SIGKILL);
            process.killed = true;
        }
    }

    TrialOutcome outcome;
    const unsigned ops = options.OpsFor(mode);
    for (unsigned index = 0; index < processes.size(); ++index) {
        auto& process = processes[index];
        int status = 0;
        while (waitpid(process.pid, &status, 0) < 0 && errno == EINTR) {
        }
        CloseFd(process.out);

        // Released, a process reports each operation; the ones it never reported are written down as
        // what stopped it. Not released, a process that failed to open has said why in a row of its
        // own, one that was ready has nothing to say, and one that was neither is written down here.
        if (ready) {
            for (auto op = static_cast<unsigned>(process.rows.size()); op < ops; ++op) {
                Row missing = row;
                missing.process_index = index;
                missing.op_index = op;
                missing.status = process.killed ? "timeout" : "error";
                missing.error = process.killed ? "killed: still running " + std::to_string(kGrace.count()) +
                                                     " s after the trial deadline"
                                               : "not reported: " + ExitDescription(status);
                process.rows.push_back(FormatRow(missing));
            }
        } else if (process.rows.empty() && !process.ready) {
            Row missing = row;
            missing.process_index = index;
            missing.status = "error";
            missing.error = process.killed ? "killed: not ready within the trial timeout"
                                           : "not ready: " + ExitDescription(status);
            process.rows.push_back(FormatRow(missing));
        }
        for (auto& line : process.rows) {
            if (const auto summary = SummariseRow(line)) {
                if (summary->status == "ok") {
                    ++outcome.ok;
                } else if (summary->status == "timeout") {
                    ++outcome.timeouts;
                } else {
                    ++outcome.errors;
                }
                outcome.makespan_s = std::max(outcome.makespan_s, summary->t_end_s.value_or(0.0));
            }
            outcome.rows.push_back(std::move(line));
        }
    }
    return outcome;
}

}  // namespace carta::zarr::bench
