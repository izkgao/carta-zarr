/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// carta-zarr-bench below its command line: the positions it reads, the rows it writes, and what it
// reads at them.
//
// A comparison between layouts rests on three things this pins. Every layout is asked for the same
// positions, which depend on the cube's shape and the seed and nothing else. Processes standing for
// different users never share a position while there are enough to go round, and say so when there
// are not. And the fingerprint of what an operation read agrees between layouts holding the same
// pixels, so that a layout read wrongly shows up in the CSV rather than as a fast time.
//
// Run with `layouts <dir>` it checks that last one against the generator's datasets, which only
// exist when uv does; without arguments it needs nothing but the committed fixtures.

#include <carta-zarr/carta_zarr.h>

#include "cold.h"
#include "options.h"
#include "record.h"
#include "workload.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

#include "support/check.h"

namespace {

using namespace carta::zarr;
using namespace carta::zarr::bench;
using carta::zarr::testing::Require;

const std::string kWide = CARTA_ZARR_PIXEL_FIXTURE_WIDE;

constexpr Mode kPositionedModes[] = {Mode::plane, Mode::spectrum, Mode::region};

const Context& SharedContext() {
    static const auto context = Context::Create();
    Require(static_cast<bool>(context), "Context::Create failed");
    return *context;
}

Image OpenDefault(const std::string& location) {
    const auto dataset = Dataset::Open(SharedContext(), location);
    Require(dataset.has_value(), location + " did not open");
    const auto image = dataset->OpenImage(dataset->descriptor().default_image_id.value_or(""));
    Require(image.has_value(), location + ": the default image did not open");
    return image.value();
}

CubeAxes Cube(std::uint64_t width, std::uint64_t height, std::uint64_t channels, std::uint64_t polarizations) {
    CubeAxes axes;
    axes.rank = 4;
    axes.x = 0;
    axes.y = 1;
    axes.spectral = 2;
    axes.polarization = 3;
    axes.width = width;
    axes.height = height;
    axes.channels = channels;
    axes.polarizations = polarizations;
    return axes;
}

std::vector<std::string> Positions(const std::vector<Operation>& operations) {
    std::vector<std::string> positions;
    for (const auto& operation : operations) {
        positions.push_back(operation.Describe());
    }
    return positions;
}

// Every operation of a trial, every process's, in process order.
std::vector<Operation> Trial(Mode mode, const CubeAxes& axes, unsigned processes, unsigned ops,
                             std::uint64_t seed = 1, unsigned trial = 0, double region_fraction = 0.05) {
    std::vector<Operation> all;
    for (unsigned process = 0; process < processes; ++process) {
        const auto some = PlanOperations(mode, axes, seed, trial, processes, process, ops, region_fraction);
        all.insert(all.end(), some.begin(), some.end());
    }
    return all;
}

template <typename T>
T Get(const Command& command) {
    Require(std::holds_alternative<T>(command), "the command line parsed as something else");
    return std::get<T>(command);
}

Command Parse(std::vector<const char*> words) {
    words.insert(words.begin(), "carta-zarr-bench");
    return ParseCommandLine(static_cast<int>(words.size()), words.data());
}

void TestSizesArePowersOf1024() {
    Require(ParseSize("4096") == std::size_t{4096}, "a plain number is bytes");
    Require(ParseSize("512K") == std::size_t{512} << 10, "K is 1024");
    Require(ParseSize("64M") == std::size_t{64} << 20, "M is 1024^2");
    Require(ParseSize("64MiB") == std::size_t{64} << 20, "MiB is M");
    Require(ParseSize("64MB") == std::size_t{64} << 20, "MB is M, since these size memory");
    Require(ParseSize("4G") == std::size_t{4} << 30, "G is 1024^3");
    Require(ParseSize("1T") == std::size_t{1} << 40, "T is 1024^4");
    for (const char* bad : {"", "M", "4X", "-1", "1.5G", "99999999999T", "4 G"}) {
        Require(!ParseSize(bad), std::string("accepted a size that is not one: ") + bad);
    }
}

void TestTheCommandLine() {
    const auto defaults = Get<RunOptions>(Parse({"run", "cube.zarr"}));
    Require(defaults.dataset == "cube.zarr" && defaults.modes.size() == 6, "run does not default to every mode");
    Require(defaults.OpsFor(Mode::animation) == 2 && defaults.animation_frames == 32,
            "an animation does not default to two runs of 32 frames");
    Require(defaults.FirstTouchCacheBytes() == std::size_t{1} << 30,
            "a first touch does not get the backend's default cache when the context's is left to TensorStore");
    Require(Get<RunOptions>(Parse({"run", "cube.zarr", "--animation-frames", "8"})).animation_frames == 8,
            "--animation-frames was lost");
    Require(Get<Usage>(Parse({"run", "cube.zarr", "--animation-frames", "0"})).error,
            "an animation of no frames was accepted");
    Require(defaults.trials == 5 && defaults.processes == 1 && !defaults.cold, "run's defaults moved");
    Require(defaults.OpsFor(Mode::spectrum) == 32 && defaults.OpsFor(Mode::region) == 1,
            "operations per trial do not default by mode");
    Require(!defaults.context.cache_bytes, "the cache is not left to TensorStore by default");

    const auto set = Get<RunOptions>(Parse({"run", "--mode=plane,open", "cube.zarr", "--cache-bytes", "0", "--ops",
                                            "3", "--read-budget-bytes=1G", "--cold", "off", "--trial-timeout", "9"}));
    Require(set.modes == std::vector<Mode>{Mode::plane, Mode::open}, "--mode was not read as a list");
    Require(set.context.cache_bytes == std::size_t{0}, "--cache-bytes 0 is not a cache of nothing");
    Require(set.FirstTouchCacheBytes() == 0, "a first touch does not get the context's cache size");
    Require(set.OpsFor(Mode::spectrum) == 3, "--ops does not apply to every mode");
    Require(set.read_budget_bytes == std::size_t{1} << 30 && set.cold == ColdMethod::off, "an option was lost");
    Require(set.trial_timeout == std::chrono::seconds(9), "--trial-timeout was lost");

    const auto shaped = Get<RunOptions>(Parse({"run", "cube.zarr", "--ops", "8,spectrum=64,region=2",
                                               "--region-fraction", "0.2", "--histogram-method", "sampled:8"}));
    Require(shaped.OpsFor(Mode::plane) == 8 && shaped.OpsFor(Mode::spectrum) == 64 &&
                shaped.OpsFor(Mode::region) == 2,
            "--ops does not give a mode its own count over the one for every mode");
    Require(Get<RunOptions>(Parse({"run", "cube.zarr", "--ops", "spectrum=64"})).OpsFor(Mode::plane) == 16,
            "a count for one mode moved another mode off its default");
    Require(shaped.region_fraction == 0.2, "--region-fraction was lost");
    Require(shaped.histogram.kind == HistogramMethod::Kind::sampled && shaped.histogram.stride == 8,
            "--histogram-method sampled:8 was lost");
    Require(defaults.histogram.kind == HistogramMethod::Kind::exact,
            "a cube histogram is not exact by default, as the backend's is");
    for (const char* bad : {"plane=0", "cube=3", "", "spectrum="}) {
        Require(Get<Usage>(Parse({"run", "cube.zarr", "--ops", bad})).error,
                std::string("--ops accepted ") + bad);
    }
    for (const char* bad : {"0", "1.5", "x", "-0.1"}) {
        Require(Get<Usage>(Parse({"run", "cube.zarr", "--region-fraction", bad})).error,
                std::string("--region-fraction accepted ") + bad);
    }

    const auto unknown = Get<Usage>(Parse({"run", "cube.zarr", "--bogus"}));
    Require(unknown.error && unknown.message.find("unknown option") != std::string::npos,
            "an unknown option is not reported as unknown");
    Require(Get<Usage>(Parse({"run", "cube.zarr", "--resume"})).error, "--resume without --csv was accepted");
    Require(Get<Usage>(Parse({"run", "cube.zarr", "--cold", "command"})).error,
            "--cold command without a command was accepted");
    Require(Get<Usage>(Parse({"run", "cube.zarr", "--mode", "plane,cube"})).error, "an unknown mode was accepted");
    Require(Get<Usage>(Parse({"run", "cube.zarr", "--processes", "0"})).error, "no processes was accepted");
    Require(Get<Usage>(Parse({"probe"})).error, "probe without a dataset was accepted");
    Require(!Get<Usage>(Parse({"run", "--help"})).error, "asking for help is not an error");
    Require(Get<ProbeOptions>(Parse({"probe", "cube.zarr", "--image", "SKY"})).image_id == "SKY",
            "probe lost its --image");
}

// Process 0 reads the same positions however many processes there are, so a measurement at one
// user and at sixteen differ in the users and not in what the first of them reads.
void TestProcessZeroReadsTheSameWhateverTheCount() {
    const auto axes = Cube(1000, 900, 300, 2);
    for (const auto mode : kPositionedModes) {
        const auto alone = Positions(PlanOperations(mode, axes, 7, 3, 1, 0, 6));
        const auto among = Positions(PlanOperations(mode, axes, 7, 3, 16, 0, 6));
        Require(alone == among, std::string(ModeName(mode)) + ": process 0 moved when processes were added");
    }
}

void TestPositionsDependOnTheSeedAndTheTrial() {
    const auto axes = Cube(1000, 900, 300, 2);
    for (const auto mode : kPositionedModes) {
        const auto base = Positions(Trial(mode, axes, 2, 4, 1, 0));
        Require(base == Positions(Trial(mode, axes, 2, 4, 1, 0)), "the same seed and trial planned differently");
        Require(base != Positions(Trial(mode, axes, 2, 4, 2, 0)), "the seed does not move the positions");
        Require(base != Positions(Trial(mode, axes, 2, 4, 1, 1)), "every trial reads the same positions");
    }
}

void TestProcessesDoNotShareAPosition() {
    const auto axes = Cube(1000, 900, 300, 2);
    for (const auto mode : {Mode::plane, Mode::spectrum}) {
        const auto all = Trial(mode, axes, 4, DefaultOps(mode));
        const auto positions = Positions(all);
        Require(std::set<std::string>(positions.begin(), positions.end()).size() == all.size(),
                std::string(ModeName(mode)) + ": two operations of a trial share a position");
        for (const auto& operation : all) {
            Require(!operation.overlap, std::string(ModeName(mode)) + ": an overlap was reported where none is");
            Require(operation.x < axes.width && operation.y < axes.height && operation.channel < axes.channels &&
                        operation.polarization < axes.polarizations,
                    std::string(ModeName(mode)) + ": a position is outside the cube");
        }
    }

    // Region boxes: 5% of the plane each, inside it, and apart from each other.
    const auto boxes = Trial(Mode::region, axes, 16, 1);
    for (std::size_t index = 0; index < boxes.size(); ++index) {
        const auto& box = boxes[index];
        const double fraction = static_cast<double>(box.width * box.height) / (axes.width * axes.height);
        Require(std::abs(fraction - 0.05) < 0.002, "a region box does not cover 5% of the plane");
        Require(box.x + box.width <= axes.width && box.y + box.height <= axes.height, "a box leaves the plane");
        Require(!box.overlap, "an overlap was reported among 16 boxes");
        for (std::size_t other = 0; other < index; ++other) {
            const auto& them = boxes[other];
            const bool apart = box.x + box.width <= them.x || them.x + them.width <= box.x ||
                               box.y + box.height <= them.y || them.y + them.height <= box.y;
            Require(apart, "two region boxes of one trial overlap");
        }
    }
}

void TestTheHistogramMethodIsSpelledAsTheBackendSpellsIt() {
    for (const char* spelling : {"exact", "binned", "sampled:4", "sampled:16"}) {
        const auto method = HistogramMethod::Parse(spelling);
        Require(method && method->Spell() == spelling, std::string("did not round-trip ") + spelling);
    }
    Require(HistogramMethod::Parse("sampled")->stride == 4, "a bare sampled is not the backend's stride of 4");
    for (const char* bad : {"", "sampled:0", "sampled:x", "sampledx", "two-pass"}) {
        Require(!HistogramMethod::Parse(bad), std::string("accepted a method there is not: ") + bad);
    }
}

void TestARegionCoversItsFraction() {
    const auto axes = Cube(1000, 900, 30, 1);
    for (const double fraction : {0.01, 0.05, 0.2, 0.5, 1.0}) {
        const auto side = std::sqrt(fraction);
        const auto cells = static_cast<unsigned>(std::floor(1.0 / side));
        const auto boxes = Trial(Mode::region, axes, cells * cells, 1, 1, 0, fraction);
        for (std::size_t index = 0; index < boxes.size(); ++index) {
            const auto& box = boxes[index];
            const double covered = static_cast<double>(box.width * box.height) / (axes.width * axes.height);
            Require(std::abs(covered - fraction) < 0.01 * fraction + 0.002,
                    "a region box does not cover its fraction of the plane");
            Require(box.x + box.width <= axes.width && box.y + box.height <= axes.height, "a box leaves the plane");
            Require(!box.overlap, "boxes that fit the grid were marked as overlapping");
            for (std::size_t other = 0; other < index; ++other) {
                const auto& them = boxes[other];
                Require(box.x + box.width <= them.x || them.x + them.width <= box.x ||
                            box.y + box.height <= them.y || them.y + them.height <= box.y,
                        "two region boxes of one trial overlap");
            }
        }
    }
}

void TestRunningOutOfPositionsIsSaid() {
    // Three planes for four operations: the fourth repeats one, and both of those say so.
    const auto planes = Trial(Mode::plane, Cube(10, 10, 3, 1), 2, 2);
    unsigned overlapping = 0;
    for (const auto& operation : planes) {
        overlapping += operation.overlap ? 1 : 0;
    }
    Require(overlapping == 2, "a repeated plane is not marked on both operations that read it");

    const auto boxes = Trial(Mode::region, Cube(1000, 900, 30, 1), 17, 1);
    Require(boxes.back().overlap, "a seventeenth box in a grid of sixteen is not marked");
}

void TestACubeHistogramSplitsTheChannels() {
    const auto axes = Cube(100, 100, 300, 4);
    const auto all = Trial(Mode::cube_histogram, axes, 3, 1);
    Require(all.size() == 3, "one cube histogram per process was not planned");
    for (unsigned process = 0; process < 3; ++process) {
        Require(all[process].channel == process * 100 && all[process].channel_count == 100,
                "the channels were not split into one run per process");
        Require(!all[process].overlap, "disjoint runs of channels were marked as overlapping");
    }
    const auto several = PlanOperations(Mode::cube_histogram, axes, 1, 0, 1, 0, 5);
    std::set<std::uint64_t> polarizations;
    for (const auto& operation : several) {
        polarizations.insert(operation.polarization);
    }
    Require(polarizations.size() == 4 && several.back().overlap,
            "repeated cube histograms do not take each polarization in turn before repeating one");
    for (const auto& operation : Trial(Mode::cube_histogram, Cube(10, 10, 2, 1), 3, 1)) {
        Require(operation.overlap && operation.channel_count == 1,
                "more processes than channels was not marked as overlapping on every one of them");
    }
}

void TestAFingerprintHasOneNaN() {
    const float quiet = std::numeric_limits<float>::quiet_NaN();
    std::uint32_t bits = 0x7FC12345u;
    float payload = 0.0f;
    std::memcpy(&payload, &bits, sizeof(payload));
    const float left[] = {1.0f, quiet, -2.0f};
    const float right[] = {1.0f, payload, -2.0f};
    Require(Fingerprint(left, 3) == Fingerprint(right, 3), "two NaNs fingerprint differently");
    const float other[] = {1.0f, 0.0f, -2.0f};
    Require(Fingerprint(left, 3) != Fingerprint(other, 3), "a NaN fingerprints as a number");
}

RunOptions SomeOptions() {
    RunOptions options;
    options.dataset = "cube.zarr";
    options.processes = 4;
    return options;
}

void TestTheRunKeyIsTheSettings() {
    const auto options = SomeOptions();
    const DatasetFacts facts{"abc123", "zstd:3", "", "1.500"};
    const auto key = RowTemplate(options, Mode::plane, ColdMethod::fadvise, facts, "run-a").run_key;
    Require(key == RowTemplate(options, Mode::plane, ColdMethod::fadvise, facts, "run-b").run_key,
            "the run key depends on the run id, so nothing could ever resume");
    Require(key == RowTemplate(SomeOptions(), Mode::plane, ColdMethod::fadvise, DatasetFacts{"abc123", "", "", ""},
                               "run-a")
                       .run_key,
            "the run key depends on what the manifest says about bytes rather than on its identity");

    auto threads = options;
    threads.context.decode_threads = 8;
    auto label = options;
    label.label = "validate";
    auto processes = options;
    processes.processes = 1;
    for (const auto& changed : {threads, label, processes}) {
        Require(key != RowTemplate(changed, Mode::plane, ColdMethod::fadvise, facts, "run-a").run_key,
                "a setting that changes the measurement does not change the run key");
    }
    Require(key != RowTemplate(options, Mode::spectrum, ColdMethod::fadvise, facts, "run-a").run_key,
            "the mode does not change the run key");
    Require(key != RowTemplate(options, Mode::plane, ColdMethod::off, facts, "run-a").run_key,
            "warm and cold share a run key");
    // The library's tuning overrides are fixed when it is built, so a test sees only its own build's:
    // that they are written down, and are what the build says.
    Require(RowTemplate(options, Mode::plane, ColdMethod::off, facts, "run-a").tuning == CARTA_ZARR_BENCH_TUNING &&
                CsvHeader().find(",tuning,") != std::string::npos,
            "a row does not say which tuning overrides the library was built with");

    // A setting that shapes one mode moves that mode's key and no other's.
    auto wider = options;
    wider.region_fraction = 0.2;
    auto binned = options;
    binned.histogram.kind = HistogramMethod::Kind::binned;
    Require(key == RowTemplate(wider, Mode::plane, ColdMethod::fadvise, facts, "run-a").run_key &&
                key == RowTemplate(binned, Mode::plane, ColdMethod::fadvise, facts, "run-a").run_key,
            "a setting of another mode changed the plane's run key");
    Require(RowTemplate(options, Mode::region, ColdMethod::fadvise, facts, "run-a").run_key !=
                RowTemplate(wider, Mode::region, ColdMethod::fadvise, facts, "run-a").run_key,
            "the region fraction does not change the region's run key");
    Require(RowTemplate(options, Mode::cube_histogram, ColdMethod::fadvise, facts, "run-a").run_key !=
                RowTemplate(binned, Mode::cube_histogram, ColdMethod::fadvise, facts, "run-a").run_key,
            "the histogram method does not change the cube histogram's run key");
    Require(key != RowTemplate(options, Mode::plane, ColdMethod::fadvise, DatasetFacts{"def456", "", "", ""}, "run-a")
                       .run_key,
            "two datasets share a run key");
}

void TestARowIsOneLine() {
    auto row = RowTemplate(SomeOptions(), Mode::region, ColdMethod::off, {}, "run");
    row.trial = 3;
    row.status = "error";
    row.error = "io_error: \"a\", then\nb";
    row.position = "pol=0;l=1:2;m=3:4";
    row.seconds = 1.25;
    const auto line = FormatRow(row);
    Require(line.find('\n') == std::string::npos, "a row holds a line break");
    const auto fields = ParseCsvLine(line);
    Require(fields.size() == ParseCsvLine(CsvHeader()).size(), "a row does not have one field per column");
    Require(std::find(fields.begin(), fields.end(), "io_error: \"a\", then b") != fields.end(),
            "a quoted field did not survive the round trip");

    const auto summary = SummariseRow(line);
    Require(summary.has_value() && summary->run_key == row.run_key && summary->trial == 3 &&
                summary->status == "error" && summary->seconds == 1.25,
            "a row does not read back as what was written");
}

void TestResumingSkipsOnlyWholeTrials() {
    const auto path =
        (std::filesystem::temp_directory_path() / ("carta-zarr-bench-test-" + std::to_string(getpid()) + ".csv"))
            .string();
    std::filesystem::remove(path);
    const auto base = RowTemplate(SomeOptions(), Mode::plane, ColdMethod::off, {}, "run");
    {
        std::string error;
        auto output = CsvOutput::Open(path, error);
        Require(output.has_value(), "a new CSV did not open: " + error);
        auto ok = base;
        ok.trial = 0;
        ok.status = "ok";
        auto timeout = base;
        timeout.trial = 0;
        timeout.status = "timeout";
        auto failed = base;
        failed.trial = 1;
        failed.status = "error";
        output->Write({FormatRow(ok), FormatRow(timeout), FormatRow(failed)});
    }
    {
        std::string error;
        const auto output = CsvOutput::Open(path, error);
        Require(output.has_value(), "the CSV did not reopen: " + error);
        const auto done = output->completed().find(base.run_key);
        Require(done != output->completed().end() && done->second == std::set<unsigned>{0},
                "resuming does not skip exactly the trials that finished without an error");
    }
    std::ifstream written(path);
    std::string header;
    std::getline(written, header);
    Require(header == CsvHeader(), "the header was not written first");
    std::size_t lines = 1;
    for (std::string line; std::getline(written, line);) {
        ++lines;
    }
    Require(lines == 4, "reopening the CSV wrote another header");

    std::ofstream(path) << "csv_version,something_else\n";
    std::string error;
    Require(!CsvOutput::Open(path, error) && !error.empty(), "a CSV of another version was appended to");
    std::filesystem::remove(path);
}

// An animation plays consecutive channels, and two of one trial never play the same ones while the
// cube has channels enough for both.
void TestAnAnimationPlaysConsecutiveChannels() {
    const auto axes = Cube(100, 90, 300, 2);
    const auto all = Trial(Mode::animation, axes, 2, 2);
    Require(all.size() == 4, "an animation trial did not plan an operation per process per run");
    for (std::size_t one = 0; one < all.size(); ++one) {
        Require(all[one].channel_count == 32 && all[one].channel + 32 <= axes.channels,
                "an animation is not 32 channels inside the cube: " + all[one].Describe());
        Require(!all[one].overlap, "an animation was marked as repeating another with channels to spare");
        for (std::size_t other = 0; other < one; ++other) {
            const auto& a = all[one];
            const auto& b = all[other];
            const bool apart = a.polarization != b.polarization || a.channel + a.channel_count <= b.channel ||
                               b.channel + b.channel_count <= a.channel;
            Require(apart, "two animations of one trial play the same planes: " + a.Describe() + " and " +
                               b.Describe());
        }
    }
    const auto short_cube = PlanOperations(Mode::animation, Cube(100, 90, 10, 1), 1, 0, 1, 0, 1, 0.05, 32);
    Require(short_cube.front().channel == 0 && short_cube.front().channel_count == 10,
            "an animation longer than the cube does not play the whole of it");
    const auto shorter = PlanOperations(Mode::animation, axes, 1, 0, 1, 0, 1, 0.05, 5);
    Require(shorter.front().channel_count == 5, "--animation-frames does not set the frames");
}

// Which operations of a trial read a chunk that another has read or is reading.
void TestSharedChunksAreMarked() {
    const auto axes = Cube(256, 256, 300, 2);
    const auto plane = [](std::uint64_t channel, std::uint64_t polarization = 0) {
        Operation operation;
        operation.mode = Mode::plane;
        operation.channel = channel;
        operation.polarization = polarization;
        return operation;
    };
    const auto marks = [&](std::vector<std::vector<Operation>> plans, std::vector<std::uint64_t> chunk) {
        MarkSharedChunks(plans, axes, chunk);
        std::vector<bool> marked;
        for (const auto& plan : plans) {
            for (const auto& operation : plan) {
                marked.push_back(operation.shares_chunks);
            }
        }
        return marked;
    };
    using Marks = std::vector<bool>;

    Require(marks({{plane(3), plane(10), plane(20)}}, {256, 256, 1, 1}) == Marks{false, false, false},
            "planes of a layout one channel deep were marked as sharing chunks");
    Require(marks({{plane(3), plane(10), plane(20)}}, {256, 256, 16, 1}) == Marks{false, true, false},
            "only the later of two planes in one chunk's depth is to be marked");
    Require(marks({{plane(3)}, {plane(10)}}, {256, 256, 16, 1}) == Marks{true, true},
            "planes of two processes in one chunk's depth are not both marked");
    Require(marks({{plane(3), plane(3, 1)}}, {256, 256, 16, 1}) == Marks{false, false},
            "planes of different polarizations were marked as sharing chunks");
    Require(marks({{plane(3), plane(3, 1)}}, {256, 256, 16, 2}) == Marks{false, true},
            "polarizations in one chunk were not marked as sharing it");

    const auto spectrum = [](std::uint64_t x, std::uint64_t y) {
        Operation operation;
        operation.mode = Mode::spectrum;
        operation.x = x;
        operation.y = y;
        return operation;
    };
    Require(marks({{spectrum(10, 10), spectrum(50, 50), spectrum(100, 10)}}, {64, 64, 300, 1}) ==
                Marks{false, true, false},
            "spectra in one spatial chunk were not marked, or ones in different chunks were");

    // Each process's run of channels ends inside a chunk the next one's begins in.
    const auto slabs = [&](std::uint64_t depth) {
        std::vector<std::vector<Operation>> plans;
        for (unsigned process = 0; process < 2; ++process) {
            plans.push_back(PlanOperations(Mode::cube_histogram, axes, 1, 0, 2, process, 1));
        }
        MarkSharedChunks(plans, axes, {64, 64, depth, 1});
        return plans[0][0].shares_chunks && plans[1][0].shares_chunks;
    };
    Require(slabs(16), "cube histograms whose runs meet inside a chunk were not marked");
    Require(!slabs(1), "cube histograms of disjoint runs of channels were marked");

    auto animations = std::vector<std::vector<Operation>>{Trial(Mode::animation, axes, 1, 2)};
    animations[0].push_back(animations[0].front());
    MarkSharedChunks(animations, axes, {256, 256, 300, 2});
    for (const auto& operation : animations[0]) {
        Require(!operation.shares_chunks, "an animation was marked: it reuses chunks on purpose");
    }
}

// What this process has asked the kernel to read, page cache or not, as /proc/self/io counts it.
// Linux only.
std::optional<std::uint64_t> BytesAskedFor() {
    std::ifstream io("/proc/self/io");
    std::string key;
    std::uint64_t value = 0;
    while (io >> key >> value) {
        if (key == "rchar:") {
            return value;
        }
    }
    return std::nullopt;
}

// A plane or a spectrum reads its chunks again however recently they were read, where a read through
// the context's cache only asks storage whether they changed. Told apart by the bytes the process
// reads, which only Linux counts. Not by making the chunks unreadable or removing them: TensorStore
// opens a chunk to check it is unchanged before it uses the cached one, so both reads would fail.
void TestAFirstTouchReadsItsChunksAgain() {
    if (!BytesAskedFor()) {
        std::cerr << "skipped: a first touch is told by /proc/self/io, which this system does not have\n";
        return;
    }
    ContextOptions caching;
    caching.cache_bytes = std::size_t{64} << 20;
    const auto context = Context::Create(caching);
    Require(context.has_value(), "Context::Create failed");
    const auto dataset = Dataset::Open(*context, kWide);
    Require(dataset.has_value(), "the wide fixture did not open");
    const auto image = dataset->OpenImage(dataset->descriptor().default_image_id.value_or(""));
    Require(image.has_value(), "the wide fixture's image did not open");
    const auto axes = CubeAxes::Of(image->descriptor());

    const auto read_twice = [&](Runner& runner, const Operation& operation) {
        Require(runner.Prepare(operation).has_value() && runner.Run(operation, {}).has_value(), "a read failed");
        Require(runner.Prepare(operation).has_value(), "a fresh pool could not be made");
        const auto before = *BytesAskedFor();
        Require(runner.Run(operation, {}).has_value(), "a read failed the second time");
        return *BytesAskedFor() - before;
    };
    Runner fresh(*context, *image, {}, std::size_t{64} << 20);
    Runner shared(*context, *image);
    constexpr std::uint64_t kSome = 4096;
    for (const auto mode : {Mode::plane, Mode::spectrum}) {
        const auto operation = PlanOperations(mode, *axes, 1, 0, 1, 0, 1).front();
        Require(read_twice(fresh, operation) >= kSome,
                std::string(ModeName(mode)) + " found its chunks in a cache rather than reading them");
    }
    // An animation of one frame reads the plane through the context's cache.
    auto frame = PlanOperations(Mode::plane, *axes, 1, 0, 1, 0, 1).front();
    frame.mode = Mode::animation;
    frame.channel_count = 1;
    Require(read_twice(shared, frame) < kSome,
            "the shared cache read the plane's chunks again, so this test shows nothing about a fresh pool");
}

// Each mode against a real image, through the library: the elements an operation reports, and a
// fingerprint of a plane that matches reading the plane directly.
void TestEachModeReads() {
    const auto image = OpenDefault(kWide);
    const auto axes = CubeAxes::Of(image.descriptor());
    Require(axes.has_value(), "the wide fixture is not a cube");
    const std::uint64_t plane = axes->width * axes->height;
    Runner runner(SharedContext(), image);

    const auto read = [&](Mode mode, unsigned processes = 1) {
        const auto operation = PlanOperations(mode, *axes, 1, 0, processes, 0, 1).front();
        const auto elements = runner.Run(operation, {});
        Require(elements.has_value(), std::string(ModeName(mode)) + " failed: " +
                                          (elements ? std::string() : elements.error().message));
        return std::make_pair(operation, *elements);
    };

    const auto [planar, plane_elements] = read(Mode::plane);
    Require(plane_elements == plane, "a plane read is not a plane");
    ReadRequest request;
    request.axes.assign(axes->rank, Range{0, 1, 1});
    request.axes[axes->x] = {0, axes->width, 1};
    request.axes[axes->y] = {0, axes->height, 1};
    request.axes[axes->spectral] = {planar.channel, 1, 1};
    request.axes[*axes->polarization] = {planar.polarization, 1, 1};
    std::vector<float> direct(plane);
    Require(image.Read(request, {direct.data(), direct.size()}).has_value(), "the plane did not read directly");
    Require(runner.Fingerprint() == Fingerprint(direct.data(), direct.size()),
            "the plane's fingerprint is not the fingerprint of the plane");

    Require(read(Mode::spectrum).second == axes->channels, "a spectrum is not every channel");

    // An animation's fingerprint is its frames', each as a plane read alone fingerprints, in order.
    const auto animation = PlanOperations(Mode::animation, *axes, 1, 0, 1, 0, 1, 0.05, 3).front();
    const auto played = runner.Run(animation, {});
    Require(played.has_value() && *played == 3 * plane, "an animation did not read three planes");
    const auto frames = runner.Fingerprint();
    std::uint64_t expected = 0xCBF29CE484222325ull;
    Operation frame = animation;
    frame.mode = Mode::plane;
    for (std::uint64_t index = 0; index < 3; ++index) {
        frame.channel = animation.channel + index;
        Require(runner.Run(frame, {}).has_value(), "a frame did not read alone");
        const auto one = runner.Fingerprint();
        for (unsigned byte = 0; byte < 8; ++byte) {
            expected ^= (one >> (8 * byte)) & 0xFFu;
            expected *= 0x100000001B3ull;
        }
    }
    Require(frames == expected, "an animation did not read the planes from its channel on, in order");
    const auto [box, region_elements] = read(Mode::region);
    Require(region_elements == box.width * box.height * axes->channels, "a region did not cover its box");
    const auto region = runner.Fingerprint();
    Require(read(Mode::cube_histogram, 2).second == plane * (axes->channels / 2),
            "a cube histogram did not cover its share of the channels");
    const auto exact = runner.Fingerprint();
    Require(exact != region, "a cube histogram fingerprinted as the region before it");

    // One pass over the same channels: its extremes and pixel count are the exact ones, but it carries
    // no counts into the fingerprint, so it cannot fingerprint as the two-pass histogram does.
    Runner one_pass(SharedContext(), image, *HistogramMethod::Parse("binned"));
    const auto histogram = PlanOperations(Mode::cube_histogram, *axes, 1, 0, 2, 0, 1).front();
    Require(one_pass.Run(histogram, {}).has_value(), "a one-pass cube histogram failed");
    Require(one_pass.Fingerprint() != exact, "a one-pass histogram fingerprinted as an exact one");

    Runner opener(ContextOptions{}, kWide, "");
    Require(opener.Run(Operation{Mode::open}, {}).has_value() && opener.image().has_value(),
            "open did not open the default image");
}

void TestTheColdMethodIsChosenOrRefused() {
    Require(ChooseColdMethod(std::nullopt, "true").method == ColdMethod::command,
            "auto does not prefer the administrator's command");
    Require(ChooseColdMethod(ColdMethod::off, "").error.empty(), "off was refused");
    if (geteuid() != 0) {
        Require(!ChooseColdMethod(ColdMethod::drop_caches, "").error.empty(),
                "drop-caches was accepted without root");
    }
    Require(DropCaches(ColdMethod::command, "true", "").empty(), "a command that succeeded was reported failing");
    Require(!DropCaches(ColdMethod::command, "false", "").empty(), "a command that failed was reported working");
}

// The generator's layouts of one cube read back the same at every position of every mode.
void TestLayoutsFingerprintAlike(const std::string& generated) {
    const auto plain = OpenDefault(generated + "/plain");
    const auto axes = CubeAxes::Of(plain.descriptor());
    Require(axes.has_value(), "the plain layout is not a cube");
    for (const char* layout : {"sharded", "flagged"}) {
        for (const char* method : {"exact", "binned", "sampled:2"}) {
            const auto histogram = *HistogramMethod::Parse(method);
            Runner left(SharedContext(), plain, histogram);
            Runner right(SharedContext(), OpenDefault(generated + "/" + layout), histogram);
            for (const auto mode : {Mode::plane, Mode::spectrum, Mode::region, Mode::cube_histogram}) {
                for (const auto& operation : Trial(mode, *axes, 3, mode == Mode::cube_histogram ? 2 : 3)) {
                    const auto where = std::string(layout) + ", " + method + ": " + ModeName(mode) + " at " +
                                       operation.Describe();
                    Require(left.Run(operation, {}).has_value() && right.Run(operation, {}).has_value(),
                            where + " failed");
                    Require(left.Fingerprint() == right.Fingerprint(), where + " reads differently from plain");
                }
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "layouts") {
            TestLayoutsFingerprintAlike(argv[2]);
            return 0;
        }
        TestSizesArePowersOf1024();
        TestTheCommandLine();
        TestProcessZeroReadsTheSameWhateverTheCount();
        TestPositionsDependOnTheSeedAndTheTrial();
        TestProcessesDoNotShareAPosition();
        TestTheHistogramMethodIsSpelledAsTheBackendSpellsIt();
        TestARegionCoversItsFraction();
        TestRunningOutOfPositionsIsSaid();
        TestACubeHistogramSplitsTheChannels();
        TestAnAnimationPlaysConsecutiveChannels();
        TestSharedChunksAreMarked();
        TestAFingerprintHasOneNaN();
        TestTheRunKeyIsTheSettings();
        TestARowIsOneLine();
        TestResumingSkipsOnlyWholeTrials();
        TestEachModeReads();
        TestAFirstTouchReadsItsChunksAgain();
        TestTheColdMethodIsChosenOrRefused();
    } catch (const std::exception& error) {
        std::cerr << "bench: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
