/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "options.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <utility>

namespace carta::zarr::bench {

namespace {

// In the order a run measures them when --mode does not say.
constexpr std::array<std::pair<Mode, std::string_view>, 6> kModes{{
    {Mode::plane, "plane"},
    {Mode::animation, "animation"},
    {Mode::spectrum, "spectrum"},
    {Mode::region, "region"},
    {Mode::cube_histogram, "cube-histogram"},
    {Mode::open, "open"},
}};

constexpr std::array<std::pair<ColdMethod, std::string_view>, 4> kColdMethods{{
    {ColdMethod::command, "command"},
    {ColdMethod::drop_caches, "drop-caches"},
    {ColdMethod::fadvise, "fadvise"},
    {ColdMethod::off, "off"},
}};

// The options of run that take a value, so that one it does not know is reported as unknown rather
// than as missing its value.
constexpr std::array<std::string_view, 19> kRunOptions{
    "--image", "--mode", "--trials", "--ops", "--seed",
    "--io-threads", "--decode-threads", "--cache-bytes", "--read-budget-bytes", "--processes",
    "--cold", "--drop-cache-cmd", "--trial-timeout", "--csv", "--label",
    "--region-fraction", "--histogram-method", "--animation-frames", "--animation-fps",
};

constexpr std::string_view kUsage = R"(usage:
  carta-zarr-bench run <dataset> [options]
  carta-zarr-bench probe <dataset> [--image ID]

probe opens the dataset as carta-backend would and prints what the library sees, as one line of JSON.
It exits non-zero when the dataset does not open.

run measures reads of the dataset and writes one CSV row per operation. plane and spectrum read
each operation through a cache of its own, so that each is a first touch; animation plays frames at a
frame rate through the shared one, as CARTA's animator does, and records which frames were late.

  --image ID                 the image to read; the dataset's default image otherwise
  --mode LIST                plane,animation,spectrum,region,cube-histogram,open (all by default)
  --trials N                 trials per mode (5)
  --ops LIST                 operations per process per trial: N for every mode, MODE=N for one,
                             or both, as in 8,spectrum=64 (plane 16, animation 2, spectrum 32,
                             region 1, cube-histogram 1, open 8)
  --animation-frames N       consecutive planes one animation operation reads (32)
  --animation-fps F          play animations at F frames a second, as CARTA's animator does; 0 reads
                             the frames back to back (5)
  --animation-prefetch       read the next run of chunks along the spectrum in the background, as a
                             backend that prefetched would
  --region-fraction F        the share of the plane a region box covers (0.05)
  --histogram-method METHOD  exact, binned or sampled:N, as the backend's --zarr_histogram_method (exact)
  --seed N                   where the random positions come from (1)
  --io-threads N             ContextOptions::io_threads, the backend's --zarr_file_io_threads (0)
  --decode-threads N         ContextOptions::decode_threads, the backend's --zarr_data_copy_threads (0)
  --cache-bytes SIZE         ContextOptions::cache_bytes, the backend's --zarr_cache_size:
                             default, 0 for none, or a size; also the size of the cache each plane
                             and spectrum operation gets, 1G for default
  --read-budget-bytes SIZE   ReadOptions::read_budget_bytes (0, the library's own)
  --processes N              users reading at once, one process each (1)
  --cold METHOD              auto, command, drop-caches, fadvise or off (auto)
  --drop-cache-cmd CMD       a shell command that empties every cache between here and the disks
  --trial-timeout SECONDS    a trial's deadline (600)
  --csv PATH                 append rows here rather than to stdout
  --label TEXT               carried into every row, for telling runs apart
  --resume                   skip trials the CSV already holds

Sizes take K, M, G or T, each a power of 1024.
)";

template <typename T, std::size_t N>
std::optional<T> Lookup(const std::array<std::pair<T, std::string_view>, N>& table, std::string_view name) {
    for (const auto& [value, spelling] : table) {
        if (spelling == name) {
            return value;
        }
    }
    return std::nullopt;
}

template <typename T, std::size_t N>
const char* Spell(const std::array<std::pair<T, std::string_view>, N>& table, T value) {
    for (const auto& [candidate, spelling] : table) {
        if (candidate == value) {
            return spelling.data();
        }
    }
    return "unknown";
}

template <typename T>
std::optional<T> ParseNumber(std::string_view text) {
    T value{};
    const auto* end = text.data() + text.size();
    const auto [stop, error] = std::from_chars(text.data(), end, value);
    if (error != std::errc() || stop != end || text.empty()) {
        return std::nullopt;
    }
    return value;
}

Usage Wrong(std::string message) {
    return Usage{std::move(message) + "\n\n" + std::string(kUsage), true};
}

// The value of an option, whether it came as --name=value or as --name value.
struct Arguments {
    std::vector<std::string_view> words;
    std::size_t next = 0;

    bool Done() const {
        return next >= words.size();
    }

    std::optional<std::string_view> Value(std::string_view& word) {
        if (const auto equals = word.find('='); equals != std::string_view::npos) {
            auto value = word.substr(equals + 1);
            word = word.substr(0, equals);
            return value;
        }
        if (Done()) {
            return std::nullopt;
        }
        return words[next++];
    }
};

// Comma-separated entries, each a count for every mode or MODE=count for one.
bool ParseOps(std::string_view list, RunOptions& options) {
    if (list.empty()) {
        return false;
    }
    while (!list.empty()) {
        const auto comma = list.find(',');
        const auto entry = list.substr(0, comma);
        list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
        const auto equals = entry.find('=');
        const auto count = ParseNumber<unsigned>(equals == std::string_view::npos ? entry : entry.substr(equals + 1));
        if (!count || *count == 0) {
            return false;
        }
        if (equals == std::string_view::npos) {
            options.ops = *count;
            continue;
        }
        const auto mode = ParseMode(entry.substr(0, equals));
        if (!mode) {
            return false;
        }
        options.mode_ops[*mode] = *count;
    }
    return true;
}

std::optional<std::vector<Mode>> ParseModes(std::string_view list) {
    std::vector<Mode> modes;
    while (!list.empty()) {
        const auto comma = list.find(',');
        const auto name = list.substr(0, comma);
        const auto mode = ParseMode(name);
        if (!mode) {
            return std::nullopt;
        }
        modes.push_back(*mode);
        list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
    }
    if (modes.empty()) {
        return std::nullopt;
    }
    return modes;
}

Command ParseRun(Arguments& arguments) {
    RunOptions options;
    for (const auto& [mode, name] : kModes) {
        options.modes.push_back(mode);
    }
    bool have_dataset = false;

    while (!arguments.Done()) {
        std::string_view word = arguments.words[arguments.next++];
        if (word.substr(0, 2) != "--") {
            if (have_dataset) {
                return Wrong("more than one dataset: " + std::string(word));
            }
            options.dataset = std::string(word);
            have_dataset = true;
            continue;
        }
        if (word == "--resume") {
            options.resume = true;
            continue;
        }
        if (word == "--animation-prefetch") {
            options.animation_prefetch = true;
            continue;
        }

        const auto value = arguments.Value(word);
        if (std::find(kRunOptions.begin(), kRunOptions.end(), word) == kRunOptions.end()) {
            return Wrong("unknown option for run: " + std::string(word));
        }
        if (!value) {
            return Wrong(std::string(word) + " needs a value");
        }
        const auto bad = [&] { return Wrong("bad value for " + std::string(word) + ": " + std::string(*value)); };
        const auto count = [&](auto& into, auto least) {
            using T = std::decay_t<decltype(into)>;
            const auto parsed = ParseNumber<T>(*value);
            if (!parsed || *parsed < static_cast<T>(least)) {
                return false;
            }
            into = *parsed;
            return true;
        };

        if (word == "--image") {
            options.image_id = std::string(*value);
        } else if (word == "--mode") {
            const auto modes = ParseModes(*value);
            if (!modes) {
                return bad();
            }
            options.modes = *modes;
        } else if (word == "--trials") {
            if (!count(options.trials, 1)) {
                return bad();
            }
        } else if (word == "--ops") {
            if (!ParseOps(*value, options)) {
                return bad();
            }
        } else if (word == "--animation-frames") {
            if (!count(options.animation_frames, 1)) {
                return bad();
            }
        } else if (word == "--animation-fps") {
            char* end = nullptr;
            const std::string text(*value);
            const double fps = std::strtod(text.c_str(), &end);
            if (text.empty() || end != text.c_str() + text.size() || !(fps >= 0.0)) {
                return bad();
            }
            options.animation_fps = fps;
        } else if (word == "--region-fraction") {
            char* end = nullptr;
            const std::string text(*value);
            const double fraction = std::strtod(text.c_str(), &end);
            if (text.empty() || end != text.c_str() + text.size() || !(fraction > 0.0 && fraction <= 1.0)) {
                return bad();
            }
            options.region_fraction = fraction;
        } else if (word == "--histogram-method") {
            const auto method = HistogramMethod::Parse(*value);
            if (!method) {
                return bad();
            }
            options.histogram = *method;
        } else if (word == "--seed") {
            if (!count(options.seed, 0)) {
                return bad();
            }
        } else if (word == "--io-threads") {
            if (!count(options.context.io_threads, 0)) {
                return bad();
            }
        } else if (word == "--decode-threads") {
            if (!count(options.context.decode_threads, 0)) {
                return bad();
            }
        } else if (word == "--cache-bytes") {
            if (*value == "default") {
                options.context.cache_bytes.reset();
            } else if (const auto bytes = ParseSize(*value)) {
                options.context.cache_bytes = *bytes;
            } else {
                return bad();
            }
        } else if (word == "--read-budget-bytes") {
            const auto bytes = ParseSize(*value);
            if (!bytes) {
                return bad();
            }
            options.read_budget_bytes = *bytes;
        } else if (word == "--processes") {
            if (!count(options.processes, 1)) {
                return bad();
            }
        } else if (word == "--cold") {
            if (*value == "auto") {
                options.cold.reset();
            } else if (const auto method = ParseColdMethod(*value)) {
                options.cold = *method;
            } else {
                return bad();
            }
        } else if (word == "--drop-cache-cmd") {
            options.drop_cache_command = std::string(*value);
        } else if (word == "--trial-timeout") {
            unsigned seconds = 0;
            if (!count(seconds, 1)) {
                return bad();
            }
            options.trial_timeout = std::chrono::seconds(seconds);
        } else if (word == "--csv") {
            options.csv_path = std::string(*value);
        } else if (word == "--label") {
            options.label = std::string(*value);
        }
    }

    if (!have_dataset) {
        return Wrong("run needs a dataset");
    }
    if (options.cold == ColdMethod::command && options.drop_cache_command.empty()) {
        return Wrong("--cold command needs --drop-cache-cmd");
    }
    if (options.resume && options.csv_path.empty()) {
        return Wrong("--resume needs --csv: there is nothing to resume from on stdout");
    }
    return options;
}

Command ParseProbe(Arguments& arguments) {
    ProbeOptions options;
    bool have_dataset = false;
    while (!arguments.Done()) {
        std::string_view word = arguments.words[arguments.next++];
        if (word.substr(0, 2) != "--") {
            if (have_dataset) {
                return Wrong("more than one dataset: " + std::string(word));
            }
            options.dataset = std::string(word);
            have_dataset = true;
            continue;
        }
        const auto value = arguments.Value(word);
        if (word != "--image") {
            return Wrong("unknown option for probe: " + std::string(word));
        }
        if (!value) {
            return Wrong("--image needs a value");
        }
        options.image_id = std::string(*value);
    }
    if (!have_dataset) {
        return Wrong("probe needs a dataset");
    }
    return options;
}

}  // namespace

const char* ModeName(Mode mode) noexcept {
    return Spell(kModes, mode);
}

std::optional<Mode> ParseMode(std::string_view name) noexcept {
    return Lookup(kModes, name);
}

unsigned DefaultOps(Mode mode) noexcept {
    switch (mode) {
        case Mode::plane:
            return 16;
        case Mode::spectrum:
            return 32;
        case Mode::region:
        case Mode::cube_histogram:
            return 1;
        case Mode::open:
            return 8;
        case Mode::animation:
            return 2;
    }
    return 1;
}

const char* ColdMethodName(ColdMethod method) noexcept {
    return Spell(kColdMethods, method);
}

std::optional<ColdMethod> ParseColdMethod(std::string_view name) noexcept {
    return Lookup(kColdMethods, name);
}

std::string HistogramMethod::Spell() const {
    switch (kind) {
        case Kind::exact:
            return "exact";
        case Kind::binned:
            return "binned";
        case Kind::sampled:
            return "sampled:" + std::to_string(stride);
    }
    return "exact";
}

std::optional<HistogramMethod> HistogramMethod::Parse(std::string_view text) noexcept {
    HistogramMethod method;
    if (text == "exact") {
        return method;
    }
    if (text == "binned") {
        method.kind = Kind::binned;
        return method;
    }
    // The backend's own default stride for a bare "sampled" is 4.
    constexpr std::string_view kSampled = "sampled";
    if (text.substr(0, kSampled.size()) != kSampled) {
        return std::nullopt;
    }
    method.kind = Kind::sampled;
    method.stride = 4;
    const auto rest = text.substr(kSampled.size());
    if (rest.empty()) {
        return method;
    }
    const auto stride = rest.front() == ':' ? ParseNumber<std::uint64_t>(rest.substr(1)) : std::nullopt;
    if (!stride || *stride == 0) {
        return std::nullopt;
    }
    method.stride = *stride;
    return method;
}

std::optional<std::size_t> ParseSize(std::string_view text) noexcept {
    std::size_t digits = 0;
    while (digits < text.size() && text[digits] >= '0' && text[digits] <= '9') {
        ++digits;
    }
    const auto number = ParseNumber<std::size_t>(text.substr(0, digits));
    if (!number) {
        return std::nullopt;
    }

    auto suffix = text.substr(digits);
    for (const std::string_view unit : {"iB", "B"}) {
        if (suffix.size() > unit.size() && suffix.substr(suffix.size() - unit.size()) == unit) {
            suffix.remove_suffix(unit.size());
            break;
        }
    }
    unsigned shift = 0;
    if (suffix.empty() || suffix == "B") {
        shift = 0;
    } else if (suffix == "K" || suffix == "k") {
        shift = 10;
    } else if (suffix == "M") {
        shift = 20;
    } else if (suffix == "G") {
        shift = 30;
    } else if (suffix == "T") {
        shift = 40;
    } else {
        return std::nullopt;
    }
    if (*number > (std::numeric_limits<std::size_t>::max() >> shift)) {
        return std::nullopt;
    }
    return *number << shift;
}

Command ParseCommandLine(int argc, const char* const* argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        arguments.words.emplace_back(argv[index]);
    }
    if (arguments.Done()) {
        return Wrong("no command");
    }
    const auto command = arguments.words[arguments.next++];
    if (command == "-h" || command == "--help" || command == "help") {
        return Usage{std::string(kUsage), false};
    }
    for (const auto word : arguments.words) {
        if (word == "-h" || word == "--help") {
            return Usage{std::string(kUsage), false};
        }
    }
    if (command == "run") {
        return ParseRun(arguments);
    }
    if (command == "probe") {
        return ParseProbe(arguments);
    }
    return Wrong("unknown command: " + std::string(command));
}

}  // namespace carta::zarr::bench
