/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_BENCH_OPTIONS_H_
#define CARTA_ZARR_BENCH_OPTIONS_H_

// What carta-zarr-bench was asked to do, parsed from its command line.

#include <carta-zarr/descriptor.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace carta::zarr::bench {

// The ways a CARTA user reads a cube, each measured on its own.
//
// plane and spectrum time a first touch: each reads through a cache pool of its own, made before the
// clock starts, so that what an earlier operation decoded never answers a later one. A cube's chunks
// span many channels or many pixels, and with a shared cache most of a trial's planes would be found
// already decoded by the one before -- a median of cache hits, which says nothing about the layout
// except that its chunks are deep. animation is where that reuse is the point, and measures it.
//
// The order is the one positions are drawn under, so a new mode goes at the end.
enum class Mode {
    plane,           // one whole l x m plane at a random channel
    spectrum,        // every channel at one random pixel
    region,          // ReduceSpectral over a box covering 5% of the plane
    cube_histogram,  // ComputeCubeHistogram over this process's share of the channels
    open,            // Context::Create, Dataset::Open and OpenImage, with nothing cached
    animation,       // consecutive planes from a random channel at a frame rate, through the shared cache
};

const char* ModeName(Mode mode) noexcept;
std::optional<Mode> ParseMode(std::string_view name) noexcept;

// How many operations one process makes in one trial, when --ops does not say: as many as keep a
// trial of each mode about as long as the others on a cube of tens of gigabytes.
unsigned DefaultOps(Mode mode) noexcept;

// How the caches between a trial and the storage are emptied before it.
enum class ColdMethod {
    command,      // the administrator's --drop-cache-cmd, the only one that can reach a server's cache
    drop_caches,  // /proc/sys/vm/drop_caches, which needs root and Linux
    fadvise,      // posix_fadvise(DONTNEED) on every file of the dataset, which needs Linux
    off,          // nothing: a warm measurement
};

const char* ColdMethodName(ColdMethod method) noexcept;
std::optional<ColdMethod> ParseColdMethod(std::string_view name) noexcept;

// How a cube histogram is computed, as carta-backend's --zarr_histogram_method names the choices.
//
// exact is the backend's default, and so the bench's: two passes, ReduceSpectral over whole planes
// for the range and then ComputeHistogram over it, reading the cube twice. binned and sampled are
// ComputeCubeHistogram's one pass, measured for reference rather than recommended -- they move where
// the bin edges land, which is a question of accuracy and not one a timing can settle.
struct HistogramMethod {
    enum class Kind { exact, binned, sampled };
    Kind kind = Kind::exact;
    // Every nth pixel along both spatial axes, for sampled.
    std::uint64_t stride = 1;

    // As the backend spells it: exact, binned, or sampled:N.
    std::string Spell() const;
    static std::optional<HistogramMethod> Parse(std::string_view text) noexcept;
};

struct RunOptions {
    std::string dataset;
    // Empty for the dataset's default image.
    std::string image_id;
    std::vector<Mode> modes;
    unsigned trials = 5;
    // Operations per process per trial: a mode's own count, then the one for every mode, then
    // DefaultOps.
    std::optional<unsigned> ops;
    std::map<Mode, unsigned> mode_ops;
    // The share of the plane one region box covers.
    double region_fraction = 0.05;
    // The planes one animation operation reads, one after another.
    unsigned animation_frames = 32;
    // Frames a second an animation is played at, each frame read no earlier than its turn; 0 reads
    // them back to back. CARTA's animator plays at 5 unless the user changes it.
    double animation_fps = 5.0;
    // Whether an animation reads the next run of chunks along the spectrum in the background as soon
    // as it enters one, as a backend that prefetched would.
    bool animation_prefetch = false;
    HistogramMethod histogram;
    std::uint64_t seed = 1;
    ContextOptions context;
    std::size_t read_budget_bytes = 0;
    unsigned processes = 1;
    // Unset for auto: the command when there is one, then drop_caches, then fadvise, then off.
    std::optional<ColdMethod> cold;
    std::string drop_cache_command;
    std::chrono::seconds trial_timeout{600};
    // Empty for stdout.
    std::string csv_path;
    std::string label;
    bool resume = false;

    // The cache pool plane and spectrum read each operation through: the size of the context's own,
    // and carta-backend's default --zarr_cache_size of 1 GiB when that is left to the library. Not
    // none, even then: a sharded layout keeps its shard index in the pool, and one that kept nothing
    // would read the index again for every chunk of a single read.
    std::size_t FirstTouchCacheBytes() const noexcept {
        return context.cache_bytes.value_or(std::size_t{1} << 30);
    }

    unsigned OpsFor(Mode mode) const noexcept {
        if (const auto own = mode_ops.find(mode); own != mode_ops.end()) {
            return own->second;
        }
        return ops.value_or(DefaultOps(mode));
    }
};

struct ProbeOptions {
    std::string dataset;
    std::string image_id;
};

// Asked for help, or asked wrongly. `error` says which: help goes to stdout and exits 0.
struct Usage {
    std::string message;
    bool error = false;
};

using Command = std::variant<RunOptions, ProbeOptions, Usage>;

Command ParseCommandLine(int argc, const char* const* argv);

// A size in bytes, with an optional binary suffix: 4096, 512K, 64M, 4G, 1T, and the same with iB or
// B after them. Every multiple is a power of 1024, because what these size is memory.
std::optional<std::size_t> ParseSize(std::string_view text) noexcept;

}  // namespace carta::zarr::bench

#endif  // CARTA_ZARR_BENCH_OPTIONS_H_
