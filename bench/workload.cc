/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "workload.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

namespace carta::zarr::bench {

namespace {

// SplitMix64: small, fast, and the same sequence everywhere.
class Stream {
public:
    Stream(std::uint64_t seed, Mode mode, unsigned trial, std::uint64_t salt)
        : _state(seed * 0x9E3779B97F4A7C15ull + (static_cast<std::uint64_t>(mode) << 48) +
                 (static_cast<std::uint64_t>(trial) << 16) + salt) {}

    std::uint64_t Next() {
        std::uint64_t z = (_state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // Uniform in [0, bound), by the multiply-shift of Lemire without its rejection step: the bias is
    // bound / 2^64, far below anything a benchmark could notice.
    std::uint64_t Below(std::uint64_t bound) {
        if (bound <= 1) {
            return 0;
        }
        return static_cast<std::uint64_t>((static_cast<unsigned __int128>(Next()) * bound) >> 64);
    }

private:
    std::uint64_t _state;
};

struct Draw {
    std::uint64_t value = 0;
    bool overlap = false;
};

// `count` values from [0, pool), distinct while the pool lasts and repeating it after. Each prefix
// of the result is the same whatever `count` is, which is what keeps process 0's positions fixed as
// processes are added.
std::vector<Draw> DistinctSample(Stream& stream, std::uint64_t pool, std::uint64_t count) {
    std::vector<Draw> draws(count);
    if (pool == 0) {
        return draws;
    }
    constexpr std::uint64_t kMaterialize = std::uint64_t{1} << 20;
    if (pool <= kMaterialize || count > pool / 2) {
        // A partial Fisher-Yates shuffle, which only ever touches the first min(count, pool) slots.
        std::vector<std::uint64_t> values(pool);
        std::iota(values.begin(), values.end(), std::uint64_t{0});
        const auto distinct = std::min(count, pool);
        for (std::uint64_t index = 0; index < distinct; ++index) {
            std::swap(values[index], values[index + stream.Below(pool - index)]);
        }
        const auto repeats = count > pool ? count - pool : 0;
        for (std::uint64_t index = 0; index < count; ++index) {
            draws[index] = {values[index % pool], index < repeats || index >= pool};
        }
        return draws;
    }
    std::unordered_set<std::uint64_t> taken;
    for (auto& draw : draws) {
        do {
            draw.value = stream.Below(pool);
        } while (!taken.insert(draw.value).second);
    }
    return draws;
}

constexpr std::uint64_t kQuietNaN32 = 0x7FC00000u;
constexpr std::uint64_t kQuietNaN64 = 0x7FF8000000000000ull;
constexpr std::uint64_t kFnvOffset = 0xCBF29CE484222325ull;
constexpr std::uint64_t kFnvPrime = 0x100000001B3ull;

void Mix(std::uint64_t& hash, std::uint64_t bits, unsigned bytes) {
    for (unsigned byte = 0; byte < bytes; ++byte) {
        hash ^= (bits >> (8 * byte)) & 0xFFu;
        hash *= kFnvPrime;
    }
}

// CARTA's automatic bin count for a plane: the square root of its pixel count, and at least two.
std::uint32_t AutomaticBins(const CubeAxes& axes) {
    const auto bins = std::sqrt(static_cast<double>(axes.width) * static_cast<double>(axes.height));
    return static_cast<std::uint32_t>(std::max(bins, 2.0));
}

// How many boxes of `side` (a share of the axis) fit along an axis of `length` pixels: at 5% of the
// plane a side is 0.2236 of the axis, so four, each in a cell of 0.25 with room to move.
std::uint64_t GridCells(std::uint64_t length, double side) {
    const auto cells = static_cast<std::uint64_t>(std::floor(1.0 / side));
    return std::clamp<std::uint64_t>(cells, 1, std::max<std::uint64_t>(length, 1));
}

}  // namespace

Result<CubeAxes> CubeAxes::Of(const ImageDescriptor& descriptor) {
    const auto& axes = descriptor.axes;
    const auto x = AxisIndex(axes, AxisRole::spatial_x);
    const auto y = AxisIndex(axes, AxisRole::spatial_y);
    const auto spectral = AxisIndex(axes, AxisRole::spectral);
    if (!x || !y || !spectral) {
        return Error{ErrorCode::invalid_argument,
                     "the bench reads cubes, and image " + descriptor.id + " has no " +
                         (!x ? "spatial x" : !y ? "spatial y" : "spectral") + " axis",
                     descriptor.id};
    }
    CubeAxes cube;
    cube.rank = axes.size();
    cube.x = *x;
    cube.y = *y;
    cube.spectral = *spectral;
    cube.polarization = AxisIndex(axes, AxisRole::polarization);
    cube.time = AxisIndex(axes, AxisRole::time);
    cube.width = axes[*x].length;
    cube.height = axes[*y].length;
    cube.channels = axes[*spectral].length;
    cube.polarizations = cube.polarization ? axes[*cube.polarization].length : 1;
    return cube;
}

std::string Operation::Describe() const {
    const auto pol = "pol=" + std::to_string(polarization);
    switch (mode) {
        case Mode::plane:
            return pol + ";chan=" + std::to_string(channel);
        case Mode::spectrum:
            return pol + ";l=" + std::to_string(x) + ";m=" + std::to_string(y);
        case Mode::region:
            return pol + ";l=" + std::to_string(x) + ":" + std::to_string(x + width) + ";m=" + std::to_string(y) +
                   ":" + std::to_string(y + height);
        case Mode::cube_histogram:
        case Mode::animation:
            return pol + ";chan=" + std::to_string(channel) + ":" + std::to_string(channel + channel_count);
        case Mode::open:
            return "";
    }
    return "";
}

std::vector<Operation> PlanOperations(Mode mode, const CubeAxes& axes, std::uint64_t seed, unsigned trial,
                                      unsigned processes, unsigned process_index, unsigned ops,
                                      double region_fraction, unsigned animation_frames) {
    std::vector<Operation> operations(ops);
    for (auto& operation : operations) {
        operation.mode = mode;
    }
    if (mode == Mode::open) {
        return operations;
    }

    // Two streams: one for which position, one for everything about it. Kept apart so that adding a
    // draw to one never shifts the other.
    Stream positions(seed, mode, trial, 1);
    Stream details(seed, mode, trial, 2);
    const std::uint64_t total = static_cast<std::uint64_t>(processes) * ops;
    const std::uint64_t first = static_cast<std::uint64_t>(process_index) * ops;

    if (mode == Mode::cube_histogram) {
        // Contiguous runs of channels, one per process; with more processes than channels, each
        // takes one channel and some of them share it.
        const bool shared = processes > axes.channels;
        std::uint64_t start = process_index % axes.channels;
        std::uint64_t end = start + 1;
        if (!shared) {
            start = axes.channels * process_index / processes;
            end = axes.channels * (process_index + 1) / processes;
        }
        const auto base = details.Below(axes.polarizations);
        for (unsigned index = 0; index < ops; ++index) {
            auto& operation = operations[index];
            operation.channel = start;
            operation.channel_count = end - start;
            operation.polarization = (base + index) % axes.polarizations;
            operation.overlap = shared || index >= axes.polarizations;
        }
        return operations;
    }

    const double side = std::sqrt(region_fraction);
    const auto cells_x = GridCells(axes.width, side);
    const auto cells_y = GridCells(axes.height, side);
    // An animation's runs of channels, each in a cell of its own with room to move.
    const auto frames = std::clamp<std::uint64_t>(animation_frames, 1, std::max<std::uint64_t>(axes.channels, 1));
    const auto runs = std::max<std::uint64_t>(axes.channels / frames, 1);
    const auto run_cell = axes.channels / runs;

    std::uint64_t pool = 0;
    switch (mode) {
        case Mode::plane:
            pool = axes.channels * axes.polarizations;
            break;
        case Mode::spectrum:
            pool = axes.width * axes.height;
            break;
        case Mode::region:
            pool = cells_x * cells_y;
            break;
        case Mode::animation:
            pool = runs * axes.polarizations;
            break;
        default:
            break;
    }
    const auto draws = DistinctSample(positions, pool, total);

    // The details are drawn for every operation of the trial in order, including other processes',
    // so that each process's are the same whatever process it is.
    const auto cell_width = axes.width / cells_x;
    const auto cell_height = axes.height / cells_y;
    const auto box_width =
        std::clamp<std::uint64_t>(std::llround(side * static_cast<double>(axes.width)), 1, cell_width);
    const auto box_height =
        std::clamp<std::uint64_t>(std::llround(side * static_cast<double>(axes.height)), 1, cell_height);

    for (std::uint64_t index = 0; index < total; ++index) {
        const auto& draw = draws[index];
        Operation operation;
        operation.mode = mode;
        operation.overlap = draw.overlap;
        switch (mode) {
            case Mode::plane:
                operation.channel = draw.value % axes.channels;
                operation.polarization = draw.value / axes.channels;
                break;
            case Mode::spectrum:
                operation.x = draw.value % axes.width;
                operation.y = draw.value / axes.width;
                operation.polarization = details.Below(axes.polarizations);
                break;
            case Mode::region: {
                operation.width = box_width;
                operation.height = box_height;
                operation.x = (draw.value % cells_x) * cell_width + details.Below(cell_width - box_width + 1);
                operation.y = (draw.value / cells_x) * cell_height + details.Below(cell_height - box_height + 1);
                operation.polarization = details.Below(axes.polarizations);
                break;
            }
            case Mode::animation:
                operation.channel_count = frames;
                operation.channel = (draw.value % runs) * run_cell + details.Below(run_cell - frames + 1);
                operation.polarization = draw.value / runs;
                break;
            default:
                break;
        }
        if (index >= first && index < first + ops) {
            operations[index - first] = operation;
        }
    }
    return operations;
}

namespace {

// The chunks an operation reads, as a range of chunk indices along each axis it moves on: the spatial
// two, the spectral and the polarization. Every other axis is read at 0 by every operation.
struct ChunkBox {
    std::array<std::uint64_t, 4> first{};
    std::array<std::uint64_t, 4> last{};

    bool Meets(const ChunkBox& other) const {
        for (std::size_t axis = 0; axis < first.size(); ++axis) {
            if (last[axis] < other.first[axis] || other.last[axis] < first[axis]) {
                return false;
            }
        }
        return true;
    }
};

ChunkBox ChunksOf(const Operation& operation, const CubeAxes& axes, const std::vector<std::uint64_t>& chunk_shape) {
    // From and to, in pixels, along x, y, spectral and polarization.
    std::array<std::uint64_t, 4> from{};
    std::array<std::uint64_t, 4> to{};
    switch (operation.mode) {
        case Mode::plane:
            from = {0, 0, operation.channel, operation.polarization};
            to = {axes.width, axes.height, operation.channel + 1, operation.polarization + 1};
            break;
        case Mode::spectrum:
            from = {operation.x, operation.y, 0, operation.polarization};
            to = {operation.x + 1, operation.y + 1, axes.channels, operation.polarization + 1};
            break;
        case Mode::region:
            from = {operation.x, operation.y, 0, operation.polarization};
            to = {operation.x + operation.width, operation.y + operation.height, axes.channels,
                  operation.polarization + 1};
            break;
        default:
            from = {0, 0, operation.channel, operation.polarization};
            to = {axes.width, axes.height, operation.channel + operation.channel_count, operation.polarization + 1};
            break;
    }
    const std::array<std::optional<std::size_t>, 4> index{axes.x, axes.y, axes.spectral, axes.polarization};
    ChunkBox box;
    for (std::size_t axis = 0; axis < index.size(); ++axis) {
        std::uint64_t chunk = 1;
        if (index[axis] && *index[axis] < chunk_shape.size()) {
            chunk = std::max<std::uint64_t>(chunk_shape[*index[axis]], 1);
        }
        box.first[axis] = from[axis] / chunk;
        box.last[axis] = (std::max(to[axis], from[axis] + 1) - 1) / chunk;
    }
    return box;
}

}  // namespace

void MarkSharedChunks(std::vector<std::vector<Operation>>& plans, const CubeAxes& axes,
                      const std::vector<std::uint64_t>& chunk_shape) {
    struct Placed {
        std::size_t process;
        std::size_t index;
        ChunkBox box;
    };
    std::vector<Placed> placed;
    for (std::size_t process = 0; process < plans.size(); ++process) {
        for (std::size_t index = 0; index < plans[process].size(); ++index) {
            const auto& operation = plans[process][index];
            if (operation.mode == Mode::animation || operation.mode == Mode::open) {
                continue;
            }
            placed.push_back({process, index, ChunksOf(operation, axes, chunk_shape)});
        }
    }
    for (const auto& one : placed) {
        bool shares = false;
        for (const auto& other : placed) {
            const bool earlier_here = other.process == one.process && other.index < one.index;
            if ((earlier_here || other.process != one.process) && one.box.Meets(other.box)) {
                shares = true;
                break;
            }
        }
        plans[one.process][one.index].shares_chunks = shares;
    }
}

Runner::Runner(const Context& context, Image image, HistogramMethod histogram, std::size_t first_touch_bytes)
    : _shared(context), _image(std::move(image)), _histogram(histogram), _first_touch_bytes(first_touch_bytes) {
    if (auto pool = context.NewCachePool(0)) {
        _keeping_nothing = *pool;
    }
    if (auto axes = CubeAxes::Of(_image->descriptor())) {
        _axes = *axes;
        // Sized here so that no operation pays for growing it.
        _pixels.resize(std::max(_axes->width * _axes->height, _axes->channels));
        const auto& chunk = _image->chunk_geometry().chunk_shape;
        if (_axes->spectral < chunk.size()) {
            _spectral_chunk = std::max<std::uint64_t>(chunk[_axes->spectral], 1);
        }
        _exact.reserve(_axes->channels * 4);
    }
}

Runner::Runner(ContextOptions context, std::string dataset, std::string image_id)
    : _context(std::move(context)), _dataset(std::move(dataset)), _image_id(std::move(image_id)) {}

Result<void> Runner::Prepare(const Operation& operation) {
    // Let go of the last one first, so that two pools' worth of chunks are never held at once.
    _first_touch.reset();
    if ((operation.mode != Mode::plane && operation.mode != Mode::spectrum) || !_shared) {
        return {};
    }
    auto pool = _shared->NewCachePool(_first_touch_bytes);
    if (!pool) {
        return std::move(pool).error();
    }
    _first_touch = *pool;
    return {};
}

Result<std::uint64_t> Runner::Run(const Operation& operation, const ReadOptions& options) {
    _pixel_count = 0;
    _frames.reset();
    _frame_stats.reset();
    _exact.clear();
    _counts.clear();
    if (operation.mode == Mode::open) {
        return Open();
    }
    if (!_axes) {
        return Error{ErrorCode::invalid_argument, "the image is not a cube", ""};
    }
    switch (operation.mode) {
        case Mode::plane:
        case Mode::spectrum: {
            auto own = options;
            if (_first_touch) {
                own.control.cache_pool = _first_touch;
            }
            return Read(operation, own);
        }
        case Mode::animation:
            return Animate(operation, options);
        case Mode::region:
            return Reduce(operation, options);
        case Mode::cube_histogram:
            return Histogram(operation, options);
        case Mode::open:
            break;
    }
    return Error{ErrorCode::invalid_argument, "unknown mode", ""};
}

// Plane after plane, as carta-backend serves a playing animation: each frame its own read, through
// the context's cache, so that a frame finds what the one before it decoded when they share chunks.
//
// Played at a frame rate, a frame is read no earlier than its turn and is late if it is not ready by
// the end of it; a frame that runs over pushes the ones after it back, as a viewer waiting on it would
// see. With prefetch, entering a run of chunks along the spectrum starts a read of the first plane of
// the next run on a thread of its own, which decodes that run's chunks into the cache while this run's
// frames play from it.
Result<std::uint64_t> Runner::Animate(const Operation& operation, const ReadOptions& options) {
    using Clock = std::chrono::steady_clock;
    std::uint64_t hash = kFnvOffset;
    std::uint64_t elements = 0;
    Operation frame = operation;
    frame.mode = Mode::plane;
    const auto& axes = *_axes;
    const auto period = _fps > 0.0 ? std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / _fps))
                                   : Clock::duration::zero();
    if (_prefetch && _prefetched.size() < axes.width * axes.height) {
        _prefetched.resize(axes.width * axes.height);
    }
    std::thread prefetcher;
    const auto prefetch = [&](std::uint64_t channel) {
        if (prefetcher.joinable()) {
            prefetcher.join();
        }
        prefetcher = std::thread([this, &axes, channel, polarization = operation.polarization, options] {
            ReadRequest request;
            request.axes.assign(axes.rank, Range{0, 1, 1});
            request.axes[axes.x] = Range{0, axes.width, 1};
            request.axes[axes.y] = Range{0, axes.height, 1};
            request.axes[axes.spectral] = Range{channel, 1, 1};
            if (axes.polarization) {
                request.axes[*axes.polarization] = Range{polarization, 1, 1};
            }
            (void)_image->Read(request, {_prefetched.data(), axes.width * axes.height}, options);
        });
    };

    FrameStats stats;
    std::vector<double> reads;
    const auto start = Clock::now();
    auto turn = start;
    for (std::uint64_t index = 0; index < operation.channel_count; ++index) {
        frame.channel = operation.channel + index;
        if (_prefetch && _spectral_chunk > 1 && (index == 0 || frame.channel % _spectral_chunk == 0)) {
            const auto next = (frame.channel / _spectral_chunk + 1) * _spectral_chunk;
            if (next < operation.channel + operation.channel_count) {
                prefetch(next);
            }
        }
        if (period > Clock::duration::zero()) {
            std::this_thread::sleep_until(turn);
        }
        const auto began = Clock::now();
        auto read = Read(frame, options);
        const auto done = Clock::now();
        if (!read) {
            if (prefetcher.joinable()) {
                prefetcher.join();
            }
            return read;
        }
        elements += *read;
        // Played at a frame rate, only the last frame is fingerprinted: a plane of a large cube takes
        // longer to hash than a frame's turn, and hashing every one would make every frame late.
        if (period == Clock::duration::zero() || index + 1 == operation.channel_count) {
            Mix(hash, bench::Fingerprint(_pixels.data(), _pixel_count), sizeof(hash));
        }

        const double seconds = std::chrono::duration<double>(done - began).count();
        if (index == 0) {
            stats.first_s = seconds;
        } else {
            reads.push_back(seconds);
            if (period > Clock::duration::zero() && done > turn + period) {
                ++stats.late;
                stats.late_max_s = std::max(stats.late_max_s, std::chrono::duration<double>(done - turn - period).count());
            }
        }
        // The next frame's turn follows this one's, or this frame's end when it ran over.
        turn = std::max(turn + period, done);
    }
    if (prefetcher.joinable()) {
        prefetcher.join();
    }
    if (!reads.empty()) {
        std::sort(reads.begin(), reads.end());
        stats.median_s = reads[reads.size() / 2];
        stats.max_s = reads.back();
    }
    _frame_stats = stats;
    _pixel_count = 0;
    _frames = hash;
    return elements;
}

Result<std::uint64_t> Runner::Read(const Operation& operation, const ReadOptions& options) {
    const auto& axes = *_axes;
    const bool plane = operation.mode == Mode::plane;
    ReadRequest request;
    request.axes.assign(axes.rank, Range{0, 1, 1});
    request.axes[axes.x] = plane ? Range{0, axes.width, 1} : Range{operation.x, 1, 1};
    request.axes[axes.y] = plane ? Range{0, axes.height, 1} : Range{operation.y, 1, 1};
    request.axes[axes.spectral] = plane ? Range{operation.channel, 1, 1} : Range{0, axes.channels, 1};
    if (axes.polarization) {
        request.axes[*axes.polarization] = Range{operation.polarization, 1, 1};
    }
    const std::size_t elements = plane ? axes.width * axes.height : axes.channels;
    auto written = _image->Read(request, {_pixels.data(), elements}, options);
    if (!written) {
        return std::move(written).error();
    }
    _pixel_count = *written;
    return static_cast<std::uint64_t>(*written);
}

Result<std::uint64_t> Runner::Reduce(const Operation& operation, const ReadOptions& options) {
    const auto& axes = *_axes;
    const RegionMask region{operation.x, operation.y, operation.width, operation.height, {}};
    SpectralReduceRequest request;
    request.planes.spectral = Range{0, axes.channels, 1};
    request.planes.polarization = operation.polarization;
    request.regions = {&region, 1};
    request.statistics = Statistic::num_pixels | Statistic::nan_count | Statistic::sum | Statistic::sum_sq |
                         Statistic::min | Statistic::max;

    _exact.assign(axes.channels * 4, std::numeric_limits<double>::quiet_NaN());
    const auto sink = [this](const SpectralBlock& block) {
        if (!block.complete) {
            return true;
        }
        for (std::uint64_t channel = 0; channel < block.channel_count; ++channel) {
            const auto totals = block.Totals(0, channel);
            auto* exact = &_exact[(block.first_channel + channel) * 4];
            exact[0] = totals.num_pixels;
            exact[1] = totals.nan_count;
            exact[2] = totals.min;
            exact[3] = totals.max;
        }
        return true;
    };
    if (auto reduced = _image->ReduceSpectral(request, sink, options); !reduced) {
        _exact.clear();
        return std::move(reduced).error();
    }
    return operation.width * operation.height * axes.channels;
}

Result<std::uint64_t> Runner::Histogram(const Operation& operation, const ReadOptions& passed) {
    auto options = passed;
    options.control.cache_pool = _keeping_nothing;
    if (_histogram.kind == HistogramMethod::Kind::exact) {
        return ExactHistogram(operation, options);
    }
    const auto& axes = *_axes;
    CubeHistogramRequest request;
    request.planes.spectral = Range{operation.channel, operation.channel_count, 1};
    request.planes.polarization = operation.polarization;
    request.bins = AutomaticBins(axes);
    request.spatial_sample = _histogram.kind == HistogramMethod::Kind::sampled ? _histogram.stride : 1;
    auto histogram = _image->ComputeCubeHistogram(request, options);
    if (!histogram) {
        return std::move(histogram).error();
    }
    const auto& totals = histogram->totals;
    _exact = {totals.num_pixels, totals.nan_count, totals.min, totals.max};
    return axes.width * axes.height * operation.channel_count;
}

// As carta-backend computes a cube histogram by default: a reduction over whole planes for the
// cube's range, then every plane binned over it and the planes added together.
Result<std::uint64_t> Runner::ExactHistogram(const Operation& operation, const ReadOptions& options) {
    const auto& axes = *_axes;
    const RegionMask plane{0, 0, axes.width, axes.height, {}};
    SpectralReduceRequest range;
    range.planes.spectral = Range{operation.channel, operation.channel_count, 1};
    range.planes.polarization = operation.polarization;
    range.regions = {&plane, 1};
    range.statistics =
        Statistic::num_pixels | Statistic::sum | Statistic::sum_sq | Statistic::min | Statistic::max;

    double pixels = 0.0;
    double lowest = std::numeric_limits<double>::quiet_NaN();
    double highest = std::numeric_limits<double>::quiet_NaN();
    const auto extremes = [&](const SpectralBlock& block) {
        if (!block.complete) {
            return true;
        }
        for (std::uint64_t channel = 0; channel < block.channel_count; ++channel) {
            const auto totals = block.Totals(0, channel);
            pixels += totals.num_pixels;
            // fmin and fmax pass over NaN, which is what a plane with nothing finite reports.
            lowest = std::fmin(lowest, totals.min);
            highest = std::fmax(highest, totals.max);
        }
        return true;
    };
    if (auto reduced = _image->ReduceSpectral(range, extremes, options); !reduced) {
        return std::move(reduced).error();
    }
    _exact = {pixels, lowest, highest};

    // The backend declines an empty or inverted range, and so does this: there is nothing to bin.
    if (lowest < highest) {
        HistogramRequest bins;
        bins.planes = range.planes;
        bins.bins = AutomaticBins(axes);
        bins.lower = lowest;
        bins.upper = highest;
        _counts.assign(bins.bins, 0);
        const auto add = [this](const HistogramBlock& block) {
            if (!block.complete) {
                return true;
            }
            for (std::uint64_t channel = 0; channel < block.channel_count; ++channel) {
                const auto* counts = block.Counts(channel);
                for (std::size_t bin = 0; bin < block.bin_count; ++bin) {
                    _counts[bin] += counts[bin];
                }
            }
            return true;
        };
        if (auto binned = _image->ComputeHistogram(bins, add, options); !binned) {
            _counts.clear();
            return std::move(binned).error();
        }
    }
    return axes.width * axes.height * operation.channel_count;
}

Result<std::uint64_t> Runner::Open() {
    auto context = Context::Create(_context);
    if (!context) {
        return std::move(context).error();
    }
    auto dataset = Dataset::Open(*context, _dataset);
    if (!dataset) {
        return std::move(dataset).error();
    }
    std::string id = _image_id;
    if (id.empty()) {
        if (!dataset->descriptor().default_image_id) {
            return Error{ErrorCode::not_found, "the dataset lists no image that opens", _dataset};
        }
        id = *dataset->descriptor().default_image_id;
    }
    auto image = dataset->OpenImage(id);
    if (!image) {
        return std::move(image).error();
    }
    _image = std::move(image).value();
    for (const auto& axis : _image->descriptor().axes) {
        _exact.push_back(static_cast<double>(axis.length));
    }
    return std::uint64_t{0};
}

std::uint64_t Runner::Fingerprint() const {
    if (_frames) {
        return *_frames;
    }
    if (_pixel_count > 0) {
        return bench::Fingerprint(_pixels.data(), _pixel_count);
    }
    auto hash = bench::Fingerprint(_exact.data(), _exact.size());
    for (const auto count : _counts) {
        Mix(hash, count, sizeof(count));
    }
    return hash;
}

std::uint64_t Fingerprint(const float* values, std::size_t count) {
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0; index < count; ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(bits));
        Mix(hash, std::isnan(values[index]) ? kQuietNaN32 : bits, sizeof(bits));
    }
    return hash;
}

std::uint64_t Fingerprint(const double* values, std::size_t count) {
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0; index < count; ++index) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(bits));
        Mix(hash, std::isnan(values[index]) ? kQuietNaN64 : bits, sizeof(bits));
    }
    return hash;
}

}  // namespace carta::zarr::bench
