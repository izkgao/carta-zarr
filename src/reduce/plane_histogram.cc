/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "plane_histogram.h"

#include "chunk_blocks.h"
#include "reduce/axis_map.h"
#include "zarr/pixel_reader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace carta::zarr::internal {
namespace {

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

Result<void> ValidateRequest(const ImageDescriptor& descriptor, const AxisMap& axes,
                             const HistogramRequest& request) {
    const auto& node = descriptor.id;
    if (request.bins == 0 || request.bins > kMaxHistogramBins) {
        return MakeError(ErrorCode::invalid_argument,
                         "A histogram needs between 1 and " + std::to_string(kMaxHistogramBins) + " bins",
                         node);
    }
    // A zero-width range would divide by zero on every pixel. The caller decides what an image with
    // no finite pixel should look like -- there is more than one defensible answer -- so this says
    // no rather than inventing one.
    if (!(request.lower < request.upper) || !std::isfinite(request.lower) || !std::isfinite(request.upper)) {
        return MakeError(ErrorCode::invalid_argument,
                         "A histogram needs a finite range with a lower bound below its upper bound", node);
    }
    if (!axes.has_polarization && request.polarization != 0) {
        return MakeError(ErrorCode::invalid_argument, "The image has no polarization axis to select", node);
    }
    if (axes.has_polarization && request.polarization >= descriptor.axes.at(axes.polarization).length) {
        return MakeError(ErrorCode::invalid_argument, "The polarization index is outside the image", node);
    }
    if (!axes.has_time && request.time != 0) {
        return MakeError(ErrorCode::invalid_argument, "The image has no time axis to select", node);
    }
    if (axes.has_time && request.time >= descriptor.axes.at(axes.time).length) {
        return MakeError(ErrorCode::invalid_argument, "The time index is outside the image", node);
    }
    return {};
}


Result<void> ValidateSpectral(const ImageDescriptor& descriptor, const AxisMap& axes, const Range& spectral) {
    const auto channels = descriptor.axes.at(axes.spectral).length;
    if (spectral.stride == 0 || spectral.count == 0 || spectral.start >= channels ||
        (spectral.count - 1) * spectral.stride > channels - 1 - spectral.start) {
        return MakeError(ErrorCode::invalid_argument, "The spectral range falls outside the image",
                         descriptor.id);
    }
    return {};
}

// A histogram whose range grows to fit whatever arrives.
//
// Doubling to one side and merging bins in pairs keeps every count: the old range becomes one half
// of the new one, and old bins 2m and 2m+1 become bin m of that half. Resolution halves each time,
// which is the only thing given up -- and only for as many doublings as the data's dynamic range
// actually needs.
class GrowingHistogram {
public:
    explicit GrowingHistogram(std::size_t bins) : _counts(bins, 0) {}

    void Add(float value) {
        const double v = value;
        if (!_seeded) {
            // A first range around the first pixel seen, and a narrow one. The range only ever
            // grows, so a guess that is too wide is permanent while one that is too narrow costs a
            // few merges and then fits.
            //
            // Anchoring it at one instead of at the pixel's own magnitude is what this used to do,
            // and on a radio image it spent almost all of the resolution on the empty space between
            // a Jansky and the hundredths of one the pixels actually are: a two-wide range over
            // 65,536 bins left about three of them inside each of a thousand target bins rather
            // than the sixty-five the default is chosen for.
            //
            // The floor is there because a first pixel of exactly zero would give a range of zero
            // width, which never grows. Starting that far down costs a couple of hundred merges
            // before the range fits -- once, against a cube of billions of pixels.
            const double magnitude = std::max<double>(std::abs(v), std::numeric_limits<float>::min());
            _lower = v - magnitude;
            _width = (2.0 * magnitude) / static_cast<double>(_counts.size());
            _seeded = true;
        }
        while (v < _lower) {
            Grow(false);
        }
        while (v >= _lower + (_width * static_cast<double>(_counts.size()))) {
            Grow(true);
        }
        auto bin = static_cast<std::size_t>((v - _lower) / _width);
        if (bin >= _counts.size()) {
            bin = _counts.size() - 1;
        }
        ++_counts[bin];
    }

    // Re-aggregate over the range the caller wants, giving each provisional bin to the target bin
    // its centre falls in. That is where the error lives: a target edge cutting through a
    // provisional bin takes all of it or none of it.
    std::vector<std::uint64_t> Aggregate(std::size_t bins, double lower, double upper) const {
        std::vector<std::uint64_t> out(bins, 0);
        std::uint64_t total = 0;
        for (const auto count : _counts) {
            total += count;
        }
        if (!(lower < upper)) {
            // Every pixel had the same value, so there is one bin it can be in.
            out.front() = total;
            return out;
        }
        const double target_width = (upper - lower) / static_cast<double>(bins);
        for (std::size_t i = 0; i < _counts.size(); ++i) {
            if (_counts[i] == 0) {
                continue;
            }
            const double centre = _lower + (_width * (static_cast<double>(i) + 0.5));
            auto bin = static_cast<std::ptrdiff_t>((centre - lower) / target_width);
            bin = std::clamp<std::ptrdiff_t>(bin, 0, static_cast<std::ptrdiff_t>(bins) - 1);
            out[static_cast<std::size_t>(bin)] += _counts[i];
        }
        return out;
    }

private:
    void Grow(bool upward) {
        const std::size_t half = _counts.size() / 2;
        std::vector<std::uint64_t> merged(_counts.size(), 0);
        for (std::size_t m = 0; m < half; ++m) {
            merged[upward ? m : half + m] = _counts[2 * m] + _counts[(2 * m) + 1];
        }
        if (!upward) {
            _lower -= _width * static_cast<double>(_counts.size());
        }
        _width *= 2.0;
        _counts.swap(merged);
    }

    std::vector<std::uint64_t> _counts;
    double _lower = 0.0;
    double _width = 1.0;
    bool _seeded = false;
};

// Everything one walk over the planes needs that does not change from read to read.
//
// Extracted because two entry points share it and the read strategy has moved more than once: how
// wide a band is, how deep a slab goes, which axis is contiguous. One copy of that, not two.
struct PlaneWalk {
    const Store* store = nullptr;
    const ImageDescriptor* descriptor = nullptr;
    const ReadOptions* options = nullptr;
    AxisMap map;
    std::size_t axis_u = 0;
    std::size_t axis_v = 0;
    std::uint64_t u_length = 0;
    std::uint64_t v_length = 0;
    std::uint64_t chunk_u = 1;
    std::uint64_t chunk_v = 1;
    std::uint64_t chunk_depth = 1;
    std::uint64_t least_channels = 1;
    std::uint64_t chunk_bytes = 1;
    std::size_t slab_budget_bytes = 0;
    std::uint64_t band_rows = 1;
    std::uint64_t layer_chunks = 1;
    Range spectral;
    std::uint64_t polarization = 0;
    std::uint64_t time = 0;
    // Take every nth pixel along both spatial axes. This does not reduce the chunks a read decodes
    // -- a chunk comes back whole however few of its pixels are wanted -- so it pays only when it
    // steps over chunks entirely.
    std::uint64_t sample = 1;
    bool apply_mask = false;
};

PlaneWalk MakeWalk(const Store& store, const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                   const AxisMap& map, const Range& spectral, std::uint64_t polarization, std::uint64_t time,
                   std::uint64_t sample, const ReadOptions& options) {
    PlaneWalk walk;
    walk.store = &store;
    walk.descriptor = &descriptor;
    walk.options = &options;
    walk.map = map;
    // The spatial axis the store varies fastest is the one to ask for first; see the reduction.
    const bool swap_spatial = SpatialYIsFastest(geometry);
    walk.axis_u = swap_spatial ? map.y : map.x;
    walk.axis_v = swap_spatial ? map.x : map.y;
    walk.u_length = descriptor.axes.at(walk.axis_u).length;
    walk.v_length = descriptor.axes.at(walk.axis_v).length;
    walk.chunk_u = std::max<std::uint64_t>(1, geometry.chunk_shape.at(walk.axis_u));
    walk.chunk_v = std::max<std::uint64_t>(1, geometry.chunk_shape.at(walk.axis_v));
    walk.chunk_depth = std::max<std::uint64_t>(1, geometry.chunk_shape.at(map.spectral));
    walk.apply_mask = options.apply_pixel_mask && descriptor.has_pixel_mask;
    walk.chunk_bytes = DecodedChunkBytes(descriptor, geometry) * (walk.apply_mask ? 2 : 1);
    walk.slab_budget_bytes = options.temporary_memory_limit_bytes != 0 ? options.temporary_memory_limit_bytes
                                                                       : DefaultReadBytes(walk.chunk_bytes);
    walk.least_channels = ((walk.chunk_depth + spectral.stride - 1) / spectral.stride);
    const std::uint64_t row_chunks = std::max<std::uint64_t>(1, ((walk.u_length - 1) / walk.chunk_u) + 1);
    const std::uint64_t column_chunks = std::max<std::uint64_t>(1, ((walk.v_length - 1) / walk.chunk_v) + 1);
    walk.layer_chunks = std::max<std::uint64_t>(1, row_chunks * column_chunks);
    // How many chunk rows one read may hold, so that a read is a budget's worth of chunk data.
    walk.band_rows = std::max<std::uint64_t>(
        1, walk.slab_budget_bytes / std::max<std::uint64_t>(1, row_chunks * walk.chunk_bytes));
    walk.spectral = spectral;
    walk.polarization = polarization;
    walk.time = time;
    walk.sample = std::max<std::uint64_t>(1, sample);
    return walk;
}

// Samples of `stride` that fall in [begin, end), as a start and a count.
void SampledRange(std::uint64_t begin, std::uint64_t end, std::uint64_t stride, std::uint64_t& start,
                  std::uint64_t& count) {
    const std::uint64_t first = (begin + stride - 1) / stride;
    const std::uint64_t last = (end + stride - 1) / stride;
    start = first * stride;
    count = last > first ? last - first : 0;
}

/**
 * Visit every plane of one channel range, a band of chunk rows at a time.
 *
 * `before_read` runs before each read after the first of the range, which is where a caller reports
 * what it has or decides to stop. `visit` receives a whole read as a pointer and three strides
 * rather than a packed buffer, because the destination comes back in the store's own order and
 * packing it would be the transpose this walk exists to avoid.
 *
 * A read, not a plane: a caller that splits the work across threads needs a piece big enough to pay
 * for the dispatch, and a plane of a few hundred thousand pixels is not one. A caller that wants
 * planes loops over `channel_count` itself, which costs it nothing.
 */
template <typename BeforeRead, typename Visit>
Result<void> WalkChannels(const PlaneWalk& walk, std::uint64_t begin, std::uint64_t end,
                          std::uint64_t& chunks_done, BeforeRead&& before_read, Visit&& visit) {
    const auto& descriptor = *walk.descriptor;
    const auto& options = *walk.options;
    const auto& node = descriptor.id;
    const auto rank = descriptor.axes.size();
    const Range spectral = walk.spectral;
    std::vector<float> pixels;
    std::vector<std::uint8_t> mask;
    std::uint64_t reads_done = 0;

    for (std::uint64_t v_begin = 0; v_begin < walk.v_length;) {
        const std::uint64_t v_end = std::min(walk.v_length, v_begin + (walk.band_rows * walk.chunk_v));
        std::uint64_t v_start = 0;
        std::uint64_t v_count = 0;
        SampledRange(v_begin, v_end, walk.sample, v_start, v_count);
        const std::uint64_t band_chunks = std::max<std::uint64_t>(
            1, (((walk.u_length - 1) / walk.chunk_u) + 1) * ((((v_end - v_begin) - 1) / walk.chunk_v) + 1));
        if (v_count == 0) {
            chunks_done += band_chunks * ((end - begin + walk.least_channels - 1) / walk.least_channels);
            v_begin = v_end;
            continue;
        }
        const std::uint64_t spectral_chunks =
            std::max<std::uint64_t>(1, walk.slab_budget_bytes / (band_chunks * walk.chunk_bytes));
        const std::uint64_t slab_channels = std::max<std::uint64_t>(1, spectral_chunks * walk.least_channels);

        for (std::uint64_t slab_begin = begin; slab_begin < end;) {
            const std::uint64_t slab_end = AlignedBlockEnd(slab_begin, std::min(slab_channels, end - slab_begin),
                                                           end, spectral.start, spectral.stride, walk.chunk_depth);
            const std::uint64_t slab_length = slab_end - slab_begin;

            if (reads_done > 0) {
                if (auto ready = before_read(chunks_done); !ready) {
                    return ready.error();
                }
            }
            ++reads_done;

            if (auto control = zarr::CheckReadControl(options, node); !control) {
                return control.error();
            }

            std::uint64_t u_start = 0;
            std::uint64_t u_count = 0;
            SampledRange(0, walk.u_length, walk.sample, u_start, u_count);

            ReadRequest read_request;
            read_request.axes.assign(rank, Range{0, 1, 1});
            read_request.axes.at(walk.axis_u) = Range{u_start, u_count, walk.sample};
            read_request.axes.at(walk.axis_v) = Range{v_start, v_count, walk.sample};
            read_request.axes.at(walk.map.spectral) =
                Range{spectral.start + (slab_begin * spectral.stride), slab_length, spectral.stride};
            if (walk.map.has_polarization) {
                read_request.axes.at(walk.map.polarization) = Range{walk.polarization, 1, 1};
            }
            if (walk.map.has_time) {
                read_request.axes.at(walk.map.time) = Range{walk.time, 1, 1};
            }

            auto selection = zarr::BuildSelection(descriptor, read_request);
            if (!selection) {
                return selection.error();
            }
            // Ask for the stored dimensions reversed, which against a destination whose dimension 0
            // is fastest is asking for no transpose at all.
            for (std::size_t i = 0; i < rank; ++i) {
                selection.value().logical_to_stored.at(i) = rank - 1 - i;
            }

            std::vector<std::uint64_t> stored_stride(rank, 1);
            std::uint64_t running = 1;
            for (std::size_t stored = rank; stored-- > 0;) {
                stored_stride.at(stored) = running;
                running *= selection.value().count.at(stored);
            }
            const std::uint64_t stride_u = stored_stride.at(descriptor.axes.at(walk.axis_u).storage_index);
            const std::uint64_t stride_v = stored_stride.at(descriptor.axes.at(walk.axis_v).storage_index);
            const std::uint64_t stride_z = stored_stride.at(descriptor.axes.at(walk.map.spectral).storage_index);
            const auto elements = static_cast<std::size_t>(running);

            pixels.resize(elements);
            if (auto read = walk.store->ReadPixelsFloat32(descriptor.id, selection.value(), pixels.data(),
                                                          pixels.size(), options);
                !read) {
                return read.error();
            }
            if (walk.apply_mask) {
                mask.resize(elements);
                if (auto read = walk.store->ReadPixelMaskBytes(descriptor.pixel_mask_id, selection.value(),
                                                               mask.data(), mask.size(), options);
                    !read) {
                    return read.error();
                }
                for (std::size_t i = 0; i < elements; ++i) {
                    if (mask[i] == 0) {
                        pixels[i] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }

            visit(slab_begin - begin, slab_length, pixels.data(), stride_u, stride_v, stride_z, u_count,
                  v_count);

            chunks_done += band_chunks * ((slab_length + walk.least_channels - 1) / walk.least_channels);
            slab_begin = slab_end;
        }
        v_begin = v_end;
    }
    return {};
}

}  // namespace

Result<void> ComputeHistogram(const Store& store, const ImageDescriptor& descriptor,
                              const ChunkGeometry& geometry, const HistogramRequest& request,
                              const HistogramSink& sink, const ReadOptions& options, WorkPool& workers) {
    const auto& node = descriptor.id;
    if (!sink) {
        return MakeError(ErrorCode::invalid_argument, "A histogram needs a sink", node);
    }
    auto axes = MapAxes(descriptor);
    if (!axes) {
        return axes.error();
    }
    const auto& map = axes.value();
    if (auto valid = ValidateRequest(descriptor, map, request); !valid) {
        return valid.error();
    }
    if (auto valid = ValidateSpectral(descriptor, map, request.spectral); !valid) {
        return valid.error();
    }

    const Range spectral = request.spectral;
    const auto walk = MakeWalk(store, descriptor, geometry, map, spectral, request.polarization, request.time,
                               1, options);

    // Emit granularity, as in the reduction: without a hint a block costs one read budget, so it is
    // as often as the walk can report without making any read smaller.
    const std::size_t bytes_per_channel = static_cast<std::size_t>(request.bins) * sizeof(std::uint64_t);
    const std::uint64_t budget_channels =
        std::max<std::uint64_t>(1, kSpectralEmitBudgetBytes / std::max<std::size_t>(1, bytes_per_channel));
    const std::uint64_t block_chunks =
        std::min(spectral.count, std::max<std::uint64_t>(1, walk.slab_budget_bytes /
                                                                std::max<std::uint64_t>(1, walk.layer_chunks *
                                                                                               walk.chunk_bytes)));
    const std::uint64_t wanted_channels =
        std::min({request.emit_every_channels == 0 ? block_chunks * walk.least_channels
                                                   : static_cast<std::uint64_t>(request.emit_every_channels),
                  budget_channels, spectral.count});

    // The caller's own sequence: divide in double, narrow the width, compare against the narrowed
    // bounds. Doing any one of those in the other type moves pixels across bin edges.
    const float width = static_cast<float>((request.upper - request.lower) / request.bins);
    const float lower = static_cast<float>(request.lower);
    const float upper = static_cast<float>(request.upper);
    const auto bins = static_cast<std::size_t>(request.bins);
    std::vector<std::uint64_t> counts;

    // How many private histograms the binning may split a plane into. Capped three ways: by the
    // pool, by memory, and -- inside the visit, where the plane's size is known -- by whether there
    // is enough work to be worth waking anyone for.
    //
    // The memory cap is what keeps a large bin count from turning a split into an allocation: a
    // caller may ask for as many as kMaxHistogramBins, and a private copy of that for every worker
    // is hundreds of megabytes for a pass that is supposed to stream.
    constexpr std::size_t kPartialBudgetBytes = 64U << 20U;
    // Below this a task is not worth its share of a dispatch, so the plane is binned in place.
    constexpr std::uint64_t kLeastPixelsPerTask = 1U << 16U;
    const std::size_t partials_by_memory =
        std::max<std::size_t>(1, kPartialBudgetBytes / std::max<std::size_t>(1, bins * sizeof(std::uint64_t)));
    const std::size_t max_tasks = std::min(workers.size(), partials_by_memory);
    // One allocation for the whole walk. Each task owns one row of it, so no two of them ever touch
    // the same bin and the sum at the end is the only place they meet.
    std::vector<std::uint64_t> partials;
    if (max_tasks > 1) {
        partials.resize(max_tasks * bins);
    }

    for (std::uint64_t block_begin = 0; block_begin < spectral.count;) {
        const std::uint64_t block_end = AlignedBlockEnd(block_begin, wanted_channels, spectral.count,
                                                        spectral.start, spectral.stride, walk.chunk_depth);
        const auto block_length = static_cast<std::size_t>(block_end - block_begin);
        counts.assign(block_length * bins, 0);

        const std::uint64_t block_spectral_chunks =
            (block_length + walk.least_channels - 1) / walk.least_channels;
        const std::uint64_t block_chunks_total =
            std::max<std::uint64_t>(1, walk.layer_chunks * block_spectral_chunks);
        std::uint64_t chunks_done = 0;

        const auto hand_over = [&](bool complete) -> Result<void> {
            HistogramBlock block;
            block.first_channel = block_begin;
            block.channel_count = block_length;
            block.counts = counts.data();
            block.bin_count = bins;
            block.complete = complete;
            block.completeness =
                complete ? 1.0 : static_cast<double>(chunks_done) / static_cast<double>(block_chunks_total);
            if (!sink(block)) {
                return MakeError(ErrorCode::cancelled, "The histogram was cancelled by its sink", node);
            }
            return {};
        };

        const auto walked = WalkChannels(
            walk, block_begin, block_end, chunks_done,
            [&](std::uint64_t) -> Result<void> { return hand_over(false); },
            [&](std::uint64_t first_channel, std::uint64_t channel_count, const float* base,
                std::uint64_t stride_u, std::uint64_t stride_v, std::uint64_t stride_z,
                std::uint64_t u_count, std::uint64_t v_count) {
              for (std::uint64_t offset = 0; offset < channel_count; ++offset) {
                const float* plane = base + (offset * stride_z);
                std::uint64_t* into =
                    counts.data() + (static_cast<std::size_t>(first_channel + offset) * bins);

                const auto bin_rows = [&](std::uint64_t v_first, std::uint64_t v_last,
                                          std::uint64_t* destination) {
                    for (std::uint64_t v = v_first; v < v_last; ++v) {
                        const float* row = plane + (v * stride_v);
                        for (std::uint64_t u = 0; u < u_count; ++u) {
                            const float value = row[u * stride_u];
                            // The caller's own rule: a pixel outside the range is not counted, and
                            // NaN fails both comparisons.
                            if (lower <= value && value <= upper) {
                                auto bin = static_cast<std::size_t>((value - lower) / width);
                                if (bin >= bins) {
                                    bin = bins - 1;
                                }
                                ++destination[bin];
                            }
                        }
                    }
                };

                // Rows, not planes: a read holding one plane is the common case for a large image,
                // so splitting by plane would leave the split with nothing to divide.
                const std::size_t tasks = PlanRowTasks(u_count, v_count, max_tasks, kLeastPixelsPerTask);
                if (tasks <= 1) {
                    bin_rows(0, v_count, into);
                    continue;
                }

                std::fill(partials.begin(), partials.begin() + static_cast<std::ptrdiff_t>(tasks * bins), 0);
                const std::uint64_t rows_per_task = (v_count + tasks - 1) / tasks;
                workers.Run(tasks, [&](std::size_t task, std::size_t) {
                    const std::uint64_t v_first = static_cast<std::uint64_t>(task) * rows_per_task;
                    if (v_first >= v_count) {
                        return;
                    }
                    bin_rows(v_first, std::min(v_first + rows_per_task, v_count),
                             partials.data() + (task * bins));
                });
                // Integer counts, so this sum is the serial loop's answer exactly -- which is what
                // lets histogram_test keep comparing against an oracle rather than a tolerance.
                for (std::size_t task = 0; task < tasks; ++task) {
                    const std::uint64_t* from = partials.data() + (task * bins);
                    for (std::size_t bin = 0; bin < bins; ++bin) {
                        into[bin] += from[bin];
                    }
                }
              }
            });
        if (!walked) {
            return walked.error();
        }

        if (auto handed = hand_over(true); !handed) {
            return handed.error();
        }
        block_begin = block_end;
    }
    return {};
}

Result<CubeHistogramResult> ComputeCubeHistogram(const Store& store, const ImageDescriptor& descriptor,
                                                 const ChunkGeometry& geometry,
                                                 const CubeHistogramRequest& request,
                                                 const ReadOptions& options, WorkPool& workers) {
    const auto& node = descriptor.id;
    if (request.bins == 0 || request.bins > kMaxHistogramBins) {
        return MakeError(ErrorCode::invalid_argument,
                         "A histogram needs between 1 and " + std::to_string(kMaxHistogramBins) + " bins",
                         node);
    }
    if (request.spatial_sample == 0) {
        return MakeError(ErrorCode::invalid_argument, "A spatial sample of zero selects nothing", node);
    }
    auto axes = MapAxes(descriptor);
    if (!axes) {
        return axes.error();
    }
    const auto& map = axes.value();
    // The polarization and time checks are the same ones a fixed-range histogram makes; the bins and
    // bounds in this stand-in are only there to get past its own checks, and nothing reads them.
    HistogramRequest shape;
    shape.spectral = request.spectral;
    shape.polarization = request.polarization;
    shape.time = request.time;
    shape.bins = 1;
    shape.lower = 0.0;
    shape.upper = 1.0;
    if (auto valid = ValidateRequest(descriptor, map, shape); !valid) {
        return valid.error();
    }
    if (auto valid = ValidateSpectral(descriptor, map, request.spectral); !valid) {
        return valid.error();
    }

    std::size_t provisional =
        request.provisional_bins == 0 ? kDefaultProvisionalBins : request.provisional_bins;
    provisional = std::min<std::size_t>(provisional, kMaxHistogramBins);
    std::size_t rounded = 2;
    while (rounded < provisional) {
        rounded *= 2;
    }
    provisional = rounded;

    const auto walk = MakeWalk(store, descriptor, geometry, map, request.spectral, request.polarization,
                               request.time, request.spatial_sample, options);
    const std::uint64_t total_chunks =
        std::max<std::uint64_t>(1, walk.layer_chunks * ((request.spectral.count + walk.least_channels - 1) /
                                                        walk.least_channels));

    // One accumulator per worker -- not per task. Tasks are claimed from a shared counter, so task
    // n is run by a different thread on every plane, and a provisional histogram is half a megabyte
    // of scattered writes: keyed by task it would be dragged from one core's cache to another's
    // once per plane, which measured slower than not splitting at all. Keyed by the worker index
    // the pool hands out, each one stays on the thread that owns it for the whole walk.
    //
    // Padded to a cache line because Add writes the range and a bin on every pixel, and two
    // accumulators sharing a line would trade it between cores a billion times over a cube this
    // size.
    struct alignas(64) Accumulator {
        explicit Accumulator(std::size_t bins) : growing(bins) {}

        GrowingHistogram growing;
        double num_pixels = 0.0;
        double nan_count = 0.0;
        double sum = 0.0;
        double sum_sq = 0.0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
    };

    // What limits the split is cache, not the pool. A provisional histogram is eight bytes a bin --
    // half a megabyte at the default resolution -- and every worker writes to its own at random
    // while streaming its share of the pixels through the same cache. Past a couple of megabytes of
    // them the pixels evict the histograms and the pass gets slower the more workers it uses.
    //
    // Measured on a 512x512x7776 ASKAP cube, warm, against 8.8 s for not splitting at all: four
    // workers 4.9 s, eight 8.7 s, twenty-eight 16.5 s. So the cap is the budget divided by what one
    // accumulator costs, which at the default resolution comes out at four.
    constexpr std::size_t kAccumulatorCacheBytes = 2U << 20U;
    // Below this a task is not worth its share of a dispatch, so the read is binned in place.
    constexpr std::uint64_t kLeastPixelsPerTask = 1U << 16U;
    const std::size_t accumulator_bytes = provisional * sizeof(std::uint64_t);
    const std::size_t by_cache =
        std::max<std::size_t>(1, kAccumulatorCacheBytes / std::max<std::size_t>(1, accumulator_bytes));
    const std::size_t max_tasks = std::min(workers.size(), by_cache);
    std::vector<Accumulator> accumulators(max_tasks, Accumulator(provisional));

    CubeHistogramResult result;
    result.minimum = std::numeric_limits<double>::infinity();
    result.maximum = -std::numeric_limits<double>::infinity();
    result.sampled = request.spatial_sample > 1;

    std::uint64_t chunks_done = 0;
    const auto walked = WalkChannels(
        walk, 0, request.spectral.count, chunks_done,
        [&](std::uint64_t done) -> Result<void> {
            if (request.progress &&
                !request.progress(static_cast<double>(done) / static_cast<double>(total_chunks))) {
                return MakeError(ErrorCode::cancelled, "The histogram was cancelled by its caller", node);
            }
            return {};
        },
        [&](std::uint64_t, std::uint64_t channel_count, const float* base, std::uint64_t stride_u,
            std::uint64_t stride_v, std::uint64_t stride_z, std::uint64_t u_count,
            std::uint64_t v_count) {
            // Rows of the whole read, numbered across its planes, rather than rows of one plane.
            // A provisional histogram is half a megabyte, so a task has to be long enough to earn
            // the cache it pulls in: split per plane, a task was a few hundred thousand pixels
            // against that half megabyte and the pass ran slower than not splitting at all.
            const auto take_rows = [&](std::uint64_t first, std::uint64_t last, Accumulator& into) {
                // The scalars go into locals and are folded in once at the end. They are touched on
                // every pixel, and leaving them in the accumulator would have the compiler reload
                // them around each call into the histogram.
                double num_pixels = 0.0;
                double nan_count = 0.0;
                double sum = 0.0;
                double sum_sq = 0.0;
                double minimum = std::numeric_limits<double>::infinity();
                double maximum = -std::numeric_limits<double>::infinity();
                for (std::uint64_t index = first; index < last; ++index) {
                    const float* row =
                        base + ((index / v_count) * stride_z) + ((index % v_count) * stride_v);
                    for (std::uint64_t u = 0; u < u_count; ++u) {
                        const float value = row[u * stride_u];
                        if (!std::isfinite(value)) {
                            nan_count += 1.0;
                            continue;
                        }
                        const double v_value = value;
                        num_pixels += 1.0;
                        sum += v_value;
                        sum_sq += v_value * v_value;
                        minimum = std::min(minimum, v_value);
                        maximum = std::max(maximum, v_value);
                        into.growing.Add(value);
                    }
                }
                into.num_pixels += num_pixels;
                into.nan_count += nan_count;
                into.sum += sum;
                into.sum_sq += sum_sq;
                into.minimum = std::min(into.minimum, minimum);
                into.maximum = std::max(into.maximum, maximum);
            };

            const std::uint64_t rows = channel_count * v_count;
            const std::size_t tasks = PlanRowTasks(u_count, rows, max_tasks, kLeastPixelsPerTask);
            if (tasks <= 1) {
                take_rows(0, rows, accumulators.front());
                return;
            }

            const std::uint64_t rows_per_task = (rows + tasks - 1) / tasks;
            // By task, not by worker: the cap above can leave fewer accumulators than the pool has
            // workers, and a task is the thing there is one accumulator for. Two tasks never run at
            // once on the same accumulator because there are never more tasks than accumulators.
            workers.Run(tasks, [&](std::size_t task, std::size_t) {
                const std::uint64_t first = static_cast<std::uint64_t>(task) * rows_per_task;
                if (first >= rows) {
                    return;
                }
                take_rows(first, std::min(first + rows_per_task, rows), accumulators[task]);
            });
        });
    if (!walked) {
        return walked.error();
    }

    for (const auto& accumulator : accumulators) {
        result.num_pixels += accumulator.num_pixels;
        result.nan_count += accumulator.nan_count;
        result.sum += accumulator.sum;
        result.sum_sq += accumulator.sum_sq;
        result.minimum = std::min(result.minimum, accumulator.minimum);
        result.maximum = std::max(result.maximum, accumulator.maximum);
    }

    if (result.num_pixels == 0.0) {
        result.minimum = std::numeric_limits<double>::quiet_NaN();
        result.maximum = std::numeric_limits<double>::quiet_NaN();
        result.counts.assign(request.bins, 0);
        return result;
    }

    // Each accumulator re-aggregates onto the same target grid and the counts are added. No two
    // provisional histograms are ever merged with each other, which is what makes their ranges
    // having drifted apart not a problem: the grid they all land on comes from the extremes, and
    // those are exact. An accumulator that saw nothing contributes zeros.
    result.counts.assign(request.bins, 0);
    for (const auto& accumulator : accumulators) {
        const auto part = accumulator.growing.Aggregate(request.bins, result.minimum, result.maximum);
        for (std::size_t bin = 0; bin < result.counts.size(); ++bin) {
            result.counts[bin] += part[bin];
        }
    }
    return result;
}

}  // namespace carta::zarr::internal
