/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "spectral_reduce.h"

#include "chunk_blocks.h"
#include "axis_map.h"
#include "reduce/tuning.h"
#include "reduce/occupancy.h"
#include "reduce/pass.h"
#include "reduce/plane_selection.h"
#include "reduce/statistic_slots.h"
#include "zarr/pixel_selection.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace carta::zarr::internal {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

Result<void> ValidateRequest(const ImageDescriptor& descriptor, const AxisMap& axes,
                             const SpectralReduceRequest& request) {
    const auto& node = descriptor.id;
    const auto region_count = request.regions.size;
    if (region_count == 0 || request.regions.data == nullptr) {
        return Error{ErrorCode::invalid_argument, "A spectral reduction needs at least one region", node};
    }
    if (region_count > kMaxSpectralRegions) {
        return Error{ErrorCode::invalid_argument,
                     "A spectral reduction accepts at most " + std::to_string(kMaxSpectralRegions) +
                         " regions, not " + std::to_string(region_count),
                     node};
    }
    if (request.statistics.empty()) {
        return Error{ErrorCode::invalid_argument, "A spectral reduction needs at least one statistic", node};
    }

    const auto width = descriptor.axes.at(axes.x).length;
    const auto height = descriptor.axes.at(axes.y).length;
    for (std::size_t i = 0; i < region_count; ++i) {
        const auto& region = request.regions.data[i];
        if (region.width == 0 || region.height == 0) {
            return Error{ErrorCode::invalid_argument, "Region " + std::to_string(i) + " is empty", node};
        }
        if (region.x_start >= width || region.width > width - region.x_start || region.y_start >= height ||
            region.height > height - region.y_start) {
            return Error{ErrorCode::invalid_argument,
                         "Region " + std::to_string(i) + " falls outside the image", node};
        }
        // Both inside the image, so their product is a pixel count and cannot overflow.
        const auto box = region.width * region.height;
        const auto& mask = region.mask;
        if (mask.data != nullptr ? mask.size != box : mask.size != 0) {
            return Error{ErrorCode::invalid_argument,
                         "Region " + std::to_string(i) + " has a mask of " + std::to_string(mask.size) +
                             " elements for a box of " + std::to_string(box),
                         node};
        }
    }

    return {};
}

// The per-pixel loop, written so that a compiler can vectorise it.
//
// Both template parameters are loop invariants that used to be runtime tests, and each one on its
// own was enough to stop vectorisation: the u stride is 1 for every image, since the walk runs
// along the axis the store varies fastest, but the compiler cannot know that and pays a multiply
// per pixel for the possibility; and a mask cost a branch per pixel. Only a raster too fragmented to
// be worth runs still takes that branch -- every other masked region reaches here as runs, which
// Occupancy::Of makes when the caller did not.
//
// The finiteness test is branchless for the same reason. A non-finite value contributes zero to the
// sums and its own identity to the extrema, which is exactly what excluding it means, so there is
// nothing an if would do that arithmetic does not.
template <bool kUnitStride, bool kMasked>
void AccumulateRow(const float* row, std::uint64_t stride, std::uint64_t count, const std::uint8_t* selected,
                   std::uint64_t mask_stride, RowTotals& totals) {
    std::uint64_t good = 0;
    std::uint64_t selected_count = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double smallest = kInfinity;
    double largest = -kInfinity;

    for (std::uint64_t i = 0; i < count; ++i) {
        if (kMasked && selected[i * mask_stride] == 0) {
            continue;
        }
        ++selected_count;
        const float value = kUnitStride ? row[i] : row[i * stride];
        const bool finite = std::isfinite(value);
        const double clean = finite ? static_cast<double>(value) : 0.0;
        good += static_cast<std::uint64_t>(finite);
        sum += clean;
        sum_sq += clean * clean;
        smallest = std::min(smallest, finite ? clean : kInfinity);
        largest = std::max(largest, finite ? clean : -kInfinity);
    }

    // Added rather than assigned: a row described by runs is several spans, and they share one set
    // of totals. A row that is one span starts from the same zeros and reaches the same answer.
    totals.good += good;
    totals.bad += selected_count - good;
    totals.sum += sum;
    totals.sum_sq += sum_sq;
    totals.smallest = std::min(totals.smallest, smallest);
    totals.largest = std::max(totals.largest, largest);
}

}  // namespace

Result<void> ReduceSpectral(const ReducibleImage& image, const SpectralReduceRequest& request,
                            const SpectralSink& sink, const ReadOptions& options) {
    const auto& descriptor = image.descriptor();
    const auto& source = image.source();
    const auto& map = image.map();
    const auto& node = descriptor.id;
    if (!sink) {
        return Error{ErrorCode::invalid_argument, "A spectral reduction needs a sink", node};
    }

    if (auto valid = ValidateRequest(descriptor, map, request); !valid) {
        return valid.error();
    }

    // Checked here as well as inside each slab request, so that a bad range is one error naming the
    // axis rather than a partial reduction that fails on some later slab.
    const auto planned = image.Plan(request.planes, 1, options);
    if (!planned) {
        return planned.error();
    }
    const auto& plan = planned.value();

    // The caller's regions, placed on the walk's axes and indexed by the chunks they occupy. Which
    // spatial axis the store varies fastest decides both -- where a region lands, and whether its
    // runs are the kind worth taking -- so the plan is asked for it once and the occupancy is told.
    // Asked of the plan rather than of the geometry: the plan applied that rule when it picked
    // axis_u, and working it out again here would be a second place for it to be got wrong.
    auto occupancy_result =
        Occupancy::Of(request.regions, plan.chunk_u, plan.chunk_v,
                      plan.SwapsSpatial() ? AxisRole::spatial_y : AxisRole::spatial_x, node);
    if (!occupancy_result) {
        return occupancy_result.error();
    }
    const auto& occupancy = occupancy_result.value();
    const auto& regions = occupancy.regions();
    // What the reduction reads, cut once against the plan's budget. They do not depend on which
    // channels a block covers, so every block walks the same ones.
    const auto footprints = occupancy.Footprints(plan.ChunksPerRead());

    // The requested statistics, in the one order a block reports them.
    const auto layout = StatisticLayout::Of(request.statistics);

    // The budget is over the chunk data a slab decodes, not the pixels it keeps. A one-pixel region
    // asks for almost nothing and still decodes an entire chunk per chunk it touches, so sizing by
    // the region's own area would let a cursor-sized request pull an unbounded amount through. That
    // is the plan's rule as much as it is this one's, which is why the plan holds it.

    // The pass walks the footprints, and counts a block's progress in the chunks they occupy
    // together. Not the plan's layer, which is the whole plane: a reduction spends its emit budget
    // against the region set it was given.
    //
    // How many channels one emitted block may hold is the pass's to work out. The hint is the
    // caller's, the budgets are the library's and the chunk alignment is the walk's; the smallest
    // wins, and the block reports what it used.
    //
    // Without a hint a block costs one budget of decoded bytes, the same invariant a piece of Read
    // carries, which is why it is free: the block spends whatever the spatial walk left over. A
    // small region leaves almost all of it, so the block spans many chunks along the spectrum. A
    // region covering the image spends the budget spatially, what is left affords one chunk along
    // the spectrum, and the block becomes the single chunk layer the walk is already reading -- the
    // results are handed over a layer at a time because that is when they are finished, not sooner.
    //
    // Emitting only at the end instead, which is what a zero used to mean, is silent for as long as
    // the whole reduction takes. One layer of a 7763x4742 image is 160 MiB and 70 ms; a thousand
    // channels of it is a minute of work with no partial answer and nowhere to cancel.
    auto pass =
        PassOverFootprints(source, plan, options, footprints, "The spectral reduction was cancelled by its sink");

    StatisticSlots accumulator;
    // One private accumulator per task, reused across slabs. See the dispatch below. "Partials" as
    // in the plane histogram, and deliberately not "sinks": a sink in this library is where a
    // finished block goes, and this one is named in the same function as the caller's SpectralSink.
    std::vector<StatisticSlots> partials;

    const auto reset_block = [&](std::uint64_t length) { accumulator.Reset(layout, request.regions.size, length); };

    // One read's worth of pixels, accumulated into the block's own totals.
    //
    // Named rather than written into the call below, for the reason plane_histogram.cc gives for
    // bin_slab, and this one carries another lambda inside it.
    //
    // The pass hands it the footprint each slab was read over. Its bounds arrive as an argument and
    // are unpacked into the names the body already used, so that the accumulation is the same text
    // it has been since it stopped doing its own reading. It stays a lambda passed as a template
    // parameter, never a std::function: the per-pixel loop inlines through it. ADR 0005.
    const auto accumulate_slab = [&](const OccupiedFootprint& footprint, const Slab& slab) {
        const std::uint64_t chunk_cv_begin = footprint.chunk_cv_begin;
        const std::uint64_t chunk_cv_end = footprint.chunk_cv_end;
        const std::uint64_t chunk_cu_begin = footprint.chunk_cu_begin;
        const std::uint64_t chunk_cu_end = footprint.chunk_cu_end;
        const std::uint64_t u_begin = footprint.slab.u_start;
        const std::uint64_t u_end = footprint.slab.u_start + footprint.slab.u_count;
        const std::uint64_t v_begin = footprint.slab.v_start;
        const std::uint64_t v_end = footprint.slab.v_start + footprint.slab.v_count;

        // Into locals so that the accumulation below is the text it was when this
        // function did its own reading.
        const float* const slab_pixels = slab.pixels;
        const std::uint64_t stride_u = slab.stride_u;
        const std::uint64_t stride_v = slab.stride_v;
        const std::uint64_t stride_z = slab.stride_z;
        const std::uint64_t slab_length = slab.channel_count;

        const std::uint64_t cv_span = chunk_cv_end - chunk_cv_begin;
        const std::uint64_t cu_span = chunk_cu_end - chunk_cu_begin;
        const std::uint64_t cells = std::max<std::uint64_t>(1, cv_span * cu_span);
        const std::uint64_t units = slab_length * cells;
        const std::size_t partial_bytes = std::max<std::size_t>(
            1, layout.BytesPerChannel(request.regions.size) * static_cast<std::size_t>(slab_length));

        // One (channel, chunk cell) unit of the accumulation, into a private partial the
        // length of this slab. Private because two units of the same channel can touch the
        // same region -- a region wider than a chunk spans several cells -- so they would
        // otherwise be adding to one double.
        const auto accumulate_unit = [&](std::uint64_t channel, std::uint64_t cell_cv,
                                         std::uint64_t cell_cu, StatisticSlots& partial) {
            const float* plane = slab_pixels + (channel * stride_z);

            const std::uint64_t chunk_cv = cell_cv;
            const std::uint64_t cell_v0 = std::max(v_begin, chunk_cv * plan.chunk_v);
            const std::uint64_t cell_v1 = std::min(v_end, (chunk_cv + 1) * plan.chunk_v);
            if (cell_v0 >= cell_v1) {
                return;
            }
            const std::uint64_t chunk_cu = cell_cu;
            const std::uint64_t cell_u0 = std::max(u_begin, chunk_cu * plan.chunk_u);
            const std::uint64_t cell_u1 = std::min(u_end, (chunk_cu + 1) * plan.chunk_u);
            if (cell_u0 >= cell_u1) {
                return;
            }

            for (const auto index : occupancy.RegionsTouching(chunk_cu, chunk_cv)) {
                const std::size_t r = index;
                const auto& region = regions.at(r);

                const std::uint64_t ru0 = std::max(cell_u0, region.u_start);
                const std::uint64_t ru1 = std::min(cell_u1, region.u_start + region.u_size);
                const std::uint64_t rv0 = std::max(cell_v0, region.v_start);
                const std::uint64_t rv1 = std::min(cell_v1, region.v_start + region.v_size);
                if (ru0 >= ru1 || rv0 >= rv1) {
                    continue;
                }

                for (std::uint64_t y = rv0; y < rv1; ++y) {
                    RowTotals totals;
                    if (region.runs != nullptr) {
                        // Every pixel of a run is selected, so each one goes
                        // through the same loop an unmasked region uses and the
                        // per-pixel test never happens.
                        const auto r = static_cast<std::size_t>(y - region.v_start);
                        for (auto k = region.run_offsets[r];
                             k < region.run_offsets[r + 1]; ++k) {
                            const std::uint64_t run_begin =
                                region.u_start + region.runs[2 * k];
                            if (run_begin >= ru1) {
                                break;  // runs ascend, so the rest are past the cell
                            }
                            const std::uint64_t run_end =
                                region.u_start + region.runs[(2 * k) + 1];
                            const std::uint64_t first = std::max(ru0, run_begin);
                            const std::uint64_t last = std::min(ru1, run_end);
                            if (first >= last) {
                                continue;
                            }
                            const float* span = plane + ((y - v_begin) * stride_v) +
                                                ((first - u_begin) * stride_u);
                            if (stride_u == 1) {
                                AccumulateRow<true, false>(span, 1, last - first, nullptr, 1,
                                                           totals);
                            } else {
                                AccumulateRow<false, false>(span, stride_u, last - first,
                                                            nullptr, 1, totals);
                            }
                        }
                    } else {
                        const float* pixel_row =
                            plane + ((y - v_begin) * stride_v) + ((ru0 - u_begin) * stride_u);
                        const std::uint8_t* selected =
                            region.mask == nullptr
                                ? nullptr
                                : region.mask +
                                      ((y - region.v_start) * region.mask_v_stride) +
                                      ((ru0 - region.u_start) * region.mask_u_stride);
                        const std::uint64_t run = ru1 - ru0;
                        const std::uint64_t mask_step = region.mask_u_stride;
                        if (stride_u == 1) {
                            if (selected == nullptr) {
                                AccumulateRow<true, false>(pixel_row, 1, run, nullptr, 1, totals);
                            } else {
                                AccumulateRow<true, true>(pixel_row, 1, run, selected, mask_step,
                                                          totals);
                            }
                        } else if (selected == nullptr) {
                            AccumulateRow<false, false>(pixel_row, stride_u, run, nullptr, 1,
                                                        totals);
                        } else {
                            AccumulateRow<false, true>(pixel_row, stride_u, run, selected,
                                                       mask_step, totals);
                        }
                    }

                    partial.Fold(r, channel, totals);
                }
            }
        };

        // The unit order is the order the nested loops used to run in -- channel, then
        // chunk row, then chunk column -- so a single task reproduces the old sums bit
        // for bit. Several tasks do not: each one sums its own contiguous run of units
        // and the partials are added in task order, which is deterministic but a
        // different association.
        // The statistics are doubles over as many as 5x10^11 values, and partial sums
        // are if anything the more accurate arrangement; the tests compare to 1e-9
        // relative for exactly this reason, and the counts and extrema are unaffected
        // because integers and min/max do not care what order they arrive in.
        // One per task, so the split is also an allocation and a memset of this size
        // once per slab, which is why the split is made here rather than once: a
        // partial is as long as the slab. See kSpectralPartialBudgetBytes.
        const auto split = image.Split(kSpectralPartialBudgetBytes, partial_bytes);
        const std::size_t tasks = split.Tasks(plan.chunk_u * plan.chunk_v, units);

        if (partials.size() < tasks) {
            partials.resize(tasks);
        }
        for (std::size_t task = 0; task < tasks; ++task) {
            partials.at(task).Reset(layout, request.regions.size, slab_length);
        }

        // Contiguous runs of units, cut the way the histograms cut their rows. They were
        // dealt out round-robin once, for no recorded reason, and that measured slower
        // rather than better balanced: on a 7763x4742 plane of 31x19 chunk cells, 5-13%
        // slower at ten threads and no different at four, and never measurably faster.
        split.Run(tasks, units, [&](std::size_t task, std::uint64_t first, std::uint64_t last) {
            StatisticSlots& partial = partials.at(task);
            for (std::uint64_t unit = first; unit < last; ++unit) {
                const std::uint64_t channel = unit / cells;
                const std::uint64_t cell = unit % cells;
                accumulate_unit(channel, chunk_cv_begin + (cell / cu_span),
                                chunk_cu_begin + (cell % cu_span), partial);
            }
        });

        accumulator.MergeInOrder(partials.data(), tasks, slab.first_channel.index);
    };

    // Out of the type and into the public block, which is the one place it happens.
    const auto hand_over = [&](SelectionChannel first_channel, [[maybe_unused]] std::uint64_t length, bool complete,
                               double completeness) {
        assert(length == accumulator.channels());
        return accumulator.HandOver(first_channel.index, complete, completeness, sink);
    };

    return pass.InBlocks(layout.BytesPerChannel(request.regions.size), request.emit_every_channels, reset_block,
                         accumulate_slab, hand_over);
}

}  // namespace carta::zarr::internal
