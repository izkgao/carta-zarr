/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "plane_histogram.h"

#include "chunk_blocks.h"
#include "axis_map.h"
#include "reduce/tuning.h"
#include "reduce/pass.h"
#include "reduce/plane_selection.h"
#include "reduce/provisional_histograms.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace carta::zarr::internal {

// The one place a HistogramBlock is handed its counts. HistogramBlock names this its friend, so the
// layout Counts reads is set here and nowhere a caller can reach.
class HistogramBlocks {
public:
    // `counts` is [channel][bin], block.bin_count wide.
    static void Hold(HistogramBlock& block, const std::uint64_t* counts) noexcept {
        block._counts = counts;
    }
};

namespace {

// The one check both histograms make. It takes the count rather than a request because their two
// requests are different types -- and because a cube histogram used to reach this by building a
// stand-in HistogramRequest with bins = 1 over [0, 1], bounds that nothing ever read.
Result<void> ValidateBins(const std::string& node, std::uint32_t bins) {
    if (bins == 0 || bins > kMaxHistogramBins) {
        return Error{ErrorCode::invalid_argument,
                     "A histogram needs between 1 and " + std::to_string(kMaxHistogramBins) + " bins",
                     node};
    }
    return {};
}

// The bounds a fixed-range histogram bins against. A cube histogram has none: its range comes from
// the data's own extremes, so there is nothing here for it to be checked against.
Result<void> ValidateRange(const std::string& node, const HistogramRequest& request) {
    // A zero-width range would divide by zero on every pixel. The caller decides what an image with
    // no finite pixel should look like -- there is more than one defensible answer -- so this says
    // no rather than inventing one.
    if (!(request.lower < request.upper) || !std::isfinite(request.lower) || !std::isfinite(request.upper)) {
        return Error{ErrorCode::invalid_argument,
                     "A histogram needs a finite range with a lower bound below its upper bound", node};
    }
    // Pixels are float, so the range is narrowed once and every pixel is binned against the
    // narrowed copy. A range that is finite and non-empty in double need not still be either: two
    // distinct doubles can narrow to one float, a bin width can underflow to zero or overflow to
    // infinity, and (value - lower) / 0 is a NaN whose conversion to a bin index is undefined.
    // Checking the narrowed values is therefore checking the ones the loop will actually use.
    if (const float lower = static_cast<float>(request.lower), upper = static_cast<float>(request.upper),
        width = static_cast<float>((request.upper - request.lower) / request.bins);
        !std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper) || !std::isfinite(width) ||
        !(width > 0.0F)) {
        return Error{ErrorCode::invalid_argument,
                     "A histogram needs a range that stays finite and non-empty, and bins that stay wider "
                     "than nothing, in the precision its pixels are counted in",
                     node};
    }
    return {};
}

}  // namespace

Result<void> ComputeHistogram(const ReducibleImage& image, const HistogramRequest& request,
                              const HistogramSink& sink, const ReadOptions& options) {
    const auto& descriptor = image.descriptor();
    const auto& source = image.source();
    const auto& node = descriptor.id;
    if (!sink) {
        return Error{ErrorCode::invalid_argument, "A histogram needs a sink", node};
    }
    if (auto valid = ValidateBins(node, request.bins); !valid) {
        return valid.error();
    }
    if (auto valid = ValidateRange(node, request); !valid) {
        return valid.error();
    }
    const auto planned = image.Plan(request.planes, 1, options);
    if (!planned) {
        return planned.error();
    }
    const auto& plan = planned.value();

    auto pass = PassOverPlane(source, plan, options, "The histogram was cancelled by its sink");

    // The caller's own sequence: divide in double, narrow the width, compare against the narrowed
    // bounds. Doing any one of those in the other type moves pixels across bin edges.
    const float width = static_cast<float>((request.upper - request.lower) / request.bins);
    const float lower = static_cast<float>(request.lower);
    const float upper = static_cast<float>(request.upper);
    const auto bins = static_cast<std::size_t>(request.bins);
    std::vector<std::uint64_t> counts;

    // How many private histograms the binning may split a plane into. Capped three ways: by the
    // pool, by memory -- see kHistogramPartialBudgetBytes -- and, inside the visit where the plane's
    // size is known, by whether there is enough work to be worth waking anyone for.
    const auto split = image.Split(kHistogramPartialBudgetBytes, bins * sizeof(std::uint64_t));
    // One allocation for the whole plan. Each task owns one row of it, so no two of them ever touch
    // the same bin and the sum at the end is the only place they meet.
    std::vector<std::uint64_t> partials;
    if (split.most() > 1) {
        partials.resize(split.most() * bins);
    }

    // One read's worth of pixels binned into the block's counts. Named rather than written
    // into the call below, because it is the longest of the three lambdas the pass is handed
    // and written in place it would bury the other two.
    const auto bin_slab = [&](const Slab& slab) {
        // Hoisted into locals so that the loops below are the same text they were when the
        // pass handed these over as eight separate arguments.
        const std::uint64_t stride_u = slab.stride_u;
        const std::uint64_t stride_v = slab.stride_v;
        const std::uint64_t stride_z = slab.stride_z;
        const std::uint64_t u_count = slab.u_count;
        const std::uint64_t v_count = slab.v_count;
        for (std::uint64_t offset = 0; offset < slab.channel_count; ++offset) {
            const float* plane = slab.pixels + (offset * stride_z);
            std::uint64_t* into = counts.data() + (static_cast<std::size_t>((slab.first_channel + offset).index) * bins);

            const auto bin_rows = [&](std::uint64_t v_first, std::uint64_t v_last, std::uint64_t* destination) {
                for (std::uint64_t v = v_first; v < v_last; ++v) {
                    const float* row = plane + (v * stride_v);
                    for (std::uint64_t u = 0; u < u_count; ++u) {
                        const float value = row[u * stride_u];
                        // The caller's own rule: a pixel outside the range is not counted, and
                        // NaN fails both comparisons.
                        if (lower <= value && value <= upper) {
                            // A range wider than FLT_MAX passes ValidateRange -- its bounds and its
                            // width each fit -- but the offset of a pixel in its upper part does not,
                            // and infinity converted to an index is undefined. Only then is the offset
                            // taken in double, where it fits; everywhere else the float sequence
                            // above is left alone, because it is the caller's.
                            const float offset = value - lower;
                            auto bin = std::isfinite(offset)
                                           ? static_cast<std::size_t>(offset / width)
                                           : static_cast<std::size_t>(
                                                 (static_cast<double>(value) - static_cast<double>(lower)) /
                                                 static_cast<double>(width));
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
            const std::size_t tasks = split.Tasks(u_count, v_count);
            if (tasks <= 1) {
                bin_rows(0, v_count, into);
                continue;
            }

            std::fill(partials.begin(), partials.begin() + static_cast<std::ptrdiff_t>(tasks * bins), 0);
            split.Run(tasks, v_count, [&](std::size_t task, std::uint64_t first, std::uint64_t last) {
                bin_rows(first, last, partials.data() + (task * bins));
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
    };

    return pass.InBlocks(
        bins * sizeof(std::uint64_t), request.emit_every_channels,
        [&](std::uint64_t length) { counts.assign(static_cast<std::size_t>(length) * bins, 0); }, bin_slab,
        [&](SelectionChannel first_channel, std::uint64_t length, bool complete, double completeness) {
            HistogramBlock block;
            // Out of the type and into the public block, which is the one place it happens.
            block.first_channel = first_channel.index;
            block.channel_count = length;
            block.bin_count = bins;
            HistogramBlocks::Hold(block, counts.data());
            block.complete = complete;
            block.completeness = completeness;
            return sink(block);
        });
}

Result<CubeHistogramResult> ComputeCubeHistogram(const ReducibleImage& image,
                                                 const CubeHistogramRequest& request,
                                                 const ReadOptions& options,
                                                 const CubeHistogramProgressCallback& progress) {
    const auto& descriptor = image.descriptor();
    const auto& source = image.source();
    const auto& node = descriptor.id;
    if (auto valid = ValidateBins(node, request.bins); !valid) {
        return valid.error();
    }
    if (request.spatial_sample == 0) {
        return Error{ErrorCode::invalid_argument, "A spatial sample of zero selects nothing", node};
    }
    const auto planned = image.Plan(request.planes, request.spatial_sample, options);
    if (!planned) {
        return planned.error();
    }
    const auto& plan = planned.value();

    ProvisionalHistograms histograms(image, request);

    auto pass = PassOverPlane(source, plan, options, "The histogram was cancelled by its caller");
    const auto walked = pass.Whole(
        [&](double fraction) {
            if (!progress) {
                return true;
            }
            CubeHistogramProgress update;
            update.progress = fraction;
            // By reference and lazily: re-aggregating on every read would cost more than the
            // binning does on a cube with thousands of them, and a caller that only draws a bar
            // never asks. Safe because the pass reports between reads, with no Add under way.
            update.snapshot = [&histograms]() { return histograms.Collect(); };
            return progress(update);
        },
        [&](const Slab& slab) { histograms.Add(slab); });
    if (!walked) {
        return walked.error();
    }

    return histograms.Collect();
}

}  // namespace carta::zarr::internal
