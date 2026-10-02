/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_REDUCE_PROVISIONAL_HISTOGRAMS_H_
#define CARTA_ZARR_SRC_REDUCE_PROVISIONAL_HISTOGRAMS_H_

// What a cube histogram counts into as it walks, and the answer it makes of that.
//
// All of this was inside ComputeCubeHistogram: the rule for how finely to bin, an accumulator per
// task, the per-pixel loop that feeds them, and the collection that adds six totals across tasks
// and re-aggregates each task's provisional histogram onto the caller's bins. The only way to reach
// any of it was a whole walk over a fixture, compared against a two-pass answer to a tolerance --
// and the public description of the first rule had already come apart from the code. Here each
// can be asked a question with a few rows of pixels and an exact answer.
//
// Not the spectral reduction's StatisticSlots, although it counts the same six statistics; that
// header says why the two stay apart.

#include "carta-zarr/reduce.h"

#include "reduce/growing_histogram.h"
#include "reduce/slab.h"
#include "reduce/task_split.h"
#include "reduce/tuning.h"
#include "reducible_image.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace carta::zarr::internal {

/**
 * The provisional histograms of one cube histogram, one per task, and the answer over them.
 *
 * Each task bins into a GrowingHistogram of its own whose range grows to fit what it sees, and
 * keeps the six statistics beside it. Nothing merges two of them until Collect, which takes the
 * extremes across all of them -- exact, however the bins fell -- and re-aggregates each one onto
 * the same target grid. No two provisional histograms are ever merged with each other, which is
 * what makes their ranges having drifted apart not a problem.
 *
 * There are never more tasks than histograms: the split that decides how many there are is the
 * same one Add divides each read with, and it is held here rather than by the caller, so the two
 * cannot be made from different budgets. ADR 0005.
 */
class ProvisionalHistograms {
public:
    // How many bins each provisional histogram holds, for a request asking for `bins` that stated
    // `provisional_bins`. See CubeHistogramRequest::provisional_bins, which this is the whole of.
    static std::size_t Resolution(std::uint32_t bins, std::uint32_t provisional_bins) {
        std::size_t provisional = provisional_bins;
        if (provisional == 0) {
            provisional = std::clamp<std::size_t>(static_cast<std::size_t>(bins) * kProvisionalBinsPerBin,
                                                  kLeastProvisionalBins, kMostProvisionalBins);
        }
        provisional = std::min<std::size_t>(provisional, kMaxHistogramBins);
        std::size_t rounded = 2;
        while (rounded < provisional) {
            rounded *= 2;
        }
        return rounded;
    }

    // The request is read once, here; its bins must already have been checked.
    //
    // What limits the split is cache, not the pool: the budget is kCubeAccumulatorCacheBytes, which
    // says why and what it was measured at. Divided by what one provisional histogram costs, it comes
    // out at four histograms at the default resolution.
    ProvisionalHistograms(const ReducibleImage& image, const CubeHistogramRequest& request)
        : _bins(request.bins),
          _sampled(request.spatial_sample > 1),
          _split(image.Split(kCubeAccumulatorCacheBytes,
                             Resolution(request.bins, request.provisional_bins) * sizeof(std::uint64_t))),
          _accumulators(_split.most(), Accumulator(Resolution(request.bins, request.provisional_bins))) {}

    ProvisionalHistograms(const ProvisionalHistograms&) = delete;
    ProvisionalHistograms& operator=(const ProvisionalHistograms&) = delete;

    // Count every pixel of one read.
    //
    // Rows of the whole read, numbered across its planes, rather than rows of one plane. A
    // provisional histogram is half a megabyte, so a task has to be long enough to earn the cache it
    // pulls in: split per plane, a task was a few hundred thousand pixels against that half megabyte
    // and the pass ran slower than not splitting at all.
    //
    // By task, not by worker: the cap can leave fewer histograms than the pool has workers, and a
    // task is the thing there is one histogram for. A read of one task binds into the first, on this
    // thread.
    void Add(const Slab& slab) {
        const std::uint64_t rows = slab.channel_count * slab.v_count;
        _split.Run(_split.Tasks(slab.u_count, rows), rows,
                   [&](std::size_t task, std::uint64_t first, std::uint64_t last) {
                       TakeRows(slab, first, last, _accumulators[task]);
                   });
    }

    // The answer over everything added so far. The extremes are exact for the pixels counted, so the
    // grid to re-aggregate onto is known at any point, not only at the last one -- which is what lets
    // a walk hand a caller a snapshot part of the way through.
    //
    // Not free: it re-aggregates every provisional histogram. Safe to call between reads, with no
    // Add under way.
    CubeHistogramResult Collect() const {
        CubeHistogramResult result;
        result.sampled = _sampled;
        auto& totals = result.totals;
        double smallest = std::numeric_limits<double>::infinity();
        double largest = -std::numeric_limits<double>::infinity();
        for (const auto& accumulator : _accumulators) {
            totals.num_pixels += accumulator.num_pixels;
            totals.nan_count += accumulator.nan_count;
            totals.sum += accumulator.sum;
            totals.sum_sq += accumulator.sum_sq;
            smallest = std::min(smallest, accumulator.minimum);
            largest = std::max(largest, accumulator.maximum);
        }

        result.counts.assign(_bins, 0);
        // Nothing finite was read, so the extrema stay at the NaN SpectralTotals starts them at: the
        // one place that answer is written, for this and for every spectral block.
        if (totals.num_pixels == 0.0) {
            return result;
        }
        totals.min = smallest;
        totals.max = largest;

        // Each provisional histogram re-aggregates onto the same target grid and the counts are
        // added. One that saw nothing contributes zeros.
        for (const auto& accumulator : _accumulators) {
            const auto part = accumulator.growing.Aggregate(_bins, totals.min, totals.max);
            for (std::size_t bin = 0; bin < result.counts.size(); ++bin) {
                result.counts[bin] += part[bin];
            }
        }
        return result;
    }

private:
    // One task's provisional histogram and the six statistics beside it.
    //
    // Tasks are claimed from a shared counter rather than divided up front, so task n is run by a
    // different thread on each read and an accumulator does move between cores as the pass
    // advances. The pool hands the body a worker index that would avoid that, and nothing here uses
    // it; whether it is worth anything is unmeasured. ADR 0005 records the state rather than
    // pretending it is settled.
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

    // Rows [first, last) of a read, numbered across its planes, into one accumulator.
    static void TakeRows(const Slab& slab, std::uint64_t first, std::uint64_t last, Accumulator& into) {
        // Hoisted so the per-pixel loop reads as it did inside the walk's visitor.
        const float* const base = slab.pixels;
        const std::uint64_t stride_u = slab.stride_u;
        const std::uint64_t stride_v = slab.stride_v;
        const std::uint64_t stride_z = slab.stride_z;
        const std::uint64_t u_count = slab.u_count;
        const std::uint64_t v_count = slab.v_count;
        // The scalars go into locals and are folded in once at the end. They are touched on every
        // pixel, and leaving them in the accumulator would have the compiler reload them around each
        // call into the histogram.
        double num_pixels = 0.0;
        double nan_count = 0.0;
        double sum = 0.0;
        double sum_sq = 0.0;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::uint64_t index = first; index < last; ++index) {
            const float* row = base + ((index / v_count) * stride_z) + ((index % v_count) * stride_v);
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
    }

    std::uint32_t _bins;
    bool _sampled;
    TaskSplit _split;
    std::vector<Accumulator> _accumulators;
};

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_PROVISIONAL_HISTOGRAMS_H_
