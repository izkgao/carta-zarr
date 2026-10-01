/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_REDUCE_TUNING_H_
#define CARTA_ZARR_SRC_REDUCE_TUNING_H_

// Numbers the reductions were tuned to, which a caller neither sets nor needs.
//
// They were public, in the header that describes what an image dataset is, which published them as
// though they were part of the contract. They are not: nothing outside src/reduce/ has ever read
// one. The limits a caller does have to respect -- kMaxHistogramBins and kMaxSpectralRegions --
// stay in carta-zarr/reduce.h, because a request that exceeds either is rejected.
//
// Four of them were measured on one machine and depend on its caches and its thread count rather
// than on what is being computed: how many pixels a task needs, and what the tasks' private
// accumulators may cost together. Each of those can be replaced when the library is built, through
// CARTA_ZARR_TUNING_OVERRIDES, so that carta-zarr-bench can measure another value on another
// machine. That is all it is for: a build that overrides one is a measurement, not a configuration,
// and none of them is a setting a deployment chooses. See ADR 0014. The others decide what a
// reduction answers -- how finely a cube histogram resolves a bin, how much one block may hold --
// and are not replaceable.

#include "carta-zarr/reduce.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace carta::zarr::internal {

// Every statistic, in the order a SpectralBlock lays them out.
inline constexpr std::array<Statistic, 6> kStatisticOrder{Statistic::num_pixels, Statistic::nan_count,
                                                          Statistic::sum, Statistic::sum_sq,
                                                          Statistic::min, Statistic::max};

// The provisional resolution a cube histogram bins at when the caller does not choose one.
//
// Sixteen provisional bins for every bin asked for, because what decides the error is how finely
// the walk resolves one target bin, not how many bins it holds in total -- and because the
// provisional histogram is eight bytes a bin and there is one per worker, so the ones nobody needs
// are paid for in cache. A thousand target bins get 16,384 of them, which is 128 kB.
//
// Held between 4,096 and 65,536. Measured on a billion-pixel ASKAP cube against the exact two-pass
// answer, with the walk on twenty-eight threads: 65,536 misplaced 0.003% of pixels in 3.64 s,
// 16,384 misplaced 0.004% in 2.40 s, 8,192 misplaced 0.006% in 2.21 s, 4,096 misplaced 0.011% in
// 2.14 s, and below that both numbers get worse at once. Sixteen is where the time has flattened
// out, with a few times the resolution the error would need.
inline constexpr std::uint32_t kProvisionalBinsPerBin = 16;
inline constexpr std::uint32_t kLeastProvisionalBins = 1u << 12;
inline constexpr std::uint32_t kMostProvisionalBins = 1u << 16;

// The memory one emitted block may occupy. emit_every_channels is reduced to fit it; see
// SpectralReduceRequest::emit_every_channels.
inline constexpr std::size_t kSpectralEmitBudgetBytes = 64u << 20;

// Below this a task is not worth its share of a dispatch, so the work is done on the calling thread
// instead. It is the floor TaskSplit hands PlanRowTasks for every reduction. All three used to have
// their own copy of the number -- two under one name and one under another, which is why the two
// looked like different rules rather than the same one written three times.
//
// A statement about pixels, whatever the thing being split is counted in: a plane histogram splits
// rows of a plane, a cube histogram splits rows across the planes of a read, and a spectral
// reduction splits chunk cells. PlanRowTasks multiplies out to pixels before it divides, so all
// three are asking the same question of the same number.
//
// It was 65,536, which split a 512 x 512 plane into four tasks however many cores there were. Measured
// on the 512x512x7776 ASKAP cube, warm, one user: at 16,384 the backend's exact cube histogram, which
// bins plane by plane, took 4.91 s against 5.50 s on a 28-core desktop and 5.15 s against 6.24 s on a
// 32-thread two-socket server; at 262,144 it took 6.99 s and 10.52 s. With eight users at once every
// value tried, 4,096 to 1,048,576, was within 6% on both, and regions within 3%. 4,096 gained a few
// percent more on both, which is less than it risks for a small read on a slower dispatch. ADR 0014.
#ifndef CARTA_ZARR_TUNING_LEAST_PIXELS_PER_TASK
#define CARTA_ZARR_TUNING_LEAST_PIXELS_PER_TASK (1u << 14)
#endif
inline constexpr std::uint64_t kLeastPixelsPerTask = CARTA_ZARR_TUNING_LEAST_PIXELS_PER_TASK;
static_assert(kLeastPixelsPerTask > 0, "CARTA_ZARR_TUNING_LEAST_PIXELS_PER_TASK must be positive");

// What every task's private accumulator may cost together, one budget per reduction, each handed to
// TaskSplit with what one accumulator costs. They are three numbers rather than one because they
// bound two different things, and the first two are not ADR 0005's cap.
//
// A spectral reduction's partials are the region totals for one slab, allocated and zeroed once per
// slab. The budget stops a reduction over thousands of regions from spending more on the split than
// on the pixels.
#ifndef CARTA_ZARR_TUNING_SPECTRAL_PARTIAL_BUDGET_BYTES
#define CARTA_ZARR_TUNING_SPECTRAL_PARTIAL_BUDGET_BYTES (16u << 20)
#endif
inline constexpr std::size_t kSpectralPartialBudgetBytes = CARTA_ZARR_TUNING_SPECTRAL_PARTIAL_BUDGET_BYTES;
static_assert(kSpectralPartialBudgetBytes > 0, "CARTA_ZARR_TUNING_SPECTRAL_PARTIAL_BUDGET_BYTES must be positive");

// A plane histogram's partials are a copy of the bins per task. A caller may ask for as many as
// kMaxHistogramBins, and a private copy of that for every worker is hundreds of megabytes for a pass
// that is supposed to stream, so this is what keeps a large bin count from turning a split into an
// allocation.
#ifndef CARTA_ZARR_TUNING_HISTOGRAM_PARTIAL_BUDGET_BYTES
#define CARTA_ZARR_TUNING_HISTOGRAM_PARTIAL_BUDGET_BYTES (64u << 20)
#endif
inline constexpr std::size_t kHistogramPartialBudgetBytes = CARTA_ZARR_TUNING_HISTOGRAM_PARTIAL_BUDGET_BYTES;
static_assert(kHistogramPartialBudgetBytes > 0, "CARTA_ZARR_TUNING_HISTOGRAM_PARTIAL_BUDGET_BYTES must be positive");

// A cube histogram's accumulators are a provisional histogram each, and what bounds them is cache
// rather than memory: this is the cap ADR 0005 is about. A provisional histogram is eight bytes a
// bin -- half a megabyte at the default resolution -- and every worker writes to its own at random
// while streaming its share of the pixels through the same cache. Past a couple of megabytes of them
// the pixels evict the histograms and the pass gets slower the more workers it uses.
//
// Measured on a 512x512x7776 ASKAP cube, warm, against 8.8 s for not splitting at all: four workers
// 4.9 s, eight 8.7 s, twenty-eight 16.5 s. At the default resolution this comes out at four.
#ifndef CARTA_ZARR_TUNING_CUBE_ACCUMULATOR_CACHE_BYTES
#define CARTA_ZARR_TUNING_CUBE_ACCUMULATOR_CACHE_BYTES (2u << 20)
#endif
inline constexpr std::size_t kCubeAccumulatorCacheBytes = CARTA_ZARR_TUNING_CUBE_ACCUMULATOR_CACHE_BYTES;
static_assert(kCubeAccumulatorCacheBytes > 0, "CARTA_ZARR_TUNING_CUBE_ACCUMULATOR_CACHE_BYTES must be positive");

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_TUNING_H_
