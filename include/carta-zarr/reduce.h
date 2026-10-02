/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_REDUCE_H_
#define CARTA_ZARR_REDUCE_H_

// Asking for an answer about pixels rather than for the pixels: statistics over regions of a
// spectrum, and histograms of a plane or of a whole cube.
//
// Separate from read.h because a consumer that only opens images and reads them needs none of it,
// and because this vocabulary is where the library has been changing: it arrived whole in the
// fourteen commits before this split, and every one of them edited the header that describes what
// an image dataset is.

#include "carta-zarr/read.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace carta::zarr {

// One statistic a spectral reduction can produce. The enumerators are bit flags so that a request
// names a set in one field.
//
// There is deliberately no output-type option here. Every one of these is an N:1 reduction over as
// many as 5 x 10^11 values, accumulation is always double, and a float32 sum at that scale silently
// stops adding: offering the choice would only let a caller ask for a wrong answer quickly.
//
// num_pixels and nan_count partition the region: num_pixels counts the pixels that sum, sum_sq, min
// and max were taken over -- finite, and not rejected by the image's pixel mask -- while nan_count
// counts every other pixel the mask selected. Their total is the number of pixels the mask
// selected, so a caller needing the denominator of a mean asks for num_pixels rather than deriving
// it from the mask.
enum class Statistic : std::uint32_t {
    num_pixels = 1u << 0,
    nan_count = 1u << 1,
    sum = 1u << 2,
    sum_sq = 1u << 3,
    min = 1u << 4,
    max = 1u << 5,
};

// A set of statistics, made by joining them with |: `Statistic::sum | Statistic::num_pixels`. A single
// statistic is the set of just it.
//
// A type of its own rather than the integer it was, because an integer could name a statistic that
// does not exist -- `statistics = 0x40` compiled, and a bit past the last one was simply never
// accumulated -- and because `set & statistic` read as a question and was a number. Nothing makes one
// from bits, so every set holds only statistics there are.
class StatisticSet {
public:
    constexpr StatisticSet() noexcept = default;
    constexpr StatisticSet(Statistic statistic) noexcept : _bits(static_cast<std::uint32_t>(statistic)) {}

    constexpr bool empty() const noexcept {
        return _bits == 0;
    }
    // Whether every statistic in `other` is in this one. The empty set is in every set.
    constexpr bool Contains(StatisticSet other) const noexcept {
        return (other._bits & ~_bits) == 0;
    }

    friend constexpr StatisticSet operator|(StatisticSet a, StatisticSet b) noexcept {
        return StatisticSet(a._bits | b._bits);
    }
    constexpr StatisticSet& operator|=(StatisticSet other) noexcept {
        _bits |= other._bits;
        return *this;
    }
    friend constexpr bool operator==(StatisticSet a, StatisticSet b) noexcept {
        return a._bits == b._bits;
    }
    friend constexpr bool operator!=(StatisticSet a, StatisticSet b) noexcept {
        return a._bits != b._bits;
    }

private:
    constexpr explicit StatisticSet(std::uint32_t bits) noexcept : _bits(bits) {}

    std::uint32_t _bits = 0;
};

// Two statistics make a set. Needed beside StatisticSet's own |, which is found only when one side
// is already a set.
inline constexpr StatisticSet operator|(Statistic a, Statistic b) noexcept {
    return StatisticSet(a) | StatisticSet(b);
}


// Which planes of an image a reduction is over: a range along the spectral coordinate, one
// polarization, and one time.
//
// The three travel together because a reduction is always over whole planes, and they are one type
// rather than three fields because they are answered together: whether an image can serve this is
// one question about that image's axes. An ordinary read says the same thing as one range per axis
// instead, which is why ReadRequest carries no plane selection.
//
// polarization and time are indices into their own axes. An image that has no such axis accepts
// zero and nothing else: index 0 of an axis an image does not have is the image itself, and any
// other index names a plane that does not exist.
struct PlaneSelection {
    // The channels the reduction is over, along the image's spectral axis. Strides are honoured.
    Range spectral;
    std::uint64_t polarization = 0;
    std::uint64_t time = 0;
};


// A 2D (x, y) mask in logical image coordinates, addressed row-major with x fastest.
//
// This is a borrowed view: the raster must stay valid until ReduceSpectral returns, and the
// library never retains it. The shape is the region's bounding box rather than the image, which is
// what makes handing over tens of thousands of regions at once affordable -- 5,792 PV boxes on a
// 4096^2 image describe themselves in 2.5 MB.
//
// A mask with no data selects the whole bounding box, and an unmasked region is written without one:
// `{x_start, y_start, width, height}`. That branch is required, not a convenience: an unrotated
// rectangle reaches a caller as a box with no raster mask at all, and it is the most common region
// shape there is.
struct RegionMask {
    std::uint64_t x_start = 0;
    std::uint64_t y_start = 0;
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    // A raster is all a region needs. The reduction turns it into runs along whichever spatial axis
    // the store varies fastest, in one pass over it -- about 2 ms for a 7763x4742 bounding box -- so
    // that it finds the chunks a region occupies from the runs, and accumulates every run with the
    // loop an unmasked region uses. A raster too fragmented to be worth runs is read as a raster.
    //
    // width * height elements when there is one, and nothing when there is not; a reduction refuses
    // any other size. It was a bare pointer, whose length a reduction could only assume, so a raster
    // cut for another box was read past its end.
    BufferView<const std::uint8_t> mask;
};

// One plane histogram request: bin every pixel of each plane over a fixed range.
//
// The bounds arrive as double and the per-pixel arithmetic happens in float, which is not an
// oversight but the rule a caller already follows: it keeps the range as double, divides by the bin
// count in double, and narrows only the resulting width to float. Comparing a pixel then happens
// against the narrowed bounds. An integer count is the one thing here that can match exactly rather
// than to a tolerance, so the sequence of roundings is copied rather than approximated.
//
// A pixel outside [lower, upper] is not counted, and neither is one that is not finite, which is
// the same rule: NaN fails both comparisons.
struct HistogramRequest {
    PlaneSelection planes;
    std::uint32_t bins = 0;
    double lower = 0.0;
    double upper = 0.0;
    // How often to hand results back; zero lets the library choose, as in SpectralReduceRequest.
    std::uint32_t emit_every_channels = 0;
};

namespace internal {
class HistogramBlocks;
}  // namespace internal

// Bin counts for a run of planes, valid only inside the sink call.
//
// Read through Counts, as a SpectralBlock is read through Series: how the channels are laid out is
// the library's alone, which is why only the library can fill one in. It was a bare pointer with the
// layout written in a comment, the one block in the library a caller indexed by hand.
struct HistogramBlock {
    // Index into the request's spectral selection, as SpectralBlock::first_channel is.
    std::uint64_t first_channel = 0;
    std::uint64_t channel_count = 0;
    std::size_t bin_count = 0;
    // As in SpectralBlock: a block whose walk takes more than one read is handed over as it fills.
    bool complete = true;
    double completeness = 1.0;

    // The bin_count counts of one channel of this block, counted from first_channel, lowest bin
    // first. channel < channel_count.
    const std::uint64_t* Counts(std::uint64_t channel) const noexcept {
        assert(channel < channel_count);
        return _counts + (static_cast<std::size_t>(channel) * bin_count);
    }

private:
    friend class internal::HistogramBlocks;

    const std::uint64_t* _counts = nullptr;
};

using HistogramSink = std::function<bool(const HistogramBlock&)>;

// The six statistics a reduction counts over a set of pixels: one region at one channel of a
// SpectralBlock, or everything a cube histogram's selection covers.
//
// What nothing was counted into reads as zero for the counts and sums and NaN for the extrema, since
// there is no smallest value of nothing -- and that is what a default-constructed one holds, so a
// total nobody filled in cannot pass for a range that was found. A caller deriving a mean from these
// sees the division it must not perform either way.
//
// In a SpectralBlock, a statistic the block does not carry reads the same way. A caller that has to
// tell "not asked for" from "nothing there" asks the block, with Carries.
//
// One type for both, rather than a set of fields per result: the cube histogram used to spell the
// same six its own way, with the extrema named minimum and maximum, and the two defaults drifted
// apart until c66699d put them back.
struct SpectralTotals {
    double num_pixels = 0.0;
    double nan_count = 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double min = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
};

// Everything one pass can say about the selection.
struct CubeHistogramResult {
    // Over every pixel the selection covers, or every one the sample kept. The extrema are exact,
    // whatever the bin edges did, and NaN when nothing finite was read.
    SpectralTotals totals;
    // `bins` counts over [totals.min, totals.max].
    std::vector<std::uint64_t> counts;
    // Whether spatial_sample kept this from being every pixel.
    bool sampled = false;
};

// What the walk hands the caller as it goes.
//
// `snapshot` answers over the pixels read so far, which is a legitimate histogram of them: the
// extremes are tracked exactly as the walk runs, so the bin edges it re-aggregates onto are the
// right ones for what it has seen. They move as more of the cube arrives, which is why it hands
// back a whole result rather than counts alone -- the caller needs the range that goes with them.
//
// It is not free: it re-aggregates every worker's provisional histogram. A caller reporting on a
// timer should call it only when it reports, not on every update.
struct CubeHistogramProgress {
    // The fraction of the walk's chunks that are done.
    double progress = 0.0;
    std::function<CubeHistogramResult()> snapshot;
};

// Called as a cube histogram's walk advances: before every read after the first, so a walk that takes
// one read never calls it, and nothing is called once the last read is done. Returning false cancels,
// which then reports cancelled.
//
// An argument of Image::ComputeCubeHistogram, as ProgressCallback is of Image::Read, rather than a
// field of CubeHistogramRequest. A request says what to compute and is data; this is the caller's
// code, run while it is computed. As a field it was the one request in the library that carried
// behaviour, which the reasoning for ProgressCallback had already ruled out.
using CubeHistogramProgressCallback = std::function<bool(const CubeHistogramProgress&)>;

// One histogram for the whole selection in a single pass, for a caller that does not know the range
// in advance.
//
// The two-pass shape exists because bin edges come from the data's own extremes, so binning has to
// wait for the pass that finds them. This does both at once: it bins into a provisional histogram
// far finer than the one asked for, doubles that histogram's range and merges its bins in pairs
// whenever a pixel falls outside -- no pixel is lost, only resolution -- and re-aggregates at the
// end over the extremes it tracked exactly along the way.
//
// What is given up is where the bin edges land. A provisional bin straddling a target edge is split
// between the two in proportion to the overlap, so what is left is the assumption that pixels are
// spread evenly inside one provisional bin -- the same assumption the caller's own percentile makes
// between target bins, and a far smaller error than giving the straddling bin to one side whole.
// The extremes, the total count and the sums are exact.
//
// The walk runs on as many threads as the context was given, each with a provisional histogram of
// its own that it re-aggregates onto the same target grid at the end, so the counts depend on the
// thread count and a caller who needs the same ones every time asks for one decode thread. It is
// not much of a dependence. On a billion-pixel ASKAP cube against the two-pass answer over the same
// range, one thread misplaced 0.007% of pixels and twenty-eight misplaced 0.004%; the two disagreed
// with each other about 0.007% of pixels, no target bin by more than 0.002 of an average one, and
// every percentile CARTA offers landed within 0.003 of a bin of the two-pass answer on both.
//
// The result is one histogram for the whole selection, not one per plane: no plane's counts can be
// settled until the last pixel has been read, and holding a provisional histogram for every plane
// of a deep cube is gigabytes. A caller that adds its planes together loses nothing by it.
struct CubeHistogramRequest {
    PlaneSelection planes;
    // The bins the caller wants back.
    std::uint32_t bins = 0;
    // The resolution the walk bins at: how many bins each of its provisional histograms holds before
    // they are re-aggregated onto `bins`. Larger is more faithful and costs eight bytes a bin for
    // every task the walk is split into.
    //
    // Zero takes the library's default, which is sixteen times the bins asked for, held between
    // 4,096 and 65,536 -- the range where tuning.h measured the error and the time to flatten out. A
    // value stated here is taken as given, up to kMaxHistogramBins, so a caller can ask for coarser
    // than the default as well as finer. Either way it is rounded up to a power of two, and to at
    // least two, so that merging in pairs leaves nothing behind.
    std::uint32_t provisional_bins = 0;
    // Take every nth pixel along both spatial axes. One reads every pixel.
    //
    // Worth knowing before reaching for it: a stride below the chunk width saves no decompression,
    // because a chunk comes back whole however few of its pixels are wanted. It pays when it steps
    // over whole chunks.
    std::uint64_t spatial_sample = 1;
};



// The largest number of bins one histogram accepts. CARTA's automatic bin count is the square root
// of the plane's pixel count, which is 32,768 for the largest image anyone has; this is a guard
// against an uninitialised count, not a capacity estimate.
inline constexpr std::uint32_t kMaxHistogramBins = 1u << 20;

// The largest number of regions one reduction accepts.
//
// This is a structural guard, not a capacity estimate: the largest legitimate request is one box
// per pixel along an image diagonal, which is 46,341 for a 32768^2 image. The bound exists so that
// a caller passing an uninitialised count gets invalid_argument instead of a 16 GB allocation.
inline constexpr std::size_t kMaxSpectralRegions = 1u << 20;


struct SpectralReduceRequest {
    PlaneSelection planes;
    // The regions, all reduced in a single pass over the pixels. Borrowed until ReduceSpectral
    // returns, as each one's raster is.
    BufferView<const RegionMask> regions;
    StatisticSet statistics;
    // How often to hand results back, as a hint rather than a contract. Zero lets the library
    // choose, which is what most callers want: it emits as often as it can without making the reads
    // any smaller, so a region covering the image reports a chunk layer at a time while a
    // cursor-sized one reports far less often, and neither pays for the difference.
    //
    // The library lowers a hint to fit a 64 MiB block budget and then to a whole number of
    // spectral chunks, because a block boundary inside a chunk would split one decode's results
    // across two blocks. A caller that wants the whole reduction in one block asks for
    // SpectralReduceRequest::planes.spectral.count. The value actually used is reported as
    // SpectralBlock::channel_count, which a caller has to read anyway.
    std::uint32_t emit_every_channels = 0;
};

namespace internal {
class StatisticSlots;
}  // namespace internal

// One contiguous run of channels, for every region and every requested statistic.
//
// Read it through Series, Totals and Carries. The values are owned by the library and are valid
// only for the duration of the call. How they are laid out is the library's alone, which is why
// only the library can fill one in.
//
// min and max are reported as NaN for a channel whose region contributed no finite pixel, since
// there is no such thing as the smallest value of nothing. sum and sum_sq are zero in that case,
// which is what they are worth, and num_pixels is zero -- so a caller deriving a mean sees the
// division it must not perform.
struct SpectralBlock {
    // Index into the request's spectral selection, not an image channel: the image channel is
    // planes.spectral.start + (first_channel + i) * planes.spectral.stride.
    std::uint64_t first_channel = 0;
    std::uint64_t channel_count = 0;
    // Whether these values are final. A reduction whose block spans more than one read hands the
    // block over as it fills, so that a caller has something to show and somewhere to stop long
    // before the last pixel of the block is read. The same channels arrive again, refined, and a
    // last time with complete set; a caller that only wants finished answers ignores the rest.
    //
    // The statistics of an unfinished block are honest over the pixels read so far: the counts and
    // sums are partial and grow, the extrema are over a subset, and anything derived from them --
    // a mean, an RMS -- is an estimate that converges. NumPixels says how much is behind them.
    bool complete = true;
    // The fraction of this block's chunks that are in the values, in [0, 1]. One when complete.
    double completeness = 1.0;
    // How many regions the block reports, which is how many the request gave.
    std::size_t region_count = 0;

    // Whether the block carries every statistic in `wanted`, which may be one. The empty set is always
    // carried.
    bool Carries(StatisticSet wanted) const noexcept {
        StatisticSet carried;
        for (std::size_t slot = 0; slot < _statistic_count; ++slot) {
            carried |= _statistics[slot];
        }
        return carried.Contains(wanted);
    }

    // One region's channel_count values of one statistic, in channel order, or nullptr when the block
    // does not carry that statistic. nullptr means nothing else: a region past region_count is a
    // mistake in the caller, not a statistic that is absent, and is not answered as one.
    const double* Series(std::size_t region, Statistic statistic) const noexcept {
        assert(region < region_count);
        for (std::size_t slot = 0; slot < _statistic_count; ++slot) {
            if (_statistics[slot] == statistic) {
                return _values + (region * _region_stride) + (slot * channel_count);
            }
        }
        return nullptr;
    }

    // Every statistic of one region at one channel of this block, counted from first_channel. A
    // statistic the block does not carry reads as SpectralTotals says. The same preconditions as
    // Series, and channel < channel_count.
    SpectralTotals Totals(std::size_t region, std::uint64_t channel) const noexcept {
        assert(region < region_count && channel < channel_count);
        SpectralTotals totals;
        const auto at = [&](Statistic statistic, double& into) {
            if (const double* series = Series(region, statistic)) {
                into = series[channel];
            }
        };
        at(Statistic::num_pixels, totals.num_pixels);
        at(Statistic::nan_count, totals.nan_count);
        at(Statistic::sum, totals.sum);
        at(Statistic::sum_sq, totals.sum_sq);
        at(Statistic::min, totals.min);
        at(Statistic::max, totals.max);
        return totals;
    }

private:
    friend class internal::StatisticSlots;

    const double* _values = nullptr;
    std::size_t _region_stride = 0;
    const Statistic* _statistics = nullptr;
    std::size_t _statistic_count = 0;
};

// Called once per block, on the thread that called ReduceSpectral. Returning false cancels the
// reduction, which then reports cancelled.
using SpectralSink = std::function<bool(const SpectralBlock&)>;

}  // namespace carta::zarr

#endif  // CARTA_ZARR_REDUCE_H_
