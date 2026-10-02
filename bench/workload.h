/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_BENCH_WORKLOAD_H_
#define CARTA_ZARR_BENCH_WORKLOAD_H_

// Where each operation of a trial reads, and the reading itself.
//
// Positions are a function of the seed, the mode, the trial and the cube's logical shape -- never of
// its layout -- so every layout of one cube is asked for the same pixels, and a difference in time is
// the layout's. They come from a generator written out here rather than from <random>, whose
// distributions are free to differ between standard libraries, so the Mac and a Linux server agree.

#include "options.h"

#include <carta-zarr/carta_zarr.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace carta::zarr::bench {

// The axes the operations address, as logical indices into ImageDescriptor::axes. An image with no
// polarization or time axis reads as one with a single plane of each.
struct CubeAxes {
    std::size_t rank = 0;
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t spectral = 0;
    std::optional<std::size_t> polarization;
    std::optional<std::size_t> time;

    std::uint64_t width = 1;
    std::uint64_t height = 1;
    std::uint64_t channels = 1;
    std::uint64_t polarizations = 1;

    // The image's axes, or an error naming the role it lacks.
    static Result<CubeAxes> Of(const ImageDescriptor& descriptor);
};

// One operation, as planned before the trial starts.
struct Operation {
    Mode mode = Mode::plane;
    std::uint64_t x = 0;
    std::uint64_t y = 0;
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    std::uint64_t channel = 0;
    std::uint64_t channel_count = 0;
    std::uint64_t polarization = 0;
    // Whether another operation of this trial, in any process, reads the same position: there were
    // more operations than distinct positions to give them.
    bool overlap = false;
    // Whether it reads a chunk that an earlier operation of its own process read, or that any
    // operation of another process reads. Its time may then be the page cache's rather than the
    // storage's -- the earlier read brought the chunk in, or the other process is bringing it in
    // now -- so it is not a first touch however fresh its cache pool. See MarkSharedChunks.
    bool shares_chunks = false;

    // Where it reads, for the CSV: semicolon-separated, so that it stays one field.
    std::string Describe() const;
};

// The operations one process makes in one trial. Process p takes the p-th run of `ops` positions from
// one sequence drawn for the whole trial, so processes never share one while there are enough, and
// process 0 reads the same positions whatever the process count.
//
// cube-histogram is the exception, since its operation covers the plane: process p takes the p-th of
// `processes` contiguous runs of channels, and each operation a polarization of its own.
//
// A region box covers `region_fraction` of the plane, square in pixels' proportion to the plane, and
// sits in a cell of a grid of as many such boxes as fit along each side. An animation reads
// `animation_frames` consecutive channels from a start in a cell of channels the same way, so that no
// two of a trial's animations play the same planes while there are enough channels for them.
std::vector<Operation> PlanOperations(Mode mode, const CubeAxes& axes, std::uint64_t seed, unsigned trial,
                                      unsigned processes, unsigned process_index, unsigned ops,
                                      double region_fraction = 0.05, unsigned animation_frames = 32);

// Sets shares_chunks on every operation of a trial, given each process's plan, indexed by process,
// and the image's chunk shape in logical axis order. An operation that reads a chunk first is never
// marked: of two that share one, the earlier in its process is the first touch, unless the other is
// another process's, which may be reading it at the same moment. animation and open are left alone,
// the one meaning to reuse chunks and the other reading none.
void MarkSharedChunks(std::vector<std::vector<Operation>>& plans, const CubeAxes& axes,
                      const std::vector<std::uint64_t>& chunk_shape);

// How an animation's frames went, the first apart: it is a cold read whatever the layout, and what a
// layout decides is how the frames after it go.
struct FrameStats {
    // The frames played, which is the channels the animation read: as many as were asked for, or every
    // channel of a cube with fewer.
    unsigned frames = 0;
    double first_s = 0.0;
    // The read times of every frame after the first.
    double median_s = 0.0;
    double max_s = 0.0;
    // Frames after the first not ready by the end of their turn, and the most any was late by. Zero
    // when the frames are read back to back, which gives them no turn to miss.
    unsigned late = 0;
    double late_max_s = 0.0;
    // Prefetches of the next run of chunks started, and how many of them the animation caught up with
    // before they had finished: ReadAheadStats's prefetches and caught_up. Zero without prefetch, or
    // when the cache cannot hold two runs. Prefetches stop once a frame is late while one is under way,
    // so the first is the more telling.
    unsigned prefetches = 0;
    unsigned late_prefetches = 0;
};

// Runs operations against one image and remembers enough of the last result to fingerprint it.
//
// The fingerprint is taken after the clock stops, and only of what every layout of the same pixels
// must agree on exactly: the pixels a read returns, with every NaN one NaN; and for a reduction or a
// histogram the pixel counts and extremes, which are exact, but not sums, whose rounding depends on
// the order the chunks were visited in. An exact histogram's counts are in it too, since binning
// over fixed bounds is exact; a one-pass histogram's are not, since ComputeCubeHistogram says they
// depend on the thread count.
//
// A cube histogram reads through a cache pool that keeps nothing, as carta-backend's cube walks do,
// so that a scan of the whole cube neither fills nor finds the context's cache. A plane or a spectrum
// reads through a pool of `first_touch_bytes` made for it by Prepare, and an animation through the
// context's own, which its frames share as the backend's do.
class Runner {
public:
    // For every mode but open, which brings its own context and image to each operation.
    Runner(const Context& context, Image image, HistogramMethod histogram = {},
           std::size_t first_touch_bytes = std::size_t{1} << 30);
    // For open.
    Runner(ContextOptions context, std::string dataset, std::string image_id);

    // How animations are played: at `fps` frames a second, 0 for back to back, and reading ahead through
    // ReadAhead when `prefetch`.
    void SetAnimation(double fps, bool prefetch) {
        _fps = fps;
        _prefetch = prefetch;
    }

    // How the last animation's frames went.
    const std::optional<FrameStats>& frame_stats() const noexcept {
        return _frame_stats;
    }

    // What an operation needs that is not part of what it measures, done before the clock starts:
    // a fresh cache pool for a plane or a spectrum, with the previous one let go of.
    Result<void> Prepare(const Operation& operation);

    // How many elements of the cube the operation covered.
    Result<std::uint64_t> Run(const Operation& operation, const ReadOptions& options);

    std::uint64_t Fingerprint() const;

    // The image last read, if there was one: an open that failed leaves none.
    const std::optional<Image>& image() const noexcept {
        return _image;
    }

private:
    Result<std::uint64_t> Read(const Operation& operation, const ReadOptions& options);
    Result<std::uint64_t> Animate(const Operation& operation, const ReadOptions& options);
    Result<std::uint64_t> Reduce(const Operation& operation, const ReadOptions& options);
    Result<std::uint64_t> Histogram(const Operation& operation, const ReadOptions& options);
    Result<std::uint64_t> ExactHistogram(const Operation& operation, const ReadOptions& options);
    Result<std::uint64_t> Open();

    std::optional<Context> _shared;
    std::optional<Image> _image;
    std::optional<CubeAxes> _axes;
    HistogramMethod _histogram;
    std::optional<CachePool> _keeping_nothing;
    std::size_t _first_touch_bytes = 0;
    std::optional<CachePool> _first_touch;
    double _fps = 0.0;
    bool _prefetch = false;
    std::optional<FrameStats> _frame_stats;
    ContextOptions _context;
    std::string _dataset;
    std::string _image_id;

    std::vector<float> _pixels;
    std::size_t _pixel_count = 0;
    // An animation's frames, each fingerprinted as it is read and the fingerprints hashed together.
    std::optional<std::uint64_t> _frames;
    // The exact statistics of a reduction or a cube histogram, in order.
    std::vector<double> _exact;
    std::vector<std::uint64_t> _counts;
};

// FNV-1a over a run of values, with every NaN hashed as the one quiet NaN.
std::uint64_t Fingerprint(const float* values, std::size_t count);
std::uint64_t Fingerprint(const double* values, std::size_t count);

}  // namespace carta::zarr::bench

#endif  // CARTA_ZARR_BENCH_WORKLOAD_H_
