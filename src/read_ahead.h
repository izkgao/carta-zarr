/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_READ_AHEAD_H_
#define CARTA_ZARR_SRC_READ_AHEAD_H_

#include "carta-zarr/descriptor.h"
#include "carta-zarr/read.h"
#include "carta-zarr/read_ahead.h"
#include "carta-zarr/result.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace carta::zarr::internal {

// The chunks a read decodes, as the first and the last chunk it touches along each axis. Two reads
// with the same box decode the same chunks, which is what makes them one run; see CONTEXT.md.
//
// Every axis counts alike. A plane is one element along the spectrum and along the polarization,
// and the chunk it lies in along each is part of which run it is in -- a chunk deep in Stokes puts
// several polarizations in one run, exactly as one deep in frequency puts several channels in it.
struct Run {
    std::vector<std::uint64_t> first;
    std::vector<std::uint64_t> last;

    bool operator==(const Run& other) const { return first == other.first && last == other.last; }
    bool operator!=(const Run& other) const { return !(*this == other); }
};

// The run a read is in. An axis with no chunk shape to speak of is one chunk.
Run RunOf(const ChunkGeometry& geometry, const ReadRequest& request);

// What one run of a whole plane of this image holds once decoded: the plane rounded out to whole
// chunks along its two spatial axes, a chunk deep along every other, at what a chunk decodes to --
// counting the flag beside it when the read applies the pixel mask. Zero for an image with no two
// spatial axes, which has no plane to animate.
std::uint64_t PlaneRunBytes(const ImageDescriptor& descriptor, const ChunkGeometry& geometry, bool apply_mask);

// The cache an image's reads keep their chunks in, and how much it holds. Images reading through the
// same one share what it holds, which is what it has to be enough for.
struct CacheShare {
    // Equal for two images whose reads go through the same cache.
    const void* identity = nullptr;
    std::uint64_t bytes = 0;
};

// What reading ahead asks of one image: which run a plane is in, what a run holds and what holds it,
// and the decode itself.
//
// The seam an image's own answers and a test's stand at. A prefetch a test can hold up until it lets
// go is the only way to say "a frame began while one was under way" without racing a real decode.
class RunSource {
public:
    RunSource() = default;
    RunSource(const RunSource&) = delete;
    RunSource& operator=(const RunSource&) = delete;
    RunSource(RunSource&&) = delete;
    RunSource& operator=(RunSource&&) = delete;
    virtual ~RunSource() = default;

    virtual Run RunOf(const ReadRequest& plane) const = 0;
    virtual std::uint64_t PlaneRunBytes() const = 0;
    virtual CacheShare Cache() const = 0;
    // Named in what is said when reading ahead is declined.
    virtual std::string Name() const = 0;
    // Decodes the run `plane` is in, so that a read of any plane of it finds its chunks decoded.
    // Blocks until it has, or `cancelled` says to stop, or it fails; says whether it finished.
    virtual bool Prefetch(const ReadRequest& plane, const std::function<bool()>& cancelled) const = 0;
};

// Reading ahead of an animation over the images `sources` answer for. What ReadAhead is, behind its
// handle; see ReadAhead and ADR 0016 for the policy.
class ReadingAhead {
public:
    using Clock = std::chrono::steady_clock;

    static constexpr std::size_t kUpcomingFrames = ReadAhead::kUpcomingFrames;

    // Reading ahead over `sources`, or buffer_too_small, saying which, when a cache they read
    // through cannot hold two runs of every one of them that reads through it; invalid_argument
    // when there are none, or one cannot say what a run of it holds.
    static Result<std::unique_ptr<ReadingAhead>> For(std::vector<std::shared_ptr<const RunSource>> sources);

    // Cancels what is under way and waits for it.
    ~ReadingAhead();
    ReadingAhead(const ReadingAhead&) = delete;
    ReadingAhead& operator=(const ReadingAhead&) = delete;
    ReadingAhead(ReadingAhead&&) = delete;
    ReadingAhead& operator=(ReadingAhead&&) = delete;

    // After a frame that began at `began` and showed `shown`, late by the caller's measure or not;
    // `upcoming` is the planes of the frames to come, nearest first, of which the first
    // kUpcomingFrames are looked at.
    //
    // A late frame that began while a prefetch was under way stops all reading ahead. Otherwise,
    // with none under way, starts one that decodes, for each image shown, the first run among
    // `upcoming` that is not the run it shows now, unless that run has been decoded ahead already.
    void Served(Clock::time_point began, bool late, const std::vector<AnimatedPlane>& shown,
                const std::vector<std::vector<AnimatedPlane>>& upcoming);

    // Stops what is under way without waiting for it, and starts nothing more.
    void Cancel();

    ReadAheadStats Stats() const;
    bool UnderWay() const;

private:
    explicit ReadingAhead(std::vector<std::shared_ptr<const RunSource>> sources);
    void Join();

    std::vector<std::shared_ptr<const RunSource>> _sources;
    // Only Served, Cancel and the destructor touch the worker, and a caller makes those from the one
    // thread that plays the animation or after it has stopped; the rest is shared with the worker.
    std::thread _worker;
    std::atomic<bool> _cancelled{false};

    mutable std::mutex _mutex;
    bool _under_way = false;
    Clock::time_point _started;
    Clock::time_point _finished;
    // The runs the prefetch under way, or the last one, is decoding: by image.
    std::map<std::size_t, Run> _prefetching;
    bool _counted_catch = false;
    // The run last decoded ahead for each image, so that it is not decoded again.
    std::map<std::size_t, Run> _requested;
    ReadAheadStats _stats;
};

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_READ_AHEAD_H_
