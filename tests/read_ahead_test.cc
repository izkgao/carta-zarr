/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Reading ahead of an animation, over images whose prefetch a test can hold up until it lets go.
//
// The policy came from carta-backend, with its tests: which run is decoded ahead and when, one at a
// time, none once a frame is late beside one, and none without room in the cache for two runs. A
// prefetch that can be held is the only way to say "a frame began while one was under way" without
// racing a real decode, so these stand at the RunSource seam; what a run of a real image is, is
// stated apart from them, of a chunk geometry.

#include "read_ahead.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "support/check.h"

namespace {

using carta::zarr::AnimatedPlane;
using carta::zarr::AxisDescriptor;
using carta::zarr::AxisRole;
using carta::zarr::ChunkGeometry;
using carta::zarr::DataType;
using carta::zarr::ErrorCode;
using carta::zarr::ImageDescriptor;
using carta::zarr::Range;
using carta::zarr::ReadRequest;
using carta::zarr::internal::CacheShare;
using carta::zarr::internal::PlaneRunBytes;
using carta::zarr::internal::ReadingAhead;
using carta::zarr::internal::Run;
using carta::zarr::internal::RunOf;
using carta::zarr::internal::RunSource;

using carta::zarr::testing::Require;

using Clock = std::chrono::steady_clock;
using Planes = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

// A request for one plane of an image whose axes are a channel and a Stokes, which is all a fake
// image has: the planes here have no pixels, only a place in a run.
ReadRequest PlaneAt(std::uint64_t channel, std::uint64_t stokes) {
    ReadRequest request;
    request.axes = {Range{channel, 1, 1}, Range{stokes, 1, 1}};
    return request;
}

AnimatedPlane Plane(std::size_t image, std::uint64_t channel, std::uint64_t stokes = 0) {
    return AnimatedPlane{image, PlaneAt(channel, stokes)};
}

// Runs of `depth` channels and one Stokes each, and a prefetch a test can hold up until it lets go
// or the prefetch is cancelled. Each has a cache of its own unless it is given one to share.
class FakeImage final : public RunSource {
public:
    explicit FakeImage(std::uint64_t depth = 4, std::uint64_t run_bytes = 100, std::uint64_t cache_bytes = 1000,
                       const void* cache = nullptr)
        : _depth(depth), _run_bytes(run_bytes), _cache{cache != nullptr ? cache : this, cache_bytes} {}

    Run RunOf(const ReadRequest& plane) const override {
        return Run{{plane.axes.at(0).start / _depth, plane.axes.at(1).start},
                   {plane.axes.at(0).start / _depth, plane.axes.at(1).start}};
    }
    std::uint64_t PlaneRunBytes() const override { return _run_bytes; }
    CacheShare Cache() const override { return _cache; }
    std::string Name() const override { return "fake/" + std::to_string(_depth); }

    bool Prefetch(const ReadRequest& plane, const std::function<bool()>& cancelled) const override {
        std::unique_lock<std::mutex> lock(_mutex);
        _prefetched.emplace_back(plane.axes.at(0).start, plane.axes.at(1).start);
        while (_holding && !cancelled()) {
            _released.wait_for(lock, std::chrono::milliseconds(1));
        }
        if (cancelled()) {
            _saw_cancel = true;
            return false;
        }
        return true;
    }

    void Hold() {
        const std::scoped_lock lock(_mutex);
        _holding = true;
    }
    void Release() {
        {
            const std::scoped_lock lock(_mutex);
            _holding = false;
        }
        _released.notify_all();
    }
    Planes Prefetched() const {
        const std::scoped_lock lock(_mutex);
        return _prefetched;
    }
    bool SawCancel() const {
        const std::scoped_lock lock(_mutex);
        return _saw_cancel;
    }

private:
    std::uint64_t _depth;
    std::uint64_t _run_bytes;
    CacheShare _cache;
    mutable std::mutex _mutex;
    mutable std::condition_variable _released;
    bool _holding = false;
    mutable bool _saw_cancel = false;
    mutable Planes _prefetched;
};

// The planes of image 0 at each channel in turn.
std::vector<std::vector<AnimatedPlane>> Upcoming(std::uint64_t from, std::uint64_t count) {
    std::vector<std::vector<AnimatedPlane>> frames;
    for (std::uint64_t channel = from; channel < from + count; ++channel) {
        frames.push_back({Plane(0, channel)});
    }
    return frames;
}

std::unique_ptr<ReadingAhead> Over(std::vector<std::shared_ptr<const RunSource>> images) {
    auto made = ReadingAhead::For(std::move(images));
    Require(made.has_value(), "reading ahead was declined: " + (made ? std::string{} : made.error().message));
    return std::move(made).value();
}

void WaitUntilIdle(const ReadingAhead& reading) {
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (reading.UnderWay() && Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Require(!reading.UnderWay(), "a prefetch never finished");
}

void WaitUntilStarted(const FakeImage& image, std::size_t count) {
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    while (image.Prefetched().size() < count && Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Require(image.Prefetched().size() >= count, "a prefetch never started");
}

void TestARunIsTheChunksAReadDecodes() {
    // x, y, channel, Stokes: chunks of 4 x 4 pixels, 2 channels and 1 Stokes.
    ChunkGeometry geometry;
    geometry.chunk_shape = {4, 4, 2, 1};
    const auto plane = [](std::uint64_t channel, std::uint64_t stokes) {
        ReadRequest request;
        request.axes = {Range{0, 10, 1}, Range{0, 5, 1}, Range{channel, 1, 1}, Range{stokes, 1, 1}};
        return request;
    };
    Require(RunOf(geometry, plane(0, 0)) == (Run{{0, 0, 0, 0}, {2, 1, 0, 0}}),
            "a plane is its spatial extent in whole chunks and the chunk it lies in along the rest");
    Require(RunOf(geometry, plane(0, 0)) == RunOf(geometry, plane(1, 0)), "two channels of one chunk are one run");
    Require(RunOf(geometry, plane(1, 0)) != RunOf(geometry, plane(2, 0)), "the next chunk of channels is another run");
    Require(RunOf(geometry, plane(0, 0)) != RunOf(geometry, plane(0, 1)),
            "a Stokes a chunk apart is another run, whatever the channel");
    geometry.chunk_shape = {4, 4, 2, 2};
    Require(RunOf(geometry, plane(0, 0)) == RunOf(geometry, plane(1, 1)),
            "a chunk two Stokes deep puts both of them in one run");

    ReadRequest offset;
    offset.axes = {Range{3, 6, 1}, Range{0, 1, 1}, Range{0, 3, 5}, Range{0, 1, 1}};
    Require(RunOf(geometry, offset) == (Run{{0, 0, 0, 0}, {2, 0, 5, 0}}),
            "a read that starts inside a chunk, and one strided over several, are their first and last chunk");

    ChunkGeometry unchunked;
    unchunked.chunk_shape = {4, 0};
    ReadRequest two_axes;
    two_axes.axes = {Range{5, 1, 1}, Range{7, 1, 1}, Range{9, 1, 1}};
    Require(RunOf(unchunked, two_axes) == (Run{{1, 0, 0}, {1, 0, 0}}),
            "an axis with no chunk shape to speak of is one chunk");
}

AxisDescriptor Axis(const char* name, AxisRole role, std::uint64_t length) {
    AxisDescriptor axis;
    axis.name = name;
    axis.role = role;
    axis.length = length;
    return axis;
}

void TestARunOfAPlaneHoldsItsChunksWhole() {
    ImageDescriptor descriptor;
    descriptor.stored_type = DataType::float32;
    descriptor.axes = {Axis("l", AxisRole::spatial_x, 10), Axis("m", AxisRole::spatial_y, 5),
                       Axis("frequency", AxisRole::spectral, 8), Axis("polarization", AxisRole::polarization, 4)};
    ChunkGeometry geometry;
    geometry.chunk_shape = {4, 4, 2, 2};
    // 3 x 2 chunks across the plane, each 4 x 4 x 2 x 2 elements of four bytes.
    Require(PlaneRunBytes(descriptor, geometry, false) == 6ULL * 256, "a run is the plane in whole chunks");
    Require(PlaneRunBytes(descriptor, geometry, true) == 6ULL * (256 + 64),
            "with the pixel mask applied a run holds the flag beside it, at a byte an element");

    descriptor.axes = {Axis("frequency", AxisRole::spectral, 8)};
    geometry.chunk_shape = {2};
    Require(PlaneRunBytes(descriptor, geometry, false) == 0, "an image with no plane has no run of one");
}

void TestNothingIsReadAheadWithoutRoomForTwoRunsOfEachImageItsCacheHolds() {
    const auto declined = [](std::vector<std::shared_ptr<const RunSource>> images, ErrorCode code) {
        auto made = ReadingAhead::For(std::move(images));
        return !made && made.error().code == code ? made.error().message : std::string{};
    };
    Require(!declined({}, ErrorCode::invalid_argument).empty(), "reading ahead of no image was made");
    Require(!declined({std::make_shared<FakeImage>(4, 0, 1000)}, ErrorCode::invalid_argument).empty(),
            "reading ahead of an image with no run was made");

    const auto message = declined({std::make_shared<FakeImage>(4, 600, 1000)}, ErrorCode::buffer_too_small);
    Require(message.find("fake/4") != std::string::npos,
            "a cache that cannot hold two runs was not declined, or not said of which image: " + message);
    Require(ReadingAhead::For({std::make_shared<FakeImage>(4, 500, 1000)}).has_value(), "two runs that fit");

    const int shared = 0;
    const auto sharing = [&](std::uint64_t depth) { return std::make_shared<FakeImage>(depth, 300, 1000, &shared); };
    Require(!declined({sharing(4), sharing(2)}, ErrorCode::buffer_too_small).empty(),
            "two images whose runs fit one cache one at a time but not together");
    Require(ReadingAhead::For({std::make_shared<FakeImage>(4, 300, 1000), std::make_shared<FakeImage>(2, 300, 1000)})
                .has_value(),
            "two images in caches of their own were held to one between them");
}

void TestTheNextRunIsDecodedOnceAndNoSooner() {
    const auto image = std::make_shared<FakeImage>();
    auto reading = Over({image});

    reading->Served(Clock::now(), false, {Plane(0, 0)}, Upcoming(1, 2));
    WaitUntilIdle(*reading);
    Require(image->Prefetched().empty(), "a run was decoded before it came within reach");

    reading->Served(Clock::now(), false, {Plane(0, 0)}, Upcoming(1, 8));
    WaitUntilIdle(*reading);
    reading->Served(Clock::now(), false, {Plane(0, 1)}, Upcoming(2, 8));
    WaitUntilIdle(*reading);
    Require(image->Prefetched() == Planes{{4, 0}}, "the next run, from its first plane, once");

    reading->Served(Clock::now(), false, {Plane(0, 4)}, Upcoming(5, 8));
    WaitUntilIdle(*reading);
    Require(image->Prefetched() == (Planes{{4, 0}, {8, 0}}), "the run after was not decoded once it was in reach");
    Require(reading->Stats().prefetches == 2, "prefetches were miscounted");
}

void TestOnlyTheFramesLookedAheadToAreLookedAt() {
    const auto image = std::make_shared<FakeImage>(ReadingAhead::kUpcomingFrames + 2);
    auto reading = Over({image});
    reading->Served(Clock::now(), false, {Plane(0, 0)}, Upcoming(1, 2 * ReadingAhead::kUpcomingFrames));
    WaitUntilIdle(*reading);
    Require(image->Prefetched().empty(), "a run further away than the frames looked ahead to was decoded");
}

void TestOnePrefetchIsUnderWayAtATime() {
    const auto image = std::make_shared<FakeImage>();
    auto reading = Over({image});
    image->Hold();
    reading->Served(Clock::now(), false, {Plane(0, 0)}, Upcoming(1, 8));
    Require(reading->UnderWay(), "no prefetch was started");
    reading->Served(Clock::now(), false, {Plane(0, 4)}, Upcoming(5, 8));
    image->Release();
    WaitUntilIdle(*reading);
    Require(image->Prefetched().size() == 1, "a prefetch started while another was under way");

    reading->Served(Clock::now(), false, {Plane(0, 5)}, Upcoming(6, 8));
    WaitUntilIdle(*reading);
    Require(image->Prefetched().back() == Planes::value_type{8, 0},
            "the run passed over was not decoded once there was room");
}

void TestAFrameLateWhileAPrefetchIsUnderWayStopsReadingAhead() {
    const auto image = std::make_shared<FakeImage>();
    auto reading = Over({image});
    image->Hold();
    const auto first = Clock::now();
    reading->Served(first, true, {Plane(0, 0)}, Upcoming(1, 8));
    Require(!reading->Stats().stopped, "a frame late with nothing under way is no reason to stop");
    Require(reading->UnderWay(), "no prefetch was started");

    reading->Served(first, true, {Plane(0, 1)}, Upcoming(2, 8));
    Require(!reading->Stats().stopped, "a frame that began before the prefetch did was taken to share its time");

    reading->Served(Clock::now(), true, {Plane(0, 2)}, Upcoming(3, 8));
    Require(reading->Stats().stopped, "a frame late while a prefetch was under way did not stop reading ahead");
    image->Release();
    WaitUntilIdle(*reading);
    reading->Served(Clock::now(), false, {Plane(0, 4)}, Upcoming(5, 8));
    WaitUntilIdle(*reading);
    Require(image->Prefetched().size() == 1, "reading ahead went on after a frame was late beside it");
}

void TestAFrameLateAfterAPrefetchFinishedIsNoReasonToStop() {
    const auto image = std::make_shared<FakeImage>();
    auto reading = Over({image});
    reading->Served(Clock::now(), false, {Plane(0, 0)}, Upcoming(1, 8));
    WaitUntilIdle(*reading);
    reading->Served(Clock::now(), true, {Plane(0, 1)}, Upcoming(2, 8));
    Require(!reading->Stats().stopped,
            "a frame that began after the prefetch had finished was taken to share its time");
}

void TestAFrameThatReachesAPrefetchUnderWayCatchesItUp() {
    const auto image = std::make_shared<FakeImage>();
    auto reading = Over({image});
    image->Hold();
    reading->Served(Clock::now(), false, {Plane(0, 0)}, Upcoming(1, 8));
    reading->Served(Clock::now(), false, {Plane(0, 3)}, Upcoming(4, 8));
    Require(reading->Stats().caught_up == 0, "a frame of the run before was taken to have caught the prefetch");
    reading->Served(Clock::now(), false, {Plane(0, 4)}, Upcoming(5, 8));
    reading->Served(Clock::now(), false, {Plane(0, 5)}, Upcoming(6, 8));
    image->Release();
    WaitUntilIdle(*reading);
    const auto stats = reading->Stats();
    Require(stats.caught_up == 1, "a prefetch a frame reached was not counted caught up, or was counted twice");
    Require(!stats.stopped, "a frame that was on time stopped reading ahead");

    reading->Served(Clock::now(), false, {Plane(0, 8)}, Upcoming(9, 8));
    WaitUntilIdle(*reading);
    Require(reading->Stats().caught_up == 1, "a run decoded before a frame reached it was counted caught up");
}

void TestCancellingStopsWhatIsUnderWay() {
    const auto image = std::make_shared<FakeImage>();
    auto reading = Over({image});
    image->Hold();
    reading->Served(Clock::now(), false, {Plane(0, 0)}, Upcoming(1, 8));
    WaitUntilStarted(*image, 1);
    reading->Cancel();
    WaitUntilIdle(*reading);
    Require(image->SawCancel(), "the prefetch under way was not told to stop");
    Require(reading->Stats().stopped, "cancelling did not stop reading ahead");
}

void TestLettingGoStopsWhatIsUnderWayAndWaitsForIt() {
    const auto image = std::make_shared<FakeImage>();
    {
        auto reading = Over({image});
        image->Hold();
        reading->Served(Clock::now(), false, {Plane(0, 0)}, Upcoming(1, 8));
        WaitUntilStarted(*image, 1);
    }
    Require(image->SawCancel(), "a prefetch outlived the reading ahead that started it");
}

void TestEachAnimatedImageHasItsNextRunDecoded() {
    const auto active = std::make_shared<FakeImage>(4);
    const auto matched = std::make_shared<FakeImage>(2);
    auto reading = Over({active, matched});
    std::vector<std::vector<AnimatedPlane>> upcoming;
    for (std::uint64_t channel = 1; channel < 8; ++channel) {
        // The matched image moves at half the active one's pace.
        upcoming.push_back({Plane(0, channel), Plane(1, channel / 2, 1)});
    }
    reading->Served(Clock::now(), false, {Plane(0, 0), Plane(1, 0, 1)}, upcoming);
    WaitUntilIdle(*reading);
    Require(active->Prefetched() == Planes{{4, 0}}, "the active image's next run was not decoded");
    Require(matched->Prefetched() == Planes{{2, 1}}, "the matched image's next run was not decoded");
    Require(reading->Stats().prefetches == 2, "a prefetch for two images was not counted for each");
}

}  // namespace

int main() {
    try {
        TestARunIsTheChunksAReadDecodes();
        TestARunOfAPlaneHoldsItsChunksWhole();
        TestNothingIsReadAheadWithoutRoomForTwoRunsOfEachImageItsCacheHolds();
        TestTheNextRunIsDecodedOnceAndNoSooner();
        TestOnlyTheFramesLookedAheadToAreLookedAt();
        TestOnePrefetchIsUnderWayAtATime();
        TestAFrameLateWhileAPrefetchIsUnderWayStopsReadingAhead();
        TestAFrameLateAfterAPrefetchFinishedIsNoReasonToStop();
        TestAFrameThatReachesAPrefetchUnderWayCatchesItUp();
        TestCancellingStopsWhatIsUnderWay();
        TestLettingGoStopsWhatIsUnderWayAndWaitsForIt();
        TestEachAnimatedImageHasItsNextRunDecoded();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "read ahead test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
