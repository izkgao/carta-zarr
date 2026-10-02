/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Reading ahead of a real image, through nothing but what a consumer sees: that the run decoded ahead
// is the one the next frames read, into the cache they read through, and that a cache with no room
// for two runs is declined in words that say why. What is decided when is tested beside the module,
// against images whose prefetch the test can hold; see read_ahead_test.cc.

#include <carta-zarr/carta_zarr.h>
#include <carta-zarr/read_ahead.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "support/check.h"

namespace {

using carta::zarr::AnimatedPlane;
using carta::zarr::AxisIndex;
using carta::zarr::AxisRole;
using carta::zarr::Context;
using carta::zarr::ContextOptions;
using carta::zarr::Dataset;
using carta::zarr::ErrorCode;
using carta::zarr::Image;
using carta::zarr::Range;
using carta::zarr::ReadAhead;
using carta::zarr::ReadOptions;
using carta::zarr::ReadRequest;

using carta::zarr::testing::Require;

// Four channels in chunks of two, so two runs along the spectrum, and two polarizations a chunk each.
const char* const kFixture = CARTA_ZARR_PIXEL_FIXTURE_WIDE;

Image OpenSky(const Context& context, const std::string& location) {
    const auto dataset = Dataset::Open(context, location);
    Require(static_cast<bool>(dataset), "the fixture did not open");
    auto image = dataset.value().OpenImage("SKY");
    Require(static_cast<bool>(image), "the fixture's image did not open");
    return std::move(image).value();
}

Context ContextHolding(std::size_t bytes) {
    ContextOptions options;
    options.cache_bytes = bytes;
    auto context = Context::Create(options);
    Require(static_cast<bool>(context), "Context::Create failed");
    return std::move(context).value();
}

// The whole plane at one channel of the first polarization and time.
ReadRequest PlaneAt(const Image& image, std::uint64_t channel) {
    const auto& axes = image.descriptor().axes;
    ReadRequest request;
    request.axes.assign(axes.size(), Range{0, 1, 1});
    for (const auto role : {AxisRole::spatial_x, AxisRole::spatial_y}) {
        const auto axis = AxisIndex(axes, role).value();
        request.axes[axis] = Range{0, axes[axis].length, 1};
    }
    request.axes[AxisIndex(axes, AxisRole::spectral).value()] = Range{channel, 1, 1};
    return request;
}

std::uint64_t PlaneElements(const Image& image) {
    const auto& axes = image.descriptor().axes;
    return axes[AxisIndex(axes, AxisRole::spatial_x).value()].length *
           axes[AxisIndex(axes, AxisRole::spatial_y).value()].length;
}

std::vector<float> ReadPlane(const Image& image, std::uint64_t channel, bool* read = nullptr) {
    std::vector<float> pixels(PlaneElements(image));
    const auto outcome = image.Read(PlaneAt(image, channel), {pixels.data(), pixels.size()});
    if (read != nullptr) {
        *read = outcome.has_value();
    } else {
        Require(outcome.has_value(), "a plane did not read");
    }
    return pixels;
}

// Played from channel 0, the run of channels 2 and 3 is decoded ahead into the cache the frames read
// through. Seen by truncating every chunk of a copy once it has been: channel 2 still reads, and reads
// what it holds, while channel 0, which nothing has decoded, does not -- so what answered was the
// cache, filled by reading ahead. ADR 0015 is why a cached chunk is not looked for again.
void TestTheNextRunIsDecodedIntoTheCacheTheFramesReadThrough() {
    const auto expected = ReadPlane(OpenSky(ContextHolding(0), kFixture), 2);

    const auto copy = std::filesystem::temp_directory_path() / ("carta-zarr-read-ahead-" + std::to_string(getpid()));
    std::filesystem::remove_all(copy);
    std::filesystem::copy(kFixture, copy, std::filesystem::copy_options::recursive);
    const auto image = OpenSky(ContextHolding(std::size_t{64} << 20), copy.string());

    auto made = ReadAhead::For({{image, ReadOptions{}}});
    Require(made.has_value(), "reading ahead was declined: " + (made ? std::string{} : made.error().message));
    auto reading = std::move(made).value();
    std::vector<std::vector<AnimatedPlane>> upcoming;
    for (std::uint64_t channel = 1; channel < 4; ++channel) {
        upcoming.push_back({AnimatedPlane{0, PlaneAt(image, channel)}});
    }
    reading.Served(ReadAhead::Clock::now(), false, {AnimatedPlane{0, PlaneAt(image, 0)}}, upcoming);
    const auto deadline = ReadAhead::Clock::now() + std::chrono::seconds(10);
    while (reading.stats().under_way && ReadAhead::Clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto stats = reading.stats();

    for (const auto& entry : std::filesystem::recursive_directory_iterator(copy / "SKY" / "c")) {
        if (entry.is_regular_file()) {
            std::filesystem::resize_file(entry.path(), 0);
        }
    }
    bool ahead_read = false;
    bool behind_read = true;
    const auto ahead = ReadPlane(image, 2, &ahead_read);
    (void)ReadPlane(image, 0, &behind_read);
    std::filesystem::remove_all(copy);

    Require(!stats.under_way && stats.prefetches == 1, "the next run was not read ahead, once");
    Require(ahead_read, "the run read ahead was not in the cache the frames read through");
    Require(ahead == expected, "the run read ahead read back differently");
    Require(!behind_read, "a run nothing decoded still read, so this test shows nothing about the cache");
}

// No room for two runs is a refusal, which says of which image and how much. The cache asked is the
// one the frames read through: the context's when they name none -- which holds nothing unless it was
// sized -- and otherwise a pool of their own.
void TestACacheWithoutRoomForTwoRunsIsDeclinedAndSaysWhy() {
    const auto context = ContextHolding(0);
    const auto image = OpenSky(context, kFixture);

    const auto declined = ReadAhead::For({{image, ReadOptions{}}});
    Require(!declined && declined.error().code == ErrorCode::buffer_too_small,
            "a context's cache that holds nothing was read ahead into");
    Require(declined.error().message.find(image.descriptor().id) != std::string::npos,
            "the refusal did not say which image: " + declined.error().message);

    ReadOptions pooled;
    pooled.control.cache_pool = context.NewCachePool(std::size_t{64} << 20).value();
    Require(ReadAhead::For({{image, pooled}}).has_value(), "a pool of the frames' own was not the cache asked");
    pooled.control.cache_pool = context.NewCachePool(1).value();
    Require(!ReadAhead::For({{image, pooled}}), "a pool of a byte was read ahead into");
    Require(!ReadAhead::For({}), "reading ahead of nothing was made");
}

}  // namespace

int main() {
    try {
        TestTheNextRunIsDecodedIntoTheCacheTheFramesReadThrough();
        TestACacheWithoutRoomForTwoRunsIsDeclinedAndSaysWhy();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "read ahead image test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
