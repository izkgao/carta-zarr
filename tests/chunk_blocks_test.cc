/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

// How large one request is allowed to be. This has been wrong three times -- once as a fixed chunk
// count that ignored how far chunk sizes vary, once as a byte budget alone that gave a 16 MiB-chunk
// image four chunks per request where a 4 MiB one got sixteen, once as eight chunks a read that left
// most of a twenty-eight-thread machine idle -- so the policy is pinned here in the unit that matters,
// which is chunks a decode thread, across the chunk sizes XRADIO actually writes.

#include "chunk_blocks.h"

#include "support/check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

using carta::zarr::internal::DefaultReadBytes;
using carta::zarr::internal::kChunksPerDecodeThread;
using carta::zarr::internal::kHeldBytesPerDecodedByte;
using carta::zarr::internal::kLeastBytesPerRead;
using carta::zarr::internal::kMostBytesPerRead;
using carta::zarr::internal::PixelsHeld;

constexpr std::uint64_t kMiB = 1u << 20;
// The five-core machine the first default was measured on, and the twenty-eight-thread one it failed.
constexpr std::size_t kFewThreads = 5;
constexpr std::size_t kManyThreads = 28;

using carta::zarr::testing::Require;

// An image with a flag, of `lengths` along its axes in order -- all a cost asks of one.
carta::zarr::ImageDescriptor Image(carta::zarr::DataType type, const std::vector<std::uint64_t>& lengths) {
    carta::zarr::ImageDescriptor image;
    image.stored_type = type;
    image.has_pixel_mask = true;
    image.pixel_mask_id = "FLAG";
    for (const auto length : lengths) {
        carta::zarr::AxisDescriptor axis;
        axis.length = length;
        image.axes.push_back(axis);
    }
    return image;
}

// The chunks a default read of a float32 chunk decoding to `chunk_bytes` affords on `threads`, when
// the library holds what they produce -- the dearer of the two ways a chunk is held.
std::uint64_t ChunksPerRead(std::uint64_t chunk_bytes, std::size_t threads) {
    const std::uint64_t held = (kHeldBytesPerDecodedByte * chunk_bytes) + chunk_bytes;
    return DefaultReadBytes(held, threads) / held;
}

// The shapes these images are written with, as (x, y, spectral) times the polarizations the chunk
// carries. A chunk holding every stokes is four times the size of one holding a single one, which
// is the whole reason a byte budget alone is not enough.
void TestRealChunkShapesGetEnoughChunks() {
    const struct {
        const char* what;
        std::uint64_t bytes;
    } cases[]{
        {"HD163296 128x128x2", 128 * 128 * 2 * 4}, {"ASKAP 512x512x1", 512 * 512 * 1 * 4},
        {"ASKAP 256x256x3", 256 * 256 * 3 * 4},    {"IRC 256x256x1 over four stokes", 256 * 256 * 1 * 4 * 4},
        {"XRADIO 256x256x6", 256 * 256 * 6 * 4},   {"XRADIO 256x256x6 over four stokes", 256 * 256 * 6 * 4 * 4},
        {"XRADIO 512x512x4", 512 * 512 * 4 * 4},   {"XRADIO 512x512x4 over four stokes", 512 * 512 * 4 * 4 * 4},
    };
    for (const auto threads : {std::size_t{1}, kFewThreads, kManyThreads}) {
        for (const auto& one : cases) {
            const auto chunks = ChunksPerRead(one.bytes, threads);
            const std::string what =
                std::string(one.what) + " on " + std::to_string(threads) + " threads gets " + std::to_string(chunks);
            // Every thread has a chunk to decode, and two each wherever the ceiling leaves room for them.
            const std::uint64_t held = (kHeldBytesPerDecodedByte + 1) * one.bytes;
            Require(chunks >= threads, what + " chunks a read, which leaves decode threads idle");
            Require(chunks >= std::min<std::uint64_t>(kChunksPerDecodeThread * threads, kMostBytesPerRead / held),
                    what + " chunks a read, short of two a thread with room under the ceiling");
        }
    }
}

// A chunk large enough that two a thread would be past the ceiling gets as many as fit, and one
// larger than the ceiling still gets the one it cannot avoid.
void TestAnOversizedChunkIsCappedRatherThanMultiplied() {
    Require(DefaultReadBytes(256 * kMiB, kManyThreads) == kMostBytesPerRead,
            "a chunk holding an eighth of the ceiling should stop at the ceiling, not multiply past it");
    Require(DefaultReadBytes(3 * 480 * kMiB, 1) == kMostBytesPerRead,
            "a chunk larger than the ceiling should not raise the budget above it; the walk's own "
            "floor of one chunk is what makes the request, and nothing can make it smaller");
}

// A small chunk does not shrink the budget: the bytes are the floor as well as the chunks.
void TestASmallChunkKeepsTheByteBudget() {
    Require(DefaultReadBytes(128 * 1024, kManyThreads) == kLeastBytesPerRead,
            "a small chunk should leave the byte budget alone");
    Require(DefaultReadBytes(0, 0) == kLeastBytesPerRead,
            "an image reporting no chunk size should fall back to the byte budget");
}

// More decode threads buy a larger read, between the floor and the ceiling: a 4 MiB chunk holding
// 16 MiB is sixteen to a read at the floor on one thread and fifty-six on twenty-eight.
void TestMoreThreadsBuyMoreChunks() {
    Require(DefaultReadBytes(16 * kMiB, 1) == kLeastBytesPerRead, "one thread's two chunks are under the floor");
    Require(DefaultReadBytes(16 * kMiB, kManyThreads) == 16 * kMiB * kChunksPerDecodeThread * kManyThreads,
            "twenty-eight threads should get two chunks each");
}

// What a chunk holds is what it decodes to, three times over, and the buffers the library allocates
// for it. Over a 7763 x 4742 x 128 cube in 512 x 512 x 64 float32 chunks -- 64 MiB each -- on
// twenty-eight threads a read holding the pixels itself affords eight at the default ceiling, and
// one the caller holds the pixels of affords ten.
void TestAChunkHoldsThreeTimesWhatItDecodesTo() {
    using carta::zarr::internal::ReadCost;

    carta::zarr::ChunkGeometry geometry;
    geometry.chunk_shape = {512, 512, 64, 1, 1};
    auto image = Image(carta::zarr::DataType::float32, {7763, 4742, 128, 1, 1});
    image.has_pixel_mask = false;
    const auto cost =
        ReadCost::Of(image, geometry, geometry, carta::zarr::ReadOptions{}, PixelsHeld::by_caller, kManyThreads);
    Require(cost.chunk_bytes == 64 * kMiB, "a 512 x 512 x 64 float32 chunk decodes to 64 MiB");
    Require(cost.Held(PixelsHeld::by_caller) == 3 * 64 * kMiB, "a chunk holds three times what it decodes to");
    Require(cost.Held(PixelsHeld::by_library) == 4 * 64 * kMiB,
            "a chunk whose pixels the library holds holds them too, four bytes an element");
    Require(cost.budget_bytes == kMostBytesPerRead,
            "two such chunks a thread are past the ceiling, so the ceiling is the budget");
    Require(cost.ChunksPerRead(PixelsHeld::by_library) == 8, "the ceiling affords eight chunks held by the library");
    Require(cost.ChunksPerRead(PixelsHeld::by_caller) == 10, "and ten whose pixels the caller holds");

    // A stated budget is the caller's, and smaller than one chunk it affords none; every walk floors
    // that at one.
    carta::zarr::ReadOptions tight;
    tight.read_budget_bytes = 64 * kMiB;
    Require(ReadCost::Of(image, geometry, geometry, tight, PixelsHeld::by_library, kManyThreads)
                    .ChunksPerRead(PixelsHeld::by_library) == 0,
            "a budget of what one chunk decodes to affords less than what it holds");

    // The folded-in flag is a buffer of a byte an element, beside the flag chunk it decodes.
    image.has_pixel_mask = true;
    const auto masked =
        ReadCost::Of(image, geometry, geometry, carta::zarr::ReadOptions{}, PixelsHeld::by_caller, kManyThreads);
    const std::uint64_t elements = 512ULL * 512ULL * 64ULL;
    Require(masked.Held(PixelsHeld::by_caller) == (3 * (masked.chunk_bytes)) + elements,
            "a masked chunk holds its decoded flag three times over and the folded-in flag once");
}

// What a chunk costs when the read will apply the image's pixel mask.
//
// This was twice the pixels, on the reasoning that a masked slab decodes a flag chunk beside the
// pixel chunk. It does -- but a flag is bool over the image's own shape, so it is one byte an
// element and not another float32. Counting it as another float32 spends a masked read's budget on
// bytes it never decodes, and the smaller read that leaves is the one thing this header exists to
// prevent.
void TestAMaskCostsOneByteAnElement() {
    using carta::zarr::internal::ChunkElements;
    using carta::zarr::internal::ReadCost;

    carta::zarr::ChunkGeometry geometry;
    geometry.chunk_shape = {256, 260, 2, 1, 1};
    const std::uint64_t elements = 256ULL * 260ULL * 2ULL;
    Require(ChunkElements(geometry) == elements, "a chunk holds the product of its extents");

    const auto image = Image(carta::zarr::DataType::float32, {512, 520, 32, 1, 1});
    carta::zarr::ReadOptions masked;
    carta::zarr::ReadOptions unmasked;
    unmasked.apply_pixel_mask = false;
    const auto chunk_bytes = [&](const carta::zarr::ImageDescriptor& of, const carta::zarr::ReadOptions& options) {
        return ReadCost::Of(of, geometry, geometry, options, PixelsHeld::by_caller, kManyThreads).chunk_bytes;
    };
    Require(chunk_bytes(image, unmasked) == elements * 4, "float32 is four bytes an element");
    Require(chunk_bytes(image, masked) == elements * 5,
            "four bytes of pixels and one of flag, so a masked float32 chunk is a quarter more");

    // The ratio is the image's, not a constant: the flag costs the same whatever the pixels are.
    Require(chunk_bytes(Image(carta::zarr::DataType::float64, {512, 520, 32, 1, 1}), masked) == elements * 9,
            "beside float64 the same flag is an eighth more, not a doubling");
    Require(chunk_bytes(Image(carta::zarr::DataType::int8, {512, 520, 32, 1, 1}), masked) == elements * 2,
            "only a one-byte image is actually doubled by its flag");
}

// A flag need share nothing with its image but the shape, and what decoding a pixel chunk brings
// with it is the flag chunks that chunk lies across, whole. In the pixels' own chunks a flag kept in
// one chunk was a byte an element beside each of them -- the coarse-flag fixture's row of two pixel
// chunks counted 100 bytes and decoded 200.
void TestAFlagIsCountedInItsOwnChunks() {
    using carta::zarr::internal::DecodedFlagBytes;
    using carta::zarr::internal::ReadCost;

    const auto image = Image(carta::zarr::DataType::float32, {10, 5, 8, 4});
    carta::zarr::ChunkGeometry pixels;
    pixels.chunk_shape = {4, 4, 2, 2};
    const auto flag = [](std::vector<std::uint64_t> shape) {
        carta::zarr::ChunkGeometry geometry;
        geometry.chunk_shape = std::move(shape);
        return geometry;
    };
    Require(DecodedFlagBytes(image, pixels, pixels) == 64, "a flag chunked alike is a byte an element");
    Require(DecodedFlagBytes(image, pixels, {}) == 64, "a flag with no layout of its own is the pixels'");
    Require(DecodedFlagBytes(image, pixels, flag({2, 4, 1, 2})) == 64,
            "a finer flag lined up with the pixels is the pixel chunk's elements, in more chunks");
    Require(DecodedFlagBytes(image, pixels, flag({10, 5, 8, 4})) == 1600,
            "a flag kept whole in one chunk is all of it beside every pixel chunk");
    // Channels 2 and 3 are one pixel chunk across two flag chunks of three, so six channels deep.
    Require(DecodedFlagBytes(image, pixels, flag({4, 4, 3, 2})) == 4ULL * 4 * 6 * 2,
            "a flag chunk boundary inside a pixel chunk was not counted on both sides");

    // The coarse-flag fixture: time, frequency, polarization, l, m of 1, 2, 3, 4, 5, the pixels in
    // chunks of 1 x 1 x 1 x 2 x 5 and the flag in one chunk of all of it.
    const auto fixture = Image(carta::zarr::DataType::float32, {1, 2, 3, 4, 5});
    carta::zarr::ChunkGeometry fixture_pixels;
    fixture_pixels.chunk_shape = {1, 1, 1, 2, 5};
    const auto cost = ReadCost::Of(fixture, fixture_pixels, flag({1, 2, 3, 4, 5}), carta::zarr::ReadOptions{},
                                   PixelsHeld::by_caller, kManyThreads);
    Require(cost.chunk_bytes == 40 + 120, "a pixel chunk of the coarse-flag fixture decodes the whole flag");
    carta::zarr::ReadOptions unmasked;
    unmasked.apply_pixel_mask = false;
    Require(ReadCost::Of(fixture, fixture_pixels, flag({1, 2, 3, 4, 5}), unmasked, PixelsHeld::by_caller, kManyThreads)
                    .chunk_bytes == 40,
            "a read that does not apply the flag does not decode it");
}

// What a walk decodes along the spectrum, which is what its progress is counted in. The chunks a
// selection spans from first to last are not that once a stride steps over whole chunks.
void TestAStrideCountsOnlyTheChunksItLandsIn() {
    using carta::zarr::internal::ChunksTouched;
    Require(ChunksTouched(1, 8, 1, 4) == 3, "channels 1 to 8 in chunks of four touch chunks 0, 1 and 2");
    Require(ChunksTouched(1, 8, 3, 4) == 6, "every third channel from 1 misses none of six chunks");
    Require(ChunksTouched(0, 3, 9, 4) == 3,
            "channels 0, 9 and 18 land in three chunks, not the five from the first to the last");
    Require(ChunksTouched(0, 0, 1, 4) == 0, "nothing selected touches nothing");
}

// Which chunks a sampled axis touches, named rather than counted, for a walk to go through. They are
// the ones ChunksTouched counts, which is what keeps a walk's reads and its plan's layer one number.
void TestASampleNamesTheChunksItTouches() {
    using carta::zarr::internal::ChunksSampled;
    using carta::zarr::internal::ChunksTouched;
    using Chunks = std::vector<std::uint64_t>;
    const auto listed = [](std::uint64_t length, std::uint64_t chunk, std::uint64_t stride) {
        const auto sampled = ChunksSampled(length, chunk, stride);
        Chunks chunks;
        for (std::uint64_t index = 0; index < sampled.size(); ++index) {
            chunks.push_back(sampled[index]);
        }
        return chunks;
    };
    Require(listed(16, 4, 8) == Chunks{0, 2}, "every eighth of 16 in chunks of four is in chunks 0 and 2");
    Require(listed(10, 4, 3) == Chunks{0, 1, 2}, "every third of 10 steps over no chunk");
    Require(listed(13, 4, 1) == Chunks{0, 1, 2, 3}, "every element touches every chunk, the partial one too");
    Require(listed(13, 4, 5) == Chunks{0, 1, 2}, "0, 5 and 10 miss the partial chunk at 12");
    Require(ChunksSampled(0, 4, 1).empty(), "an axis of nothing touches nothing");
    for (std::uint64_t length = 1; length <= 40; ++length) {
        for (std::uint64_t chunk = 1; chunk <= 9; ++chunk) {
            for (std::uint64_t stride = 1; stride <= 12; ++stride) {
                const auto samples = ((length - 1) / stride) + 1;
                Require(ChunksSampled(length, chunk, stride).size() == ChunksTouched(0, samples, stride, chunk),
                        "named and counted disagree at length " + std::to_string(length) + ", chunk " +
                            std::to_string(chunk) + ", stride " + std::to_string(stride));
            }
        }
    }
}

}  // namespace

int main() {
    try {
        TestRealChunkShapesGetEnoughChunks();
        TestAnOversizedChunkIsCappedRatherThanMultiplied();
        TestASmallChunkKeepsTheByteBudget();
        TestMoreThreadsBuyMoreChunks();
        TestAChunkHoldsThreeTimesWhatItDecodesTo();
        TestAMaskCostsOneByteAnElement();
        TestAFlagIsCountedInItsOwnChunks();
        TestASampleNamesTheChunksItTouches();
        TestAStrideCountsOnlyTheChunksItLandsIn();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "chunk blocks test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
