/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// How large one request is allowed to be. This has been wrong twice -- once as a fixed chunk count
// that ignored how far chunk sizes vary, once as a byte budget alone that gave a 16 MiB-chunk image
// four chunks per request where a 4 MiB one got sixteen -- so the policy is pinned here in the unit
// that matters, which is chunks, across the chunk sizes XRADIO actually writes.

#include "chunk_blocks.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "support/check.h"

namespace {

using carta::zarr::internal::DefaultReadBytes;
using carta::zarr::internal::kMaxDecodedBytesPerRead;
using carta::zarr::internal::kMinChunksPerRead;

constexpr std::uint64_t kMiB = 1u << 20;

using carta::zarr::testing::Require;

std::uint64_t ChunksPerRead(std::uint64_t chunk_bytes) {
    return DefaultReadBytes(chunk_bytes) / chunk_bytes;
}

// The shapes these images are written with, as (x, y, spectral) times the polarizations the chunk
// carries. A chunk holding every stokes is four times the size of one holding a single one, which
// is the whole reason a byte budget alone is not enough.
void TestRealChunkShapesGetEnoughChunks() {
    const struct {
        const char* what;
        std::uint64_t bytes;
    } cases[]{
        {"HD163296 128x128x2", 128 * 128 * 2 * 4},
        {"ASKAP 512x512x1", 512 * 512 * 1 * 4},
        {"ASKAP 256x256x3", 256 * 256 * 3 * 4},
        {"IRC 256x256x1 over four stokes", 256 * 256 * 1 * 4 * 4},
        {"XRADIO 256x256x6", 256 * 256 * 6 * 4},
        {"XRADIO 256x256x6 over four stokes", 256 * 256 * 6 * 4 * 4},
        {"XRADIO 512x512x4", 512 * 512 * 4 * 4},
        {"XRADIO 512x512x4 over four stokes", 512 * 512 * 4 * 4 * 4},
    };
    for (const auto& one : cases) {
        const auto chunks = ChunksPerRead(one.bytes);
        Require(chunks >= kMinChunksPerRead,
                std::string(one.what) + " gets only " + std::to_string(chunks) +
                    " chunks per request, which does not keep the decode pool busy");
    }
}

// A chunk large enough that eight of them would be past the ceiling gets as many as fit, and one
// larger than the ceiling still gets the one it cannot avoid.
void TestAnOversizedChunkIsCappedRatherThanMultiplied() {
    Require(DefaultReadBytes(64 * kMiB) == kMaxDecodedBytesPerRead,
            "a chunk of an eighth of the ceiling should stop at the ceiling, not multiply past it");
    Require(DefaultReadBytes(480 * kMiB) == kMaxDecodedBytesPerRead,
            "a chunk larger than the ceiling should not raise the budget above it; the walk's own "
            "floor of one chunk is what makes the request, and nothing can make it smaller");
}

// A small chunk does not shrink the budget: the bytes are the cap and the chunks are the floor.
void TestASmallChunkKeepsTheByteBudget() {
    Require(DefaultReadBytes(128 * 1024) == carta::zarr::internal::kDecodedBytesPerRead,
            "a small chunk should leave the byte budget alone");
    Require(DefaultReadBytes(0) == carta::zarr::internal::kDecodedBytesPerRead,
            "an image reporting no chunk size should fall back to the byte budget");
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
    using carta::zarr::internal::DecodedChunkBytes;

    carta::zarr::ChunkGeometry geometry;
    geometry.chunk_shape = {256, 260, 2, 1, 1};
    const std::uint64_t elements = 256ULL * 260ULL * 2ULL;
    Require(ChunkElements(geometry) == elements, "a chunk holds the product of its extents");

    carta::zarr::ImageDescriptor image;
    image.stored_type = carta::zarr::DataType::float32;
    Require(DecodedChunkBytes(image, geometry, false) == elements * 4, "float32 is four bytes an element");
    Require(DecodedChunkBytes(image, geometry, true) == elements * 5,
            "four bytes of pixels and one of flag, so a masked float32 chunk is a quarter more");

    // The ratio is the image's, not a constant: the flag costs the same whatever the pixels are.
    carta::zarr::ImageDescriptor doubles;
    doubles.stored_type = carta::zarr::DataType::float64;
    Require(DecodedChunkBytes(doubles, geometry, true) == elements * 9,
            "beside float64 the same flag is an eighth more, not a doubling");

    carta::zarr::ImageDescriptor bytes;
    bytes.stored_type = carta::zarr::DataType::int8;
    Require(DecodedChunkBytes(bytes, geometry, true) == elements * 2,
            "only a one-byte image is actually doubled by its flag");
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
        TestAMaskCostsOneByteAnElement();
        TestASampleNamesTheChunksItTouches();
        TestAStrideCountsOnlyTheChunksItLandsIn();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "chunk blocks test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
