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
#include <stdexcept>
#include <string>

namespace {

using carta::zarr::internal::DefaultReadBytes;
using carta::zarr::internal::kMaxDecodedBytesPerRead;
using carta::zarr::internal::kMinChunksPerRead;

constexpr std::uint64_t kMiB = 1u << 20;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

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

}  // namespace

int main() {
    try {
        TestRealChunkShapesGetEnoughChunks();
        TestAnOversizedChunkIsCappedRatherThanMultiplied();
        TestASmallChunkKeepsTheByteBudget();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "chunk blocks test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
