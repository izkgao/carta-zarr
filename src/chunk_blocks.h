/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_CHUNK_BLOCKS_H_
#define CARTA_ZARR_SRC_CHUNK_BLOCKS_H_

#include <algorithm>
#include <cstdint>

namespace carta::zarr::internal {

// Enough chunks in one storage request to keep the decode pool busy.
//
// Measured on two cubes by reading one cursor column in differently sized pieces. One chunk per
// request is 5 to 7 times slower than reading the whole column at once, and four is still about
// twice as slow -- not because a request costs anything much, but because a request holding one
// chunk has nothing to decode in parallel. From sixteen chunks upwards the difference disappears,
// and at sixty-four a split read is as fast as an unsplit one or faster.
//
// This is a fact about the storage, not a caller's tuning knob, which is why splitting a read is
// decided here rather than by whoever is asking for pixels.
inline constexpr std::uint64_t kMinChunksPerRead = 64;

// The end of a block of selected indices that begins at `begin` and would like to be `desired`
// long, moved so that it lands on a chunk boundary.
//
// Splitting inside a chunk would make one decode serve two blocks, and every caller here decodes a
// chunk exactly once. When the desired length does not reach the end of the chunk it started in,
// the block grows to that chunk's end rather than shrinking to nothing.
inline std::uint64_t AlignedBlockEnd(std::uint64_t begin, std::uint64_t desired, std::uint64_t total,
                                     std::uint64_t axis_start, std::uint64_t stride, std::uint64_t chunk) {
    if (desired == 0) {
        desired = 1;
    }
    const std::uint64_t target = std::min(begin + desired, total);
    if (target >= total || chunk == 0 || stride == 0) {
        return target;
    }
    const auto chunk_of = [&](std::uint64_t index) { return (axis_start + (index * stride)) / chunk; };
    // The first selected index at or after an absolute coordinate.
    const auto first_at = [&](std::uint64_t absolute) -> std::uint64_t {
        if (absolute <= axis_start) {
            return 0;
        }
        return ((absolute - axis_start) + stride - 1) / stride;
    };

    const auto chunk_index = chunk_of(target);
    if (chunk_of(target - 1) != chunk_index) {
        return target;
    }
    const auto chunk_first = first_at(chunk_index * chunk);
    if (chunk_first > begin) {
        return chunk_first;
    }
    return std::min(first_at((chunk_index + 1) * chunk), total);
}

// How many chunks a strided selection spans along one axis.
inline std::uint64_t ChunksSpanned(std::uint64_t start, std::uint64_t count, std::uint64_t stride,
                                   std::uint64_t chunk) {
    if (chunk == 0 || count == 0) {
        return 1;
    }
    const auto last = start + ((count - 1) * (stride == 0 ? 1 : stride));
    return (last / chunk) - (start / chunk) + 1;
}

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_CHUNK_BLOCKS_H_
