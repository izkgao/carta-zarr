/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_CHUNK_BLOCKS_H_
#define CARTA_ZARR_SRC_CHUNK_BLOCKS_H_

#include "carta-zarr/types.h"

#include <algorithm>
#include <cstdint>

namespace carta::zarr::internal {

// How much decompressed chunk data one storage request should pull through.
//
// Two things set this, and they pull in opposite directions. A request that is too small stops
// holding enough chunks to decode in parallel -- see kMinChunksPerRead below, which is that floor
// in the unit it is really in. Above about 100 MiB a piece takes longer than the interval a caller
// wants to report progress on, which is the reason for splitting at all.
//
// 64 MiB sits between them for the images these were measured on: sixteen chunks of a 4 MiB one,
// and about a third of a second of decoding at the rates they decompress at. DefaultReadBytes
// raises it when a chunk is large enough that this many bytes would not buy enough chunks.
//
// This is a byte budget rather than a chunk count because chunk sizes are not comparable across
// real images -- the cubes measured here span 0.12 MiB to 480 MiB per decoded chunk, a factor of
// nearly four thousand. A fixed count of 64 would mean 8 MiB pieces on one of them and 30 GiB
// pieces on another, and the one with the largest chunks would end up never splitting at all.
inline constexpr std::size_t kDecodedBytesPerRead = 64u << 20;

// The chunks one request should cover, below which the decode pool runs short of work.
//
// This is the same floor the budget above describes, stated in the unit it is actually in. Holding
// everything else fixed and varying only the decode concurrency settles it: on a 1 MiB chunk image
// a whole-plane profile took 97.8 ms at 64 chunks per request, 109.2 at 16, 119.8 at 8, 152.5 at 4
// and 462.2 at 1 -- while the same sweep with the pool limited to one thread was flat at 410-459 ms
// throughout. A request holding one chunk performs exactly as if there were no pool, because there
// is nothing to spread over it.
//
// So a budget in bytes alone is not enough: 64 MiB is sixteen chunks of a 4 MiB image and four of a
// 16 MiB one, and XRADIO writes both -- 512x512x4 is 4 MiB with one polarization and 16 MiB with
// four of them in the chunk. Eight is where the curve is within a quarter of flat on a five-core
// machine. A machine with more decode threads wants more, so this is a floor, not a target.
inline constexpr std::uint64_t kMinChunksPerRead = 8;

// The ceiling on raising the budget to reach that floor. An image whose chunk is already a large
// fraction of what a request should hold cannot be given eight of them, and past this both reasons
// for splitting at all -- bounded memory, and a piece short enough to report on -- are lost anyway.
inline constexpr std::size_t kMaxDecodedBytesPerRead = 256u << 20;

// What one request may decode when the caller has not said otherwise.
inline std::uint64_t DefaultReadBytes(std::uint64_t chunk_bytes) {
    const std::uint64_t wanted = std::max<std::uint64_t>(1, chunk_bytes) * kMinChunksPerRead;
    return std::min<std::uint64_t>(kMaxDecodedBytesPerRead,
                                   std::max<std::uint64_t>(kDecodedBytesPerRead, wanted));
}

// Bytes one chunk of this image decompresses to.
inline std::uint64_t DecodedChunkBytes(const ImageDescriptor& descriptor, const ChunkGeometry& geometry) {
    std::uint64_t elements = 1;
    for (const auto length : geometry.chunk_shape) {
        elements *= std::max<std::uint64_t>(1, length);
    }
    std::uint64_t element_bytes = 4;
    switch (descriptor.stored_type) {
        case DataType::boolean:
        case DataType::int8:
        case DataType::uint8: element_bytes = 1; break;
        case DataType::int16:
        case DataType::uint16:
        case DataType::float16: element_bytes = 2; break;
        case DataType::int64:
        case DataType::uint64:
        case DataType::float64:
        case DataType::complex64: element_bytes = 8; break;
        case DataType::complex128: element_bytes = 16; break;
        default: element_bytes = 4; break;
    }
    return std::max<std::uint64_t>(1, elements * element_bytes);
}

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
