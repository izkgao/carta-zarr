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
// Two things set this, and they pull in opposite directions. Below about 16 MiB a request stops
// holding enough chunks to decode in parallel: measured by reading one cursor column of 1 MiB
// chunks in pieces, one chunk per request is 7x slower than one unsplit read and four chunks is
// still 2x, not because a request costs anything but because a request holding one chunk has
// nothing to spread over the decode threads. Above about 100 MiB a piece takes longer than the
// interval a caller wants to report progress on, which is the reason for splitting at all.
//
// 64 MiB sits between them with room on both sides: four times the floor, and about a third of a
// second of decoding at the rates these images decompress at.
//
// This is a byte budget rather than a chunk count because chunk sizes are not comparable across
// real images -- the four cubes measured here span 0.12 MiB to 48 MiB per chunk, a factor of 400.
// A fixed count of 64 would mean 8 MiB pieces on one of them and 3 GiB pieces on another, and the
// one with the largest chunks would end up never splitting at all.
inline constexpr std::size_t kDecodedBytesPerRead = 64u << 20;

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
