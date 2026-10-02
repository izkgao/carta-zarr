/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_CHUNK_BLOCKS_H_
#define CARTA_ZARR_SRC_CHUNK_BLOCKS_H_

#include "carta-zarr/read.h"
#include "carta-zarr/reduce.h"

#include "pixel_mask.h"
#include "zarr/data_type.h"

#include <algorithm>
#include <cstdint>
#include <vector>

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

// Elements in one chunk of this image.
// How many units of `chunks_per_unit` chunks a read that may decode `chunks_per_read` chunks affords.
// Never zero: a budget smaller than a single unit still reads one, because a chunk is the smallest
// thing that can be decoded and refusing to read is the worse answer.
//
// A unit is at least one chunk; one of none is read as one. A free function rather than only the
// pass plan's, because the occupancy asks it too, and it takes the one number rather than the plan
// so that the occupancy can ask it with nothing linked behind it -- ADR 0006.
inline std::uint64_t UnitsAffordable(std::uint64_t chunks_per_read, std::uint64_t chunks_per_unit) {
    return std::max<std::uint64_t>(1, chunks_per_read / std::max<std::uint64_t>(1, chunks_per_unit));
}

inline std::uint64_t ChunkElements(const ChunkGeometry& geometry) {
    std::uint64_t elements = 1;
    for (const auto length : geometry.chunk_shape) {
        elements *= std::max<std::uint64_t>(1, length);
    }
    return elements;
}

// Bytes one chunk of this image decompresses to.
//
// A type the table does not name is counted as four bytes rather than refused. This is a read
// budget: getting it wrong makes a read the wrong size, which costs time, while refusing here would
// close an image over a question nobody asked.
inline std::uint64_t DecodedChunkBytes(const ImageDescriptor& descriptor, const ChunkGeometry& geometry) {
    const std::uint64_t elements = ChunkElements(geometry);
    const auto* const info = zarr::FindDataType(descriptor.stored_type);
    const std::uint64_t element_bytes = info != nullptr ? info->element_bytes : 4;
    return std::max<std::uint64_t>(1, elements * element_bytes);
}

// The same, counting the flag chunk a read decodes beside the pixels when it will apply the image's
// pixel mask.
//
// One byte an element rather than another copy of the pixels: a flag is boolean over the image's own
// shape by construction -- RequireUsableFlag holds it to both -- so beside a float32 chunk it is a
// quarter of one, not a second one. Whichever way this is wrong it is wrong in the units the budget
// is spent in, and a read sized too small is the mistake this header exists to prevent: on a 1 MiB
// chunk image a whole-plane profile took 97.8 ms at 64 chunks per request and 462.2 at one.
inline std::uint64_t DecodedChunkBytes(const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                                       bool apply_mask) {
    const std::uint64_t pixels = DecodedChunkBytes(descriptor, geometry);
    return apply_mask ? pixels + ChunkElements(geometry) : pixels;
}

// What one read of this image costs, and how much of that it may spend at once: the three answers
// every walk starts from, whether it is Image::Read cutting pieces or a reduction planning a pass.
//
// They were written out twice, word for word -- whether the flag is folded in, what a chunk
// decodes to counting it, and the caller's budget or the library's own -- and the two copies have
// to agree, because a read and a reduction that sized themselves against different costs would
// split the same image differently for no reason either could give.
struct ReadCost {
    bool apply_mask = false;
    std::uint64_t chunk_bytes = 1;
    std::size_t budget_bytes = 0;

    static ReadCost Of(const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                       const ReadOptions& options) {
        ReadCost cost;
        cost.apply_mask = AppliesPixelMask(options, descriptor);
        cost.chunk_bytes = DecodedChunkBytes(descriptor, geometry, cost.apply_mask);
        cost.budget_bytes = options.read_budget_bytes != 0 ? options.read_budget_bytes
                                                                      : DefaultReadBytes(cost.chunk_bytes);
        return cost;
    }
};

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

// How many chunks a strided selection touches along one axis: the ones that hold a selected
// element, which is what a walk decodes and so what its progress is counted in.
//
// Not ChunksSpanned. A stride of a chunk or more puts every selected element in a chunk of its own
// and can step over whole chunks between them, which a span from first to last would count. Below
// a chunk, no step is long enough to skip one, and the two agree.
inline std::uint64_t ChunksTouched(std::uint64_t start, std::uint64_t count, std::uint64_t stride,
                                   std::uint64_t chunk) {
    if (count == 0) {
        return 0;
    }
    if (chunk == 0 || stride >= chunk) {
        return count;
    }
    return ChunksSpanned(start, count, stride, chunk);
}

// The chunks along an axis of `length` elements that a sample of every `stride`th element from the
// first has an element in, in order: the ones ChunksTouched counts, named. A whole-plane pass walks
// these, so that a chunk the sample steps over is neither read nor counted.
inline std::vector<std::uint64_t> ChunksSampled(std::uint64_t length, std::uint64_t chunk, std::uint64_t stride) {
    std::vector<std::uint64_t> chunks;
    if (length == 0) {
        return chunks;
    }
    chunk = std::max<std::uint64_t>(1, chunk);
    stride = std::max<std::uint64_t>(1, stride);
    const std::uint64_t samples = ((length - 1) / stride) + 1;
    if (stride < chunk) {
        // No step is long enough to skip a chunk, so they are every chunk up to the last sample's.
        const std::uint64_t last = ((samples - 1) * stride) / chunk;
        chunks.reserve(static_cast<std::size_t>(last + 1));
        for (std::uint64_t index = 0; index <= last; ++index) {
            chunks.push_back(index);
        }
        return chunks;
    }
    // Every sample is in a chunk of its own.
    chunks.reserve(static_cast<std::size_t>(samples));
    for (std::uint64_t sample = 0; sample < samples; ++sample) {
        chunks.push_back((sample * stride) / chunk);
    }
    return chunks;
}

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_CHUNK_BLOCKS_H_
