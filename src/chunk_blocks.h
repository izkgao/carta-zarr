/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_ZARR_SRC_CHUNK_BLOCKS_H_
#define CARTA_ZARR_SRC_CHUNK_BLOCKS_H_

#include "carta-zarr/read.h"
#include "carta-zarr/reduce.h"

#include "pixel_mask.h"
#include "zarr/data_type.h"

#include <algorithm>
#include <cstdint>

namespace carta::zarr::internal {

// What a chunk in flight holds, for every byte it decodes to.
//
// A read budget bounds memory, and decoded bytes are not all of what a read holds: while a chunk is
// decoded, its compressed bytes and the codec's own buffer are resident beside the decoded copy.
// Measured peak resident memory, over a 7763 x 4742 x 128 cube in 512 x 512 x 64 chunks of blosc
// zstd, is `base + destination + chunks x (2 x decoded + compressed)` at one, two and four chunks a
// read, and four decode threads or twenty-eight made no difference to it -- so what a read holds
// is decided by how many chunks it decodes, not by the machine. Compressed bytes are at most about
// the decoded ones, so three is the worst a chunk holds.
//
// Three on every image, rather than a figure per codec: a cheaper codec is held under the budget it
// was given, while a figure worked out from TensorStore's codecs would go quietly wrong when they
// change. It is a statement of what the budget means, not a tuning, so nothing overrides it.
inline constexpr std::uint64_t kHeldBytesPerDecodedByte = 3;

// The chunks one read should decode at once for each thread decoding them, below which the decode
// pool runs short of work.
//
// On a 1 MiB chunk image a whole-plane profile took 98 ms at 64 chunks per request and 462 ms at
// one, which is what it took with the pool limited to one thread: a request holding one chunk has
// nothing to spread over the pool. A count fixed for every machine does not hold: eight a read was
// within a quarter of flat on a five-core machine, and on twenty-eight threads it left most of them
// idle and plane reads of 64 MiB chunks ran 2.4 times slower. Two a thread keeps every thread one
// chunk ahead of the one it is decoding.
inline constexpr std::uint64_t kChunksPerDecodeThread = 2;

// The least a read holds by default, whatever its chunks and threads. On small chunks this, not the
// thread count, sets the read: 256 MiB is sixty-four 1 MiB chunks, which keeps every decode thread
// busy and a piece short enough to report progress on.
inline constexpr std::uint64_t kLeastBytesPerRead = 256u << 20;

// The most a read holds by default. Two chunks a thread of 64 MiB ones would hold over 10 GiB on
// twenty-eight threads; this keeps them to ten. What a deployment short of memory wants instead it
// sets as ReadOptions::read_budget_bytes.
inline constexpr std::uint64_t kMostBytesPerRead = 2u << 30;

// What one read may hold when the caller has not said otherwise, given what one chunk of it holds
// and how many threads decode its chunks. No thread count is taken as one.
inline std::uint64_t DefaultReadBytes(std::uint64_t held_per_chunk, std::size_t decode_threads) {
    const std::uint64_t wanted = std::max<std::uint64_t>(1, held_per_chunk) * kChunksPerDecodeThread *
                                 std::max<std::uint64_t>(1, decode_threads);
    return std::min<std::uint64_t>(kMostBytesPerRead, std::max<std::uint64_t>(kLeastBytesPerRead, wanted));
}

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

// Elements in one chunk of this image.
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

// The most flag chunks one pixel chunk along an axis of `length` elements lies across: one when the
// two are chunked alike, more when the flag's are shorter or their boundaries fall inside a pixel
// chunk. Asked of every pixel chunk rather than bounded, since a bound that assumes the worst
// alignment doubles the count of two chunk shapes that line up; an axis is at most a few thousand
// chunks long. An axis with no chunk shape to speak of is one chunk, of either.
inline std::uint64_t FlagChunksAcross(std::uint64_t length, std::uint64_t pixel_chunk, std::uint64_t flag_chunk) {
    // Chunked alike needs no walk, and every read asks this of every axis.
    if (flag_chunk == 0 || length == 0 || flag_chunk == pixel_chunk) {
        return 1;
    }
    const std::uint64_t step = pixel_chunk == 0 ? length : pixel_chunk;
    std::uint64_t most = 1;
    for (std::uint64_t first = 0; first < length; first += step) {
        const std::uint64_t last = std::min(first + step, length) - 1;
        most = std::max(most, (last / flag_chunk) - (first / flag_chunk) + 1);
    }
    return most;
}

// Bytes of flag that decoding one pixel chunk of this image decodes beside it: every flag chunk the
// pixel chunk lies across, whole, at a byte an element. `flag_geometry` without a chunk shape is
// taken to be the pixels'.
//
// One byte an element rather than another copy of the pixels: a flag is boolean over the image's own
// shape by construction -- RequireUsableFlag holds it to both -- so beside a float32 chunk it is a
// quarter of one, not a second one. But a flag need share nothing else with its image, and its chunks
// are what is decoded. Chunked alike, or finer and lined up, that comes to the pixel chunk's own
// elements. Coarser, it is more: a flag kept whole in one chunk is all of it beside every pixel
// chunk, and counting it in the pixels' chunks said a read of the coarse-flag fixture's two-chunk row
// decoded 100 bytes where it decoded 200.
//
// This is the most one pixel chunk can bring, and a read of several that share a flag chunk decodes
// that chunk once, so a flag coarser than its pixels is over-counted in any read of more than one.
// That is the side to be wrong on. Every walk here counts what it may decode in pixel chunks, and an
// over-count costs a read some of the chunks it could have decoded in parallel, where an under-count
// spends memory the caller's budget said not to.
inline std::uint64_t DecodedFlagBytes(const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                                      const ChunkGeometry& flag_geometry) {
    const auto& flag = flag_geometry.chunk_shape.empty() ? geometry : flag_geometry;
    const auto chunk = [](const ChunkGeometry& of, std::size_t axis) -> std::uint64_t {
        return axis < of.chunk_shape.size() ? of.chunk_shape[axis] : 0;
    };
    std::uint64_t bytes = 1;
    for (std::size_t axis = 0; axis < descriptor.axes.size(); ++axis) {
        const auto flag_chunk = chunk(flag, axis);
        bytes *= FlagChunksAcross(descriptor.axes[axis].length, chunk(geometry, axis), flag_chunk) *
                 std::max<std::uint64_t>(1, flag_chunk);
    }
    return bytes;
}

// Who holds the pixels a read produces: the caller, whose destination they go straight into and which
// no budget can shrink, or the library, in a buffer of its own that the budget pays for.
enum class PixelsHeld { by_caller, by_library };

// What one read of this image holds, and how much it may hold at once: the answers every walk starts
// from, whether it is Image::Read cutting pieces or a reduction planning a pass. A read and a
// reduction that sized themselves against different costs would split the same image differently for
// no reason either could give, so both ask this.
//
// A chunk here is a pixel chunk. What it decodes to counts the flag it brings, in the flag's own
// chunks (see DecodedFlagBytes); what it holds is that, kHeldBytesPerDecodedByte over, plus the
// buffers the library allocates for its share of the read -- the folded-in flag a byte an element,
// and the pixels when the library rather than the caller holds them. The default budget aims at
// `decode_threads`, the threads of the context the read goes through.
struct ReadCost {
    bool apply_mask = false;
    std::uint64_t chunk_bytes = 1;
    std::uint64_t held_bytes = 1;
    std::size_t budget_bytes = 0;

    static ReadCost Of(const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                       const ChunkGeometry& flag_geometry, const ReadOptions& options, PixelsHeld pixels,
                       std::size_t decode_threads) {
        ReadCost cost;
        cost.apply_mask = AppliesPixelMask(options, descriptor);
        cost.chunk_bytes = DecodedChunkBytes(descriptor, geometry) +
                           (cost.apply_mask ? DecodedFlagBytes(descriptor, geometry, flag_geometry) : 0);
        cost._chunk_elements = ChunkElements(geometry);
        cost.held_bytes = cost.Held(pixels);
        cost.budget_bytes = options.read_budget_bytes != 0 ? options.read_budget_bytes
                                                           : DefaultReadBytes(cost.held_bytes, decode_threads);
        return cost;
    }

    // What one chunk holds when `pixels` holds what it produces, against the same budget. A read whose
    // pixels go to its caller still holds some itself when it has to gather them first.
    std::uint64_t Held(PixelsHeld pixels) const {
        return (kHeldBytesPerDecodedByte * chunk_bytes) + (apply_mask ? _chunk_elements : 0) +
               (pixels == PixelsHeld::by_library ? _chunk_elements * sizeof(float) : 0);
    }

    // How many chunks the budget affords when `pixels` holds what they produce. Zero when it affords
    // less than one, which every caller floors at one through UnitsAffordable: a chunk is the least
    // that can be decoded.
    std::uint64_t ChunksPerRead(PixelsHeld pixels) const {
        return budget_bytes / std::max<std::uint64_t>(1, Held(pixels));
    }

private:
    std::uint64_t _chunk_elements = 1;
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
        // Rounded up without adding stride - 1, which wraps for a stride near the top of the range.
        const std::uint64_t offset = absolute - axis_start;
        return (offset / stride) + static_cast<std::uint64_t>(offset % stride != 0);
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

// How many chunks a strided selection touches along one axis: the ones that hold a selected
// element, which is what a walk decodes and so what its progress is counted in.
//
// Not the chunks from the first selected element to the last. A stride of a chunk or more puts every
// selected element in a chunk of its own and can step over whole chunks between them, which a span
// would count and nothing decodes. Below a chunk, no step is long enough to skip one, and the two
// agree. There was a ChunksSpanned beside this, and piece sizing asked it rather than this, so a
// strided read was sized for chunks it stepped over; this is the one answer now.
inline std::uint64_t ChunksTouched(std::uint64_t start, std::uint64_t count, std::uint64_t stride,
                                   std::uint64_t chunk) {
    if (count == 0) {
        return 0;
    }
    if (chunk == 0 || stride >= chunk) {
        return count;
    }
    // Below a chunk the chunks touched are every one from the first selected element's to the last's.
    const auto last = start + ((count - 1) * (stride == 0 ? 1 : stride));
    return (last / chunk) - (start / chunk) + 1;
}

// The chunks along an axis of `length` elements that a sample of every `stride`th element from the
// first has an element in, in order: the ones ChunksTouched counts, named. A whole-plane pass walks
// these, so that a chunk the sample steps over is neither read nor counted.
//
// Worked out as they are asked for rather than listed, so that a walk asking for them allocates
// nothing.
class SampledChunks {
public:
    SampledChunks(std::uint64_t length, std::uint64_t chunk, std::uint64_t stride)
        : _chunk(std::max<std::uint64_t>(1, chunk)), _stride(std::max<std::uint64_t>(1, stride)) {
        if (length == 0) {
            return;
        }
        const std::uint64_t samples = ((length - 1) / _stride) + 1;
        // Below a chunk no step is long enough to skip one, so they are every chunk up to the last
        // sample's; otherwise every sample is in a chunk of its own.
        _every = _stride < _chunk;
        _size = _every ? (((samples - 1) * _stride) / _chunk) + 1 : samples;
    }

    std::uint64_t size() const { return _size; }
    bool empty() const { return _size == 0; }
    // The index-th of them, for index < size().
    std::uint64_t operator[](std::uint64_t index) const { return _every ? index : (index * _stride) / _chunk; }

private:
    std::uint64_t _chunk;
    std::uint64_t _stride;
    std::uint64_t _size = 0;
    bool _every = true;
};

inline SampledChunks ChunksSampled(std::uint64_t length, std::uint64_t chunk, std::uint64_t stride) {
    return SampledChunks(length, chunk, stride);
}

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_CHUNK_BLOCKS_H_
