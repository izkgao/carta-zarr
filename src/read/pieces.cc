/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

// What a read decides before it reads anything.
//
// Separate from pieces_read.cc so that the strategy stays linkable without a store: PlanPieces and
// the arithmetic beside it reach no further than the descriptor and the geometry, and the test that
// pins them compiles this translation unit alone. The same division pass.cc and pass_read.cc make,
// and for the same reason.

#include "read/pieces.h"

#include "chunk_blocks.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace carta::zarr::internal {
namespace {

// The axis a split read is cut along: the slowest-varying one that selects more than a single
// element. The destination is dense in logical order with axis 0 fastest, so cutting there and
// nowhere else is what makes each finished piece extend a prefix instead of leaving holes.
std::optional<std::size_t> SlowestSelectedAxis(const ReadRequest& request) {
    for (std::size_t i = request.axes.size(); i-- > 0;) {
        if (request.axes.at(i).count > 1) {
            return i;
        }
    }
    return std::nullopt;
}

// The chunks of `geometry` that `range` of `axis` decodes, which a stride of a chunk or more makes
// fewer than the chunks its first and last element span. An axis with no chunk to speak of is one.
std::uint64_t ChunksAlong(const ChunkGeometry& geometry, const ReadRequest& request, std::size_t axis) {
    const auto chunk = axis < geometry.chunk_shape.size() ? geometry.chunk_shape.at(axis) : 0;
    const auto& range = request.axes.at(axis);
    return chunk == 0 ? 1 : std::max<std::uint64_t>(1, ChunksTouched(range.start, range.count, range.stride, chunk));
}

// The chunks a read of `request` decodes: the product of what each axis touches.
std::uint64_t ChunksOf(const ChunkGeometry& geometry, const ReadRequest& request) {
    std::uint64_t chunks = 1;
    for (std::size_t axis = 0; axis < request.axes.size(); ++axis) {
        chunks *= ChunksAlong(geometry, request, axis);
    }
    return chunks;
}

// How many selected indices of `axis` hold `chunks` of its chunks, so that a run of them decodes about
// that many. A stride of a chunk or more puts every selected index in a chunk of its own, so an index
// is a chunk. Below that a chunk holds about chunk / stride of them; AlignedBlockEnd rounds the run
// out to a whole chunk, so a low estimate costs nothing.
std::uint64_t IndicesFor(const ChunkGeometry& geometry, const ReadRequest& request, std::size_t axis,
                         std::uint64_t chunks) {
    const auto chunk = axis < geometry.chunk_shape.size() ? geometry.chunk_shape.at(axis) : 0;
    const auto stride = std::max<std::uint64_t>(1, request.axes.at(axis).stride);
    if (chunk != 0 && stride >= chunk) {
        return chunks;
    }
    return std::max<std::uint64_t>(1, (chunks * std::max<std::uint64_t>(1, chunk)) / stride);
}

// `request` cut along `axis` into runs of about `indices` selected indices, each rounded out to a chunk
// boundary so that no chunk is decoded by two of them. Calls `take(run, first_index)` for each in order.
template <typename Take>
void CutAlong(const ChunkGeometry& geometry, const ReadRequest& request, std::size_t axis, std::uint64_t indices,
              Take&& take) {
    const auto chunk = axis < geometry.chunk_shape.size() ? geometry.chunk_shape.at(axis) : 0;
    const auto& cut = request.axes.at(axis);
    for (std::uint64_t begin = 0; begin < cut.count;) {
        const std::uint64_t end = AlignedBlockEnd(begin, indices, cut.count, cut.start, cut.stride, chunk);
        ReadRequest run = request;
        run.axes.at(axis) = Range{cut.start + (begin * cut.stride), end - begin, cut.stride};
        take(std::move(run), begin);
        begin = end;
    }
}

// The parts of `request` a read holding at most `affordable` chunks at once decodes it in: along the
// slowest axis that still spans more than one chunk, as many of its chunks as the rest of the request
// leaves room for, and the next axis in turn wherever even one of them is too many. Down to a single
// chunk, which is as far as anything can be cut.
void Segment(const ChunkGeometry& geometry, const ReadRequest& request, std::uint64_t affordable,
             std::vector<ReadRequest>& segments) {
    const auto chunks = ChunksOf(geometry, request);
    std::optional<std::size_t> axis;
    for (std::size_t i = request.axes.size(); i-- > 0;) {
        if (ChunksAlong(geometry, request, i) > 1) {
            axis = i;
            break;
        }
    }
    if (chunks <= affordable || !axis) {
        segments.push_back(request);
        return;
    }
    const auto across = chunks / ChunksAlong(geometry, request, *axis);
    const auto runs = UnitsAffordable(affordable, across);
    CutAlong(geometry, request, *axis, IndicesFor(geometry, request, *axis, runs), [&](ReadRequest run, std::uint64_t) {
        if (ChunksOf(geometry, run) > affordable) {
            Segment(geometry, run, affordable, segments);
        } else {
            segments.push_back(std::move(run));
        }
    });
}

}  // namespace

std::vector<Piece> PlanPieces(const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                              const ChunkGeometry& flag_geometry, const ReadRequest& request,
                              const ReadOptions& options, std::size_t decode_threads, bool reports_progress) {
    const auto axis = SlowestSelectedAxis(request);
    if (!axis) {
        return {Piece{request, 0, {}}};
    }

    // The flag is decoded beside the pixels when this read will apply it, so what a chunk holds counts
    // it, in the flag's own chunks, which a piece decodes whole. The pixels go to the caller's
    // destination, which the budget does not pay for -- unless a piece has to be gathered in parts,
    // when the library holds each part's pixels itself.
    const auto cost = ReadCost::Of(descriptor, geometry, flag_geometry, options, PixelsHeld::by_caller, decode_threads);
    auto affordable = std::max<std::uint64_t>(1, cost.ChunksPerRead(PixelsHeld::by_caller));
    // TensorStore decodes no more chunks at once than the context has decode threads, so a read holds
    // what that many chunks hold however many it asks for. One the budget affords that many of has
    // nothing to gain from being cut -- a plane of 4 MiB chunks read whole held 80 MiB beside its
    // destination, and cut into pieces it ran half as fast again -- unless a caller is watching it,
    // and then it is cut so that there is a piece to report.
    if (!reports_progress && affordable >= std::max<std::size_t>(1, decode_threads)) {
        affordable = std::numeric_limits<std::uint64_t>::max();
    }
    const auto gathered = std::max<std::uint64_t>(1, cost.ChunksPerRead(PixelsHeld::by_library));

    // Every axis faster than the cut is whole in every piece, so one index of the cut axis is worth
    // their product of the destination -- and a piece of indices [begin, end) fills that many times
    // [begin, end) of it, which is what keeps the finished part a prefix.
    std::uint64_t elements_per_index = 1;
    for (std::size_t i = 0; i < *axis; ++i) {
        elements_per_index *= request.axes.at(i).count;
    }

    // The other axes already contribute whatever they span, so a plane read needs far fewer rows per
    // piece than a single-pixel column needs channels.
    const auto across = ChunksOf(geometry, request) / ChunksAlong(geometry, request, *axis);
    const auto indices = IndicesFor(geometry, request, *axis, UnitsAffordable(affordable, across));

    std::vector<Piece> pieces;
    CutAlong(geometry, request, *axis, indices, [&](ReadRequest run, std::uint64_t begin) {
        Piece piece{std::move(run), begin * elements_per_index, {}};
        // The least a piece can be is one chunk along the cut, and the whole of every other axis. When
        // that is still more than a read affords, it is read in parts that are not runs of the
        // destination, each gathered by the library and put in place.
        if (ChunksOf(geometry, piece.request) > affordable) {
            Segment(geometry, piece.request, gathered, piece.segments);
        }
        pieces.push_back(std::move(piece));
    });
    return pieces;
}

ReadRequest OneElementPerChunk(const ChunkGeometry& geometry, const ReadRequest& request) {
    ReadRequest sample = request;
    for (std::size_t axis = 0; axis < sample.axes.size(); ++axis) {
        const std::uint64_t chunk = axis < geometry.chunk_shape.size() ? geometry.chunk_shape.at(axis) : 0;
        auto& range = sample.axes.at(axis);
        if (chunk == 0 || range.stride >= chunk) {
            continue;
        }
        const std::uint64_t first = range.start / chunk;
        const std::uint64_t last = (range.start + ((range.count - 1) * range.stride)) / chunk;
        range = Range{first * chunk, last - first + 1, chunk};
    }
    return sample;
}

}  // namespace carta::zarr::internal
