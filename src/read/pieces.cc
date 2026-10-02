/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
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

// How many elements of the split axis one piece should cover, so that the piece pulls roughly the
// budgeted amount of decompressed chunk data through. The other axes already contribute whatever
// they span, so a plane read needs far fewer rows per piece than a single-pixel column needs
// channels -- and an image with very large chunks gets pieces of one chunk rather than pieces it
// could never afford.
std::uint64_t UnitsPerPiece(const ReadRequest& request, const ChunkGeometry& geometry, std::size_t axis,
                            const ReadCost& cost) {
    std::uint64_t other_chunks = 1;
    for (std::size_t i = 0; i < request.axes.size(); ++i) {
        if (i == axis) {
            continue;
        }
        const auto chunk = i < geometry.chunk_shape.size() ? geometry.chunk_shape.at(i) : 0;
        const auto& range = request.axes.at(i);
        other_chunks *= ChunksSpanned(range.start, range.count, range.stride, chunk);
    }
    const auto row_bytes = cost.chunk_bytes * other_chunks;
    // At least one chunk: a piece smaller than that would decode the same chunk twice.
    const auto chunks = std::max<std::uint64_t>(1, cost.budget_bytes / std::max<std::uint64_t>(1, row_bytes));
    const auto chunk = axis < geometry.chunk_shape.size() ? geometry.chunk_shape.at(axis) : 0;
    const auto stride = std::max<std::uint64_t>(1, request.axes.at(axis).stride);
    // AlignedBlockEnd rounds this out to a whole chunk, so a low estimate costs nothing.
    return std::max<std::uint64_t>(1, (chunks * std::max<std::uint64_t>(1, chunk)) / stride);
}

}  // namespace

std::vector<Piece> PlanPieces(const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                              const ReadRequest& request, const ReadOptions& options, bool watching) {
    const auto axis = SlowestSelectedAxis(request);
    if (!axis || (!watching && options.read_budget_bytes == 0)) {
        return {Piece{request, 0}};
    }

    // The flag is decoded beside the pixels when this read will apply it, so both halves of the
    // sizing count it: the budget the library chooses for itself, and the per-row cost that budget
    // is divided by. ReadCost counts it in both, from the one chunk cost, which is what keeps a piece
    // from being sized against a cost the read does not have.
    const auto units_per_piece = UnitsPerPiece(request, geometry, *axis, ReadCost::Of(descriptor, geometry, options));
    const auto chunk = *axis < geometry.chunk_shape.size() ? geometry.chunk_shape.at(*axis) : 0;

    // Every axis faster than the cut is whole in every piece, so one unit of the cut axis is worth
    // their product of the destination -- and a piece of units [begin, end) fills that many times
    // [begin, end) of it, which is what keeps the finished part a prefix.
    std::uint64_t elements_per_unit = 1;
    for (std::size_t i = 0; i < *axis; ++i) {
        elements_per_unit *= request.axes.at(i).count;
    }

    const auto& cut = request.axes.at(*axis);
    std::vector<Piece> pieces;
    for (std::uint64_t begin = 0; begin < cut.count;) {
        // Rounded out to a chunk boundary, so that no chunk is decoded by two pieces.
        const std::uint64_t end = AlignedBlockEnd(begin, units_per_piece, cut.count, cut.start, cut.stride, chunk);
        Piece piece{request, begin * elements_per_unit};
        auto& range = piece.request.axes.at(*axis);
        range.start = cut.start + (begin * cut.stride);
        range.count = end - begin;
        pieces.push_back(std::move(piece));
        begin = end;
    }
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
