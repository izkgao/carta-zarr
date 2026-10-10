/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_ZARR_SRC_READ_PIECES_H_
#define CARTA_ZARR_SRC_READ_PIECES_H_

#include "carta-zarr/descriptor.h"
#include "carta-zarr/read.h"
#include "carta-zarr/result.h"

#include "pixel_source.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace carta::zarr::internal {

/**
 * One piece of an ordinary read. See CONTEXT.md for what a piece is.
 *
 * What to read -- the caller's request, narrowed along the axis the read is cut on -- and where in the
 * destination it lands. A piece fills the destination from `first_element` for as many elements as
 * its request selects, and the pieces of a read fill it end to end, so the finished part is always a
 * prefix.
 *
 * A piece is read straight into the destination unless even the least of them -- one chunk along the
 * cut, and the whole of every other axis -- holds more than a read affords. Then `segments` are the
 * parts it is read in, cut along the other axes, each gathered in a buffer of the library's and put in
 * place. A segment is not a run of the destination, which is why it is not a piece: progress is
 * reported when the whole piece is in, and the prefix holds.
 */
struct Piece {
    ReadRequest request;
    std::uint64_t first_element = 0;
    std::vector<ReadRequest> segments;
};

/**
 * Cut a read into pieces.
 *
 * Pure: it reaches no further than the descriptor, the geometry and what the caller asked for, so
 * the strategy is checkable without a store, a transport or a directory tree.
 *
 * Every read is cut to fit its budget -- the caller's, or the library's own when it states none -- so
 * that what a read holds is bounded whether or not anybody watches it. What it holds is the chunks
 * TensorStore decodes at once, which is no more than `decode_threads` of them: a read nobody watches
 * (`reports_progress` false) whose budget affords that many is not cut at all, while one that is
 * watched is cut to the chunks the budget affords, so that it has pieces to report. A read that fits,
 * or that has no axis selecting more than one element, is one piece covering everything, so that the
 * loop reading it is the same loop either way.
 *
 * Where to cut, how much one piece may cover, where its end is rounded out to a chunk boundary, and
 * the segments of a piece too large to read whole all happen here. Whether the flag is decoded beside
 * the pixels, which the sizing counts, is asked of AppliesPixelMask rather than of the caller; what it
 * costs is asked of the flag's own layout, `flag_geometry`, which need not be the pixels'.
 *
 * The request has been checked against the descriptor already: it is a selection of this image.
 */
std::vector<Piece> PlanPieces(const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                              const ChunkGeometry& flag_geometry, const ReadRequest& request,
                              const ReadOptions& options, std::size_t decode_threads, bool reports_progress);

/**
 * Read a densely packed float32 result, one piece at a time.
 *
 * Everything an ordinary read does apart from being reached through a handle: it validates the
 * request against the descriptor, plans the pieces, reads each one's flag and pixels in that order,
 * folds the flag in, and reports progress. The caller supplies the source and translates whatever
 * comes back; it does not need to know that any of this happened.
 *
 * It takes the source, the descriptor and the two geometries, which are all it uses, rather than the
 * ReducibleImage a reduction is handed: that also carries an axis map and a pool, and a read has no
 * use for either.
 *
 * The flag is read before the pixels, which is the opposite of what a pass does and is the reason
 * ADR 0005 gives for this staying outside one: the destination is the caller's, so a mask that
 * cannot be read must not leave a piece of it updated. A piece read in segments keeps that for each
 * segment; the segments of it already put in place stay there, past the finished prefix.
 *
 * Reports invalid_argument when the destination cannot hold the selection.
 */
Result<std::size_t> ReadInPieces(const PixelSource& source, const ImageDescriptor& descriptor,
                                 const ChunkGeometry& geometry, const ChunkGeometry& flag_geometry,
                                 const ReadRequest& request, BufferView<float> destination, const ReadOptions& options,
                                 std::size_t decode_threads, const ProgressCallback& progress);

/**
 * One selected element in every chunk that `request` touches, and no more.
 *
 * Reading a single element of a chunk decodes all of it, so this request decodes exactly the chunks
 * a read of `request` would, into whatever cache the read goes through, while asking for a
 * destination of one float a chunk rather than one a pixel. A plane of a 7763 x 4742 image in
 * 512 x 512 chunks is 160 of them, where reading the plane itself is 147 MB to allocate and fill.
 *
 * Along an axis the chunks a request touches are consecutive unless its stride is a chunk or more,
 * in which case every element it selects is in a chunk of its own already and the range is kept as
 * it is. Otherwise the range becomes the first element of each chunk from the one its start is in to
 * the one its last element is in -- an element of the chunk in either case, though not one the
 * request selected.
 *
 * Pure, and the request has been checked against the descriptor already.
 */
ReadRequest OneElementPerChunk(const ChunkGeometry& geometry, const ReadRequest& request);

/**
 * Decode the chunks a read of `request` would decode, keeping them wherever options.control's
 * cache pool says, and hand back nothing but how many there were.
 *
 * A read of OneElementPerChunk(request) through ReadInPieces, so it is checked and cancelled
 * exactly as a read is. With the mask applied the flag's chunks are decoded too, which a read that
 * follows will want, sampled by `flag_geometry` rather than by the pixels': a flag chunked finer
 * than its image, sampled where the pixels are, had half its chunks left to read from storage. The
 * count is of both.
 */
Result<std::uint64_t> PrefetchChunks(const PixelSource& source, const ImageDescriptor& descriptor,
                                     const ChunkGeometry& geometry, const ChunkGeometry& flag_geometry,
                                     const ReadRequest& request, const ReadOptions& options,
                                     std::size_t decode_threads);

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_READ_PIECES_H_
