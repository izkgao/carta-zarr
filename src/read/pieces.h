/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
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
 */
struct Piece {
    ReadRequest request;
    std::uint64_t first_element = 0;
};

/**
 * Cut a read into pieces.
 *
 * Pure: it reaches no further than the descriptor, the geometry and what the caller asked for, so
 * the strategy is checkable without a store, a transport or a directory tree.
 *
 * A read is cut when there is a reason to cut it, and either reason is enough on its own. Somebody
 * to report progress to is one -- `watching` says whether there is, which is the whole of what this
 * ever asked about the callback. A stated read budget is the other: it says how much the read may
 * hold at once, and splitting to fit is a better answer than refusing to read at all. A read with
 * neither reason, or with no axis selecting more than one element, is one piece covering everything,
 * so that the loop reading it is the same loop either way.
 *
 * Where to cut, how much one piece may cover, and where its end is rounded out to a chunk boundary
 * all happen here. They used to be handed to the one caller as a plan to carry out -- the cut axis,
 * the unit count, the elements per unit, the chunk extent -- and the caller did the arithmetic,
 * rewrote each piece's range and worked out where it landed. Whether the flag is decoded beside the
 * pixels, which the sizing counts, is asked of AppliesPixelMask rather than of the caller.
 *
 * The request has been checked against the descriptor already: it is a selection of this image.
 */
std::vector<Piece> PlanPieces(const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                              const ReadRequest& request, const ReadOptions& options, bool watching);

/**
 * Read a densely packed float32 result, one piece at a time.
 *
 * Everything an ordinary read does apart from being reached through a handle: it validates the
 * request against the descriptor, plans the pieces, reads each one's flag and pixels in that order,
 * folds the flag in, and reports progress. The caller supplies the source and translates whatever
 * comes back; it does not need to know that any of this happened.
 *
 * It takes the source, the descriptor and the geometry, which are all it uses, rather than the
 * ReducibleImage a reduction is handed: that also carries an axis map and a pool, and a read has no
 * use for either.
 *
 * The flag is read before the pixels, which is the opposite of what a pass does and is the reason
 * ADR 0005 gives for this staying outside one: the destination is the caller's, so a mask that
 * cannot be read must not leave a piece of it updated.
 *
 * Reports buffer_too_small when the destination cannot hold the selection, or when a piece's flag
 * buffer exceeds a ceiling that no further splitting gets under.
 */
Result<std::size_t> ReadInPieces(const PixelSource& source, const ImageDescriptor& descriptor,
                                 const ChunkGeometry& geometry, const ReadRequest& request,
                                 BufferView<float> destination, const ReadOptions& options,
                                 const ProgressCallback& progress);

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
 * A read of OneElementPerChunk(request) through ReadInPieces, so it is checked, cancelled and masked
 * exactly as a read is: with the mask applied the flag's chunks are decoded too, which a read that
 * follows will want.
 */
Result<std::uint64_t> PrefetchChunks(const PixelSource& source, const ImageDescriptor& descriptor,
                                     const ChunkGeometry& geometry, const ReadRequest& request,
                                     const ReadOptions& options);


}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_READ_PIECES_H_
