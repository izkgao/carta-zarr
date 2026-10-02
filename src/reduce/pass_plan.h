/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_REDUCE_PASS_PLAN_H_
#define CARTA_ZARR_SRC_REDUCE_PASS_PLAN_H_

// What a pass decides before it reads anything, apart from the walk that carries it out.
//
// The plan is pure and the walk is not, and they were one header: anything that needed to know how a
// pass is cut -- the block emitter, above all -- took the slab reader, the walk's templates and the
// pixel source seam along with it. The walk is in pass.h, which includes this.

#include "carta-zarr/read.h"

#include "axis_map.h"
#include "chunk_blocks.h"
#include "reduce/plane_selection.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace carta::zarr::internal {

/**
 * An index into a run of channels, tagged by which run.
 *
 * Two indices into different runs do not convert to each other, and that is the whole of why this
 * exists. e97066f asked the slab reader for an index into the block being filled, where the reader
 * takes an index into the pass's whole spectral selection; every block after the first re-read the
 * first block's channels, and the totals hid it -- each block still read the right number of pixels
 * and produced a full count. Both numbers were std::uint64_t, so nothing could have said otherwise.
 *
 * The same fact was documented in three places instead (here, at SlabRequest, and at the block
 * emitter that Pass has since absorbed).
 * Three comments describing one trap is the sign the type should be carrying it.
 *
 * An index plus a count is an index into the same run; two indices into one run are a count apart.
 * Writing the wrong one on purpose is still possible -- `SelectionChannel{relative}` compiles -- but
 * it can no longer happen by assignment, which is how it happened.
 */
template <typename Tag>
struct ChannelIndex {
    std::uint64_t index = 0;

    friend bool operator<(ChannelIndex a, ChannelIndex b) noexcept { return a.index < b.index; }
    friend bool operator==(ChannelIndex a, ChannelIndex b) noexcept { return a.index == b.index; }
    friend ChannelIndex operator+(ChannelIndex a, std::uint64_t channels) noexcept {
        return ChannelIndex{a.index + channels};
    }
    friend std::uint64_t operator-(ChannelIndex a, ChannelIndex b) noexcept { return a.index - b.index; }
};

// An index into the pass's own spectral selection: channel `index` of the image is
// `planes.spectral.start + index * planes.spectral.stride`.
using SelectionChannel = ChannelIndex<struct SelectionChannelTag>;

// An index into the channel range one walk of a pass covers -- a block, or the whole run -- which is
// what a visitor accumulating into a block of its own indexes by.
using BlockChannel = ChannelIndex<struct BlockChannelTag>;

/**
 * Everything a pass decides before a byte is read: which axis is contiguous, how wide a band is,
 * how deep a slab goes, and how much a read may decode.
 *
 * Pure, and separated from the pass for the same reason `PlanRowTasks` is separated from the pool:
 * it is the half with an answer worth checking, and checking it needs no store, no transport and no
 * fixture. The read strategy has moved more than once, and every time it moved it moved in three
 * places at once.
 */
class PassPlan;

PassPlan PlanPass(const ImageDescriptor& descriptor, const ChunkGeometry& geometry, const AxisMap& map,
                  const CheckedPlanes& planes, std::uint64_t sample, const ReadOptions& options);

class PassPlan {
public:
    const ImageDescriptor* descriptor = nullptr;
    AxisMap map;
    // u is the spatial axis the store varies fastest; v is the other one.
    std::size_t axis_u = 0;
    std::size_t axis_v = 0;
    std::uint64_t u_length = 0;
    std::uint64_t v_length = 0;
    std::uint64_t chunk_u = 1;
    std::uint64_t chunk_v = 1;
    // What one chunk costs to decode, counting the flag beside it when this read applies the mask,
    // and how much of that a single read may hold. Both are answers rather than steps towards one:
    // ADR 0005 turns on the first, and the second is the caller's own ceiling when it stated one.
    std::uint64_t chunk_bytes = 1;
    std::size_t slab_budget_bytes = 0;
    std::uint64_t band_rows = 1;
    std::uint64_t layer_chunks = 1;
    // The planes this pass is over, already checked against the descriptor above.
    PlaneSelection planes;
    // Take every nth pixel along both spatial axes. This does not reduce the chunks a read decodes
    // -- a chunk comes back whole however few of its pixels are wanted -- so it pays only when it
    // steps over chunks entirely.
    std::uint64_t sample = 1;
    bool apply_mask = false;

    // Whether u is the image's y rather than its x -- so a caller holding coordinates in the image's
    // own axes knows whether to swap them into the walk's.
    //
    // Asked rather than re-derived: which spatial axis a store varies fastest is one rule, the plan
    // applies it to pick axis_u, and a reduction that worked it out again from the geometry would be
    // a second place for it to be got wrong.
    bool SwapsSpatial() const noexcept { return axis_u == map.y; }

    // How many chunks along the spectrum the selected channels [begin, end) touch.
    //
    // A walk counts its progress in chunks, and the spectral axis is the one where a selection's
    // stride makes that not simply a division. It is asked of the run rather than of its length: a
    // run that starts part-way into one chunk and ends part-way into another touches a chunk more
    // than its length divides into. The walk's slabs end on chunk boundaries and each counts what it
    // touched, so a total taken from the length alone came up short, and progress went past one with
    // a read still to come.
    std::uint64_t ChunksTouched(SelectionChannel begin, SelectionChannel end) const {
        return ::carta::zarr::internal::ChunksTouched(planes.spectral.start + (begin.index * planes.spectral.stride),
                                                      end - begin, planes.spectral.stride, _chunk_depth);
    }

    // The chunks a walk over the channels [begin, end) decodes, when each spectral layer of what it
    // walks occupies `layer_chunks`: what its progress is a fraction of. Never zero, so it can be
    // divided by -- a selection holds at least one channel, and one that occupies nothing still
    // reports against a whole.
    //
    // A block's completeness and a cube histogram's progress are both this number, and each used to
    // multiply it out for itself.
    std::uint64_t ChunksCovering(std::uint64_t layer_chunks, SelectionChannel begin, SelectionChannel end) const {
        return std::max<std::uint64_t>(1, layer_chunks * ChunksTouched(begin, end));
    }

    // How many chunks one read may decode: the budget in the units everything else here counts in.
    // Zero when the budget is smaller than a single chunk, which UnitsAffordable floors at one.
    //
    // Public because it is the whole of the budget a reader of the occupancy needs, and handing that
    // over as one number rather than a PassPlan keeps the occupancy testable with nothing linked
    // behind it -- ADR 0006.
    std::uint64_t ChunksPerRead() const noexcept { return slab_budget_bytes / std::max<std::uint64_t>(1, chunk_bytes); }

    // How many units of `chunks_per_unit` chunks one read's budget affords -- the rule in
    // chunk_blocks.h, asked with this plan's budget.
    //
    // A unit is at least one chunk. The one caller that can mean none -- a region set occupying
    // nothing -- says so itself, in EmitChannels; this used to answer it with the budget's byte
    // count standing in for a count of units.
    std::uint64_t UnitsAffordable(std::uint64_t chunks_per_unit) const {
        return ::carta::zarr::internal::UnitsAffordable(ChunksPerRead(), chunks_per_unit);
    }

    // How many channels one slab may hold when its spatial footprint occupies `footprint_chunks`
    // chunks of each spectral chunk it touches.
    std::uint64_t SlabChannels(std::uint64_t footprint_chunks) const {
        return std::max<std::uint64_t>(1, UnitsAffordable(footprint_chunks) * _least_channels);
    }

    // The end of a slab that begins at `begin` and would like to be `desired` channels long, moved
    // onto a chunk boundary so that no decode serves two slabs.
    SelectionChannel AlignedSlabEnd(SelectionChannel begin, std::uint64_t desired, SelectionChannel end) const {
        return SelectionChannel{AlignedBlockEnd(begin.index, desired, end.index, planes.spectral.start,
                                                planes.spectral.stride, _chunk_depth)};
    }

    // How many channels one emitted block may hold.
    //
    // The hint is the caller's, the budgets are the library's, and the chunk alignment is the
    // plan's; the smallest wins and the block reports what it used. Without a hint a block costs one
    // budget of decoded bytes -- the same invariant a piece of Read carries -- so it is free: the
    // block spends whatever the spatial walk left over. A small region leaves almost all of it and
    // the block spans many chunks along the spectrum; a region covering the image spends the budget
    // spatially and the block becomes the single chunk layer the walk is already reading.
    //
    // `layer_chunks` is the chunks one spectral layer of whatever the caller is walking occupies,
    // which is the plan's own for a whole plane and the region set's for a reduction.
    std::uint64_t EmitChannels(std::uint64_t layer_chunks, std::size_t bytes_per_channel, std::uint32_t hint) const;

private:
    friend PassPlan PlanPass(const ImageDescriptor& descriptor, const ChunkGeometry& geometry, const AxisMap& map,
                             const CheckedPlanes& planes, std::uint64_t sample, const ReadOptions& options);

    // Steps towards the answers above rather than answers themselves, and the two a caller used to
    // divide by itself: the chunk-count rule was written out in six places and the slab-sizing rule
    // in two. Nothing asserts either directly -- what a test has to say about them it says through
    // SlabChannels and ChunksTouched, which are the questions a caller actually asks.
    std::uint64_t _chunk_depth = 1;
    std::uint64_t _least_channels = 1;
};

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_PASS_PLAN_H_
