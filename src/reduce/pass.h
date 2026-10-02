/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_REDUCE_PASS_H_
#define CARTA_ZARR_SRC_REDUCE_PASS_H_

// The walk a pass makes over an image, slab by slab, and the slab it hands a visitor. What it walks
// by -- the plan -- is in pass_plan.h.

#include "carta-zarr/read.h"
#include "carta-zarr/result.h"

#include "pixel_source.h"
#include "reduce/footprint.h"
#include "reduce/pass_plan.h"
#include "zarr/pixel_selection.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace carta::zarr::internal {

/**
 * One read of the pass, handed to the visitor.
 *
 * A pointer and three strides rather than a packed buffer, because the destination comes back in
 * the store's own order and packing it would be the transpose the pass exists to avoid.
 *
 * A read, not a plane: a visitor that splits the work across threads needs a piece big enough to
 * pay for the dispatch, and a plane of a few hundred thousand pixels is not one. A visitor that
 * wants planes loops over `channel_count` itself, which costs it nothing.
 */
struct Slab {
    // Where this slab starts in the channel range its walk was given, which is what a visitor
    // accumulating into a block of its own indexes by. Set by the walk, not by the reader: the
    // reader is given an absolute index and has nothing to measure a relative one against.
    BlockChannel first_channel;
    std::uint64_t channel_count = 0;
    const float* pixels = nullptr;
    std::uint64_t stride_u = 1;
    std::uint64_t stride_v = 1;
    std::uint64_t stride_z = 1;
    std::uint64_t u_count = 0;
    std::uint64_t v_count = 0;
};

// Samples of `stride` that fall in [begin, end), as a start and a count.
inline void SampledRange(std::uint64_t begin, std::uint64_t end, std::uint64_t stride, std::uint64_t& start,
                         std::uint64_t& count) {
    const std::uint64_t first = (begin + stride - 1) / stride;
    const std::uint64_t last = (end + stride - 1) / stride;
    start = first * stride;
    count = last > first ? last - first : 0;
}

/**
 * Walks one spatial footprint along the spectrum, a slab at a time, and the whole plane as bands of
 * them: what a Pass reads with, and nothing a reduction or a test reaches past the Pass for.
 *
 * Which footprints to visit is not the walk's. A whole-plane pass bands the plane itself; a
 * reduction's come from its occupancy, which cuts the chunk runs its regions occupy -- see
 * Occupancy::Footprints. Those two are genuinely different walks and stay that way: joining them
 * into one traversal would make the plane's bands a special case of a region set, for a caller that
 * has no regions. What they had in common was everything inside one footprint -- how deep a slab goes, where
 * it is cut so that no decode serves two slabs, when the caller is told what it has, where
 * cancellation is checked, and how progress is counted -- and that was written out twice, in full,
 * with the subtlety intact in both copies: `reads_done` guards the report so that a footprint taken
 * in a single read still reports once at the end rather than before it starts.
 *
 * `reads_done` and `chunks_done` are the Pass's, because their spans are: a block resets them, a
 * whole run keeps them. They used to be every reduction's, passed down by reference, and the rule
 * about them was written in a comment at each place that did.
 *
 * `visit` is a template parameter and must not become a `std::function`: the per-pixel loop inlines
 * through it, and 54731c1 measured a quarter of a reduction riding on that. ADR 0005. `report` is
 * one call per slab, which is the same footing `PixelSource` stands on.
 *
 * Holds the buffers, so a pass allocates once rather than once per footprint or per block.
 *
 * Its interface is protected, and a Pass is the one thing that derives from it: a class of its own
 * rather than a member of Pass because the read itself is out of line in pass_read.cc, and Pass is a
 * template.
 */
class SlabWalk {
public:
    SlabWalk(const SlabWalk&) = delete;
    SlabWalk& operator=(const SlabWalk&) = delete;

protected:
    SlabWalk(const PixelSource& source, const PassPlan& plan, const ReadOptions& options)
        : _source(source), _plan(plan), _options(options) {}
    ~SlabWalk() = default;

    template <typename Report, typename Visit>
    Result<void> Over(const SlabFootprint& footprint, SelectionChannel begin, SelectionChannel end,
                      std::uint64_t& reads_done, std::uint64_t& chunks_done, Report&& report, Visit&& visit) {
        const std::uint64_t slab_channels = _plan.SlabChannels(footprint.chunks);

        for (SelectionChannel slab_begin = begin; slab_begin < end;) {
            const SelectionChannel slab_end =
                _plan.AlignedSlabEnd(slab_begin, std::min(slab_channels, end - slab_begin), end);
            const std::uint64_t slab_length = slab_end - slab_begin;

            // What is in hand before spending another budget. A footprint that takes one read never
            // gets here, so a small one still reports once -- at the end, through its caller.
            if (reads_done > 0) {
                if (auto ready = report(chunks_done); !ready) {
                    return ready.error();
                }
            }
            ++reads_done;

            if (auto control = zarr::CheckReadControl(_options.control, _plan.descriptor->id); !control) {
                return control.error();
            }

            SlabRequest slab_request;
            slab_request.u_start = footprint.u_start;
            slab_request.u_count = footprint.u_count;
            slab_request.u_stride = footprint.u_stride;
            slab_request.v_start = footprint.v_start;
            slab_request.v_count = footprint.v_count;
            slab_request.v_stride = footprint.v_stride;
            slab_request.channel_index = slab_begin;
            slab_request.channel_count = slab_length;

            auto slab = ReadSlab(slab_request);
            if (!slab) {
                return slab.error();
            }
            // What was read is absolute; what the visitor indexes by is relative to this walk. The
            // subtraction is the only place the two meet, and it is the only place it can be: the
            // reader never sees `begin`, so it cannot compute this and does not try.
            slab.value().first_channel = BlockChannel{slab_begin - begin};

            visit(slab.value());

            chunks_done += footprint.chunks * _plan.ChunksTouched(slab_begin, slab_end);
            slab_begin = slab_end;
        }
        return {};
    }

    /**
     * Visit every plane of the channels [begin, end), a band of chunk rows at a time -- and a band in
     * pieces along its rows when one row is wider than a read may decode.
     *
     * A piece sampling steps over entirely is not read, but its chunks are counted as it is passed,
     * so that what `before_read` is told reaches the whole.
     */
    template <typename BeforeRead, typename Visit>
    Result<void> OverBands(SelectionChannel begin, SelectionChannel end, std::uint64_t& reads_done,
                           std::uint64_t& chunks_done, BeforeRead&& before_read, Visit&& visit) {
        const std::uint64_t row_chunks = std::max<std::uint64_t>(1, ((_plan.u_length - 1) / _plan.chunk_u) + 1);
        for (std::uint64_t v_begin = 0; v_begin < _plan.v_length;) {
            const std::uint64_t v_end = std::min(_plan.v_length, v_begin + (_plan.band_rows * _plan.chunk_v));
            const std::uint64_t band_rows = (((v_end - v_begin) - 1) / _plan.chunk_v) + 1;
            SlabFootprint band;
            SampledRange(v_begin, v_end, _plan.sample, band.v_start, band.v_count);
            if (band.v_count == 0) {
                chunks_done += row_chunks * band_rows * _plan.ChunksTouched(begin, end);
                v_begin = v_end;
                continue;
            }
            band.u_stride = _plan.sample;
            band.v_stride = _plan.sample;

            // A chunk row wider than a read is read in pieces along it, as Occupancy::Footprints
            // reads a run: band_rows floors at one, so without this the smallest read is a whole
            // chunk row, however many budgets wide that is.
            const std::uint64_t segment_chunks = _plan.UnitsAffordable(band_rows);
            for (std::uint64_t first = 0; first < row_chunks;) {
                const std::uint64_t width = std::min(segment_chunks, row_chunks - first);
                SlabFootprint segment = band;
                SampledRange(first * _plan.chunk_u, std::min(_plan.u_length, (first + width) * _plan.chunk_u),
                             _plan.sample, segment.u_start, segment.u_count);
                segment.chunks = width * band_rows;
                first += width;
                if (segment.u_count == 0) {
                    chunks_done += segment.chunks * _plan.ChunksTouched(begin, end);
                    continue;
                }
                if (auto walked = Over(segment, begin, end, reads_done, chunks_done, before_read, visit); !walked) {
                    return walked.error();
                }
            }
            v_begin = v_end;
        }
        return {};
    }

private:
    // What a slab is asked for and read into, and the read itself: the walk's own, and nothing a
    // caller of it names. They were declared beside it for anyone to use, and only it ever did.

    // One slab to read, in the pass's own axes.
    struct SlabRequest {
        std::uint64_t u_start = 0;
        std::uint64_t u_count = 0;
        std::uint64_t u_stride = 1;
        std::uint64_t v_start = 0;
        std::uint64_t v_count = 0;
        std::uint64_t v_stride = 1;
        // Which channels to read. A SelectionChannel rather than a plain number because Slab carries a
        // BlockChannel, and handing one where the other belongs is the mistake this pair of types exists
        // to refuse -- see ChannelIndex.
        SelectionChannel channel_index;
        std::uint64_t channel_count = 0;
    };

    // Reused across slabs, so that a pass allocates once rather than once per read.
    struct SlabBuffers {
        std::vector<float> pixels;
        std::vector<std::uint8_t> mask;
    };

    /**
     * Read one slab and hand back how to walk it.
     *
     * This is what every pass over a cube has in common, whatever order it visits chunks in: ask for
     * the stored dimensions reversed so the plane arrives untransposed, derive the strides of what
     * comes back, read the pixels, and -- when the image has a flag the caller did not decline -- read
     * that too and fold it into the pixels, because a flagged pixel and a NaN pixel mean the same thing
     * to everything downstream.
     *
     * The returned Slab points into this walk's buffers, so it is valid until the next read.
     */
    Result<Slab> ReadSlab(const SlabRequest& request);

    const PixelSource& _source;
    const PassPlan& _plan;
    const ReadOptions& _options;
    SlabBuffers _buffers;
};

// What a pass walks, spatially: the whole plane, in bands of chunk rows.
struct PlaneBands {};

// What a pass walks, spatially: the footprints a region set's occupancy cut, each read along the
// spectrum in turn. Any range whose elements carry the SlabFootprint they are read over as `.slab`;
// the element is handed back to the visitor beside each slab read over it, because a reduction
// accumulating regions needs to know which chunk cells the slab covers.
template <typename Footprints>
struct FootprintsOf {
    const Footprints* footprints;
};

/**
 * One pass over an image, from the first read to the last hand-over.
 *
 * A reduction says what the pass walks -- the plane, or a region set's footprints -- and what to do
 * with each slab, and the pass does everything between: where a block is cut, when what is in hand
 * is handed over, how far along that is, and when a caller saying no stops it. Made by
 * PassOverPlane and PassOverFootprints.
 *
 * Which layer progress is a fraction of is the shape's own and not a caller's to state. It is the
 * chunks one spectral layer of what is walked occupies -- the plane's, or the sum of the footprints'
 * -- and it is both what the emit budget is spent against and the unit progress is counted in. A
 * reduction that gave one for the other read the right pixels and reported a bar that lied, so the
 * pass works it out from what it was given to walk, and there is nothing left to give wrongly.
 *
 * The two walks stay two. Joining them would make the plane's bands a special case of a region set,
 * for a caller that has no regions; what they share is everything inside one footprint, which is
 * SlabWalk's.
 *
 * Every callback is a template parameter and none may become a std::function: the per-pixel loop
 * inlines through `visit`, and 54731c1 measured a quarter of a reduction riding on that. ADR 0005.
 */
template <typename Shape>
class Pass : private SlabWalk {
public:
    Pass(const Pass&) = delete;
    Pass& operator=(const Pass&) = delete;

    /**
     * Walk the whole selection, a block at a time, handing each over as it fills.
     *
     * `bytes_per_channel` and `hint` size the block, as PassPlan::EmitChannels describes.
     *
     * `reset(length)` prepares the accumulator for a block of `length` channels. `visit(slab)` -- or
     * `visit(footprint, slab)` over footprints -- accumulates one read, its `first_channel` counted
     * from the block's start.
     *
     * `hand_over(first_channel, length, complete, completeness)` fills the reduction's own block and
     * calls its sink, returning what the sink returned: once before every read of a block after its
     * first, with how much of the block's chunks are in it, and once more when the block is finished.
     * `first_channel` is a SelectionChannel, and turning it into the plain number a public block
     * carries is the one place the type is left behind.
     *
     * A sink returning false cancels, with the message the pass was made with.
     */
    template <typename Reset, typename Visit, typename HandOver>
    Result<void> InBlocks(std::size_t bytes_per_channel, std::uint32_t hint, Reset&& reset, Visit&& visit,
                          HandOver&& hand_over) {
        const SelectionChannel end_of_selection{_plan.planes.spectral.count};
        // A shape occupying no chunks -- a region set whose mask selects nothing -- is the one place
        // the two uses of the layer want opposite things. Spending the emit budget, zero is the honest
        // answer: there is nothing to read, so the whole selection is one block rather than pieces
        // sized for chunks nobody will decode. As the denominator of a part-filled block's
        // completeness it must never be zero.
        const std::uint64_t emit_channels = _plan.EmitChannels(_layer_chunks, bytes_per_channel, hint);
        const std::uint64_t layer = std::max<std::uint64_t>(1, _layer_chunks);

        for (SelectionChannel begin{}; begin < end_of_selection;) {
            const SelectionChannel end = _plan.AlignedSlabEnd(begin, emit_channels, end_of_selection);
            const std::uint64_t length = end - begin;
            reset(length);

            // What the completeness of a part-filled block is a fraction of, and what the walk has of
            // it. Both are the block's: they start again when a block does.
            const std::uint64_t chunks_total = _plan.ChunksCovering(layer, begin, end);
            std::uint64_t chunks_done = 0;
            std::uint64_t reads_done = 0;

            const auto deliver = [&](bool complete) -> Result<void> {
                const double completeness =
                    complete ? 1.0 : static_cast<double>(chunks_done) / static_cast<double>(chunks_total);
                if (!hand_over(begin, length, complete, completeness)) {
                    return Cancelled();
                }
                return {};
            };

            if (auto walked = Walk(
                    begin, end, reads_done, chunks_done, [&](std::uint64_t) -> Result<void> { return deliver(false); },
                    visit);
                !walked) {
                return walked.error();
            }
            if (auto handed = deliver(true); !handed) {
                return handed.error();
            }
            begin = end;
        }
        return {};
    }

    /**
     * Walk the whole selection as one run, telling `progress(fraction)` how far along it is: before
     * every read after the first, so a run that takes one read never calls it, and never once the
     * last read is done. `fraction` is the chunks read of the chunks the run covers. Returning false
     * cancels, with the message the pass was made with.
     *
     * `visit` is as InBlocks has it, its `first_channel` counted from the start of the selection.
     */
    template <typename Progress, typename Visit>
    Result<void> Whole(Progress&& progress, Visit&& visit) {
        const SelectionChannel end_of_selection{_plan.planes.spectral.count};
        const std::uint64_t chunks_total =
            _plan.ChunksCovering(std::max<std::uint64_t>(1, _layer_chunks), SelectionChannel{}, end_of_selection);
        std::uint64_t chunks_done = 0;
        std::uint64_t reads_done = 0;
        return Walk(
            SelectionChannel{}, end_of_selection, reads_done, chunks_done,
            [&](std::uint64_t done) -> Result<void> {
                if (!progress(static_cast<double>(done) / static_cast<double>(chunks_total))) {
                    return Cancelled();
                }
                return {};
            },
            visit);
    }

private:
    template <typename Footprints>
    friend Pass<FootprintsOf<Footprints>> PassOverFootprints(const PixelSource& source, const PassPlan& plan,
                                                             const ReadOptions& options, const Footprints& footprints,
                                                             std::string cancelled);
    friend Pass<PlaneBands> PassOverPlane(const PixelSource& source, const PassPlan& plan, const ReadOptions& options,
                                          std::string cancelled);

    Pass(const PixelSource& source, const PassPlan& plan, const ReadOptions& options, Shape shape,
         std::uint64_t layer_chunks, std::string cancelled)
        : SlabWalk(source, plan, options),
          _plan(plan),
          _shape(shape),
          _layer_chunks(layer_chunks),
          _cancelled(std::move(cancelled)) {}

    Error Cancelled() const { return Error{ErrorCode::cancelled, _cancelled, _plan.descriptor->id}; }

    // The channels [begin, end) of every footprint of the shape. The footprints of one walk share
    // `reads_done`, so a walk taken in a single read still reports once -- at the end, through its
    // caller -- however many footprints it is made of.
    template <typename Report, typename Visit>
    Result<void> Walk(SelectionChannel begin, SelectionChannel end, std::uint64_t& reads_done,
                      std::uint64_t& chunks_done, Report&& report, Visit& visit) {
        if constexpr (std::is_same_v<Shape, PlaneBands>) {
            return OverBands(begin, end, reads_done, chunks_done, report, visit);
        } else {
            for (const auto& footprint : *_shape.footprints) {
                if (auto walked = Over(footprint.slab, begin, end, reads_done, chunks_done, report,
                                       [&](const Slab& slab) { visit(footprint, slab); });
                    !walked) {
                    return walked.error();
                }
            }
            return {};
        }
    }

    const PassPlan& _plan;
    Shape _shape;
    // What one spectral layer of the shape occupies, unclamped: zero for a region set occupying
    // nothing. See InBlocks.
    std::uint64_t _layer_chunks;
    std::string _cancelled;
};

// A pass over the whole plane. Its layer is the plan's own. `cancelled` is what a caller's stop is
// reported as; ADR 0011 keeps it one code.
inline Pass<PlaneBands> PassOverPlane(const PixelSource& source, const PassPlan& plan, const ReadOptions& options,
                                      std::string cancelled) {
    return Pass<PlaneBands>(source, plan, options, PlaneBands{}, plan.layer_chunks, std::move(cancelled));
}

// A pass over a region set's footprints, which must outlive it. Its layer is what they occupy
// together, which is the region set's and not the plane's.
template <typename Footprints>
Pass<FootprintsOf<Footprints>> PassOverFootprints(const PixelSource& source, const PassPlan& plan,
                                                  const ReadOptions& options, const Footprints& footprints,
                                                  std::string cancelled) {
    std::uint64_t layer_chunks = 0;
    for (const auto& footprint : footprints) {
        layer_chunks += footprint.slab.chunks;
    }
    return Pass<FootprintsOf<Footprints>>(source, plan, options, FootprintsOf<Footprints>{&footprints}, layer_chunks,
                                          std::move(cancelled));
}

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_PASS_H_
