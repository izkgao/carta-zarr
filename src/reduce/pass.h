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
#include <cstdint>
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
    // Where this slab starts in the channel range this RunPass was given, which is what a visitor
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
 * Walks one spatial footprint along the spectrum, a slab at a time.
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
 * `reads_done` and `chunks_done` are the caller's because their spans are: a reduction resets them
 * per emitted block, a whole-plane pass keeps them for the call.
 *
 * `visit` is a template parameter and must not become a `std::function`: the per-pixel loop inlines
 * through it, and 54731c1 measured a quarter of a reduction riding on that. ADR 0005. `report` is
 * one call per slab, which is the same footing `PixelSource` stands on.
 *
 * Holds the buffers, so a walk allocates once rather than once per footprint.
 */
class SlabWalk {
public:
    SlabWalk(const PixelSource& source, const PassPlan& plan, const ReadOptions& options)
        : _source(source), _plan(plan), _options(options) {}

    SlabWalk(const SlabWalk&) = delete;
    SlabWalk& operator=(const SlabWalk&) = delete;

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

/**
 * Visit every plane of one channel range, a band of chunk rows at a time -- and a band in pieces
 * along its rows when one row is wider than a read may decode.
 *
 * `before_read` runs before each read after the first of the range, which is where a caller reports
 * what it has or decides to stop. `visit` receives one `Slab`.
 *
 * Both are template parameters and neither may become a `std::function`: the per-pixel loop inlines
 * through `visit`, and 54731c1 measured a quarter of a reduction riding on that. ADR 0005.
 */
template <typename BeforeRead, typename Visit>
Result<void> RunPass(const PixelSource& source, const PassPlan& plan, const ReadOptions& options,
                     SelectionChannel begin, SelectionChannel end, std::uint64_t& chunks_done,
                     BeforeRead&& before_read, Visit&& visit) {
    SlabWalk walk(source, plan, options);
    std::uint64_t reads_done = 0;

    const std::uint64_t row_chunks = std::max<std::uint64_t>(1, ((plan.u_length - 1) / plan.chunk_u) + 1);
    for (std::uint64_t v_begin = 0; v_begin < plan.v_length;) {
        const std::uint64_t v_end = std::min(plan.v_length, v_begin + (plan.band_rows * plan.chunk_v));
        const std::uint64_t band_rows = (((v_end - v_begin) - 1) / plan.chunk_v) + 1;
        SlabFootprint band;
        SampledRange(v_begin, v_end, plan.sample, band.v_start, band.v_count);
        if (band.v_count == 0) {
            chunks_done += row_chunks * band_rows * plan.ChunksTouched(begin, end);
            v_begin = v_end;
            continue;
        }
        band.u_stride = plan.sample;
        band.v_stride = plan.sample;

        // A chunk row wider than a read is read in pieces along it, as Occupancy::Footprints reads a
        // run: band_rows floors at one, so without this the smallest read is a whole chunk row,
        // however many budgets wide that is.
        const std::uint64_t segment_chunks = plan.UnitsAffordable(band_rows);
        for (std::uint64_t first = 0; first < row_chunks;) {
            const std::uint64_t width = std::min(segment_chunks, row_chunks - first);
            SlabFootprint segment = band;
            SampledRange(first * plan.chunk_u, std::min(plan.u_length, (first + width) * plan.chunk_u), plan.sample,
                         segment.u_start, segment.u_count);
            segment.chunks = width * band_rows;
            first += width;
            if (segment.u_count == 0) {
                chunks_done += segment.chunks * plan.ChunksTouched(begin, end);
                continue;
            }
            if (auto walked = walk.Over(segment, begin, end, reads_done, chunks_done, before_read, visit);
                !walked) {
                return walked.error();
            }
        }
        v_begin = v_end;
    }
    return {};
}
}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_PASS_H_
