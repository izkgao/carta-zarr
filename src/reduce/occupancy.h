/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_REDUCE_OCCUPANCY_H_
#define CARTA_ZARR_SRC_REDUCE_OCCUPANCY_H_

#include "carta-zarr/descriptor.h"
#include "carta-zarr/reduce.h"
#include "carta-zarr/result.h"

#include "reduce/footprint.h"
#include "reduce/region_runs.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace carta::zarr::internal {

class Occupancy;

// One region placed on the axes the walk reads in: u is the spatial axis the store varies fastest
// and v is the other, so that a plane arrives with u contiguous and never has to be transposed on
// the way in. A caller's x and y are mapped onto these once, at the top of the reduction.
//
// Placed rather than "as the walk sees it", which is what it used to be called: a pass never knows
// what a region is, so this is what a region looks like after the placement rather than something
// the walk holds.
class PlacedRegion {
public:
    // The rows of this region inside the box [u0, u1) x [v0, v1), half-open. None when the region
    // misses the box along either axis, so that a caller adding up a row at a time adds nothing for
    // a region that has no pixels there.
    struct Rows {
        std::uint64_t first = 0;
        std::uint64_t last = 0;
    };
    Rows RowsWithin(std::uint64_t u0, std::uint64_t u1, std::uint64_t v0, std::uint64_t v1) const noexcept {
        const std::uint64_t first_u = std::max(u0, u_start);
        const std::uint64_t last_u = std::min(u1, u_start + u_size);
        const std::uint64_t first_v = std::max(v0, v_start);
        const std::uint64_t last_v = std::min(v1, v_start + v_size);
        if (first_u >= last_u || first_v >= last_v) {
            return {};
        }
        return {first_v, last_v};
    }

    /**
     * Every span of row `y` this region selects inside [u0, u1), in order, as
     * `span(first, last, mask, mask_step)` over the pixels [first, last).
     *
     * `mask` is null when every pixel of the span is selected -- a run, or a region that is its whole
     * box -- and otherwise points at the raster's byte for `first`, the next pixel's `mask_step` bytes
     * on. Which of the three a region is, how its runs are encoded and which way its raster's rows
     * lie are all said here and nowhere else.
     *
     * `y` must be one of the region's rows. `span` is a template parameter and must not become a
     * std::function: a reduction's per-pixel loop is inside it. ADR 0005.
     */
    template <typename Span>
    void ForEachSpan(std::uint64_t y, std::uint64_t u0, std::uint64_t u1, Span&& span) const {
        const std::uint64_t first = std::max(u0, u_start);
        const std::uint64_t last = std::min(u1, u_start + u_size);
        if (first >= last) {
            return;
        }
        if (runs != nullptr) {
            const auto row = static_cast<std::size_t>(y - v_start);
            for (auto k = run_offsets[row]; k < run_offsets[row + 1]; ++k) {
                const std::uint64_t run_begin = u_start + runs[2 * k];
                if (run_begin >= last) {
                    break;  // runs ascend, so the rest are past the span
                }
                const std::uint64_t run_end = u_start + runs[(2 * k) + 1];
                const std::uint64_t from = std::max(first, run_begin);
                const std::uint64_t to = std::min(last, run_end);
                if (from < to) {
                    span(from, to, static_cast<const std::uint8_t*>(nullptr), std::uint64_t{1});
                }
            }
            return;
        }
        const std::uint8_t* selected =
            mask == nullptr ? nullptr : mask + ((y - v_start) * mask_v_stride) + ((first - u_start) * mask_u_stride);
        span(first, last, selected, mask == nullptr ? std::uint64_t{1} : mask_u_stride);
    }

private:
    // Occupancy places a region and reads these to bucket it; nothing else does. A reduction asks
    // RowsWithin and ForEachSpan, so the encoding below can change without it noticing.
    friend class Occupancy;

    std::uint64_t u_start = 0;
    std::uint64_t v_start = 0;
    std::uint64_t u_size = 0;
    std::uint64_t v_size = 0;
    // Steps through the raster for one pixel of u and of v. One of them is 1; which one depends on
    // whether the caller's rows run along u or across it.
    const std::uint8_t* mask = nullptr;
    std::uint64_t mask_u_stride = 1;
    std::uint64_t mask_v_stride = 1;
    // Runs along u, indexed by v, made by Occupancy::Of from the raster and owned by it. Null for a
    // region that is its whole box, and for a raster too fragmented to be worth them, which is then
    // read through `mask`.
    const std::uint32_t* runs = nullptr;
    const std::uint64_t* run_offsets = nullptr;
};

// A maximal run of consecutive chunk columns that at least one region touches, in bounding-box
// grid coordinates.
struct ColumnRun {
    std::uint64_t first = 0;
    std::uint64_t last = 0;  // inclusive

    bool operator==(const ColumnRun& other) const {
        return first == other.first && last == other.last;
    }
};

// The regions touching one chunk, as a view into the index that holds them. Valid for as long as
// the Occupancy is.
//
// A view rather than a pair of iterators because the accumulation reads one per chunk cell and
// wants to write a range-for; the same shape BufferView carries, for the same reason.
struct RegionRefs {
    const std::uint32_t* data = nullptr;
    std::size_t size = 0;

    const std::uint32_t* begin() const noexcept {
        return data;
    }
    const std::uint32_t* end() const noexcept {
        return data + size;
    }
};

/**
 * One footprint of a reduction: what the walk reads, and the chunks it spans.
 *
 * The chunks are what the accumulation looks regions up by, and the slab is what the walk is handed.
 * They describe one rectangle, and used to be written out twice beside each other -- a SlabFootprint
 * for the walk and a second struct of the same bounds for the accumulation -- because the walk's type
 * has no use for chunk coordinates and the accumulation could not do without them.
 */
struct OccupiedFootprint {
    SlabFootprint slab;
    // The chunk cells the footprint spans, half-open, in the chunk grid's own coordinates.
    std::uint64_t chunk_cu_begin = 0;
    std::uint64_t chunk_cu_end = 0;
    std::uint64_t chunk_cv_begin = 0;
    std::uint64_t chunk_cv_end = 0;
};

/**
 * Which chunks a set of regions occupies, and which of them touch each one.
 *
 * Occupies rather than covers: the bounding box of a thin cut laid along the diagonal is the whole
 * image, while the cut touches one chunk per row. Bucketing by the box would read every chunk to
 * reach the band, so the region's own mask decides instead -- one pass over bytes the caller already
 * holds, against the chunks they would otherwise stand for.
 *
 * Built once per reduction and read for the whole of it. Without it the walk is chunks x regions
 * intersection tests -- 6.5 x 10^8 for a WSU diagonal -- which would put the region count back on
 * the critical path the spectral reduction exists to clear.
 *
 * It takes the caller's regions in the caller's own x and y, because placing them onto the walk's
 * axes is part of the same question: which axis the store varies fastest is what decides both where
 * a region lands and which way the runs made from its raster lie.
 *
 * Takes chunk_u, chunk_v and the fastest spatial axis rather than a PassPlan, because those are
 * what the question is about and what a test can stand up with nothing linked behind it -- ADR 0006,
 * and the same reasoning CheckedPlanes gives for taking a descriptor rather than a ReducibleImage.
 */
class Occupancy {
public:
    // Reports invalid_argument for a region set touching more chunks than one reduction can index.
    //
    // A region's raster is turned into runs here, along u, so that the walk
    // takes the unmasked loop for it and its chunks are found from the runs; see region_runs.h. That
    // is one pass over the raster per call, about 2 ms for a 7763x4742 bounding box.
    //
    // Assumes at least one region: an empty set is refused a step earlier, where the rest of the
    // request is checked.
    static Result<Occupancy> Of(BufferView<const RegionMask> regions, std::uint64_t chunk_u, std::uint64_t chunk_v,
                                AxisRole fastest_spatial_axis, const std::string& node);

    // The caller's regions on the walk's axes, in the order they were given. The accumulation
    // indexes this by what RegionsTouching hands back.
    const std::vector<PlacedRegion>& regions() const noexcept {
        return _regions;
    }

    // The chunks one spectral layer of the whole region set occupies. Zero when the regions select
    // nothing at all, which a mask of zeroes does.
    //
    // The same number as the chunks of every footprint added up, which is how a pass over them counts
    // it: occupancy_test holds the two equal.
    std::uint64_t LayerChunks() const noexcept {
        return _layer_chunks;
    }

    /**
     * What a reduction reads: the occupied chunk runs, cut into footprints one read can decode.
     *
     * Rows below one that repeat its runs exactly are read with it, so that a solid rectangle becomes
     * a few large requests while a diagonal stays one chunk per row; and a run wider than a read is
     * cut into pieces a read can hold. Each footprint is clamped to the regions' own pixels, and counts
     * the chunks it occupies in one spectral layer -- which is what the walk sizes its slabs by and
     * counts its progress in.
     *
     * Formed here rather than by the reduction because every step of it is a question about this
     * index: which rows repeat, how wide a run is, where the bounding box ends. Asked from outside,
     * that took eight accessors onto the representation and a second copy of the footprint's bounds.
     *
     * Takes the budget as the one number it needs -- how many chunks a read may decode, zero when a
     * single chunk is more than the budget -- rather than a PassPlan, for the reason Of takes a chunk
     * shape rather than one. It does not depend on which channels are being read, so a reduction
     * forms these once and walks them for every block it emits.
     */
    std::vector<OccupiedFootprint> Footprints(std::uint64_t chunks_per_read) const;

    // The regions touching one chunk. Read once per chunk cell by the accumulation, so it is inline
    // and does nothing but two lookups.
    RegionRefs RegionsTouching(std::uint64_t chunk_cu, std::uint64_t chunk_cv) const {
        const auto cell = Cell(chunk_cu, chunk_cv);
        const auto first = _offsets.at(cell);
        const auto last = _offsets.at(cell + 1);
        return RegionRefs{_entries.data() + first, static_cast<std::size_t>(last - first)};
    }

    // The compressed-row index itself. Exposed because it is what Of promises -- the incidences of
    // one chunk are contiguous and in region order, which is what the counting sort is for -- and a
    // test has nothing else to say that against.
    const std::vector<std::uint64_t>& offsets() const noexcept {
        return _offsets;
    }
    const std::vector<std::uint32_t>& entries() const noexcept {
        return _entries;
    }

    // The regions point into the runs this made, so it moves and is never copied.
    Occupancy(Occupancy&&) noexcept = default;
    Occupancy& operator=(Occupancy&&) noexcept = default;
    Occupancy(const Occupancy&) = delete;
    Occupancy& operator=(const Occupancy&) = delete;
    ~Occupancy() = default;

private:
    Occupancy() = default;

    std::size_t Cell(std::uint64_t chunk_cu, std::uint64_t chunk_cv) const noexcept {
        return static_cast<std::size_t>(((chunk_cv - _chunk_cv0) * _columns) + (chunk_cu - _chunk_cu0));
    }

    std::vector<PlacedRegion> _regions;
    // The runs made from rasters, which PlacedRegion points into. Moving the vector moves each one's
    // storage along with it, so the pointers survive a move of the Occupancy.
    std::vector<RegionRuns> _made_runs;
    // The chunk shape the grid is cut by, kept because a footprint is clamped in pixels.
    std::uint64_t _chunk_u = 1;
    std::uint64_t _chunk_v = 1;
    // The union bounding box of every region, in pixels of the walk's own axes, half-open, and the
    // chunk grid covering it.
    std::uint64_t _u0 = 0;
    std::uint64_t _v0 = 0;
    std::uint64_t _u1 = 0;
    std::uint64_t _v1 = 0;
    std::uint64_t _chunk_cu0 = 0;
    std::uint64_t _chunk_cv0 = 0;
    std::uint64_t _columns = 0;
    std::uint64_t _rows = 0;
    std::vector<std::uint64_t> _offsets;
    std::vector<std::uint32_t> _entries;
    // The occupied runs of each chunk row, which is what footprints are cut from instead of the
    // bounding box.
    std::vector<std::vector<ColumnRun>> _runs_per_row;
    std::uint64_t _layer_chunks = 0;
};

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_OCCUPANCY_H_
