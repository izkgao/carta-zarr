/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Building an Occupancy. Split from occupancy.h the way pass_read.cc is split from pass.h: what the
// accumulation reads per chunk cell has to be inline, and this is paid once per reduction.

#include "occupancy.h"

#include "chunk_blocks.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace carta::zarr::internal {
namespace {

// A region is bucketed once per chunk its bounding box touches. This bound turns a request whose
// regions each cover the whole image -- legal, just not what this API is for -- into an error
// rather than an allocation nothing can serve.
//
// Not in reduce/tuning.h, which holds the numbers a caller neither sets nor needs: exceeding this
// one is refused. Not in carta-zarr/reduce.h either, beside the limits a caller does respect,
// because it counts incidences -- regions times the chunks each occupies -- and a caller holding no
// chunk shape cannot tell in advance whether it is near it.
constexpr std::size_t kMaxChunkIncidences = 1u << 26;

}  // namespace

Result<Occupancy> Occupancy::Of(BufferView<const RegionMask> regions, std::uint64_t chunk_u, std::uint64_t chunk_v,
                                AxisRole fastest_spatial_axis, const std::string& node) {
    const std::size_t region_count = regions.size;
    // The walk follows the store. Of the two spatial axes the one written last varies fastest, so
    // asking for it first is what keeps a plane from being transposed on its way into the
    // destination; everything below is in terms of that axis (u) and the other one (v).
    const bool swap_spatial = fastest_spatial_axis == AxisRole::spatial_y;

    Occupancy occupancy;
    occupancy._chunk_u = chunk_u;
    occupancy._chunk_v = chunk_v;
    occupancy._regions.reserve(region_count);
    for (std::size_t i = 0; i < region_count; ++i) {
        const auto& given = regions.data[i];
        // The raster keeps whatever order the caller wrote it in; only the steps through it change.
        PlacedRegion region;
        region.u_start = swap_spatial ? given.y_start : given.x_start;
        region.v_start = swap_spatial ? given.x_start : given.y_start;
        region.u_size = swap_spatial ? given.height : given.width;
        region.v_size = swap_spatial ? given.width : given.height;
        region.mask = given.mask.data;
        region.mask_u_stride = swap_spatial ? given.width : 1;
        region.mask_v_stride = swap_spatial ? 1 : given.width;
        // A raster is given runs, along u: which axis that is was the caller's to know once, and is
        // not any more. The raster stays beside them for a fragmented one, which comes back refused
        // and is read as the raster it is.
        if (given.mask.data != nullptr) {
            RegionRuns made;
            if (RunsAlongU(given.mask.data, given.width, given.height, !swap_spatial, made)) {
                occupancy._made_runs.push_back(std::move(made));
                region.runs = occupancy._made_runs.back().runs.data();
                region.run_offsets = occupancy._made_runs.back().offsets.data();
            }
        }
        occupancy._regions.push_back(region);
    }

    const PlacedRegion* const placed = occupancy._regions.data();

    occupancy._u0 = std::numeric_limits<std::uint64_t>::max();
    occupancy._v0 = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < region_count; ++i) {
        const auto& region = placed[i];
        occupancy._u0 = std::min(occupancy._u0, region.u_start);
        occupancy._v0 = std::min(occupancy._v0, region.v_start);
        occupancy._u1 = std::max(occupancy._u1, region.u_start + region.u_size);
        occupancy._v1 = std::max(occupancy._v1, region.v_start + region.v_size);
    }

    occupancy._chunk_cu0 = occupancy._u0 / chunk_u;
    occupancy._chunk_cv0 = occupancy._v0 / chunk_v;
    occupancy._columns = ((occupancy._u1 - 1) / chunk_u) - occupancy._chunk_cu0 + 1;
    occupancy._rows = ((occupancy._v1 - 1) / chunk_v) - occupancy._chunk_cv0 + 1;

    const auto cells = static_cast<std::size_t>(occupancy._columns * occupancy._rows);
    if (cells > std::numeric_limits<std::uint32_t>::max()) {
        return Error{ErrorCode::invalid_argument,
                     "The regions span more chunks than one reduction can index", node};
    }
    occupancy._offsets.assign(cells + 1, 0);

    // The chunk span of one region's bounding box.
    const auto span = [&](const PlacedRegion& region, std::uint64_t& cx0, std::uint64_t& cx1, std::uint64_t& cy0,
                          std::uint64_t& cy1) {
        cx0 = region.u_start / chunk_u;
        cx1 = (region.u_start + region.u_size - 1) / chunk_u;
        cy0 = region.v_start / chunk_v;
        cy1 = (region.v_start + region.v_size - 1) / chunk_v;
    };

    // The chunks a region actually occupies, which is not the same as the chunks its bounding box
    // covers. A thin rectangle laid along the diagonal has a bounding box the size of the image and
    // sets a thousandth of it; bucketing by the box would read every chunk to reach the band. The
    // mask decides instead, at the cost of one pass over it -- bytes the caller already holds, and
    // a fraction of what reading the chunks they would otherwise stand for costs.
    //
    // A region that is its whole box has nothing to narrow, and costs one row of each chunk row.
    std::vector<std::uint8_t> occupied;

    // Walk one region's chunk rows, marking the columns each row occupies and handing the marked
    // cells over. How a row is marked is `mark_row`, which is the only part for_each_cell supplies.
    //
    // `found` stops a row scan once every column of the band is accounted for, and the raster
    // scanner reads `occupied` directly to skip a column already marked: without that it rescans
    // pixels whose answer is settled, which on a region whose box is the image is tens of megabytes
    // of them.
    const auto scan_rows = [&](const PlacedRegion& region, auto&& mark_row, auto&& visit) {
        std::uint64_t cx0 = 0;
        std::uint64_t cx1 = 0;
        std::uint64_t cy0 = 0;
        std::uint64_t cy1 = 0;
        span(region, cx0, cx1, cy0, cy1);

        const std::uint64_t columns = cx1 - cx0 + 1;
        occupied.assign(static_cast<std::size_t>(columns), 0);
        for (auto cy = cy0; cy <= cy1; ++cy) {
            std::fill(occupied.begin(), occupied.end(), std::uint8_t{0});
            std::uint64_t found = 0;
            const auto mark = [&](std::uint64_t column) {
                auto& seen = occupied.at(static_cast<std::size_t>(column - cx0));
                if (seen == 0) {
                    seen = 1;
                    ++found;
                }
            };
            const auto marked = [&](std::uint64_t column) {
                return occupied.at(static_cast<std::size_t>(column - cx0)) != 0;
            };

            const std::uint64_t y_first = std::max(region.v_start, cy * chunk_v);
            const std::uint64_t y_last = std::min(region.v_start + region.v_size, (cy + 1) * chunk_v);
            for (std::uint64_t y = y_first; y < y_last && found < columns; ++y) {
                mark_row(y, mark, marked);
            }

            for (auto cx = cx0; cx <= cx1; ++cx) {
                if (marked(cx)) {
                    visit(cx, cy);
                }
            }
        }
    };

    // Every chunk cell one region occupies. The region says which pixels of a row it selects; what is
    // decided here is only which chunk columns those fall in. A span whose every pixel is selected --
    // a run, or a box -- occupies every column it crosses. A span read through the raster occupies a
    // column once a selected pixel is found in it, and a column already marked is not looked at again.
    //
    // A box needs no pass of its own: its first row in a chunk row marks every column of it, and the
    // scan stops there.
    const auto for_each_cell = [&](const PlacedRegion& region, auto&& visit) {
        const std::uint64_t u_end = region.u_start + region.u_size;
        scan_rows(
            region,
            [&](std::uint64_t y, auto&& mark, auto&& marked) {
                region.ForEachSpan(
                    y, region.u_start, u_end,
                    [&](std::uint64_t first, std::uint64_t last, const std::uint8_t* selected, std::uint64_t step) {
                        if (selected == nullptr) {
                            for (auto column = first / chunk_u; column <= (last - 1) / chunk_u; ++column) {
                                mark(column);
                            }
                            return;
                        }
                        std::uint64_t x = first;
                        while (x < last) {
                            const std::uint64_t column = x / chunk_u;
                            const std::uint64_t boundary = std::min(last, (column + 1) * chunk_u);
                            if (!marked(column)) {
                                for (std::uint64_t k = x; k < boundary; ++k) {
                                    if (selected[(k - first) * step] != 0) {
                                        mark(column);
                                        break;
                                    }
                                }
                            }
                            x = boundary;
                        }
                    });
            },
            visit);
    };

    // Counting sort needs the counts before it can place anything, but the mask is the most
    // expensive thing here to look at twice -- a region whose box is the image is tens of megabytes
    // of it. So the incidences are recorded once, in region order, and the two passes read that.
    std::vector<std::uint32_t> incidence_cells;
    std::vector<std::uint64_t> region_first(region_count + 1, 0);
    for (std::size_t i = 0; i < region_count; ++i) {
        for_each_cell(placed[i], [&](std::uint64_t cx, std::uint64_t cy) {
            incidence_cells.push_back(static_cast<std::uint32_t>(occupancy.Cell(cx, cy)));
        });
        if (incidence_cells.size() > kMaxChunkIncidences) {
            return Error{ErrorCode::invalid_argument,
                         "The regions together touch more chunks than one reduction can index", node};
        }
        region_first.at(i + 1) = incidence_cells.size();
    }

    for (const auto cell : incidence_cells) {
        ++occupancy._offsets.at(static_cast<std::size_t>(cell) + 1);
    }
    for (std::size_t cell = 0; cell < cells; ++cell) {
        occupancy._offsets.at(cell + 1) += occupancy._offsets.at(cell);
    }

    occupancy._entries.resize(incidence_cells.size());
    std::vector<std::uint64_t> cursor(occupancy._offsets.begin(), occupancy._offsets.end() - 1);
    for (std::size_t i = 0; i < region_count; ++i) {
        for (auto k = region_first.at(i); k < region_first.at(i + 1); ++k) {
            const auto cell = static_cast<std::size_t>(incidence_cells.at(static_cast<std::size_t>(k)));
            occupancy._entries.at(static_cast<std::size_t>(cursor.at(cell)++)) = static_cast<std::uint32_t>(i);
        }
    }

// The occupied runs of each chunk row.
//
// Reading the bounding box would be correct and still wrong: the box of a diagonal cut is the whole
// image, while the cut touches one chunk per row. On a 4096^2 image that is 256 chunks decoded to
// use 16. The runs are what the walk reads instead, so the cost follows the regions rather than the
// rectangle that happens to contain them.
    occupancy._runs_per_row.resize(static_cast<std::size_t>(occupancy._rows));
    for (std::uint64_t row = 0; row < occupancy._rows; ++row) {
        auto& runs = occupancy._runs_per_row.at(static_cast<std::size_t>(row));
        for (std::uint64_t column = 0; column < occupancy._columns; ++column) {
            const auto cell = static_cast<std::size_t>((row * occupancy._columns) + column);
            if (occupancy._offsets.at(cell + 1) == occupancy._offsets.at(cell)) {
                continue;
            }
            if (!runs.empty() && runs.back().last + 1 == column) {
                runs.back().last = column;
            } else {
                runs.push_back(ColumnRun{column, column});
            }
        }
    }

    // The chunks one spectral layer of the whole region set occupies. Not the plan's layer, which
    // is the whole plane: a reduction spends its emit budget against the region set it was given.
    for (const auto& row_runs : occupancy._runs_per_row) {
        for (const auto& run : row_runs) {
            occupancy._layer_chunks += run.last - run.first + 1;
        }
    }

    return occupancy;
}

std::vector<OccupiedFootprint> Occupancy::Footprints(std::uint64_t chunks_per_read) const {
    std::vector<OccupiedFootprint> footprints;
    for (std::uint64_t row = 0; row < _rows;) {
        const auto& runs = _runs_per_row.at(static_cast<std::size_t>(row));
        if (runs.empty()) {
            ++row;
            continue;
        }

        // Rows below that repeat this row's runs exactly are read with it, so that a solid rectangle
        // becomes a few large requests while a diagonal stays one chunk per row -- as many rows as a
        // read affords at the width of the widest run in them.
        std::uint64_t widest = 0;
        for (const auto& run : runs) {
            widest = std::max(widest, run.last - run.first + 1);
        }
        const std::uint64_t band_limit = UnitsAffordable(chunks_per_read, widest);
        std::uint64_t band_end = row + 1;
        while (band_end < _rows && (band_end - row) < band_limit &&
               _runs_per_row.at(static_cast<std::size_t>(band_end)) == runs) {
            ++band_end;
        }
        const std::uint64_t band_rows = band_end - row;

        const std::uint64_t chunk_cv_begin = _chunk_cv0 + row;
        const std::uint64_t chunk_cv_end = _chunk_cv0 + band_end;
        const std::uint64_t v_begin = std::max(_v0, chunk_cv_begin * _chunk_v);
        const std::uint64_t v_end = std::min(_v1, chunk_cv_end * _chunk_v);

        // A run wider than a read is read in pieces, not in one request. Without this the smallest
        // request is a whole chunk row of the region: 157 chunks of a 80000-pixel-wide region is
        // 628 MiB, ten times the budget it was supposed to respect, and nothing to report or cancel
        // from until all of it lands.
        const std::uint64_t segment_limit = UnitsAffordable(chunks_per_read, band_rows);
        for (const auto& whole : runs) {
            for (std::uint64_t first = whole.first; first <= whole.last;) {
                const std::uint64_t width = std::min(segment_limit, whole.last - first + 1);

                OccupiedFootprint footprint;
                footprint.chunk_cv_begin = chunk_cv_begin;
                footprint.chunk_cv_end = chunk_cv_end;
                footprint.chunk_cu_begin = _chunk_cu0 + first;
                footprint.chunk_cu_end = footprint.chunk_cu_begin + width;

                const std::uint64_t u_begin = std::max(_u0, footprint.chunk_cu_begin * _chunk_u);
                const std::uint64_t u_end = std::min(_u1, footprint.chunk_cu_end * _chunk_u);
                footprint.slab.u_start = u_begin;
                footprint.slab.u_count = u_end - u_begin;
                footprint.slab.v_start = v_begin;
                footprint.slab.v_count = v_end - v_begin;
                footprint.slab.chunks = width * band_rows;
                footprints.push_back(footprint);

                first += width;
            }
        }
        row = band_end;
    }
    return footprints;
}

}  // namespace carta::zarr::internal
