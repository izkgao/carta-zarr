/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Which chunks a set of regions occupies, stated directly.
//
// Every claim here used to be reached by running a whole spectral reduction over a 4 x 5 x 2 x 3
// fixture and counting how many times a result was handed over -- two tests that said in their own
// comments that they stop testing anything if the chunk shape ever changes. Two of the refusals
// below were reachable from no test at all.
//
// This target links nothing. The question is regions and a chunk shape in, an index out.

#include "reduce/occupancy.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "support/check.h"

namespace {

using carta::zarr::AxisRole;
using carta::zarr::ErrorCode;
using carta::zarr::RegionMask;
using carta::zarr::internal::Occupancy;
using carta::zarr::internal::PlacedRegion;

using carta::zarr::testing::Require;

RegionMask Box(std::uint64_t x, std::uint64_t y, std::uint64_t width, std::uint64_t height) {
    RegionMask region;
    region.x_start = x;
    region.y_start = y;
    region.width = width;
    region.height = height;
    return region;
}

Occupancy Built(const std::vector<RegionMask>& regions, std::uint64_t chunk_u, std::uint64_t chunk_v,
                AxisRole fastest = AxisRole::spatial_x) {
    auto built = Occupancy::Of({regions.data(), regions.size()}, chunk_u, chunk_v, fastest, "TEST");
    Require(static_cast<bool>(built),
            "Occupancy::Of failed: " + (built ? std::string{} : built.error().message));
    return std::move(built.value());
}

// A read that could decode every chunk there is, so that no footprint is cut. Cutting changes a
// footprint's shape and never what the footprints cover between them.
constexpr std::uint64_t kUnbounded = std::numeric_limits<std::uint64_t>::max();

// The chunk cells the occupancy's footprints cover, as "cu,cv" pairs in row order -- what a reduction
// reads, rather than the representation it is read from.
//
// Checked against the index on the way: every covered cell is one the index has regions for, and
// between them the covered cells hold every incidence the index does. So the runs the footprints are
// cut from and the index the accumulation looks regions up in describe one set of chunks.
std::string Occupied(const Occupancy& occupancy) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> cells;
    std::size_t incidences = 0;
    for (const auto& footprint : occupancy.Footprints(kUnbounded)) {
        for (auto cv = footprint.chunk_cv_begin; cv < footprint.chunk_cv_end; ++cv) {
            for (auto cu = footprint.chunk_cu_begin; cu < footprint.chunk_cu_end; ++cu) {
                const auto touching = occupancy.RegionsTouching(cu, cv).size;
                Require(touching > 0, "a footprint covered a chunk no region touches");
                incidences += touching;
                cells.emplace_back(cv, cu);
            }
        }
    }
    Require(incidences == occupancy.entries().size(), "the index holds incidences outside every footprint");

    std::sort(cells.begin(), cells.end());
    std::string text;
    for (const auto& [cv, cu] : cells) {
        text += (text.empty() ? "" : " ") + std::to_string(cu) + "," + std::to_string(cv);
    }
    return text;
}

// Every span one row of a placed region selects inside [u0, u1), in order: "first-last" for a span
// whose every pixel is selected, and "first-last@offset/step" for one read through the raster, the
// offset counted from the raster's first byte.
std::string Spans(const PlacedRegion& region, std::uint64_t y, std::uint64_t u0, std::uint64_t u1,
                  const std::uint8_t* raster = nullptr) {
    std::string text;
    region.ForEachSpan(y, u0, u1,
                       [&](std::uint64_t first, std::uint64_t last, const std::uint8_t* mask, std::uint64_t step) {
                           text += (text.empty() ? "" : " ") + std::to_string(first) + "-" + std::to_string(last);
                           if (mask != nullptr) {
                               text += "@" + std::to_string(mask - raster) + "/" + std::to_string(step);
                           }
                       });
    return text;
}

std::string RowsOf(const PlacedRegion& region, std::uint64_t u0, std::uint64_t u1, std::uint64_t v0, std::uint64_t v1) {
    const auto rows = region.RowsWithin(u0, u1, v0, v1);
    return std::to_string(rows.first) + "-" + std::to_string(rows.last);
}

// A raster whose pixel (x, y) of the bounding box is set when the chunk it falls in is on the
// diagonal of the chunk grid.
std::vector<std::uint8_t> DiagonalRaster(std::uint64_t width, std::uint64_t height, std::uint64_t chunk) {
    std::vector<std::uint8_t> raster(static_cast<std::size_t>(width * height), 0);
    for (std::uint64_t y = 0; y < height; ++y) {
        for (std::uint64_t x = 0; x < width; ++x) {
            if (x / chunk == y / chunk) {
                raster.at(static_cast<std::size_t>((y * width) + x)) = 1;
            }
        }
    }
    return raster;
}

void TestABoxOccupiesEveryChunkItsBoundingBoxTouches() {
    // x in [2, 8) and y in [3, 8) over a 4 x 4 chunk grid: two chunk columns, two chunk rows, and
    // no mask to narrow either.
    const auto occupancy = Built({Box(2, 3, 6, 5)}, 4, 4);

    Require(Occupied(occupancy) == "0,0 1,0 0,1 1,1", "a box did not occupy its whole bounding box");

    // Read as one footprint, clamped to the region's own pixels rather than to the chunks around them.
    const auto footprints = occupancy.Footprints(kUnbounded);
    Require(footprints.size() == 1, "a box whose rows are alike was not read as one footprint");
    const auto& slab = footprints.front().slab;
    Require(slab.u_start == 2 && slab.u_count == 6 && slab.v_start == 3 && slab.v_count == 5,
            "the footprint was not clamped to the region's own pixels");
    Require(slab.chunks == 4, "a 2 x 2 footprint did not count four chunks");
    Require(occupancy.LayerChunks() == 4, "a 2 x 2 box did not report four chunks in a layer");
}

// Asked through what a reduction asks -- the rows a region has, and the span of each -- rather than
// through the fields of the placement, which are the occupancy's own. The steps through the raster
// are TestAFragmentedRasterIsReadThroughItsBytes's.
void TestThePlacementFollowsTheFastestSpatialAxis() {
    const auto straight = Built({Box(2, 3, 6, 5)}, 4, 4, AxisRole::spatial_x);
    const auto& as_written = straight.regions().at(0);
    Require(RowsOf(as_written, 0, 100, 0, 100) == "3-8" && Spans(as_written, 3, 0, 100) == "2-8",
            "x was not placed on u when the store varies x fastest");

    // The same region on a store that varies y fastest: x and y swap.
    const auto swapped = Built({Box(2, 3, 6, 5)}, 4, 4, AxisRole::spatial_y);
    const auto& placed = swapped.regions().at(0);
    Require(RowsOf(placed, 0, 100, 0, 100) == "2-8" && Spans(placed, 2, 0, 100) == "3-8",
            "y was not placed on u when the store varies y fastest");
}

void TestAMaskNarrowsTheOccupancyBelowTheBoundingBox() {
    // The whole reason this module exists: a cut along the diagonal has a bounding box of sixteen
    // chunks and occupies four of them.
    const auto raster = DiagonalRaster(16, 16, 4);
    auto region = Box(0, 0, 16, 16);
    region.mask = {raster.data(), raster.size()};

    const auto occupancy = Built({region}, 4, 4);

    Require(Occupied(occupancy) == "0,0 1,1 2,2 3,3", "the mask did not narrow the occupancy to the diagonal");
    Require(occupancy.LayerChunks() == 4, "a diagonal did not report one chunk per row");
    // No two rows of a diagonal are alike, so none is read with another, whatever a read affords.
    Require(occupancy.Footprints(kUnbounded).size() == 4, "a diagonal was read in bands of unlike rows");
}

// A region given as a raster alone reaches the walk as runs along u, whichever axis that is, so that
// the accumulation takes the loop an unmasked region uses. The diagonal is block-symmetric, so its
// runs are the same either way: line l holds the one run [4 * (l / 4), 4 * (l / 4) + 4).
void TestARasterReachesTheWalkAsRuns() {
    const auto raster = DiagonalRaster(16, 16, 4);
    auto region = Box(0, 0, 16, 16);
    region.mask = {raster.data(), raster.size()};

    for (const auto fastest : {AxisRole::spatial_x, AxisRole::spatial_y}) {
        const auto occupancy = Built({region}, 4, 4, fastest);
        const auto& placed = occupancy.regions().at(0);
        // A span with no raster byte to read is a run; one read through the raster would say where.
        for (std::uint64_t line = 0; line < 16; ++line) {
            const auto expected = std::to_string(4 * (line / 4)) + "-" + std::to_string((4 * (line / 4)) + 4);
            Require(Spans(placed, line, 0, 16, raster.data()) == expected,
                    "line " + std::to_string(line) + " is not the one run " + expected + ": " +
                        Spans(placed, line, 0, 16, raster.data()));
        }
        Require(Occupied(occupancy) == "0,0 1,1 2,2 3,3", "the runs did not narrow the occupancy to the diagonal");
    }
}

// A raster too fragmented to be worth runs is read as the raster it is, and occupies what it did.
void TestAFragmentedRasterStaysARaster() {
    std::vector<std::uint8_t> board(16 * 16, 0);
    for (std::uint64_t y = 0; y < 16; ++y) {
        for (std::uint64_t x = 0; x < 16; ++x) {
            board.at(static_cast<std::size_t>((y * 16) + x)) = (x + y) % 2 == 0 ? 1 : 0;
        }
    }
    auto region = Box(0, 0, 16, 16);
    region.mask = {board.data(), board.size()};

    for (const auto fastest : {AxisRole::spatial_x, AxisRole::spatial_y}) {
        const auto occupancy = Built({region}, 4, 4, fastest);
        const auto& placed = occupancy.regions().at(0);
        const auto expected = fastest == AxisRole::spatial_x ? "0-16@0/1" : "0-16@0/16";
        Require(Spans(placed, 0, 0, 16, board.data()) == expected, "a checkerboard was not left a raster");
        Require(occupancy.LayerChunks() == 16, "a checkerboard touches every chunk of its box");
    }
}

void TestTheIncidencesOfOneChunkAreContiguousAndInRegionOrder() {
    // Three regions over one 8 x 8 chunk grid of 4 x 4 chunks. Region 0 covers everything, region 1
    // the top-left chunk, region 2 the bottom-right one. The counting sort is what puts 0 before 1
    // in the first chunk's entries, and the accumulation reads them in that order.
    const auto occupancy = Built({Box(0, 0, 8, 8), Box(0, 0, 4, 4), Box(4, 4, 4, 4)}, 4, 4);

    const auto top_left = occupancy.RegionsTouching(0, 0);
    Require(top_left.size == 2 && top_left.data[0] == 0 && top_left.data[1] == 1,
            "the top-left chunk's regions were not 0 then 1");
    const auto bottom_right = occupancy.RegionsTouching(1, 1);
    Require(bottom_right.size == 2 && bottom_right.data[0] == 0 && bottom_right.data[1] == 2,
            "the bottom-right chunk's regions were not 0 then 2");
    const auto top_right = occupancy.RegionsTouching(1, 0);
    Require(top_right.size == 1 && top_right.data[0] == 0, "the top-right chunk was touched by more than region 0");

    // The same claim against the index itself: offsets rise, and there is one entry per incidence.
    const auto& offsets = occupancy.offsets();
    Require(offsets.size() == 5, "a 2 x 2 grid did not produce five offsets");
    for (std::size_t cell = 0; cell + 1 < offsets.size(); ++cell) {
        Require(offsets.at(cell) <= offsets.at(cell + 1), "the offsets were not non-decreasing");
    }
    Require(offsets.back() == occupancy.entries().size(), "the last offset did not account for every entry");
    Require(occupancy.entries().size() == 6, "four chunks of region 0 plus one each of 1 and 2 is six incidences");
}

void TestAMaskThatSelectsNothingOccupiesNothing() {
    const std::vector<std::uint8_t> raster(16 * 16, 0);
    auto region = Box(0, 0, 16, 16);
    region.mask = {raster.data(), raster.size()};

    const auto occupancy = Built({region}, 4, 4);

    Require(occupancy.entries().empty(), "a mask of zeroes produced incidences");
    Require(Occupied(occupancy).empty(), "a mask of zeroes occupied a chunk");
    Require(occupancy.LayerChunks() == 0, "a mask of zeroes did not report an empty layer");
    Require(occupancy.Footprints(1).empty(), "a mask of zeroes gave a reduction something to read");
}

// Rows that occupy the same chunk columns are read together, as many of them as a read affords, so
// that a solid rectangle is a few large requests rather than one per chunk row.
void TestAlikeRowsAreReadTogether() {
    // 16 x 16 pixels over 4 x 4 chunks: four rows of four, every one alike.
    const auto occupancy = Built({Box(0, 0, 16, 16)}, 4, 4);

    const auto whole = occupancy.Footprints(kUnbounded);
    Require(whole.size() == 1 && whole.front().slab.chunks == 16,
            "a solid rectangle a read can hold was not read as one footprint");

    // A read of eight chunks holds two rows of four.
    const auto halves = occupancy.Footprints(8);
    Require(halves.size() == 2, "a read of eight chunks did not take the rectangle two rows at a time");
    for (const auto& footprint : halves) {
        Require(footprint.chunk_cv_end - footprint.chunk_cv_begin == 2 && footprint.slab.chunks == 8,
                "a band of two rows of four did not count eight chunks");
    }
}

// No footprint is more than one read can decode. A run wider than that is cut, rather than read in
// one request -- which is what a region 80000 pixels wide once did, asking for 157 chunks, 628 MiB,
// ten times the budget, with nothing to report or cancel from until it landed.
void TestNoFootprintIsMoreThanOneRead() {
    const auto occupancy = Built({Box(0, 0, 16, 16)}, 4, 4);

    for (const std::uint64_t per_read : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3},
                                         std::uint64_t{5}, std::uint64_t{16}}) {
        const auto footprints = occupancy.Footprints(per_read);
        std::uint64_t covered = 0;
        for (const auto& footprint : footprints) {
            // A read smaller than a chunk still reads one: a chunk is the least that can be decoded.
            Require(footprint.slab.chunks <= std::max<std::uint64_t>(1, per_read),
                    "a footprint was more than one read can decode, at " + std::to_string(per_read));
            const auto cells = (footprint.chunk_cu_end - footprint.chunk_cu_begin) *
                               (footprint.chunk_cv_end - footprint.chunk_cv_begin);
            Require(footprint.slab.chunks == cells, "a footprint did not count the chunks it spans");
            covered += cells;
        }
        Require(covered == occupancy.LayerChunks(),
                "cutting footprints to a read changed what they cover, at " + std::to_string(per_read));
    }

    // Two chunks a read: each row of four is cut in two.
    const auto cut = occupancy.Footprints(2);
    Require(cut.size() == 8, "four rows of four chunks were not cut into eight pieces of two");
}

// kMaxChunkIncidences, the other refusal, is not here, and the measurement is why: the smallest
// input that trips it is 64 regions over a 1024 x 1024 chunk grid plus one more, which took 2.06 s
// and 815 MB of resident memory to reach the check. The refusal exists to stop an allocation
// nothing can serve, and a test of it has to make that allocation first. It was run once by hand
// against this file's Occupancy::Of and reported "The regions together touch more chunks than one
// reduction can index"; that is recorded here rather than paid for on every ctest run.

void TestAGridTooLargeToIndexIsRefused() {
    // 65,536 chunks on a side is 2^32 cells, one more than a cell number can hold. Refused before
    // anything is allocated for it.
    const std::vector<RegionMask> regions{Box(0, 0, 65536, 65536)};
    const auto refused = Occupancy::Of({regions.data(), regions.size()}, 1, 1, AxisRole::spatial_x, "TEST");
    Require(!refused && refused.error().code == ErrorCode::invalid_argument,
            "a grid of 2^32 chunks was accepted");
}

// ---------------------------------------------------------------------------------------------
// Which pixels of a chunk cell a region selects, a row at a time.
//
// The occupancy above says which cells a region touches; a reduction then has to know which pixels
// of each cell to add up, and that used to be worked out again inside the accumulation -- the run
// encoding, the raster's strides, and the clipping to the cell -- where nothing but a whole
// reduction against an oracle could see it.

// A region that is its whole box is one span a row, clipped to the cell, and no rows at all in a cell
// it misses along either axis.
void TestABoxIsOneSpanARowClippedToTheCell() {
    const auto occupancy = Built({Box(2, 3, 6, 5)}, 4, 4);
    const auto& region = occupancy.regions().at(0);
    Require(RowsOf(region, 0, 4, 4, 8) == "4-8", "the box's rows in the cell below its top");
    Require(Spans(region, 4, 0, 4) == "2-4", "the box clipped to the left cell, not " + Spans(region, 4, 0, 4));
    Require(Spans(region, 4, 4, 8) == "4-8", "the box across the whole of the next");
    Require(RowsOf(region, 8, 12, 0, 4) == "0-0", "a cell the box misses along u has no rows");
    Require(RowsOf(region, 0, 4, 8, 12) == "0-0", "and one it misses along v has none either");
}

// A raster with runs few enough to be worth them is read as its runs, every pixel of each selected,
// each clipped to the cell and none that falls outside it. Two runs in a line of 48 pixels is few
// enough; in a line of 12 it would be fragmented, and read as the raster.
void TestRunsAreClippedToTheCell() {
    std::vector<std::uint8_t> raster(48, 0);
    for (const std::uint64_t x : {1, 2, 3, 6, 7, 8, 9, 10}) {
        raster.at(static_cast<std::size_t>(x)) = 1;
    }
    auto box = Box(0, 0, 48, 1);
    box.mask = {raster.data(), raster.size()};
    const auto occupancy = Built({box}, 4, 4);
    const auto& region = occupancy.regions().at(0);
    Require(Spans(region, 0, 0, 48) == "1-4 6-11", "the row's two runs, not " + Spans(region, 0, 0, 48, raster.data()));
    Require(Spans(region, 0, 0, 4) == "1-4", "the first run alone in the first cell");
    Require(Spans(region, 0, 4, 8) == "6-8", "the second clipped to the second cell, and the first left out");
    Require(Spans(region, 0, 11, 12).empty(), "nothing in a cell between the runs and the edge");
}

// A raster too fragmented for runs is read through its bytes, and which way its rows lie decides the
// steps. A store that varies y fastest -- which is what XRADIO writes, m last -- puts the caller's
// columns along v and its rows along u, so one pixel along u is a whole raster row on.
void TestAFragmentedRasterIsReadThroughItsBytes() {
    std::vector<std::uint8_t> board(16 * 16, 0);
    for (std::uint64_t y = 0; y < 16; ++y) {
        for (std::uint64_t x = 0; x < 16; ++x) {
            board.at(static_cast<std::size_t>((y * 16) + x)) = (x + y) % 2 == 0 ? 1 : 0;
        }
    }
    auto box = Box(0, 0, 16, 16);
    box.mask = {board.data(), board.size()};

    // u is x: row y = 5 of the cell starting at x = 4 is byte 5 * 16 + 4, the next one on.
    const auto along_x = Built({box}, 4, 4, AxisRole::spatial_x);
    Require(Spans(along_x.regions().at(0), 5, 4, 8, board.data()) == "4-8@84/1",
            "a raster along x, not " + Spans(along_x.regions().at(0), 5, 4, 8, board.data()));

    // u is y: row v = x = 5 of the cell starting at y = 4 is byte 4 * 16 + 5, sixteen on.
    const auto along_y = Built({box}, 4, 4, AxisRole::spatial_y);
    Require(Spans(along_y.regions().at(0), 5, 4, 8, board.data()) == "4-8@69/16",
            "a raster across y, not " + Spans(along_y.regions().at(0), 5, 4, 8, board.data()));
}

// The diagonal is runs whichever axis the store varies fastest: line 5 is the one run [4, 8).
void TestADiagonalIsOneRunALineEitherWay() {
    const auto raster = DiagonalRaster(16, 16, 4);
    auto box = Box(0, 0, 16, 16);
    box.mask = {raster.data(), raster.size()};
    for (const auto fastest : {AxisRole::spatial_x, AxisRole::spatial_y}) {
        const auto occupancy = Built({box}, 4, 4, fastest);
        Require(Spans(occupancy.regions().at(0), 5, 0, 16) == "4-8", "line 5 of the diagonal is the run [4, 8)");
    }
}

}  // namespace

int main() {
    try {
        TestABoxOccupiesEveryChunkItsBoundingBoxTouches();
        TestThePlacementFollowsTheFastestSpatialAxis();
        TestAMaskNarrowsTheOccupancyBelowTheBoundingBox();
        TestARasterReachesTheWalkAsRuns();
        TestAFragmentedRasterStaysARaster();
        TestTheIncidencesOfOneChunkAreContiguousAndInRegionOrder();
        TestAMaskThatSelectsNothingOccupiesNothing();
        TestAlikeRowsAreReadTogether();
        TestNoFootprintIsMoreThanOneRead();
        TestAGridTooLargeToIndexIsRefused();
        TestABoxIsOneSpanARowClippedToTheCell();
        TestRunsAreClippedToTheCell();
        TestAFragmentedRasterIsReadThroughItsBytes();
        TestADiagonalIsOneRunALineEitherWay();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "occupancy test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
