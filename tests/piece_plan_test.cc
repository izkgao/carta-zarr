/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// What an ordinary read decides before it reads anything: whether to cut itself into pieces, where,
// and how big each one should be.
//
// Until this existed those decisions could only be observed through a read against a directory tree
// -- a test would count progress callbacks and reason backwards to how the read must have been cut.
// None of this needs a store, a transport or a fixture, which is the point.

#include "read/pieces.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "support/check.h"

namespace {

using carta::zarr::AxisRole;
using carta::zarr::ChunkGeometry;
using carta::zarr::ImageDescriptor;
using carta::zarr::Range;
using carta::zarr::ReadOptions;
using carta::zarr::ReadRequest;
using carta::zarr::internal::Piece;
using carta::zarr::internal::PlanPieces;

using carta::zarr::testing::Require;

// An image in the logical order XRADIO reports: x, y, spectral, polarization, time.
ImageDescriptor MakeImage(std::uint64_t x, std::uint64_t y, std::uint64_t channels, bool has_mask = false) {
    ImageDescriptor descriptor;
    descriptor.id = "SKY";
    descriptor.stored_type = carta::zarr::DataType::float32;
    descriptor.has_pixel_mask = has_mask;
    descriptor.pixel_mask_id = has_mask ? "FLAG" : "";
    const struct {
        const char* name;
        AxisRole role;
        std::uint64_t length;
    } axes[]{{"l", AxisRole::spatial_x, x},
             {"m", AxisRole::spatial_y, y},
             {"frequency", AxisRole::spectral, channels},
             {"polarization", AxisRole::polarization, 1},
             {"time", AxisRole::time, 1}};
    for (std::size_t i = 0; i < 5; ++i) {
        carta::zarr::AxisDescriptor axis;
        axis.name = axes[i].name;
        axis.role = axes[i].role;
        axis.length = axes[i].length;
        axis.storage_index = std::vector<std::size_t>{3, 4, 1, 2, 0}.at(i);
        descriptor.axes.push_back(axis);
    }
    return descriptor;
}

ChunkGeometry MakeGeometry(std::uint64_t chunk_x, std::uint64_t chunk_y, std::uint64_t chunk_z) {
    ChunkGeometry geometry;
    geometry.chunk_shape = {chunk_x, chunk_y, chunk_z, 1, 1};
    return geometry;
}

ReadRequest WholeImage(const ImageDescriptor& descriptor) {
    ReadRequest request;
    for (const auto& axis : descriptor.axes) {
        request.axes.push_back(Range{0, axis.length, 1});
    }
    return request;
}

std::uint64_t ElementCount(const ReadRequest& request) {
    std::uint64_t elements = 1;
    for (const auto& range : request.axes) {
        elements *= range.count;
    }
    return elements;
}

std::vector<Piece> Pieces(const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                          const ReadRequest& request, const ReadOptions& options, bool watching = false) {
    return PlanPieces(descriptor, geometry, request, options, watching);
}

bool SameRange(const Range& a, const Range& b) {
    return a.start == b.start && a.count == b.count && a.stride == b.stride;
}

// The axes on which a piece's request differs from the whole read's.
std::vector<std::size_t> CutAxes(const Piece& piece, const ReadRequest& whole) {
    std::vector<std::size_t> cut;
    for (std::size_t i = 0; i < whole.axes.size(); ++i) {
        if (!SameRange(piece.request.axes.at(i), whole.axes.at(i))) {
            cut.push_back(i);
        }
    }
    return cut;
}

// The pieces fill the destination end to end, each starting where the one before it stopped and the
// last stopping at the end -- which is what makes the finished part a prefix, and what a progress
// report counts.
void RequireTheyFillTheDestination(const std::vector<Piece>& pieces, const ReadRequest& whole,
                                   const std::string& what) {
    std::uint64_t filled = 0;
    for (const auto& piece : pieces) {
        Require(piece.first_element == filled, what + ": a piece did not start where the one before it stopped");
        filled += ElementCount(piece.request);
    }
    Require(filled == ElementCount(whole), what + ": the pieces did not fill the destination");
}

// A read nobody is watching and nobody has put a ceiling on is issued exactly as it was asked for.
// The plan still describes it, as one piece covering everything, so the loop that reads it has one
// shape rather than two.
void TestAnUnconstrainedReadIsOnePiece() {
    const auto image = MakeImage(64, 64, 8);
    const auto geometry = MakeGeometry(32, 64, 1);
    const auto request = WholeImage(image);

    const auto pieces = Pieces(image, geometry, request, {});
    Require(pieces.size() == 1, "a read with no reason to be cut was cut");
    Require(CutAxes(pieces.front(), request).empty() && pieces.front().first_element == 0,
            "an uncut read should be one piece covering everything");
}

// Either reason on its own is enough. The memory ceiling is the half that used to be ignored unless
// a progress callback came with it, which meant a caller who said how much memory the read could
// have and did not care to watch was refused instead of served.
//
// A watched read is cut to the library's own budget, so it takes one larger than that budget to show:
// 8192 planes of 64 x 64 float32 is 128 MiB against 64. Planning allocates nothing, so the size costs
// nothing here. The same read unwatched and unbounded is one piece, which is the difference watching
// makes.
void TestEitherReasonCutsTheRead() {
    const auto large = MakeImage(64, 64, 8192);
    const auto large_geometry = MakeGeometry(32, 64, 1);
    const auto large_request = WholeImage(large);
    Require(Pieces(large, large_geometry, large_request, ReadOptions{}).size() == 1,
            "a read nobody watches and nothing bounds was cut");
    const auto watched = Pieces(large, large_geometry, large_request, ReadOptions{}, /*watching=*/true);
    Require(watched.size() > 1, "a watched read larger than the budget was not cut");
    RequireTheyFillTheDestination(watched, large_request, "a watched cut");

    const auto image = MakeImage(64, 64, 8);
    ReadOptions bounded;
    bounded.read_budget_bytes = 4096;
    Require(Pieces(image, MakeGeometry(32, 64, 1), WholeImage(image), bounded).size() > 1,
            "a read with a memory ceiling was not cut");
}

// The cut goes on the slowest-varying axis that selects more than one element, because the
// destination is dense with axis 0 fastest and that is the only axis whose pieces extend a prefix.
void TestTheCutGoesOnTheSlowestSelectedAxis() {
    const auto image = MakeImage(64, 64, 8);
    // Chunks of 32 x 16, so one plane is eight of them, 16 KiB, and one row of chunks along m is two.
    const auto geometry = MakeGeometry(32, 16, 1);
    const auto request = WholeImage(image);
    // Axes 3 and 4 are degenerate here, so the spectrum at index 2 is the slowest one selected. A
    // ceiling of one plane makes each channel a piece.
    ReadOptions a_plane;
    a_plane.read_budget_bytes = 64 * 64 * sizeof(float);
    const auto pieces = Pieces(image, geometry, request, a_plane);
    Require(pieces.size() == 8, "a ceiling of one plane did not make a piece of each channel");
    for (const auto& piece : pieces) {
        Require(CutAxes(piece, request) == std::vector<std::size_t>{2},
                "the cut should have gone on the spectral axis and nowhere else");
        // One channel is worth a plane of the destination.
        Require(piece.first_element == piece.request.axes.at(2).start * 64 * 64,
                "a piece of the spectrum did not land a plane per channel in");
    }
    RequireTheyFillTheDestination(pieces, request, "a spectrum cut");

    // Narrow the spectrum to one channel and the only axis left with more than one element is m. A
    // ceiling of one row of chunks makes each sixteen rows a piece.
    auto one_channel = request;
    one_channel.axes.at(2) = Range{0, 1, 1};
    ReadOptions a_row_of_chunks;
    a_row_of_chunks.read_budget_bytes = 2 * 32 * 16 * sizeof(float);
    const auto narrowed = Pieces(image, geometry, one_channel, a_row_of_chunks);
    Require(narrowed.size() == 4, "a ceiling of one row of chunks did not cut the plane into four");
    for (const auto& piece : narrowed) {
        Require(CutAxes(piece, one_channel) == std::vector<std::size_t>{1},
                "with one channel the cut should move to m and nowhere else");
        Require(piece.first_element == piece.request.axes.at(1).start * 64, "one m is worth a row of the destination");
    }
    RequireTheyFillTheDestination(narrowed, one_channel, "a plane cut");
}

// A request that selects a single element of every axis has nowhere to be cut, so it is one piece
// whatever the caller asked for. A read that cannot be cut is where buffer_too_small comes from.
void TestAReadWithNowhereToCutIsOnePiece() {
    const auto image = MakeImage(64, 64, 8);
    const auto geometry = MakeGeometry(32, 64, 1);
    ReadRequest single;
    single.axes.assign(5, Range{0, 1, 1});

    ReadOptions bounded;
    bounded.read_budget_bytes = 1;
    Require(Pieces(image, geometry, single, bounded, /*watching=*/true).size() == 1,
            "a single-element read has nowhere to be cut");
}

// A piece is measured in chunks, not in elements: the budget buys whole chunks along the cut axis
// because asking for part of one decodes all of it anyway -- so a piece ends on a chunk boundary.
void TestATighterCeilingBuysFewerChunks() {
    const auto image = MakeImage(64, 64, 32);
    const auto geometry = MakeGeometry(64, 64, 4);
    const auto request = WholeImage(image);

    ReadOptions generous;
    generous.read_budget_bytes = 64 * 64 * 4 * sizeof(float) * 8;
    ReadOptions tight;
    tight.read_budget_bytes = 64 * 64 * 4 * sizeof(float);

    const auto wide = Pieces(image, geometry, request, generous);
    const auto narrow = Pieces(image, geometry, request, tight);
    Require(narrow.size() > wide.size(), "a tighter ceiling should buy smaller pieces");
    for (const auto* pieces : {&wide, &narrow}) {
        for (const auto& piece : *pieces) {
            const auto& range = piece.request.axes.at(2);
            Require(range.start % 4 == 0 && (range.start + range.count == 32 || range.count % 4 == 0),
                    "a piece did not begin and end on a chunk boundary of the spectrum");
        }
        RequireTheyFillTheDestination(*pieces, request, "a budgeted cut");
    }
}

// Where a piece ends is the arithmetic that most needed moving: the end is rounded to a chunk
// boundary so that no chunk is decoded by two pieces -- the last element one piece reads and the
// first the next one reads never share a chunk.
//
// It takes a request that does not begin on a chunk boundary to show. Starting at zero, a piece a
// whole number of chunks long ends on a boundary without being moved there, and cutting at the naive
// end passes every other test in this repository.
void TestNoChunkIsReadByTwoPieces() {
    const auto image = MakeImage(64, 64, 32);
    const auto geometry = MakeGeometry(64, 64, 4);
    ReadOptions tight;
    tight.read_budget_bytes = 64 * 64 * 4 * sizeof(float);

    const auto check = [&](const Range& spectrum, const std::string& what) {
        auto request = WholeImage(image);
        request.axes.at(2) = spectrum;
        const auto pieces = Pieces(image, geometry, request, tight);
        Require(pieces.size() > 1, what + " under a one-chunk budget was not cut");
        for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
            const auto& here = pieces.at(i).request.axes.at(2);
            const auto& next = pieces.at(i + 1).request.axes.at(2);
            Require(here.stride == spectrum.stride && next.stride == spectrum.stride,
                    what + ": a piece lost the request's stride");
            const auto last = here.start + ((here.count - 1) * here.stride);
            Require(last / 4 < next.start / 4, what + ": two pieces read from one chunk of the spectrum");
            Require(next.start == last + here.stride, what + ": a piece skipped or repeated a selected channel");
        }
        RequireTheyFillTheDestination(pieces, request, what);
    };

    check(Range{2, 28, 1}, "a read starting mid-chunk");
    check(Range{1, 15, 2}, "a strided read");  // channels 1, 3, ..., 29
}

// The flag is decoded beside the pixels, so a read that will apply it costs more per chunk and the
// same ceiling has to buy less. Counting it on one side of the division and not the other would
// size pieces against a cost the read does not have.
void TestApplyingTheFlagCostsTheBudget() {
    const auto image = MakeImage(64, 64, 32, /*has_mask=*/true);
    const auto geometry = MakeGeometry(64, 64, 1);
    const auto request = WholeImage(image);

    ReadOptions declined;
    declined.read_budget_bytes = 64 * 64 * sizeof(float) * 6;
    declined.apply_pixel_mask = false;
    ReadOptions applied = declined;
    applied.apply_pixel_mask = true;

    const auto without = Pieces(image, geometry, request, declined);
    const auto with = Pieces(image, geometry, request, applied);
    Require(with.size() > without.size(),
            "a read that also decodes the flag cannot afford as much of the spectrum per piece");
}

// A stride of a chunk or more along an axis the read is not cut on steps over whole chunks, and the
// chunks it steps over are not decoded -- so they are not what a piece's budget is spent on. Every
// other column of 16-wide chunks is two chunks across the image, not the three its first and last
// element span, and a budget of 24 chunks buys three channels a piece rather than two.
void TestChunksAStrideStepsOverCostNothing() {
    const auto image = MakeImage(64, 64, 32);
    const auto geometry = MakeGeometry(16, 16, 1);
    auto request = WholeImage(image);
    request.axes.at(0) = Range{0, 2, 32};  // columns 0 and 32: chunks 0 and 2
    ReadOptions budget;
    budget.read_budget_bytes = 24 * 16 * 16 * sizeof(float);

    const auto pieces = Pieces(image, geometry, request, budget);
    for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
        Require(pieces.at(i).request.axes.at(2).count == 3,
                "a piece holds " + std::to_string(pieces.at(i).request.axes.at(2).count) +
                    " channels: two chunks across and four down is eight a channel, and 24 buys three");
    }
    RequireTheyFillTheDestination(pieces, request, "a read striding over whole chunks");
}

// The same along the axis the read is cut on: every eighth channel of a spectrum chunked four deep is
// one chunk a channel, so a budget of two chunks is two channels a piece -- not the one it was when a
// piece's length was its chunks times the chunk depth divided by the stride.
void TestAStrideOverWholeChunksAlongTheCutIsAChunkAnElement() {
    const auto image = MakeImage(64, 64, 32);
    const auto geometry = MakeGeometry(64, 64, 4);
    auto request = WholeImage(image);
    request.axes.at(2) = Range{0, 4, 8};  // channels 0, 8, 16 and 24, a chunk each
    ReadOptions budget;
    budget.read_budget_bytes = 2 * 64 * 64 * 4 * sizeof(float);

    const auto pieces = Pieces(image, geometry, request, budget);
    Require(pieces.size() == 2, "four channels a chunk apart under a two-chunk budget are two pieces, not " +
                                    std::to_string(pieces.size()));
    RequireTheyFillTheDestination(pieces, request, "a strided spectrum");
}

}  // namespace

int main() {
    try {
        TestAnUnconstrainedReadIsOnePiece();
        TestEitherReasonCutsTheRead();
        TestTheCutGoesOnTheSlowestSelectedAxis();
        TestAReadWithNowhereToCutIsOnePiece();
        TestATighterCeilingBuysFewerChunks();
        TestNoChunkIsReadByTwoPieces();
        TestApplyingTheFlagCostsTheBudget();
        TestChunksAStrideStepsOverCostNothing();
        TestAStrideOverWholeChunksAlongTheCutIsAChunkAnElement();
    } catch (const std::exception& error) {
        std::cerr << "piece plan test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "carta-zarr piece plan tests passed\n";
    return 0;
}
