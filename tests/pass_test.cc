/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// What a pass decides before it reads anything.
//
// This is the half of the pass with an answer worth checking on its own: how deep a slab goes, how
// wide a band is, which spatial axis is the inner one, and how much a read may decode. Every one of
// those has moved at least once, and until now each could only be observed through a reduction's
// output against a directory tree on disk -- so the tests that cover them assert how many times a
// result was handed over and reason backwards to what the walk must have read.
//
// None of it needs a store, a transport or a fixture, which is the point.

#include "axis_map.h"
#include "reduce/pass.h"

#include "support/check.h"

#include "support/synthetic_pixel_source.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

namespace {

using carta::zarr::AxisRole;
using carta::zarr::ChunkGeometry;
using carta::zarr::ImageDescriptor;
using carta::zarr::Range;
using carta::zarr::ReadOptions;
using carta::zarr::internal::CheckedPlanes;
using carta::zarr::internal::MapAxes;
using carta::zarr::internal::PassPlan;
using carta::zarr::internal::PlanPass;
using carta::zarr::internal::SelectionChannel;
using carta::zarr::internal::RunPass;
using carta::zarr::internal::Slab;
using carta::zarr::testing::SyntheticPixelSource;

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
        // The stored order XRADIO writes: time, frequency, polarization, l, m -- so m is last and
        // varies fastest.
        axis.storage_index = std::vector<std::size_t>{3, 4, 1, 2, 0}.at(i);
        descriptor.axes.push_back(axis);
    }
    return descriptor;
}

ChunkGeometry MakeGeometry(std::uint64_t chunk_x, std::uint64_t chunk_y, std::uint64_t chunk_z,
                           AxisRole fastest) {
    ChunkGeometry geometry;
    geometry.fastest_spatial_axis = fastest;
    geometry.chunk_shape = {chunk_x, chunk_y, chunk_z, 1, 1};
    return geometry;
}

PassPlan Plan(const ImageDescriptor& descriptor, const ChunkGeometry& geometry, const Range& spectral,
              const ReadOptions& options, std::uint64_t sample = 1) {
    const auto map = MapAxes(descriptor);
    Require(static_cast<bool>(map), "MapAxes failed on a well-formed image");
    const auto planes = CheckedPlanes::Of(descriptor, map.value(), {spectral, 0, 0});
    Require(static_cast<bool>(planes), "the spectral range does not fit this image");
    return PlanPass(descriptor, geometry, map.value(), planes.value(), sample, options);
}

// Which spatial axis the pass walks along is the store's decision, not the image's. Reading a plane
// with the other one fastest transposes every chunk on the way into the destination.
void TestTheInnerAxisFollowsTheStore() {
    const auto image = MakeImage(512, 520, 32);
    const Range spectral{0, 32, 1};
    const ReadOptions options;

    const auto m_fastest = Plan(image, MakeGeometry(256, 260, 2, AxisRole::spatial_y), spectral, options);
    Require(m_fastest.axis_u == 1 && m_fastest.axis_v == 0,
            "when the store varies y fastest, y is the pass's inner axis");
    Require(m_fastest.u_length == 520 && m_fastest.v_length == 512, "the lengths follow the axes");
    Require(m_fastest.chunk_u == 260 && m_fastest.chunk_v == 256, "so do the chunk extents");

    const auto l_fastest = Plan(image, MakeGeometry(256, 260, 2, AxisRole::spatial_x), spectral, options);
    Require(l_fastest.axis_u == 0 && l_fastest.axis_v == 1,
            "when the store varies x fastest, x is the pass's inner axis");
    Require(l_fastest.u_length == 512 && l_fastest.v_length == 520, "the lengths follow the axes");
}

// A budget is in decoded chunk bytes, and a masked read decodes the flag too -- at the flag's own
// cost, which is the thing this used to get wrong. RequireUsableFlag holds a flag to bool over the
// image's shape, so beside a float32 chunk it is a quarter of one and not a second one. Counting it
// as a second one shrank every masked read by the difference.
void TestAMaskCostsWhatTheFlagCosts() {
    const Range spectral{0, 32, 1};
    const auto geometry = MakeGeometry(256, 260, 2, AxisRole::spatial_y);
    const std::uint64_t chunk_elements = 256ULL * 260ULL * 2ULL;
    ReadOptions options;
    options.apply_pixel_mask = true;

    const auto plain = Plan(MakeImage(512, 520, 32, false), geometry, spectral, options);
    const auto masked = Plan(MakeImage(512, 520, 32, true), geometry, spectral, options);
    Require(masked.apply_mask && !plain.apply_mask, "only an image with a flag applies one");
    Require(plain.chunk_bytes == chunk_elements * 4, "a float32 chunk is four bytes an element");
    Require(masked.chunk_bytes == plain.chunk_bytes + chunk_elements,
            "the flag beside it is one byte an element, so a masked float32 chunk costs a quarter "
            "more and not twice as much");

    ReadOptions declined;
    declined.apply_pixel_mask = false;
    const auto not_applied = Plan(MakeImage(512, 520, 32, true), geometry, spectral, declined);
    Require(!not_applied.apply_mask && not_applied.chunk_bytes == plain.chunk_bytes,
            "an image with a flag the caller declined costs what an unmasked one costs");
}

// The caller's ceiling is taken as given; without one the pass asks chunk_blocks for the policy.
void TestTheCallersCeilingWins() {
    const auto image = MakeImage(512, 520, 32);
    const auto geometry = MakeGeometry(256, 260, 2, AxisRole::spatial_y);
    const Range spectral{0, 32, 1};

    ReadOptions limited;
    limited.read_budget_bytes = 1u << 20;
    const auto small = Plan(image, geometry, spectral, limited);
    Require(small.slab_budget_bytes == (1u << 20), "a stated limit is the budget");

    const auto def = Plan(image, geometry, spectral, ReadOptions{});
    Require(def.slab_budget_bytes ==
                carta::zarr::internal::DefaultReadBytes(
                    carta::zarr::internal::DecodedChunkBytes(image, geometry)),
            "without a limit the budget is the one chunk_blocks measured");
}

// How many selected channels one chunk of the spectral axis holds is what keeps a slab from ending
// inside a chunk and making one decode serve two slabs. Asked as ChunksTouched, which is the question
// a walk actually puts to the plan: n channels from the start is one chunk, and one more is two.
void TestASlabIsCountedInChunksOfTheSpectralAxis() {
    const auto image = MakeImage(512, 520, 64);
    const auto geometry = MakeGeometry(256, 260, 8, AxisRole::spatial_y);

    const auto holds = [&](const Range& spectral, std::uint64_t channels, const std::string& what) {
        const auto plan = Plan(image, geometry, spectral, ReadOptions{});
        using carta::zarr::internal::SelectionChannel;
        Require(plan.ChunksTouched(SelectionChannel{0}, SelectionChannel{channels}) == 1,
                what + ": that many channels is one chunk");
        Require(plan.ChunksTouched(SelectionChannel{0}, SelectionChannel{channels + 1}) == 2,
                what + ": one more than that is two");
    };

    holds(Range{0, 64, 1}, 8, "eight channels to a chunk, read every one");
    holds(Range{0, 32, 2}, 4, "eight channels to a chunk, every second one selected, is four");
    holds(Range{0, 8, 8}, 1, "a stride of a whole chunk selects one channel from each");
    holds(Range{0, 4, 16}, 1, "a stride wider than a chunk still selects one, never none");
}

// layer_chunks is the chunks in one spectral layer, and band_rows is how many chunk rows of it one
// read may hold. Both feed the progress a caller sees, so a wrong one is a bar that lies.
void TestALayerIsCountedInWholeChunks() {
    const auto image = MakeImage(512, 520, 32);
    // Two chunks along each spatial axis.
    const auto geometry = MakeGeometry(256, 260, 2, AxisRole::spatial_y);
    const auto plan = Plan(image, geometry, Range{0, 32, 1}, ReadOptions{});
    Require(plan.layer_chunks == 4, "two chunks each way is four to a layer");

    // An axis that does not divide by its chunk still counts the partial chunk.
    const auto ragged = Plan(MakeImage(513, 520, 32), geometry, Range{0, 32, 1}, ReadOptions{});
    Require(ragged.layer_chunks == 6, "513 pixels over a 256-wide chunk is three, not two");
}

// A budget below one chunk row does not produce a band of zero rows, which would read nothing and
// never advance.
void TestABandIsNeverEmpty() {
    const auto image = MakeImage(512, 520, 32);
    const auto geometry = MakeGeometry(256, 260, 2, AxisRole::spatial_y);
    ReadOptions tiny;
    tiny.read_budget_bytes = 1;
    const auto plan = Plan(image, geometry, Range{0, 32, 1}, tiny);
    Require(plan.band_rows >= 1, "a band holds at least one chunk row however small the budget");
    Require(plan.SlabChannels(plan.layer_chunks) >= 1, "and a slab at least one channel");
}

// A region set that occupies no chunk at all -- a mask of zeroes -- costs nothing per layer, so
// nothing spatial bounds how many channels one emitted block holds. The budget used to answer "how
// many units of no chunks" with its own byte count, which is large enough to look right until the
// budget is smaller than the spectrum: here it is one byte, and the block was two channels.
void TestARegionSetOccupyingNothingBoundsNoBlock() {
    const auto image = MakeImage(512, 520, 32);
    const auto geometry = MakeGeometry(256, 260, 2, AxisRole::spatial_y);
    ReadOptions tiny;
    tiny.read_budget_bytes = 1;
    const auto plan = Plan(image, geometry, Range{0, 32, 1}, tiny);
    Require(plan.EmitChannels(0, 8, 0) == 32, "a block over no chunks was bounded by the read budget");
    // Anything that does occupy a chunk is still held to the read budget, which affords one layer.
    Require(plan.EmitChannels(1, 8, 0) == 2, "a block over one chunk was not held to one chunk layer");
}

// Sampling of zero would select nothing and divide by nothing; the plan floors it at one.
void TestSamplingFloorsAtOne() {
    const auto image = MakeImage(512, 520, 32);
    const auto geometry = MakeGeometry(256, 260, 2, AxisRole::spatial_y);
    Require(Plan(image, geometry, Range{0, 32, 1}, ReadOptions{}, 0).sample == 1,
            "a sample of zero reads every pixel rather than none");
    Require(Plan(image, geometry, Range{0, 32, 1}, ReadOptions{}, 4).sample == 4,
            "a sample the caller stated is kept");
}

// The sampled range is the one the pass asks the store for. Its edges are where an off-by-one costs
// a row of the image.
void TestSampledRangePicksTheMultiplesInside() {
    std::uint64_t start = 0;
    std::uint64_t count = 0;
    carta::zarr::internal::SampledRange(0, 10, 1, start, count);
    Require(start == 0 && count == 10, "a stride of one selects the whole range");

    carta::zarr::internal::SampledRange(0, 10, 3, start, count);
    Require(start == 0 && count == 4, "0, 3, 6 and 9 fall inside [0, 10)");

    carta::zarr::internal::SampledRange(4, 10, 3, start, count);
    Require(start == 6 && count == 2, "the first multiple of three at or after 4 is 6, then 9");

    carta::zarr::internal::SampledRange(7, 9, 3, start, count);
    Require(count == 0, "a range holding no multiple selects nothing rather than one");
}

// ---------------------------------------------------------------------------------------------
// Running the pass against pixels that were never written down.
//
// Everything below was unreachable while the only adapter read from a directory tree: a chunk read
// twice is invisible in a reduction's answer, no reduce test sets a cancellation or a deadline, and
// the largest committed fixture is 512x520x4.

// value = l * 1000 + m + channel * 1e6, so a pixel says where it came from.
float Encoded(const std::vector<std::uint64_t>& logical) {
    return static_cast<float>((logical.at(2) * 1000000) + (logical.at(0) * 1000) + logical.at(1));
}

struct Walked {
    std::uint64_t pixels = 0;
    double sum = 0.0;
    std::uint64_t slabs = 0;
};

Walked WalkEverything(const SyntheticPixelSource& source, const PassPlan& plan, const ReadOptions& options,
                      std::uint64_t channels) {
    Walked walked;
    std::uint64_t chunks_done = 0;
    const auto outcome = RunPass(
        source, plan, options, SelectionChannel{}, SelectionChannel{channels}, chunks_done, [](std::uint64_t) -> carta::zarr::Result<void> { return {}; },
        [&](const Slab& slab) {
            ++walked.slabs;
            for (std::uint64_t z = 0; z < slab.channel_count; ++z) {
                const float* plane = slab.pixels + (z * slab.stride_z);
                for (std::uint64_t v = 0; v < slab.v_count; ++v) {
                    const float* row = plane + (v * slab.stride_v);
                    for (std::uint64_t u = 0; u < slab.u_count; ++u) {
                        walked.sum += row[u * slab.stride_u];
                        ++walked.pixels;
                    }
                }
            }
        });
    Require(static_cast<bool>(outcome), "the pass failed");
    return walked;
}

// The invariant every piece of this was built around and nothing could assert: a pass decodes each
// chunk once. A fixture-driven test cannot see it, because a walk that reads a chunk twice and adds
// its pixels once still produces the right answer -- it is only slower.
void TestEachChunkIsReadOnce() {
    const auto image = MakeImage(512, 520, 32);
    const auto geometry = MakeGeometry(128, 130, 4, AxisRole::spatial_y);
    ReadOptions options;
    // Small enough that the pass has to split along the chunk rows and along the spectrum at once.
    options.read_budget_bytes = 4 * 128 * 130 * 4 * 4;
    const auto plan = Plan(image, geometry, Range{0, 32, 1}, options);
    SyntheticPixelSource source(image, geometry, Encoded);

    const auto walked = WalkEverything(source, plan, options, 32);
    Require(walked.slabs > 1, "this budget should have split the walk; if it did not, raise the image size");
    Require(source.most_hits_on_one_chunk() == 1,
            "a pass decodes each chunk once -- a slab that ends inside a chunk makes one decode serve "
            "two slabs and this is the only place that shows");
    Require(source.chunks_touched() == (512 / 128) * (520 / 130) * (32 / 4),
            "and it touches every chunk the selection covers, exactly the once");
}

// Every selected pixel, once, and the value the store would have given.
void TestThePassVisitsEveryPixelOnce() {
    const auto image = MakeImage(64, 40, 8);
    const auto geometry = MakeGeometry(16, 20, 2, AxisRole::spatial_y);
    ReadOptions options;
    options.read_budget_bytes = 16 * 20 * 2 * 4;
    const auto plan = Plan(image, geometry, Range{0, 8, 1}, options);
    SyntheticPixelSource source(image, geometry, Encoded);

    const auto walked = WalkEverything(source, plan, options, 8);
    Require(walked.pixels == 64ULL * 40ULL * 8ULL, "every pixel of the selection, once");

    double expected = 0.0;
    for (std::uint64_t z = 0; z < 8; ++z) {
        for (std::uint64_t l = 0; l < 64; ++l) {
            for (std::uint64_t m = 0; m < 40; ++m) {
                expected += Encoded({l, m, z, 0, 0});
            }
        }
    }
    Require(std::abs(walked.sum - expected) <= 1e-6 * expected, "and the values the store would have given");
}

// The pixel mask is folded in before the visitor sees anything, which is what lets every reduction
// treat a flagged pixel as a NaN without knowing there was a flag.
void TestAFlaggedPixelArrivesAsNaN() {
    const auto image = MakeImage(32, 20, 4, true);
    const auto geometry = MakeGeometry(16, 20, 2, AxisRole::spatial_y);
    ReadOptions options;
    const auto plan = Plan(image, geometry, Range{0, 4, 1}, options);
    Require(plan.apply_mask, "an image with a flag applies it by default");

    SyntheticPixelSource source(image, geometry, Encoded);
    // Every third pixel along l is bad.
    source.set_flags([](const std::vector<std::uint64_t>& logical) { return (logical.at(0) % 3) != 0; });

    std::uint64_t good = 0;
    std::uint64_t bad = 0;
    std::uint64_t chunks_done = 0;
    const auto outcome = RunPass(
        source, plan, options, SelectionChannel{}, SelectionChannel{4}, chunks_done, [](std::uint64_t) -> carta::zarr::Result<void> { return {}; },
        [&](const Slab& slab) {
            for (std::uint64_t z = 0; z < slab.channel_count; ++z) {
                const float* plane = slab.pixels + (z * slab.stride_z);
                for (std::uint64_t v = 0; v < slab.v_count; ++v) {
                    const float* row = plane + (v * slab.stride_v);
                    for (std::uint64_t u = 0; u < slab.u_count; ++u) {
                        std::isnan(row[u * slab.stride_u]) ? ++bad : ++good;
                    }
                }
            }
        });
    Require(static_cast<bool>(outcome), "the pass failed");
    Require(source.mask_reads() == source.pixel_reads(), "a masked pass reads a flag for every slab");
    // l in [0, 32) has eleven multiples of three.
    Require(bad == 11ULL * 20ULL * 4ULL, "every flagged pixel arrives as NaN");
    Require(good == (32ULL - 11ULL) * 20ULL * 4ULL, "and no unflagged one does");
}

// Cancellation and the deadline are checked at every storage operation, and no reduce test sets
// either -- there was no way to reach the second read of a fixture small enough to run.
void TestCancellationStopsThePass() {
    const auto image = MakeImage(256, 260, 16);
    const auto geometry = MakeGeometry(64, 65, 2, AxisRole::spatial_y);
    ReadOptions options;
    options.read_budget_bytes = 64 * 65 * 2 * 4;
    int reads = 0;
    options.control.cancellation_requested = [&]() { return reads >= 2; };
    const auto plan = Plan(image, geometry, Range{0, 16, 1}, options);
    SyntheticPixelSource source(image, geometry, Encoded);

    std::uint64_t chunks_done = 0;
    const auto outcome = RunPass(
        source, plan, options, SelectionChannel{}, SelectionChannel{16}, chunks_done, [](std::uint64_t) -> carta::zarr::Result<void> { return {}; },
        [&](const Slab&) { ++reads; });
    Require(!outcome && outcome.error().code == carta::zarr::ErrorCode::cancelled,
            "a cancelled pass reports cancelled");
    Require(reads == 2, "and stops at the read after the one that asked");
}

void TestAnExpiredDeadlineStopsThePass() {
    const auto image = MakeImage(64, 40, 8);
    const auto geometry = MakeGeometry(16, 20, 2, AxisRole::spatial_y);
    ReadOptions options;
    options.control.deadline = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    const auto plan = Plan(image, geometry, Range{0, 8, 1}, options);
    SyntheticPixelSource source(image, geometry, Encoded);

    std::uint64_t chunks_done = 0;
    const auto outcome = RunPass(
        source, plan, options, SelectionChannel{}, SelectionChannel{8}, chunks_done, [](std::uint64_t) -> carta::zarr::Result<void> { return {}; },
        [](const Slab&) {});
    Require(!outcome && outcome.error().code == carta::zarr::ErrorCode::cancelled,
            "a pass past its deadline reports cancelled before reading anything");
    Require(source.pixel_reads() == 0, "and does not read");
}

// A failure from the source is the caller's failure, not a partial answer.
void TestAReadFailureStopsThePass() {
    const auto image = MakeImage(128, 130, 8);
    const auto geometry = MakeGeometry(32, 65, 2, AxisRole::spatial_y);
    ReadOptions options;
    options.read_budget_bytes = 32 * 65 * 2 * 4;
    const auto plan = Plan(image, geometry, Range{0, 8, 1}, options);
    SyntheticPixelSource source(image, geometry, Encoded);
    source.fail_read(2, carta::zarr::ErrorCode::io_error);

    int visits = 0;
    std::uint64_t chunks_done = 0;
    const auto outcome = RunPass(
        source, plan, options, SelectionChannel{}, SelectionChannel{8}, chunks_done, [](std::uint64_t) -> carta::zarr::Result<void> { return {}; },
        [&](const Slab&) { ++visits; });
    Require(!outcome && outcome.error().code == carta::zarr::ErrorCode::io_error,
            "the source's error is the pass's error");
    Require(visits == 1, "and nothing is visited after it");
}

// The size the design comments argue about, which no fixture reaches. The point is not the answer
// but that the splitting runs at all: a 4096-square plane over one budget is many bands.
void TestALargePlaneSplitsIntoBands() {
    const auto image = MakeImage(4096, 4096, 4);
    const auto geometry = MakeGeometry(512, 512, 1, AxisRole::spatial_y);
    ReadOptions options;
    // Eight chunks to a read, which is the floor chunk_blocks measured.
    options.read_budget_bytes = 8 * 512 * 512 * 4;
    const auto plan = Plan(image, geometry, Range{0, 4, 1}, options);
    SyntheticPixelSource source(image, geometry, Encoded);
    // This one is about the splitting, not the pixels, so it does not pay for them.
    source.set_constant(1.0F);

    std::uint64_t slabs = 0;
    std::uint64_t chunks_done = 0;
    const auto outcome = RunPass(
        source, plan, options, SelectionChannel{}, SelectionChannel{4}, chunks_done,
        [](std::uint64_t) -> carta::zarr::Result<void> { return {}; }, [&](const Slab&) { ++slabs; });
    Require(static_cast<bool>(outcome), "the pass failed on a large plane");
    Require(source.elements_read() == 4096ULL * 4096ULL * 4ULL, "every pixel of a sixty-seven megapixel cube");
    Require(slabs > 8, "and it took many reads to do it");
    Require(source.most_hits_on_one_chunk() == 1, "still decoding each chunk once at this size");
    Require(source.chunks_touched() == 8ULL * 8ULL * 4ULL, "over every chunk of it");
}

// A chunk row wider than the budget is read in pieces along it, not as one band a row deep. Without
// this the smallest read a whole-plane pass makes is a full chunk row: eight chunks against a
// budget of two here, and on a 32768-wide image of 4 MiB chunks, 256 MiB against 64.
void TestAChunkRowWiderThanTheBudgetIsSplitAlongIt() {
    const auto image = MakeImage(16, 64, 2);
    // m is fastest, so u runs along the 64 pixels: eight chunks to a row.
    const auto geometry = MakeGeometry(8, 8, 1, AxisRole::spatial_y);
    ReadOptions options;
    options.read_budget_bytes = 2 * 8 * 8 * 4;
    const auto plan = Plan(image, geometry, Range{0, 2, 1}, options);
    Require(plan.u_length == 64, "this test needs u along the wide axis");
    SyntheticPixelSource source(image, geometry, Encoded);

    const auto walked = WalkEverything(source, plan, options, 2);
    for (const auto size : source.pixel_destinations()) {
        Require(size <= 2U * 8U * 8U, "a read held more than its budget's two chunks");
    }
    Require(walked.pixels == 16ULL * 64ULL * 2ULL, "every pixel of the selection, once");
    Require(source.most_hits_on_one_chunk() == 1, "and each chunk decoded once");
    Require(source.chunks_touched() == 2ULL * 8ULL * 2ULL, "over every chunk of it");
}

// Sampling steps over pixels, and a piece of a row it steps over entirely is not read at all -- but
// still counted, so that progress reaches the whole.
void TestASampledRowSplitsAlongItToo() {
    const auto image = MakeImage(16, 64, 1);
    const auto geometry = MakeGeometry(8, 8, 1, AxisRole::spatial_y);
    ReadOptions options;
    options.read_budget_bytes = 1 * 8 * 8 * 4;
    const auto plan = Plan(image, geometry, Range{0, 1, 1}, options, 16);
    SyntheticPixelSource source(image, geometry, Encoded);

    Walked walked;
    std::uint64_t chunks_done = 0;
    const auto outcome = RunPass(
        source, plan, options, SelectionChannel{}, SelectionChannel{1}, chunks_done,
        [](std::uint64_t) -> carta::zarr::Result<void> { return {}; },
        [&](const Slab& slab) { walked.pixels += slab.u_count * slab.v_count; });
    Require(static_cast<bool>(outcome), "the pass failed");
    // u = 0, 16, 32, 48 and v = 0: one pixel in every other chunk along the row.
    Require(walked.pixels == 4, "every sixteenth pixel each way");
    Require(source.pixel_reads() == 4, "and only the pieces holding one are read");
    Require(chunks_done == plan.layer_chunks, "while every chunk of the plane is counted");
}

}  // namespace

int main() {
    try {
        TestTheInnerAxisFollowsTheStore();
        TestAMaskCostsWhatTheFlagCosts();
        TestTheCallersCeilingWins();
        TestASlabIsCountedInChunksOfTheSpectralAxis();
        TestALayerIsCountedInWholeChunks();
        TestABandIsNeverEmpty();
        TestARegionSetOccupyingNothingBoundsNoBlock();
        TestSamplingFloorsAtOne();
        TestSampledRangePicksTheMultiplesInside();
        TestEachChunkIsReadOnce();
        TestThePassVisitsEveryPixelOnce();
        TestAFlaggedPixelArrivesAsNaN();
        TestCancellationStopsThePass();
        TestAnExpiredDeadlineStopsThePass();
        TestAReadFailureStopsThePass();
        TestALargePlaneSplitsIntoBands();
        TestAChunkRowWiderThanTheBudgetIsSplitAlongIt();
        TestASampledRowSplitsAlongItToo();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pass test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
