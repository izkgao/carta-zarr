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

#include "reduce/pass.h"

#include "axis_map.h"
#include "support/check.h"
#include "support/synthetic_pixel_source.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {

using carta::zarr::AxisRole;
using carta::zarr::ChunkGeometry;
using carta::zarr::ImageDescriptor;
using carta::zarr::Range;
using carta::zarr::ReadOptions;
using carta::zarr::internal::CheckedPlanes;
using carta::zarr::internal::MapAxes;
using carta::zarr::internal::PassOverFootprints;
using carta::zarr::internal::PassOverPlane;
using carta::zarr::internal::PassPlan;
using carta::zarr::internal::PlanPass;
using carta::zarr::internal::SelectionChannel;
using carta::zarr::internal::Slab;
using carta::zarr::internal::SlabFootprint;
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

constexpr const char* kCancelled = "The test's pass was cancelled";

// The whole selection as one run, with nobody asking how far along it is.
template <typename Visit>
carta::zarr::Result<void> WalkWhole(const SyntheticPixelSource& source, const PassPlan& plan,
                                    const ReadOptions& options, Visit&& visit) {
    auto pass = PassOverPlane(source, plan, options, kCancelled);
    return pass.Whole([](double) { return true; }, visit);
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

    // Sampled, a layer is the chunks the sample has a pixel in. Every sixteenth pixel of a 64-pixel
    // axis in chunks of 8 falls in chunks 0, 2, 4 and 6, and of a 16-pixel one in chunk 0 alone: four
    // of the sixteen there are.
    const auto sampled = Plan(MakeImage(16, 64, 1), MakeGeometry(8, 8, 1, AxisRole::spatial_y), Range{0, 1, 1},
                              ReadOptions{}, 16);
    Require(sampled.layer_chunks == 4, "a sampled layer of " + std::to_string(sampled.layer_chunks) +
                                           " chunks, not the four the sample reads");
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

Walked WalkEverything(const SyntheticPixelSource& source, const PassPlan& plan, const ReadOptions& options) {
    Walked walked;
    const auto outcome = WalkWhole(source, plan, options, [&](const Slab& slab) {
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

    const auto walked = WalkEverything(source, plan, options);
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

    const auto walked = WalkEverything(source, plan, options);
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
    const auto outcome = WalkWhole(source, plan, options, [&](const Slab& slab) {
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

    const auto outcome = WalkWhole(source, plan, options, [&](const Slab&) { ++reads; });
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

    const auto outcome = WalkWhole(source, plan, options, [](const Slab&) {});
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
    const auto outcome = WalkWhole(source, plan, options, [&](const Slab&) { ++visits; });
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
    const auto outcome = WalkWhole(source, plan, options, [&](const Slab&) { ++slabs; });
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

    const auto walked = WalkEverything(source, plan, options);
    for (const auto size : source.pixel_destinations()) {
        Require(size <= 2U * 8U * 8U, "a read held more than its budget's two chunks");
    }
    Require(walked.pixels == 16ULL * 64ULL * 2ULL, "every pixel of the selection, once");
    Require(source.most_hits_on_one_chunk() == 1, "and each chunk decoded once");
    Require(source.chunks_touched() == 2ULL * 8ULL * 2ULL, "over every chunk of it");
}

// Sampling steps over pixels, and a chunk it steps over entirely is neither read nor counted: what a
// pass is a fraction of is the chunks it decodes.
void TestASampledRowSplitsAlongItToo() {
    const auto image = MakeImage(16, 64, 1);
    const auto geometry = MakeGeometry(8, 8, 1, AxisRole::spatial_y);
    ReadOptions options;
    options.read_budget_bytes = 1 * 8 * 8 * 4;
    const auto plan = Plan(image, geometry, Range{0, 1, 1}, options, 16);
    SyntheticPixelSource source(image, geometry, Encoded);

    Walked walked;
    std::vector<double> reported;
    auto pass = PassOverPlane(source, plan, options, kCancelled);
    const auto outcome = pass.Whole(
        [&](double fraction) {
            reported.push_back(fraction);
            return true;
        },
        [&](const Slab& slab) { walked.pixels += slab.u_count * slab.v_count; });
    Require(static_cast<bool>(outcome), "the pass failed");
    // u = 0, 16, 32, 48 and v = 0: one pixel in every other chunk along the row.
    Require(walked.pixels == 4, "every sixteenth pixel each way");
    Require(source.pixel_reads() == 4, "and only the pieces holding one are read");
    // Sixteen chunks to the plane, and four the sample has a pixel in: every other one of the first
    // row of eight, and none of the second. Each read after the first is told the chunks before it of
    // those four. Counting the chunks stepped over as well, as this used to, told it 2/16, 4/16 and
    // 6/16 and never reached the whole: the last eight were passed after the last read.
    Require(reported == std::vector<double>{1.0 / 4, 2.0 / 4, 3.0 / 4},
            "the fractions were not of the chunks the sample reads");
}

// ---------------------------------------------------------------------------------------------
// The blocks a pass cuts, and what it says as it fills them.
//
// These were the block emitter's, driven by a walk that read nothing and only said how far it got,
// because the emitter and the walk were two things and the caller joined them. The pass is one
// thing now, so they read real pixels -- and can say what the emitter's could not: that the
// channels a block's walk reads are the channels the block is handed over as.

// One hand-over, as the sink saw it.
struct Handed {
    SelectionChannel first_channel;
    std::uint64_t length = 0;
    bool complete = false;
    double completeness = 0.0;
};

// A footprint as a region set's occupancy would cut it. The pass asks nothing of one but its slab,
// and hands it back to the visitor beside every read made over it.
struct Footprint {
    SlabFootprint slab;
    int which = 0;
};

// The image channel a slab's first pixel came from, which Encoded writes into it.
std::uint64_t ChannelOf(const Slab& slab, std::uint64_t z) {
    return static_cast<std::uint64_t>(slab.pixels[z * slab.stride_z] / 1.0e6F);
}

// A block is never cut inside a spectral chunk, however small a block the caller asks for: a decode
// serving two blocks would split one read's results across them. So a hint of one channel against a
// chunk four deep still emits blocks of four.
//
// And what a block's walk reads is the block it is handed over as: a later block's slab starts at
// zero, counted from the block, and holds that block's channels rather than the first block's --
// which is e97066f, where every block after the first re-read the first one's.
void TestBlocksAreCutOnChunkBoundaries() {
    const auto image = MakeImage(64, 64, 32);
    const auto geometry = MakeGeometry(64, 64, 4, AxisRole::spatial_y);
    const ReadOptions options;
    const auto plan = Plan(image, geometry, Range{0, 32, 1}, options);
    SyntheticPixelSource source(image, geometry, Encoded);

    std::vector<Handed> handed;
    std::vector<std::uint64_t> relative;
    std::vector<std::uint64_t> read_channels;
    auto pass = PassOverPlane(source, plan, options, kCancelled);
    const auto outcome = pass.InBlocks(
        sizeof(double), 1, [](std::uint64_t) {},
        [&](const Slab& slab) {
            relative.push_back(slab.first_channel.index);
            read_channels.push_back(ChannelOf(slab, 0));
        },
        [&](SelectionChannel first, std::uint64_t length, bool complete, double completeness) {
            handed.push_back({first, length, complete, completeness});
            return true;
        });
    Require(static_cast<bool>(outcome), "an uncancelled pass should succeed");

    Require(handed.size() == 8,
            "32 channels in blocks of a 4-deep chunk is eight blocks, not " + std::to_string(handed.size()));
    Require(read_channels.size() == handed.size(), "every block should have been read in one slab");
    SelectionChannel expected{};
    for (std::size_t i = 0; i < handed.size(); ++i) {
        Require(handed.at(i).first_channel == expected,
                "block " + std::to_string(i) + " should start at " + std::to_string(expected.index));
        Require(handed.at(i).length == 4, "block " + std::to_string(i) + " should hold four channels");
        Require(handed.at(i).complete && handed.at(i).completeness == 1.0,
                "a block read in one slab is handed over once, finished");
        Require(relative.at(i) == 0, "a slab's first channel is counted from its block");
        Require(read_channels.at(i) == expected.index, "block " + std::to_string(i) + " read channel " +
                                                           std::to_string(read_channels.at(i)) +
                                                           " rather than its own first");
        expected = expected + 4U;
    }
    Require(expected == SelectionChannel{32}, "the blocks should tile the selection");
}

// A 64 x 64 plane of 16 x 16 chunks is a layer of sixteen, eight channels deep in one chunk, under a
// budget of four chunks: one block, read as four bands of one chunk row each.
PassPlan FourBandsOfOneBlock(const ImageDescriptor& image, const ChunkGeometry& geometry, ReadOptions& options) {
    options.read_budget_bytes = 4 * 16 * 16 * 8 * sizeof(float);
    return Plan(image, geometry, Range{0, 8, 1}, options);
}

// A block whose walk takes more than one read is handed over as it fills, before every read after
// its first, and what it says about itself is the fraction of its own chunks that are in it --
// counted in the layer the emit budget was spent against, which is the same number for both.
void TestAPartFilledBlockReportsItsOwnChunks() {
    const auto image = MakeImage(64, 64, 8);
    const auto geometry = MakeGeometry(16, 16, 8, AxisRole::spatial_y);
    ReadOptions options;
    const auto plan = FourBandsOfOneBlock(image, geometry, options);
    SyntheticPixelSource source(image, geometry, Encoded);

    std::vector<Handed> handed;
    auto pass = PassOverPlane(source, plan, options, kCancelled);
    const auto outcome = pass.InBlocks(
        sizeof(double), 0, [](std::uint64_t) {}, [](const Slab&) {},
        [&](SelectionChannel first, std::uint64_t length, bool complete, double completeness) {
            handed.push_back({first, length, complete, completeness});
            return true;
        });
    Require(static_cast<bool>(outcome), "an uncancelled pass should succeed");
    Require(source.pixel_reads() == 4, "four bands is four reads, not " + std::to_string(source.pixel_reads()));

    Require(handed.size() == 4,
            "three reports and one finish is four hand-overs, not " + std::to_string(handed.size()));
    Require(!handed.at(0).complete && handed.at(0).completeness == 4.0 / 16.0, "four chunks of sixteen");
    Require(!handed.at(1).complete && handed.at(1).completeness == 8.0 / 16.0, "eight chunks of sixteen");
    Require(!handed.at(2).complete && handed.at(2).completeness == 12.0 / 16.0, "twelve chunks of sixteen");
    Require(handed.at(3).complete && handed.at(3).completeness == 1.0,
            "a finished block says one exactly, not sixteen sixteenths");
    for (const auto& one : handed) {
        Require(one.first_channel == SelectionChannel{} && one.length == 8, "every hand-over describes the same block");
    }
}

// A sink saying no stops the pass there, whether it says it to a part-filled block or to a finished
// one, and the pass reports cancelled -- with the message it was made with -- rather than a short
// answer that looks complete.
void TestASinkThatSaysNoCancels() {
    // Refusing the first finished block: the second is never read.
    {
        const auto image = MakeImage(64, 64, 32);
        const auto geometry = MakeGeometry(64, 64, 4, AxisRole::spatial_y);
        const ReadOptions options;
        const auto plan = Plan(image, geometry, Range{0, 32, 1}, options);
        SyntheticPixelSource source(image, geometry, Encoded);
        std::uint64_t hand_overs = 0;
        auto pass = PassOverPlane(source, plan, options, kCancelled);
        const auto outcome = pass.InBlocks(
            sizeof(double), 1, [](std::uint64_t) {}, [](const Slab&) {},
            [&](SelectionChannel, std::uint64_t, bool, double) {
                ++hand_overs;
                return false;
            });
        Require(!outcome, "a sink returning false should fail the pass");
        Require(outcome.error().code == carta::zarr::ErrorCode::cancelled, "and it should report cancelled");
        Require(outcome.error().message == kCancelled, "in the words the pass was made with");
        Require(hand_overs == 1 && source.pixel_reads() == 1, "and it should stop at the first block");
    }

    // Refusing a part-filled one: the rest of that block is not read either.
    {
        const auto image = MakeImage(64, 64, 8);
        const auto geometry = MakeGeometry(16, 16, 8, AxisRole::spatial_y);
        ReadOptions options;
        const auto plan = FourBandsOfOneBlock(image, geometry, options);
        SyntheticPixelSource source(image, geometry, Encoded);
        auto pass = PassOverPlane(source, plan, options, kCancelled);
        const auto outcome = pass.InBlocks(
            sizeof(double), 0, [](std::uint64_t) {}, [](const Slab&) {},
            [](SelectionChannel, std::uint64_t, bool, double) { return false; });
        Require(!outcome && outcome.error().code == carta::zarr::ErrorCode::cancelled,
                "refusing a part-filled block should cancel too");
        Require(source.pixel_reads() == 1, "and the rest of that block should not be read");
    }
}

// A run is told how far along it is before every read after its first, and saying no there stops it
// the same way, before the read it was asked about.
void TestARunThatSaysNoCancels() {
    const auto image = MakeImage(64, 64, 8);
    const auto geometry = MakeGeometry(16, 16, 8, AxisRole::spatial_y);
    ReadOptions options;
    const auto plan = FourBandsOfOneBlock(image, geometry, options);
    SyntheticPixelSource source(image, geometry, Encoded);

    std::vector<double> reported;
    auto pass = PassOverPlane(source, plan, options, kCancelled);
    const auto outcome = pass.Whole(
        [&](double fraction) {
            reported.push_back(fraction);
            return false;
        },
        [](const Slab&) {});
    Require(!outcome && outcome.error().code == carta::zarr::ErrorCode::cancelled,
            "a run told no should report cancelled");
    Require(outcome.error().message == kCancelled, "in the words the pass was made with");
    Require(reported == std::vector<double>{4.0 / 16.0}, "having been told once, after the first of four reads");
    Require(source.pixel_reads() == 1, "and read nothing after it");
}

// The hint is the caller's and the chunk alignment is the plan's, but the emit budget is the
// library's and it wins when a channel is expensive enough. Asked for the whole selection in one
// block, a pass whose channels cost 64 MiB each gets one at a time.
void TestTheBudgetLowersTheCallersHint() {
    const auto image = MakeImage(64, 64, 64);
    const auto geometry = MakeGeometry(64, 64, 1, AxisRole::spatial_y);
    const ReadOptions options;
    const auto plan = Plan(image, geometry, Range{0, 64, 1}, options);

    const auto lengths = [&](std::size_t bytes_per_channel) {
        SyntheticPixelSource source(image, geometry, Encoded);
        source.set_constant(1.0F);
        std::vector<std::uint64_t> handed;
        auto pass = PassOverPlane(source, plan, options, kCancelled);
        const auto outcome = pass.InBlocks(
            bytes_per_channel, 64, [](std::uint64_t) {}, [](const Slab&) {},
            [&](SelectionChannel, std::uint64_t length, bool, double) {
                handed.push_back(length);
                return true;
            });
        Require(static_cast<bool>(outcome), "an uncancelled pass should succeed");
        return handed;
    };
    Require(lengths(sizeof(double)) == std::vector<std::uint64_t>{64}, "a cheap channel takes the hint as given");
    const auto expensive = lengths(64U << 20U);
    Require(expensive.size() == 64 && expensive.front() == 1,
            "a channel costing 64 MiB should be emitted one at a time, not " + std::to_string(expensive.front()));
}

// The footprints of one block share its count of reads: a block made of two footprints, each taken
// in a single read, is handed over once between them and once finished -- not once per footprint
// -- and the fraction it reports is of the chunks the footprints occupy together.
//
// Each read is handed to the visitor beside the footprint it was read over.
void TestTheFootprintsOfABlockShareItsReads() {
    const auto image = MakeImage(64, 64, 4);
    const auto geometry = MakeGeometry(32, 32, 4, AxisRole::spatial_y);
    const ReadOptions options;
    const auto plan = Plan(image, geometry, Range{0, 4, 1}, options);
    SyntheticPixelSource source(image, geometry, Encoded);

    // Opposite corners of a 2 x 2 grid of chunks. u is m, v is l, because m varies fastest.
    std::vector<Footprint> footprints(2);
    footprints.at(0).slab = SlabFootprint{0, 32, 1, 0, 32, 1, 1};
    footprints.at(0).which = 0;
    footprints.at(1).slab = SlabFootprint{32, 32, 1, 32, 32, 1, 1};
    footprints.at(1).which = 1;

    std::vector<Handed> handed;
    std::vector<int> visited;
    auto pass = PassOverFootprints(source, plan, options, footprints, kCancelled);
    const auto outcome = pass.InBlocks(
        sizeof(double), 0, [](std::uint64_t) {},
        [&](const Footprint& footprint, const Slab& slab) {
            visited.push_back(footprint.which);
            const auto l = footprint.slab.v_start;
            const auto m = footprint.slab.u_start;
            Require(slab.pixels[0] == Encoded({l, m, 0, 0, 0}),
                    "a slab arrived beside a footprint it was not read over");
        },
        [&](SelectionChannel first, std::uint64_t length, bool complete, double completeness) {
            handed.push_back({first, length, complete, completeness});
            return true;
        });
    Require(static_cast<bool>(outcome), "an uncancelled pass should succeed");
    Require(visited == std::vector<int>{0, 1}, "each footprint read once, in order");
    Require(handed.size() == 2, "one report between two reads and one finish, not " + std::to_string(handed.size()));
    Require(!handed.at(0).complete && handed.at(0).completeness == 0.5,
            "the first footprint is one chunk of the two they occupy, not of the plane's four");
    Require(handed.at(1).complete && handed.at(1).completeness == 1.0, "and the block finishes complete");
}

// Footprints occupying no chunk at all -- what a mask of zeroes leaves a reduction with.
//
// Reachable from the public interface: a raster mask of zeroes selects nothing, which the request
// checks do not refuse -- a caller with a region that this frame happens not to cover is asking a
// legitimate question, and the answer is zero counts and no extrema.
//
// The layer is zero, and its two uses want opposite things about it. Spending the emit budget, zero
// is the honest answer: there is nothing to read, so the whole selection is one block. As the
// denominator of a part-filled block's completeness it must never be zero.
void TestFootprintsOccupyingNothing() {
    const auto image = MakeImage(64, 64, 64);
    const auto geometry = MakeGeometry(64, 64, 1, AxisRole::spatial_y);
    ReadOptions options;
    // A budget of four chunks, so that a layer of one chunk is visibly cut and a layer of none is
    // visibly not.
    options.read_budget_bytes = 4 * 64 * 64 * sizeof(float);
    const auto plan = Plan(image, geometry, Range{0, 64, 1}, options);

    const auto blocks = [&](const std::vector<Footprint>& footprints, std::uint64_t& reads) {
        SyntheticPixelSource source(image, geometry, Encoded);
        source.set_constant(1.0F);
        std::vector<Handed> handed;
        auto pass = PassOverFootprints(source, plan, options, footprints, kCancelled);
        const auto outcome = pass.InBlocks(
            sizeof(double), 0, [](std::uint64_t) {}, [](const Footprint&, const Slab&) {},
            [&](SelectionChannel first, std::uint64_t length, bool complete, double completeness) {
                handed.push_back({first, length, complete, completeness});
                return true;
            });
        Require(static_cast<bool>(outcome), "an uncancelled pass should succeed");
        reads = source.pixel_reads();
        return handed;
    };

    std::vector<Footprint> one(1);
    one.at(0).slab = SlabFootprint{0, 64, 1, 0, 64, 1, 1};
    std::uint64_t reads = 0;
    const auto occupied = blocks(one, reads);
    Require(occupied.front().length == 4, "a layer of one chunk should be cut by a four-chunk budget, not into " +
                                              std::to_string(occupied.front().length));

    const auto empty = blocks({}, reads);
    Require(empty.size() == 1, "a single block should be handed over once, not " + std::to_string(empty.size()));
    Require(empty.at(0).complete && empty.at(0).completeness == 1.0, "the one hand-over should be complete");
    Require(empty.at(0).length == 64, "and should cover the whole selection");
    Require(reads == 0, "having read nothing");
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
        TestBlocksAreCutOnChunkBoundaries();
        TestAPartFilledBlockReportsItsOwnChunks();
        TestASinkThatSaysNoCancels();
        TestARunThatSaysNoCancels();
        TestTheBudgetLowersTheCallersHint();
        TestTheFootprintsOfABlockShareItsReads();
        TestFootprintsOccupyingNothing();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "pass test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
