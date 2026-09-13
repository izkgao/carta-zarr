/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Multi-region spectral reduction against the same fixture the pixel reads use. Every expectation
// here is computed from the fixture's own encoding rather than from a previous run of the library,
// so a reduction that reads the right pixels in the wrong order, double-counts a region that spans
// two chunks, or silently keeps a flagged pixel gets a different answer than the oracle.

#include <carta-zarr/carta_zarr.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The same image written both ways round: XRADIO puts m last, so a plane is contiguous along m,
// and the other one puts l last. Everything below runs against both, because the walk decides what
// to ask the store for from which axis it varies fastest -- a decision only one of the two can
// exercise at a time, and the wrong answer for either is a silently transposed read.
const char* const kFixtures[]{CARTA_ZARR_PIXEL_FIXTURE, CARTA_ZARR_PIXEL_FIXTURE_L_FASTEST};

constexpr std::uint64_t kL = 4;
constexpr std::uint64_t kM = 5;
constexpr std::uint64_t kFrequency = 2;
constexpr std::uint64_t kPolarization = 3;

// The chunk the generator deletes covers frequency 1, polarization 2 and l in [2, 4).
bool InMissingChunk(std::uint64_t l, std::uint64_t frequency, std::uint64_t polarization) {
    return frequency == 1 && polarization == 2 && l >= 2;
}

double ExpectedValue(std::uint64_t l, std::uint64_t m, std::uint64_t frequency, std::uint64_t polarization) {
    return static_cast<double>((frequency * 1000) + (polarization * 100) + (l * 10) + m);
}

// The generator marks a pixel bad where (l + m) is a multiple of three.
bool ExpectedFlag(std::uint64_t l, std::uint64_t m) {
    return ((l + m) % 3) != 0;
}

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void RequireClose(double actual, double expected, const std::string& message) {
    if (std::isnan(expected)) {
        Require(std::isnan(actual), message + ": expected NaN, got " + std::to_string(actual));
        return;
    }
    Require(std::isfinite(actual) && std::abs(actual - expected) <= 1e-9 * (1.0 + std::abs(expected)),
            message + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

carta::zarr::Image OpenSky(const char* fixture, const carta::zarr::OpenOptions& options = {}) {
    Require(std::filesystem::exists(fixture),
            "the pixel fixture is missing; run tests/data/generate_zarr_fixtures.py");
    const auto context = carta::zarr::Context::Create(options);
    Require(static_cast<bool>(context), "Context::Create failed");
    auto dataset = carta::zarr::Dataset::Open(context.value(), fixture);
    Require(static_cast<bool>(dataset), "Dataset::Open failed on the pixel fixture");
    auto image = dataset.value().OpenImage("SKY");
    Require(static_cast<bool>(image), "SKY could not be opened");
    return image.value();
}

// The oracle: what one region over one plane is worth, derived from the fixture's encoding alone.
struct Totals {
    double num_pixels = 0.0;
    double nan_count = 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double min = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
};

Totals Expected(const carta::zarr::RegionMask& region, std::uint64_t frequency, std::uint64_t polarization) {
    Totals totals;
    for (std::uint64_t y = region.y_start; y < region.y_start + region.height; ++y) {
        for (std::uint64_t x = region.x_start; x < region.x_start + region.width; ++x) {
            if (region.mask != nullptr &&
                region.mask[((y - region.y_start) * region.width) + (x - region.x_start)] == 0) {
                continue;
            }
            // A flagged pixel and a pixel in the deleted chunk are both absent, and the reduction
            // must not be able to tell them apart.
            if (!ExpectedFlag(x, y) || InMissingChunk(x, frequency, polarization)) {
                totals.nan_count += 1.0;
                continue;
            }
            const double value = ExpectedValue(x, y, frequency, polarization);
            totals.num_pixels += 1.0;
            totals.sum += value;
            totals.sum_sq += value * value;
            totals.min = std::isnan(totals.min) ? value : std::min(totals.min, value);
            totals.max = std::isnan(totals.max) ? value : std::max(totals.max, value);
        }
    }
    return totals;
}

carta::zarr::StatisticSet AllStatistics() {
    return carta::zarr::Statistic::num_pixels | carta::zarr::Statistic::nan_count |
           carta::zarr::Statistic::sum | carta::zarr::Statistic::sum_sq | carta::zarr::Statistic::min |
           carta::zarr::Statistic::max;
}

// One reduction, flattened into [region][statistic][channel] over the whole spectral selection, so
// that a test can compare values without also re-implementing the block bookkeeping.
struct Collected {
    std::size_t region_count = 0;
    std::size_t channel_count = 0;
    std::vector<carta::zarr::Statistic> statistics;
    std::vector<double> values;
    std::vector<std::uint64_t> block_lengths;

    double At(std::size_t region, carta::zarr::Statistic statistic, std::size_t channel) const {
        std::size_t slot = statistics.size();
        for (std::size_t i = 0; i < statistics.size(); ++i) {
            if (statistics.at(i) == statistic) {
                slot = i;
            }
        }
        if (slot == statistics.size()) {
            throw std::runtime_error("the block did not report the requested statistic");
        }
        return values.at(((region * statistics.size()) + slot) * channel_count + channel);
    }

    // What each unfinished hand-over of a block said region 0 had counted so far, against the
    // channel the block starts at.
    std::vector<std::pair<std::uint64_t, double>> partial_counts;
};

Collected Collect(const carta::zarr::Image& sky, const carta::zarr::SpectralReduceRequest& request,
                  const carta::zarr::ReadOptions& options) {
    Collected collected;
    collected.region_count = request.region_count;
    collected.channel_count = static_cast<std::size_t>(request.spectral.count);
    std::uint64_t next_channel = 0;
    const auto result = sky.ReduceSpectral(request, [&](const carta::zarr::SpectralBlock& block) {
        Require(block.first_channel == next_channel, "blocks should tile the spectral selection in order");
        Require(block.channel_count > 0, "a block should carry at least one channel");
        Require(block.value_count ==
                    collected.region_count * block.statistic_count * static_cast<std::size_t>(block.channel_count),
                "a block's value count should match its strides");
        if (!block.complete) {
            Require(block.completeness > 0.0 && block.completeness < 1.0,
                    "an unfinished block should report a fraction of itself");
            for (std::size_t s = 0; s < block.statistic_count; ++s) {
                if (block.statistics[s] == carta::zarr::Statistic::num_pixels) {
                    collected.partial_counts.emplace_back(
                        block.first_channel, block.values[s * block.statistic_stride]);
                }
            }
            return true;
        }
        Require(block.completeness == 1.0, "a finished block is all of itself");
        next_channel += block.channel_count;
        collected.block_lengths.push_back(block.channel_count);
        if (collected.statistics.empty()) {
            collected.statistics.assign(block.statistics, block.statistics + block.statistic_count);
            collected.values.assign(collected.region_count * block.statistic_count * collected.channel_count, 0.0);
        }
        for (std::size_t r = 0; r < collected.region_count; ++r) {
            for (std::size_t s = 0; s < block.statistic_count; ++s) {
                for (std::uint64_t c = 0; c < block.channel_count; ++c) {
                    const double value =
                        block.values[(r * block.region_stride) + (s * block.statistic_stride) + c];
                    collected.values.at(((r * block.statistic_count) + s) * collected.channel_count +
                                        static_cast<std::size_t>(block.first_channel + c)) = value;
                }
            }
        }
        return true;
    }, options);
    Require(static_cast<bool>(result),
            std::string("the reduction failed: ") + (result.has_value() ? "" : result.error().message));
    Require(next_channel == request.spectral.count, "the blocks should cover the whole spectral selection");
    return collected;
}

Collected Collect(const carta::zarr::Image& sky, const carta::zarr::SpectralReduceRequest& request) {
    return Collect(sky, request, carta::zarr::ReadOptions{});
}

void CheckAgainstOracle(const Collected& collected, const std::vector<carta::zarr::RegionMask>& regions,
                        std::uint64_t polarization, const std::string& label) {
    for (std::size_t r = 0; r < regions.size(); ++r) {
        for (std::uint64_t f = 0; f < collected.channel_count; ++f) {
            const auto expected = Expected(regions.at(r), f, polarization);
            const std::string where =
                label + " region " + std::to_string(r) + " channel " + std::to_string(f);
            RequireClose(collected.At(r, carta::zarr::Statistic::num_pixels, f), expected.num_pixels,
                         where + " num_pixels");
            RequireClose(collected.At(r, carta::zarr::Statistic::nan_count, f), expected.nan_count,
                         where + " nan_count");
            RequireClose(collected.At(r, carta::zarr::Statistic::sum, f), expected.sum, where + " sum");
            RequireClose(collected.At(r, carta::zarr::Statistic::sum_sq, f), expected.sum_sq, where + " sum_sq");
            RequireClose(collected.At(r, carta::zarr::Statistic::min, f), expected.min, where + " min");
            RequireClose(collected.At(r, carta::zarr::Statistic::max, f), expected.max, where + " max");
        }
    }
}

carta::zarr::SpectralReduceRequest WholeSpectrum(const std::vector<carta::zarr::RegionMask>& regions,
                                                 std::uint64_t polarization) {
    carta::zarr::SpectralReduceRequest request;
    request.spectral = {0, kFrequency, 1};
    request.polarization = polarization;
    request.regions = regions.data();
    request.region_count = regions.size();
    request.statistics = AllStatistics();
    return request;
}

// The fixture's l chunk is two pixels wide, so a region wider than two pixels is the case that
// tells a correct walk from one that visits a region once per chunk and forgets to clip it.
void TestRegionsSpanningChunks(const carta::zarr::Image& sky) {
    const std::vector<carta::zarr::RegionMask> regions{
        {0, 0, kL, kM, nullptr},  // the whole plane, across both l chunks
        {1, 1, 2, 2, nullptr},    // straddles the chunk boundary at l = 2
        {3, 4, 1, 1, nullptr},    // a single pixel, which is what a cursor profile asks for
    };
    const auto collected = Collect(sky, WholeSpectrum(regions, 0));
    Require(collected.statistics.size() == 6, "all six statistics should be reported");
    CheckAgainstOracle(collected, regions, 0, "spanning");
}

// Every pixel of polarization 2, frequency 1, l >= 2 lives in the chunk the generator deleted.
void TestMissingChunkHasNoFinitePixels(const carta::zarr::Image& sky) {
    const std::vector<carta::zarr::RegionMask> regions{{2, 0, 2, kM, nullptr}};
    const auto collected = Collect(sky, WholeSpectrum(regions, 2));
    CheckAgainstOracle(collected, regions, 2, "missing chunk");
    RequireClose(collected.At(0, carta::zarr::Statistic::num_pixels, 1), 0.0,
                 "a deleted chunk contributes no finite pixels");
    RequireClose(collected.At(0, carta::zarr::Statistic::nan_count, 1), static_cast<double>(2 * kM),
                 "every pixel of a deleted chunk is absent");
    Require(std::isnan(collected.At(0, carta::zarr::Statistic::min, 1)),
            "the smallest value of nothing is not a number");
    Require(std::isnan(collected.At(0, carta::zarr::Statistic::max, 1)),
            "the largest value of nothing is not a number");
}

// A raster mask has to select pixels inside the bounding box, and a null mask has to behave exactly
// as a full one does -- the null branch is what an unrotated rectangle arrives as.
void TestRasterMaskAndNullMaskAgree(const carta::zarr::Image& sky) {
    std::vector<std::uint8_t> checkerboard(kL * kM);
    for (std::uint64_t m = 0; m < kM; ++m) {
        for (std::uint64_t l = 0; l < kL; ++l) {
            checkerboard.at((m * kL) + l) = static_cast<std::uint8_t>((l + m) % 2);
        }
    }
    const std::vector<std::uint8_t> full(kL * kM, 1);
    const std::vector<carta::zarr::RegionMask> regions{
        {0, 0, kL, kM, checkerboard.data()},
        {0, 0, kL, kM, full.data()},
        {0, 0, kL, kM, nullptr},
    };
    const auto collected = Collect(sky, WholeSpectrum(regions, 1));
    CheckAgainstOracle(collected, regions, 1, "masked");
    for (std::uint64_t f = 0; f < kFrequency; ++f) {
        RequireClose(collected.At(2, carta::zarr::Statistic::sum, f),
                     collected.At(1, carta::zarr::Statistic::sum, f),
                     "a null mask should select what a full mask selects");
    }
}

// The statistics a caller did not ask for must not appear, and the ones they did must arrive in the
// order the block declares rather than the order they were named.
void TestOnlyRequestedStatisticsAreReported(const carta::zarr::Image& sky) {
    const std::vector<carta::zarr::RegionMask> regions{{0, 0, kL, kM, nullptr}};
    auto request = WholeSpectrum(regions, 0);
    request.statistics = carta::zarr::Statistic::sum | carta::zarr::Statistic::num_pixels;
    const auto collected = Collect(sky, request);
    Require(collected.statistics.size() == 2, "only the two requested statistics should be reported");
    Require(collected.statistics.at(0) == carta::zarr::Statistic::num_pixels &&
                collected.statistics.at(1) == carta::zarr::Statistic::sum,
            "a block reports statistics in the library's order, not the caller's");
    for (std::uint64_t f = 0; f < kFrequency; ++f) {
        RequireClose(collected.At(0, carta::zarr::Statistic::sum, f), Expected(regions.at(0), f, 0).sum,
                     "sum over the whole plane");
    }
}

// Strides skip channels rather than sampling a dense read afterwards, and first_channel indexes the
// selection rather than the image.
void TestStrideSelectsChannels(const carta::zarr::Image& sky) {
    const std::vector<carta::zarr::RegionMask> regions{{0, 0, kL, kM, nullptr}};
    auto request = WholeSpectrum(regions, 0);
    request.spectral = {1, 1, 2};
    const auto collected = Collect(sky, request);
    Require(collected.channel_count == 1, "a stride of two over two channels selects one");
    RequireClose(collected.At(0, carta::zarr::Statistic::sum, 0), Expected(regions.at(0), 1, 0).sum,
                 "a strided selection should start at the requested channel");
}

// emit_every_channels is a hint that the block reports back. One channel per block is inside every
// budget here, so the hint survives intact and the reduction arrives in two pieces.
void TestEmitGranularityIsReported(const carta::zarr::Image& sky) {
    const std::vector<carta::zarr::RegionMask> regions{{0, 0, kL, kM, nullptr}};
    auto request = WholeSpectrum(regions, 0);
    request.emit_every_channels = 1;
    const auto collected = Collect(sky, request);
    Require(collected.block_lengths.size() == kFrequency, "one block per channel was asked for");
    for (const auto length : collected.block_lengths) {
        Require(length == 1, "each block should carry the one channel that was asked for");
    }
    CheckAgainstOracle(collected, regions, 0, "streamed");

    auto whole = WholeSpectrum(regions, 0);
    const auto in_one_block = Collect(sky, whole);
    Require(in_one_block.block_lengths.size() == 1,
            "this whole fixture costs far less than one read budget, so the library should see no "
            "reason to split it");
    for (std::uint64_t f = 0; f < kFrequency; ++f) {
        RequireClose(collected.At(0, carta::zarr::Statistic::sum, f),
                     in_one_block.At(0, carta::zarr::Statistic::sum, f),
                     "streaming should not change the answer");
    }
}

// Without a hint the library emits once per budget of decoded chunk data. A region big enough to
// spend the whole budget on one spectral layer therefore reports a layer at a time, which is the
// case that matters: on a real image that is the difference between a partial profile every 70 ms
// and one silent block lasting minutes.
// A block that takes more than one read is handed over as it fills, so that a caller has something
// to show and somewhere to stop before its last pixel arrives. The unfinished values are partial
// sums over the chunks read so far, so what is checked here is that they converge and that the
// finished block is unaffected -- in particular that putting the extremum identities back after a
// hand-over leaves the walk able to keep accumulating into them.
void TestAnUnfinishedBlockIsHandedOver(const carta::zarr::Image& sky) {
    const std::vector<carta::zarr::RegionMask> regions{{0, 0, kL, kM, nullptr}};
    carta::zarr::ReadOptions options;
    // One chunk of the fixture, so the plane's two chunks cannot be read together.
    options.temporary_memory_limit_bytes = 40;
    const auto collected = Collect(sky, WholeSpectrum(regions, 0), options);
    Require(!collected.partial_counts.empty(),
            "a budget of one chunk should take more than one read for a plane two chunks wide; if "
            "the fixture's chunk shape changed, this no longer splits and the test stops testing it");

    bool saw_growth = false;
    for (const auto& [first_channel, counted] : collected.partial_counts) {
        const double finished = collected.At(0, carta::zarr::Statistic::num_pixels, first_channel);
        Require(counted <= finished, "an unfinished block cannot have counted more than the finished one");
        saw_growth = saw_growth || counted < finished;
    }
    Require(saw_growth, "at least one hand-over should have happened before the block was complete");

    CheckAgainstOracle(collected, regions, 0, "handed over as it filled");
}

// The run-length form of a raster, which is what a caller is expected to hand over and what the
// backend derives from casacore's mask. Written the obvious way on purpose: it is the reference the
// library's own use of the runs is checked against.
struct Runs {
    std::vector<std::uint32_t> runs;
    std::vector<std::uint64_t> offsets;
};

// `along_y` walks columns instead of rows, which is what an image whose store varies fastest along
// m needs: the runs have to lie along the axis the pixels are contiguous in.
Runs RunsOf(const std::vector<std::uint8_t>& raster, std::uint64_t width, std::uint64_t height, bool along_y) {
    const std::uint64_t outer = along_y ? width : height;
    const std::uint64_t inner = along_y ? height : width;
    const auto at = [&](std::uint64_t o, std::uint64_t i) {
        const auto x = along_y ? o : i;
        const auto y = along_y ? i : o;
        return raster.at(static_cast<std::size_t>((y * width) + x));
    };
    Runs out;
    out.offsets.push_back(0);
    for (std::uint64_t o = 0; o < outer; ++o) {
        std::uint64_t i = 0;
        while (i < inner) {
            while (i < inner && at(o, i) == 0) {
                ++i;
            }
            if (i == inner) {
                break;
            }
            const auto begin = static_cast<std::uint32_t>(i);
            while (i < inner && at(o, i) != 0) {
                ++i;
            }
            out.runs.push_back(begin);
            out.runs.push_back(static_cast<std::uint32_t>(i));
        }
        out.offsets.push_back(out.runs.size() / 2);
    }
    return out;
}

// Fill in a region's runs the way the image requires them.
void Attach(carta::zarr::RegionMask& region, const Runs& runs, carta::zarr::AxisRole axis) {
    region.row_runs = runs.runs.data();
    region.row_run_offsets = runs.offsets.data();
    region.run_axis = axis;
}

// Runs are the selection when they are given, and they have to select what the raster would. The
// pattern here is chosen so that a row is two runs with a hole between them, a row is no runs at
// all, and a run crosses the chunk boundary at l = 2 -- the three shapes a wrong index or a wrong
// clip would get away with on a solid rectangle.
void TestRunsSelectTheSamePixelsAsTheRaster(const carta::zarr::Image& sky) {
    std::vector<std::uint8_t> raster(static_cast<std::size_t>(kL) * static_cast<std::size_t>(kM), 0);
    for (std::uint64_t y = 0; y < kM; ++y) {
        for (std::uint64_t x = 0; x < kL; ++x) {
            // Row 2 and column 1 are cleared so that whichever way the runs have to lie, one of
            // them is empty.
            const bool set = y != 2 && x != 1 && ((x + (2 * y)) % 3) != 0;
            raster.at(static_cast<std::size_t>((y * kL) + x)) = set ? 1 : 0;
        }
    }
    const auto axis = sky.chunk_geometry().fastest_spatial_axis;
    const auto runs = RunsOf(raster, kL, kM, axis == carta::zarr::AxisRole::spatial_y);
    std::size_t empty_rows = 0;
    std::size_t split_rows = 0;
    for (std::size_t r = 0; r + 1 < runs.offsets.size(); ++r) {
        const auto count = runs.offsets.at(r + 1) - runs.offsets.at(r);
        empty_rows += count == 0 ? 1 : 0;
        split_rows += count > 1 ? 1 : 0;
    }
    Require(empty_rows > 0, "the pattern should leave one line of the region empty");
    Require(split_rows > 0, "the pattern should leave one line split by a hole");

    const std::vector<carta::zarr::RegionMask> by_raster{{0, 0, kL, kM, raster.data()}};
    std::vector<carta::zarr::RegionMask> by_runs{{0, 0, kL, kM, nullptr}};
    Attach(by_runs.at(0), runs, axis);

    const auto from_raster = Collect(sky, WholeSpectrum(by_raster, 0));
    const auto from_runs = Collect(sky, WholeSpectrum(by_runs, 0));
    CheckAgainstOracle(from_raster, by_raster, 0, "raster");
    Require(from_raster.values.size() == from_runs.values.size(), "both forms should report the same shape");
    for (std::size_t i = 0; i < from_raster.values.size(); ++i) {
        RequireClose(from_runs.values.at(i), from_raster.values.at(i),
                     "runs and raster should select the same pixels at value " + std::to_string(i));
    }
}

// And the runs narrow the chunk index the same way the raster does, without the raster being there.
void TestRunsNarrowTheChunksTheWalkReads(const carta::zarr::Image& sky) {
    std::vector<std::uint8_t> raster(static_cast<std::size_t>(kL) * static_cast<std::size_t>(kM), 0);
    for (std::uint64_t y = 0; y < kM; ++y) {
        for (std::uint64_t x = 0; x < 2; ++x) {  // the left chunk only
            raster.at(static_cast<std::size_t>((y * kL) + x)) = 1;
        }
    }
    const auto axis = sky.chunk_geometry().fastest_spatial_axis;
    const auto runs = RunsOf(raster, kL, kM, axis == carta::zarr::AxisRole::spatial_y);
    std::vector<carta::zarr::RegionMask> regions{{0, 0, kL, kM, nullptr}};
    Attach(regions.at(0), runs, axis);

    carta::zarr::ReadOptions options;
    options.temporary_memory_limit_bytes = 40;  // one chunk of the fixture
    const auto collected = Collect(sky, WholeSpectrum(regions, 0), options);
    Require(collected.partial_counts.empty(),
            "runs confined to one chunk should make the walk read one chunk; a hand-over on the way "
            "means it read the second one too");
}

// A region is indexed by the chunks its mask occupies, not by the chunks its bounding box covers.
// The two differ whenever a region is thin and slanted: a three-pixel-wide rectangle along the
// diagonal of an image has a bounding box the size of the image, and bucketing by the box reads
// every chunk to reach the band.
//
// The fixture's plane is two chunks wide, so a mask confined to one of them should make the walk
// read one chunk and not two. With a budget of a single chunk that is observable: two chunks cannot
// be read together, so reading both would hand the block over unfinished on the way.
void TestAMaskNarrowsTheChunksTheWalkReads(const carta::zarr::Image& sky) {
    std::vector<std::uint8_t> raster(static_cast<std::size_t>(kL) * static_cast<std::size_t>(kM), 0);
    for (std::uint64_t y = 0; y < kM; ++y) {
        for (std::uint64_t x = 0; x < 2; ++x) {  // the left chunk only
            raster.at(static_cast<std::size_t>((y * kL) + x)) = 1;
        }
    }
    const std::vector<carta::zarr::RegionMask> regions{{0, 0, kL, kM, raster.data()}};

    carta::zarr::ReadOptions options;
    options.temporary_memory_limit_bytes = 40;  // one chunk of the fixture
    const auto collected = Collect(sky, WholeSpectrum(regions, 0), options);
    Require(collected.partial_counts.empty(),
            "the walk should have read the one chunk the mask occupies; a hand-over on the way means "
            "it read the second chunk too, which only the bounding box asked for");
    CheckAgainstOracle(collected, regions, 0, "mask-narrowed");

    // The same bounding box with nothing to narrow it does read both, which is what makes the check
    // above a measurement rather than a coincidence.
    const std::vector<carta::zarr::RegionMask> whole_box{{0, 0, kL, kM, nullptr}};
    const auto unnarrowed = Collect(sky, WholeSpectrum(whole_box, 0), options);
    Require(!unnarrowed.partial_counts.empty(),
            "a region with no mask covers its whole bounding box, which here is two chunks");
}

void TestABigRegionIsEmittedALayerAtATime(const carta::zarr::Image& sky) {
    const std::vector<carta::zarr::RegionMask> regions{{0, 0, kL, kM, nullptr}};
    const auto& chunk = sky.chunk_geometry().chunk_shape;
    Require(chunk.at(2) == 1,
            "this test assumes the fixture's frequency chunk is one channel deep, so that a layer "
            "is a channel and the block count can be predicted");

    carta::zarr::ReadOptions options;
    // The plane is two chunks of forty bytes, doubled because the mask is read alongside: one
    // budget buys exactly one layer.
    options.temporary_memory_limit_bytes = 2 * 40 * 2;
    const auto collected = Collect(sky, WholeSpectrum(regions, 0), options);
    Require(collected.block_lengths.size() == kFrequency,
            "a budget worth one layer should emit one layer per block; if the fixture's chunk shape "
            "changed, this limit no longer matches a layer and the test stops testing the split");
    for (const auto length : collected.block_lengths) {
        Require(length == 1, "each block should carry the single channel its layer holds");
    }
    CheckAgainstOracle(collected, regions, 0, "layer at a time");
}

void TestSinkCancels(const carta::zarr::Image& sky) {
    const std::vector<carta::zarr::RegionMask> regions{{0, 0, kL, kM, nullptr}};
    auto request = WholeSpectrum(regions, 0);
    request.emit_every_channels = 1;
    std::size_t blocks = 0;
    const auto result = sky.ReduceSpectral(request, [&](const carta::zarr::SpectralBlock&) {
        ++blocks;
        return false;
    });
    Require(!result, "a sink that returns false should cancel the reduction");
    Require(result.error().code == carta::zarr::ErrorCode::cancelled, "cancelling should report cancelled");
    Require(blocks == 1, "a cancelled reduction should stop after the block that refused");
}

void TestRejectedRequests(const carta::zarr::Image& sky) {
    const std::vector<carta::zarr::RegionMask> regions{{0, 0, kL, kM, nullptr}};
    const auto sink = [](const carta::zarr::SpectralBlock&) { return true; };
    const auto rejects = [&](carta::zarr::SpectralReduceRequest request, const std::string& what) {
        const auto result = sky.ReduceSpectral(request, sink);
        Require(!result, what + " should be rejected");
        Require(result.error().code == carta::zarr::ErrorCode::invalid_argument,
                what + " should be an invalid argument");
    };

    auto no_regions = WholeSpectrum(regions, 0);
    no_regions.region_count = 0;
    rejects(no_regions, "a reduction with no regions");

    auto too_many = WholeSpectrum(regions, 0);
    too_many.region_count = carta::zarr::kMaxSpectralRegions + 1;
    rejects(too_many, "a region count past the structural bound");

    auto no_statistics = WholeSpectrum(regions, 0);
    no_statistics.statistics = 0;
    rejects(no_statistics, "a reduction with no statistics");

    const std::vector<carta::zarr::RegionMask> outside{{kL - 1, 0, 2, kM, nullptr}};
    auto past_the_edge = WholeSpectrum(outside, 0);
    rejects(past_the_edge, "a region hanging off the image");

    const std::vector<carta::zarr::RegionMask> empty{{0, 0, 0, kM, nullptr}};
    rejects(WholeSpectrum(empty, 0), "a region with no width");

    auto past_the_last_channel = WholeSpectrum(regions, 0);
    past_the_last_channel.spectral = {0, kFrequency + 1, 1};
    rejects(past_the_last_channel, "a spectral range past the last channel");

    auto past_the_last_polarization = WholeSpectrum(regions, kPolarization);
    rejects(past_the_last_polarization, "a polarization past the last one");
}

}  // namespace

// The wide fixture, whose chunks are big enough for the reduction to split its units between
// workers. See generate_wide_pixel_fixture: value = (f * P + p) * 1e6 + (l / 8) * 1000 + (m / 8),
// exactly representable in a float32.
namespace wide {

constexpr std::uint64_t kL = 512;
constexpr std::uint64_t kM = 520;
constexpr std::uint64_t kFrequency = 4;
constexpr std::uint64_t kPolarization = 2;

double ExpectedValue(std::uint64_t l, std::uint64_t m, std::uint64_t frequency, std::uint64_t polarization) {
    const auto plane = (frequency * kPolarization) + polarization;
    return static_cast<double>((plane * 1000000) + ((l / 8) * 1000) + (m / 8));
}

// What the reduction only does on a fixture this size. Below 65,536 pixels to a chunk PlanRowTasks
// answers one however many workers there are, so the private sinks and the merge that adds them up
// never ran on the small fixtures.
//
// The counts and the extremes are exact at any thread count; the sums are re-associated once the
// work is split, so they are held to a rounding rather than to the bit.
void TestAWideRegionSplitsAndStillAgrees(const char* fixture) {
    const std::uint64_t polarization = 1;
    const std::vector<carta::zarr::RegionMask> regions{
        {0, 0, kL, kM, nullptr},        // the whole plane
        {100, 60, 300, 400, nullptr},   // an interior box across several chunks
    };

    // The oracle, recomputed from the fixture's encoding.
    std::vector<double> expected_pixels(regions.size() * kFrequency, 0.0);
    std::vector<double> expected_sum(regions.size() * kFrequency, 0.0);
    std::vector<double> expected_min(regions.size() * kFrequency, std::numeric_limits<double>::infinity());
    std::vector<double> expected_max(regions.size() * kFrequency, -std::numeric_limits<double>::infinity());
    for (std::size_t r = 0; r < regions.size(); ++r) {
        const auto& region = regions[r];
        for (std::uint64_t f = 0; f < kFrequency; ++f) {
            const auto at = (r * kFrequency) + f;
            for (std::uint64_t y = region.y_start; y < region.y_start + region.height; ++y) {
                for (std::uint64_t x = region.x_start; x < region.x_start + region.width; ++x) {
                    // A region's x runs along l and its y along m, which is the order the image
                    // axes are in and the opposite of how the generator's formula reads.
                    const double value = ExpectedValue(x, y, f, polarization);
                    expected_pixels[at] += 1.0;
                    expected_sum[at] += value;
                    expected_min[at] = std::min(expected_min[at], value);
                    expected_max[at] = std::max(expected_max[at], value);
                }
            }
        }
    }

    for (const unsigned int threads : {1U, 4U, 16U}) {
        carta::zarr::OpenOptions options;
        options.decode_threads = threads;
        const auto sky = OpenSky(fixture, options);

        carta::zarr::SpectralReduceRequest request;
        request.spectral = {0, kFrequency, 1};
        request.polarization = polarization;
        request.regions = regions.data();
        request.region_count = regions.size();
        request.statistics = AllStatistics();
        const auto collected = Collect(sky, request);

        for (std::size_t r = 0; r < regions.size(); ++r) {
            for (std::uint64_t f = 0; f < kFrequency; ++f) {
                const auto at = (r * kFrequency) + f;
                const std::string where = "region " + std::to_string(r) + " channel " + std::to_string(f) +
                                          " on " + std::to_string(threads) + " threads";
                RequireClose(collected.At(r, carta::zarr::Statistic::num_pixels, f), expected_pixels[at],
                             where + " pixels");
                RequireClose(collected.At(r, carta::zarr::Statistic::nan_count, f), 0.0, where + " nan");
                RequireClose(collected.At(r, carta::zarr::Statistic::min, f), expected_min[at], where + " min");
                RequireClose(collected.At(r, carta::zarr::Statistic::max, f), expected_max[at], where + " max");
                const double sum = collected.At(r, carta::zarr::Statistic::sum, f);
                Require(std::abs(sum - expected_sum[at]) <= 1e-9 * (1.0 + std::abs(expected_sum[at])),
                        where + " sum: expected " + std::to_string(expected_sum[at]) + ", got " +
                            std::to_string(sum));
            }
        }
    }
}

}  // namespace wide

int main() {
    std::vector<carta::zarr::AxisRole> fast_axes;
    for (const char* const fixture : kFixtures) {
        try {
            const auto sky = OpenSky(fixture);
            fast_axes.push_back(sky.chunk_geometry().fastest_spatial_axis);
            TestRegionsSpanningChunks(sky);
            TestMissingChunkHasNoFinitePixels(sky);
            TestRasterMaskAndNullMaskAgree(sky);
            TestOnlyRequestedStatisticsAreReported(sky);
            TestStrideSelectsChannels(sky);
            TestEmitGranularityIsReported(sky);
            TestABigRegionIsEmittedALayerAtATime(sky);
            TestAnUnfinishedBlockIsHandedOver(sky);
            TestAMaskNarrowsTheChunksTheWalkReads(sky);
            TestRunsSelectTheSamePixelsAsTheRaster(sky);
            TestRunsNarrowTheChunksTheWalkReads(sky);
            TestSinkCancels(sky);
            TestRejectedRequests(sky);
        } catch (const std::exception& error) {
            std::cerr << "spectral reduce test failed on " << fixture << ": " << error.what() << "\n";
            return 1;
        }
    }
    try {
        wide::TestAWideRegionSplitsAndStillAgrees(CARTA_ZARR_PIXEL_FIXTURE_WIDE);
    } catch (const std::exception& error) {
        std::cerr << "spectral reduce test failed on the wide fixture: " << error.what() << "\n";
        return 1;
    }
    try {
        Require(fast_axes.size() == 2 && fast_axes.at(0) != fast_axes.at(1),
                "the two fixtures should disagree about which spatial axis the store varies fastest; "
                "if they agree, one of them was regenerated wrongly and half the walk is untested");
    } catch (const std::exception& error) {
        std::cerr << "spectral reduce test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
