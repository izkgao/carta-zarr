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

const char* const kFixture = CARTA_ZARR_PIXEL_FIXTURE;

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

carta::zarr::Image OpenSky() {
    Require(std::filesystem::exists(kFixture),
            "the pixel fixture is missing; run tests/data/generate_zarr_fixtures.py");
    const auto context = carta::zarr::Context::Create();
    Require(static_cast<bool>(context), "Context::Create failed");
    auto dataset = carta::zarr::Dataset::Open(context.value(), kFixture);
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

int main() {
    try {
        const auto sky = OpenSky();
        TestRegionsSpanningChunks(sky);
        TestMissingChunkHasNoFinitePixels(sky);
        TestRasterMaskAndNullMaskAgree(sky);
        TestOnlyRequestedStatisticsAreReported(sky);
        TestStrideSelectsChannels(sky);
        TestEmitGranularityIsReported(sky);
    TestABigRegionIsEmittedALayerAtATime(sky);
    TestAnUnfinishedBlockIsHandedOver(sky);
        TestSinkCancels(sky);
        TestRejectedRequests(sky);
    } catch (const std::exception& error) {
        std::cerr << "spectral reduce test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
