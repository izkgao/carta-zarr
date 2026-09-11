/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Plane histograms, against counts recomputed from the fixture's encoding. An integer count is the
// one thing here that can be checked exactly rather than to a tolerance, so it is.

#include <carta-zarr/carta_zarr.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char* const kFixtures[]{CARTA_ZARR_PIXEL_FIXTURE, CARTA_ZARR_PIXEL_FIXTURE_L_FASTEST};

constexpr std::uint64_t kL = 4;
constexpr std::uint64_t kM = 5;
constexpr std::uint64_t kFrequency = 2;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// The chunk the generator deletes covers frequency 1, polarization 2 and l in [2, 4).
bool InMissingChunk(std::uint64_t l, std::uint64_t frequency, std::uint64_t polarization) {
    return frequency == 1 && polarization == 2 && l >= 2;
}

bool ExpectedFlag(std::uint64_t l, std::uint64_t m) {
    return ((l + m) % 3) != 0;
}

float ExpectedValue(std::uint64_t l, std::uint64_t m, std::uint64_t frequency, std::uint64_t polarization) {
    return static_cast<float>((frequency * 1000) + (polarization * 100) + (l * 10) + m);
}

carta::zarr::Image OpenSky(const char* fixture) {
    Require(std::filesystem::exists(fixture),
            "the pixel fixture is missing; run tests/data/generate_zarr_fixtures.py");
    const auto context = carta::zarr::Context::Create();
    Require(static_cast<bool>(context), "Context::Create failed");
    auto dataset = carta::zarr::Dataset::Open(context.value(), fixture);
    Require(static_cast<bool>(dataset), "Dataset::Open failed on the pixel fixture");
    auto image = dataset.value().OpenImage("SKY");
    Require(static_cast<bool>(image), "SKY could not be opened");
    return image.value();
}

// The same rule the library follows, written out here so the two cannot drift together.
std::vector<std::uint64_t> Expected(std::uint64_t frequency, std::uint64_t polarization, float lower,
                                    float upper, std::size_t bins) {
    std::vector<std::uint64_t> counts(bins, 0);
    const float width = (upper - lower) / static_cast<float>(bins);
    for (std::uint64_t m = 0; m < kM; ++m) {
        for (std::uint64_t l = 0; l < kL; ++l) {
            if (!ExpectedFlag(l, m) || InMissingChunk(l, frequency, polarization)) {
                continue;  // a flagged pixel and one in the deleted chunk are both absent
            }
            const float value = ExpectedValue(l, m, frequency, polarization);
            if (lower <= value && value <= upper) {
                auto bin = static_cast<std::size_t>((value - lower) / width);
                counts.at(std::min(bin, bins - 1)) += 1;
            }
        }
    }
    return counts;
}

struct Collected {
    std::vector<std::vector<std::uint64_t>> per_channel;
    std::size_t partial_blocks = 0;
};

Collected Collect(const carta::zarr::Image& sky, const carta::zarr::HistogramRequest& request,
                  const carta::zarr::ReadOptions& options) {
    Collected collected;
    collected.per_channel.assign(static_cast<std::size_t>(request.spectral.count), {});
    std::uint64_t next_channel = 0;
    const auto result = sky.ComputeHistogram(request, [&](const carta::zarr::HistogramBlock& block) {
        Require(block.first_channel == next_channel, "blocks should tile the spectral selection in order");
        Require(block.bin_count == request.bins, "a block should report the bins that were asked for");
        if (!block.complete) {
            Require(block.completeness > 0.0 && block.completeness < 1.0,
                    "an unfinished block should report a fraction of itself");
            ++collected.partial_blocks;
            return true;
        }
        for (std::uint64_t c = 0; c < block.channel_count; ++c) {
            const auto* row = block.counts + (static_cast<std::size_t>(c) * block.bin_count);
            collected.per_channel.at(static_cast<std::size_t>(block.first_channel + c))
                .assign(row, row + block.bin_count);
        }
        next_channel += block.channel_count;
        return true;
    }, options);
    Require(static_cast<bool>(result),
            std::string("the histogram failed: ") + (result.has_value() ? "" : result.error().message));
    Require(next_channel == request.spectral.count, "the blocks should cover the whole spectral selection");
    return collected;
}

carta::zarr::HistogramRequest WholeSpectrum(std::uint64_t polarization, float lower, float upper,
                                            std::uint32_t bins) {
    carta::zarr::HistogramRequest request;
    request.spectral = {0, kFrequency, 1};
    request.polarization = polarization;
    request.lower = lower;
    request.upper = upper;
    request.bins = bins;
    return request;
}

// The wide case: a range that holds every value, so the counts say how many pixels survived the
// flag and the deleted chunk.
void TestCountsMatchTheOracle(const carta::zarr::Image& sky) {
    for (std::uint64_t polarization = 0; polarization < 3; ++polarization) {
        const auto request = WholeSpectrum(polarization, 0.0F, 2000.0F, 16);
        const auto collected = Collect(sky, request, {});
        for (std::uint64_t f = 0; f < kFrequency; ++f) {
            const auto expected = Expected(f, polarization, 0.0F, 2000.0F, 16);
            const auto& actual = collected.per_channel.at(static_cast<std::size_t>(f));
            for (std::size_t bin = 0; bin < expected.size(); ++bin) {
                Require(actual.at(bin) == expected.at(bin),
                        "bin " + std::to_string(bin) + " of channel " + std::to_string(f) +
                            " polarization " + std::to_string(polarization) + ": expected " +
                            std::to_string(expected.at(bin)) + ", got " + std::to_string(actual.at(bin)));
            }
        }
    }
}

// The narrow case: a range that excludes most of the plane, which is what tells a walk that counts
// every pixel from one that honours the bounds.
void TestPixelsOutsideTheRangeAreNotCounted(const carta::zarr::Image& sky) {
    const auto request = WholeSpectrum(0, 10.0F, 25.0F, 4);
    const auto collected = Collect(sky, request, {});
    std::uint64_t total = 0;
    for (const auto count : collected.per_channel.at(0)) {
        total += count;
    }
    const auto expected = Expected(0, 0, 10.0F, 25.0F, 4);
    std::uint64_t expected_total = 0;
    for (const auto count : expected) {
        expected_total += count;
    }
    Require(total == expected_total, "a narrow range should count only what falls inside it");
    Require(expected_total > 0 && expected_total < kL * kM,
            "this range should keep some of the plane and drop some, or it tests nothing");
}

// The deleted chunk leaves a plane with fewer pixels than any other, which is the only place a
// missing chunk can show up in a count.
void TestTheMissingChunkIsNotCounted(const carta::zarr::Image& sky) {
    const auto collected = Collect(sky, WholeSpectrum(2, 0.0F, 4000.0F, 8), {});
    std::uint64_t first = 0;
    std::uint64_t second = 0;
    for (const auto count : collected.per_channel.at(0)) {
        first += count;
    }
    for (const auto count : collected.per_channel.at(1)) {
        second += count;
    }
    Require(second < first, "the plane whose chunk was deleted should count fewer pixels");
}

// A block that takes more than one read is handed over as it fills, as in the reduction.
void TestAnUnfinishedBlockIsHandedOver(const carta::zarr::Image& sky) {
    carta::zarr::ReadOptions options;
    options.temporary_memory_limit_bytes = 40;  // one chunk of the fixture
    const auto collected = Collect(sky, WholeSpectrum(0, 0.0F, 2000.0F, 16), options);

    // The walk reads a band of the slower spatial axis at a time, so it can only split when the
    // plane spans more than one chunk along that axis. The fixture's chunk is 2 by 5, so one of the
    // two storage orders can split and the other cannot -- which is worth stating rather than
    // asserting blindly, because the counts below are checked either way.
    const auto& chunk = sky.chunk_geometry().chunk_shape;
    const bool y_is_fast = sky.chunk_geometry().fastest_spatial_axis == carta::zarr::AxisRole::spatial_y;
    const std::uint64_t slow_length = y_is_fast ? kL : kM;
    const std::uint64_t slow_chunk = y_is_fast ? chunk.at(0) : chunk.at(1);
    if (slow_length > slow_chunk) {
        Require(collected.partial_blocks > 0,
                "a budget of one chunk should take more than one read when the plane spans more than "
                "one chunk along the slower axis; if the fixture's chunk shape changed, this no "
                "longer splits and the test stops testing it");
    } else {
        Require(collected.partial_blocks == 0,
                "a plane that is one chunk deep along the slower axis is one read, so nothing should "
                "have been handed over early");
    }
    const auto expected = Expected(0, 0, 0.0F, 2000.0F, 16);
    for (std::size_t bin = 0; bin < expected.size(); ++bin) {
        Require(collected.per_channel.at(0).at(bin) == expected.at(bin),
                "splitting the walk should not change a count");
    }
}

void TestSinkCancels(const carta::zarr::Image& sky) {
    int calls = 0;
    const auto result = sky.ComputeHistogram(WholeSpectrum(0, 0.0F, 2000.0F, 16),
                                             [&](const carta::zarr::HistogramBlock&) {
                                                 ++calls;
                                                 return false;
                                             });
    Require(!result && result.error().code == carta::zarr::ErrorCode::cancelled,
            "a sink that says stop should report cancelled");
    Require(calls == 1, "a sink that says stop should not be asked again");
}

void TestRejectedRequests(const carta::zarr::Image& sky) {
    const auto rejects = [&](const carta::zarr::HistogramRequest& request, const std::string& what) {
        const auto result = sky.ComputeHistogram(request, [](const carta::zarr::HistogramBlock&) { return true; });
        Require(!result && result.error().code == carta::zarr::ErrorCode::invalid_argument, what);
    };
    rejects(WholeSpectrum(0, 0.0F, 2000.0F, 0), "zero bins should be rejected");
    rejects(WholeSpectrum(0, 5.0F, 5.0F, 8), "an empty range should be rejected rather than divided by");
    rejects(WholeSpectrum(0, 10.0F, 1.0F, 8), "an inverted range should be rejected");
    auto outside = WholeSpectrum(0, 0.0F, 10.0F, 8);
    outside.spectral = {0, kFrequency + 1, 1};
    rejects(outside, "a spectral range outside the image should be rejected");
}

}  // namespace

int main() {
    for (const char* const fixture : kFixtures) {
        try {
            const auto sky = OpenSky(fixture);
            TestCountsMatchTheOracle(sky);
            TestPixelsOutsideTheRangeAreNotCounted(sky);
            TestTheMissingChunkIsNotCounted(sky);
            TestAnUnfinishedBlockIsHandedOver(sky);
            TestSinkCancels(sky);
            TestRejectedRequests(sky);
        } catch (const std::exception& error) {
            std::cerr << "histogram test failed on " << fixture << ": " << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}
