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
#include <limits>
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

// The same rule the library follows, written out here so the two cannot drift together.
std::vector<std::uint64_t> Expected(std::uint64_t frequency, std::uint64_t polarization, double range_lower,
                                    double range_upper, std::size_t bins) {
    std::vector<std::uint64_t> counts(bins, 0);
    const float width = static_cast<float>((range_upper - range_lower) / bins);
    const float lower = static_cast<float>(range_lower);
    const float upper = static_cast<float>(range_upper);
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

carta::zarr::HistogramRequest WholeSpectrum(std::uint64_t polarization, double lower, double upper,
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

// One pass over the whole selection, with the range discovered on the way. What it keeps exact is
// the extremes, the counts and the sums; what it gives up is where the bin edges land.
void TestOnePassMatchesTheTwoPassAnswer(const carta::zarr::Image& sky) {
    const std::uint64_t polarization = 1;
    carta::zarr::CubeHistogramRequest request;
    request.spectral = {0, kFrequency, 1};
    request.polarization = polarization;
    request.bins = 12;
    const auto one_pass = sky.ComputeCubeHistogram(request);
    Require(static_cast<bool>(one_pass),
            std::string("the one-pass histogram failed: ") +
                (one_pass.has_value() ? "" : one_pass.error().message));
    const auto& result = one_pass.value();

    // The extremes and the sums, from the fixture's encoding.
    double expected_pixels = 0.0;
    double expected_nan = 0.0;
    double expected_sum = 0.0;
    double expected_min = std::numeric_limits<double>::infinity();
    double expected_max = -std::numeric_limits<double>::infinity();
    for (std::uint64_t f = 0; f < kFrequency; ++f) {
        for (std::uint64_t m = 0; m < kM; ++m) {
            for (std::uint64_t l = 0; l < kL; ++l) {
                if (!ExpectedFlag(l, m) || InMissingChunk(l, f, polarization)) {
                    expected_nan += 1.0;
                    continue;
                }
                const double value = ExpectedValue(l, m, f, polarization);
                expected_pixels += 1.0;
                expected_sum += value;
                expected_min = std::min(expected_min, value);
                expected_max = std::max(expected_max, value);
            }
        }
    }
    Require(result.num_pixels == expected_pixels, "the finite pixel count should be exact");
    Require(result.nan_count == expected_nan, "the absent pixel count should be exact");
    Require(std::abs(result.sum - expected_sum) <= 1e-9 * (1.0 + std::abs(expected_sum)),
            "the sum should agree to a rounding");
    Require(result.minimum == expected_min, "the minimum should be exact, whatever the bins did");
    Require(result.maximum == expected_max, "the maximum should be exact, whatever the bins did");
    Require(!result.sampled, "nothing was sampled");

    std::uint64_t total = 0;
    for (const auto count : result.counts) {
        total += count;
    }
    Require(static_cast<double>(total) == expected_pixels,
            "every finite pixel should land in some bin; the provisional range grows until it does");

    // Against the two-pass answer over the range this pass discovered. They need not agree bin for
    // bin -- a provisional bin straddling a target edge goes to one side -- so what is required is
    // that no bin is off by more than the fixture's largest provisional bin could hold, which for
    // values this far apart is nothing.
    auto fixed = WholeSpectrum(polarization, result.minimum, result.maximum, request.bins);
    const auto two_pass = Collect(sky, fixed, {});
    for (std::size_t bin = 0; bin < request.bins; ++bin) {
        std::uint64_t summed = 0;
        for (std::uint64_t f = 0; f < kFrequency; ++f) {
            summed += two_pass.per_channel.at(static_cast<std::size_t>(f)).at(bin);
        }
        Require(result.counts.at(bin) == summed,
                "bin " + std::to_string(bin) + ": one pass said " + std::to_string(result.counts.at(bin)) +
                    ", two passes said " + std::to_string(summed));
    }
}

// The provisional range starts around the first pixel seen and doubles its way out. A selection
// whose values span far more than that first guess is the case that exercises it.
void TestTheProvisionalRangeGrowsToFit(const carta::zarr::Image& sky) {
    carta::zarr::CubeHistogramRequest request;
    request.spectral = {0, kFrequency, 1};
    request.polarization = 0;
    request.bins = 4;
    // Few enough bins that the merging on every doubling is visible rather than hidden in noise.
    request.provisional_bins = 8;
    const auto result = sky.ComputeCubeHistogram(request);
    Require(static_cast<bool>(result), "a coarse provisional histogram should still work");
    std::uint64_t total = 0;
    for (const auto count : result.value().counts) {
        total += count;
    }
    Require(static_cast<double>(total) == result.value().num_pixels,
            "merging on a doubling must not drop a pixel");
}

// Sampling reads fewer pixels, and says so.
void TestSamplingTakesFewerPixels(const carta::zarr::Image& sky) {
    carta::zarr::CubeHistogramRequest request;
    request.spectral = {0, kFrequency, 1};
    request.polarization = 0;
    request.bins = 8;
    const auto every = sky.ComputeCubeHistogram(request);
    Require(static_cast<bool>(every), "the unsampled pass should work");

    request.spatial_sample = 2;
    const auto sampled = sky.ComputeCubeHistogram(request);
    Require(static_cast<bool>(sampled), "the sampled pass should work");
    Require(sampled.value().sampled, "a sampled result should say so");
    Require(sampled.value().num_pixels + sampled.value().nan_count <
                every.value().num_pixels + every.value().nan_count,
            "taking every second pixel along both axes should look at fewer of them");
    Require(sampled.value().minimum >= every.value().minimum &&
                sampled.value().maximum <= every.value().maximum,
            "a sample cannot find an extreme that is not there");
}

void TestOnePassRejectsAndCancels(const carta::zarr::Image& sky) {
    carta::zarr::CubeHistogramRequest request;
    request.spectral = {0, kFrequency, 1};
    request.bins = 0;
    Require(!sky.ComputeCubeHistogram(request), "zero bins should be rejected");
    request.bins = 8;
    request.spatial_sample = 0;
    Require(!sky.ComputeCubeHistogram(request), "a sample of zero should be rejected");
}

}  // namespace

// decode_threads sizes the worker pool that bins the pixels as well as TensorStore's read pool, so
// the same question asked of two differently sized pools must come back with the same counts. This
// fixture is far below the split's threshold, so what this pins is the wiring rather than the
// split itself -- work_pool_test covers the split, which no fixture this small can reach.
// What one pass promises whatever the thread count is. Not the counts: each worker keeps a
// provisional histogram of its own and re-aggregates it onto the target grid at the end, so a
// provisional bin straddling a target edge can go to a different side than it did on one thread.
// The extremes, the totals and the sums are the parts the loader turns into a BasicStats, and those
// hold exactly -- except the sums, which are re-associated and so agree only to a rounding.
void TestOnePassKeepsItsContractAtAnyThreadCount(const char* fixture) {
    std::vector<carta::zarr::CubeHistogramResult> answers;
    for (const unsigned int threads : {1U, 2U, 8U}) {
        carta::zarr::OpenOptions options;
        options.decode_threads = threads;
        const auto sky = OpenSky(fixture, options);
        carta::zarr::CubeHistogramRequest request;
        request.spectral = {0, kFrequency, 1};
        request.polarization = 1;
        request.bins = 12;
        auto result = sky.ComputeCubeHistogram(request);
        Require(static_cast<bool>(result), "the one-pass histogram failed");
        answers.push_back(std::move(result.value()));
    }
    const auto& first = answers.front();
    for (const auto& answer : answers) {
        Require(answer.num_pixels == first.num_pixels, "the finite pixel count must not move with the threads");
        Require(answer.nan_count == first.nan_count, "the absent pixel count must not move with the threads");
        Require(answer.minimum == first.minimum, "the minimum must not move with the threads");
        Require(answer.maximum == first.maximum, "the maximum must not move with the threads");
        Require(std::abs(answer.sum - first.sum) <= 1e-12 * (1.0 + std::abs(first.sum)),
                "the sum should agree to a rounding");
        Require(std::abs(answer.sum_sq - first.sum_sq) <= 1e-12 * (1.0 + std::abs(first.sum_sq)),
                "the sum of squares should agree to a rounding");
        std::uint64_t total = 0;
        for (const auto count : answer.counts) {
            total += count;
        }
        Require(static_cast<double>(total) == answer.num_pixels,
                "every finite pixel should still land in some bin, whoever binned it");
    }
}

void TestThreadCountDoesNotChangeTheCounts(const char* fixture) {
    std::vector<std::vector<std::uint64_t>> answers;
    for (const unsigned int threads : {1U, 2U, 8U}) {
        carta::zarr::OpenOptions options;
        options.decode_threads = threads;
        const auto sky = OpenSky(fixture, options);
        const auto collected = Collect(sky, WholeSpectrum(0, 0.0F, 2000.0F, 16), {});
        std::vector<std::uint64_t> flat;
        for (const auto& channel : collected.per_channel) {
            flat.insert(flat.end(), channel.begin(), channel.end());
        }
        answers.push_back(std::move(flat));
    }
    for (std::size_t index = 1; index < answers.size(); ++index) {
        Require(answers[index] == answers.front(),
                "the counts changed with the thread count, which they must not");
    }
}

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
            TestOnePassMatchesTheTwoPassAnswer(sky);
            TestTheProvisionalRangeGrowsToFit(sky);
            TestSamplingTakesFewerPixels(sky);
            TestOnePassRejectsAndCancels(sky);
            TestThreadCountDoesNotChangeTheCounts(fixture);
            TestOnePassKeepsItsContractAtAnyThreadCount(fixture);
        } catch (const std::exception& error) {
            std::cerr << "histogram test failed on " << fixture << ": " << error.what() << "\n";
            return 1;
        }
    }
    return 0;
}
