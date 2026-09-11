/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Pixel reads against a fixture whose every value spells its own logical coordinates. The point of
// that encoding is that the axis permutation cannot pass by accident: the fixture's axes all have
// different lengths, so a wrong order changes the shape, and a right shape with a wrong order
// changes the values.

#include <carta-zarr/carta_zarr.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

const char* const kFixture = CARTA_ZARR_PIXEL_FIXTURE;

// Axis lengths of the fixture, in the logical order the library reports.
constexpr std::uint64_t kL = 4;
constexpr std::uint64_t kM = 5;
constexpr std::uint64_t kFrequency = 2;
constexpr std::uint64_t kPolarization = 3;
constexpr std::uint64_t kTime = 1;

// The chunk the generator deletes covers frequency 1, polarization 2 and l in [2, 4).
bool InMissingChunk(std::uint64_t l, std::uint64_t frequency, std::uint64_t polarization) {
    return frequency == 1 && polarization == 2 && l >= 2;
}

float ExpectedValue(std::uint64_t l, std::uint64_t m, std::uint64_t frequency, std::uint64_t polarization,
                    std::uint64_t time) {
    return static_cast<float>((time * 10000) + (frequency * 1000) + (polarization * 100) + (l * 10) + m);
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

carta::zarr::ReadRequest WholeImage(const carta::zarr::ImageDescriptor& descriptor) {
    carta::zarr::ReadRequest request;
    for (const auto& axis : descriptor.axes) {
        request.axes.push_back(carta::zarr::Range{0, axis.length, 1});
    }
    return request;
}

std::size_t LogicalOffset(std::uint64_t l, std::uint64_t m, std::uint64_t frequency, std::uint64_t polarization) {
    // Axis 0 is the fastest-varying destination dimension.
    return static_cast<std::size_t>(l + (kL * (m + (kM * (frequency + (kFrequency * polarization))))));
}

void TestAxesAndGeometry(const carta::zarr::Image& sky) {
    const auto& axes = sky.descriptor().axes;
    Require(axes.size() == 5, "the fixture image should report five axes");
    const std::vector<std::pair<std::string, std::uint64_t>> expected{
        {"l", kL}, {"m", kM}, {"frequency", kFrequency}, {"polarization", kPolarization}, {"time", kTime}};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        Require(axes.at(i).name == expected.at(i).first,
                "logical axis " + std::to_string(i) + " should be " + expected.at(i).first);
        Require(axes.at(i).length == expected.at(i).second, "axis " + expected.at(i).first + " has the wrong length");
    }

    // Stored order is (time, frequency, polarization, l, m), so every axis moves.
    const auto& geometry = sky.chunk_geometry();
    Require(geometry.transpose_required, "a stored order differing from the logical order is a transpose");
    Require(geometry.chunk_shape == std::vector<std::uint64_t>{2, 5, 1, 1, 1},
            "chunk shape should be reported in logical order");
    Require(geometry.grid_shape == std::vector<std::uint64_t>{2, 1, 2, 3, 1}, "chunk grid shape is wrong");
    // Nothing in this fixture is sharded, so one I/O request still fetches one chunk.
    Require(!geometry.sharded && geometry.shard_shape == geometry.chunk_shape,
            "an unsharded array should report its chunk shape as the I/O granularity");
    Require(geometry.compressor == "zstd", "the fixture is written with zstd");
}

// Raw pixels, with the mask deliberately not applied, so that the stored values and the flag can
// be checked independently of each other.
carta::zarr::ReadOptions Unmasked() {
    carta::zarr::ReadOptions options;
    options.apply_pixel_mask = false;
    return options;
}

void TestWholeImage(const carta::zarr::Image& sky) {
    const auto request = WholeImage(sky.descriptor());
    std::vector<float> pixels(kL * kM * kFrequency * kPolarization * kTime);
    const auto read = sky.Read(request, {pixels.data(), pixels.size() * sizeof(float)}, Unmasked());
    Require(static_cast<bool>(read), "reading the whole image failed");
    Require(read.value() == pixels.size(), "the whole-image read reported the wrong element count");

    std::size_t missing = 0;
    for (std::uint64_t p = 0; p < kPolarization; ++p) {
        for (std::uint64_t f = 0; f < kFrequency; ++f) {
            for (std::uint64_t m = 0; m < kM; ++m) {
                for (std::uint64_t l = 0; l < kL; ++l) {
                    const float value = pixels.at(LogicalOffset(l, m, f, p));
                    const std::string where = "at l=" + std::to_string(l) + " m=" + std::to_string(m) +
                                              " frequency=" + std::to_string(f) +
                                              " polarization=" + std::to_string(p);
                    if (InMissingChunk(l, f, p)) {
                        // A chunk that was never written resolves to the array's fill value, which
                        // this fixture declares as NaN.
                        Require(std::isnan(value), "a deleted chunk should read as the fill value " + where);
                        ++missing;
                    } else {
                        Require(value == ExpectedValue(l, m, f, p, 0), "wrong pixel " + where);
                    }
                }
            }
        }
    }
    Require(missing == 2 * kM, "exactly one deleted chunk's worth of pixels should be missing");
}

void TestSubsetAndStride(const carta::zarr::Image& sky) {
    // Every other l, every other m, one plane. A stride the reader silently ignored would return
    // neighbouring values instead of these.
    carta::zarr::ReadRequest request;
    request.axes = {{0, 2, 2}, {1, 2, 2}, {0, 1, 1}, {1, 1, 1}, {0, 1, 1}};
    std::vector<float> pixels(4);
    const auto read = sky.Read(request, {pixels.data(), pixels.size() * sizeof(float)}, Unmasked());
    Require(static_cast<bool>(read), "a strided read failed");
    Require(read.value() == pixels.size(), "a strided read reported the wrong element count");

    const std::vector<std::pair<std::uint64_t, std::uint64_t>> selected{{0, 1}, {2, 1}, {0, 3}, {2, 3}};
    for (std::size_t i = 0; i < selected.size(); ++i) {
        Require(pixels.at(i) == ExpectedValue(selected.at(i).first, selected.at(i).second, 0, 1, 0),
                "a strided read returned the wrong element at offset " + std::to_string(i));
    }
}

void TestPixelMask(const carta::zarr::Image& sky) {
    Require(sky.descriptor().has_pixel_mask, "the fixture image declares a flag variable");
    Require(sky.descriptor().pixel_mask_id == "FLAG", "the pixel mask should name the flag variable");

    const auto request = WholeImage(sky.descriptor());
    std::vector<std::uint8_t> mask(kL * kM * kFrequency * kPolarization * kTime);
    const auto read = sky.ReadPixelMask(request, {mask.data(), mask.size()});
    Require(static_cast<bool>(read), "reading the pixel mask failed");
    Require(read.value() == mask.size(), "the mask read reported the wrong element count");

    for (std::uint64_t p = 0; p < kPolarization; ++p) {
        for (std::uint64_t f = 0; f < kFrequency; ++f) {
            for (std::uint64_t m = 0; m < kM; ++m) {
                for (std::uint64_t l = 0; l < kL; ++l) {
                    const bool good = mask.at(LogicalOffset(l, m, f, p)) != 0;
                    Require(good == ExpectedFlag(l, m), "wrong mask value at l=" + std::to_string(l) +
                                                            " m=" + std::to_string(m));
                }
            }
        }
    }
}

void TestMaskFusion(const carta::zarr::Image& sky) {
    const auto request = WholeImage(sky.descriptor());
    std::vector<float> pixels(kL * kM * kFrequency * kPolarization * kTime);
    // Masking is the default, so this is the plain two-argument read.
    const auto read = sky.Read(request, {pixels.data(), pixels.size() * sizeof(float)});
    Require(static_cast<bool>(read), "a masked read failed");

    for (std::uint64_t p = 0; p < kPolarization; ++p) {
        for (std::uint64_t f = 0; f < kFrequency; ++f) {
            for (std::uint64_t m = 0; m < kM; ++m) {
                for (std::uint64_t l = 0; l < kL; ++l) {
                    const float value = pixels.at(LogicalOffset(l, m, f, p));
                    const std::string where = "at l=" + std::to_string(l) + " m=" + std::to_string(m);
                    if (!ExpectedFlag(l, m) || InMissingChunk(l, f, p)) {
                        Require(std::isnan(value), "a flagged or missing pixel should read as NaN " + where);
                    } else {
                        Require(value == ExpectedValue(l, m, f, p, 0), "a good pixel should survive masking " + where);
                    }
                }
            }
        }
    }
}

void TestRejectedRequests(const carta::zarr::Image& sky) {
    const auto whole = WholeImage(sky.descriptor());
    std::vector<float> pixels(kL * kM * kFrequency * kPolarization * kTime);
    const carta::zarr::MutableBufferView buffer{pixels.data(), pixels.size() * sizeof(float)};

    const auto expect_rejected = [&](const carta::zarr::ReadRequest& request, const std::string& what) {
        const auto read = sky.Read(request, buffer);
        Require(!read, what + " should be rejected");
        Require(read.error().code == carta::zarr::ErrorCode::invalid_argument,
                what + " should be reported as an invalid argument, not an I/O failure");
    };

    auto past_end = whole;
    past_end.axes.at(0).start = kL;
    expect_rejected(past_end, "a start past the end of an axis");

    auto too_long = whole;
    too_long.axes.at(1).count = kM + 1;
    expect_rejected(too_long, "a count running past the end of an axis");

    auto strided_past_end = whole;
    strided_past_end.axes.at(0) = {0, kL, 2};
    expect_rejected(strided_past_end, "a stride carrying the last element past the end");

    auto zero_stride = whole;
    zero_stride.axes.at(0).stride = 0;
    expect_rejected(zero_stride, "a zero stride");

    auto empty = whole;
    empty.axes.at(2).count = 0;
    expect_rejected(empty, "an empty selection");

    auto wrong_rank = whole;
    wrong_rank.axes.pop_back();
    expect_rejected(wrong_rank, "a request naming fewer axes than the image has");

    // A buffer that cannot hold the result is caught before any bytes are read.
    std::vector<float> small(2);
    const auto short_buffer = sky.Read(whole, {small.data(), small.size() * sizeof(float)});
    Require(!short_buffer, "a destination that is too small should be rejected");
    Require(short_buffer.error().code == carta::zarr::ErrorCode::invalid_argument,
            "a short destination is an invalid argument");
}

void TestReadControls(const carta::zarr::Image& sky) {
    const auto request = WholeImage(sky.descriptor());
    const std::size_t elements = kL * kM * kFrequency * kPolarization * kTime;
    std::vector<float> pixels(elements, 123.0F);

    carta::zarr::ReadOptions cancelled;
    cancelled.cancellation_requested = [] { return true; };
    const auto cancelled_read = sky.Read(request, {pixels.data(), pixels.size() * sizeof(float)}, cancelled);
    Require(!cancelled_read && cancelled_read.error().code == carta::zarr::ErrorCode::cancelled,
            "a cancelled read was not rejected");
    Require(std::all_of(pixels.begin(), pixels.end(), [](float value) { return value == 123.0F; }),
            "a cancelled read modified its destination");

    carta::zarr::ReadOptions expired;
    expired.deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    const auto expired_read = sky.Read(request, {pixels.data(), pixels.size() * sizeof(float)}, expired);
    Require(!expired_read && expired_read.error().code == carta::zarr::ErrorCode::cancelled,
            "a read past its deadline was not rejected");

    carta::zarr::ReadOptions over_budget;
    over_budget.temporary_memory_limit_bytes = elements - 1;
    const auto budget_read = sky.Read(request, {pixels.data(), pixels.size() * sizeof(float)}, over_budget);
    Require(!budget_read && budget_read.error().code == carta::zarr::ErrorCode::buffer_too_small,
            "a masked read over its temporary memory budget was not rejected");
    Require(std::all_of(pixels.begin(), pixels.end(), [](float value) { return value == 123.0F; }),
            "a masked read rejected for memory budget modified its destination");
}

// The header promises that one handle may be read from any number of threads. This cannot prove
// the absence of a race, but it does fail loudly if the shared state a read touches is not actually
// read-only, and it pins the contract next to the code that has to keep it.
void TestConcurrentReads(const carta::zarr::Image& sky) {
    constexpr int kThreads = 8;
    const auto request = WholeImage(sky.descriptor());
    const std::size_t elements = kL * kM * kFrequency * kPolarization * kTime;

    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            for (int repeat = 0; repeat < 4; ++repeat) {
                std::vector<float> pixels(elements);
                const auto read = sky.Read(request, {pixels.data(), pixels.size() * sizeof(float)}, Unmasked());
                if (!read || read.value() != elements) {
                    ++failures;
                    return;
                }
                for (std::uint64_t p = 0; p < kPolarization; ++p) {
                    for (std::uint64_t f = 0; f < kFrequency; ++f) {
                        for (std::uint64_t m = 0; m < kM; ++m) {
                            for (std::uint64_t l = 0; l < kL; ++l) {
                                const float value = pixels.at(LogicalOffset(l, m, f, p));
                                const bool ok = InMissingChunk(l, f, p) ? std::isnan(value)
                                                                        : value == ExpectedValue(l, m, f, p, 0);
                                if (!ok) {
                                    ++failures;
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    Require(failures == 0, "concurrent reads of one image handle disagreed with a serial read");
}

// A read that reports its progress must produce the same pixels as one that does not, must report
// a prefix that is already final, and must stop when the callback says so.
void TestProgressiveRead(const carta::zarr::Image& sky) {
    const auto request = WholeImage(sky.descriptor());
    const std::size_t total = kL * kM * kFrequency * kPolarization * kTime;

    std::vector<float> expected(total);
    Require(static_cast<bool>(sky.Read(request, {expected.data(), expected.size() * sizeof(float)}, Unmasked())),
            "the reference read failed");

    std::vector<float> pixels(total, -1.0f);
    std::vector<std::size_t> reported;
    auto options = Unmasked();
    // A piece is sized by how much chunk data it decodes, so a small limit is what makes this
    // fixture -- forty bytes per chunk -- produce more than one. Without it the whole image fits in
    // a single piece and the split below is never exercised.
    options.temporary_memory_limit_bytes = 160;
    options.progress = [&](std::size_t written, std::size_t elements_total) {
        Require(elements_total == total, "progress should report the request's own element count");
        Require(written > 0 && written <= total, "progress should report a prefix of the destination");
        Require(reported.empty() || written > reported.back(), "the finished prefix should only grow");
        // The point of a prefix rather than a count: what has been reported is already readable.
        for (std::size_t i = 0; i < written; ++i) {
            const bool matches = pixels.at(i) == expected.at(i) ||
                                 (std::isnan(pixels.at(i)) && std::isnan(expected.at(i)));
            Require(matches, "a reported prefix should already hold its final values");
        }
        reported.push_back(written);
        return true;
    };
    const auto read = sky.Read(request, {pixels.data(), pixels.size() * sizeof(float)}, options);
    Require(static_cast<bool>(read), "a progressive read failed");
    Require(read.value() == total, "a progressive read reported the wrong element count");
    Require(!reported.empty() && reported.back() == total, "the last progress report should cover everything");
    for (std::size_t i = 0; i < total; ++i) {
        const bool matches = pixels.at(i) == expected.at(i) || (std::isnan(pixels.at(i)) && std::isnan(expected.at(i)));
        Require(matches, "a progressive read should return what an ordinary one returns");
    }

    Require(reported.size() > 1,
            "the read should have been split; if kDecodedBytesPerRead or the fixture's chunk shape "
            "changed, raise temporary_memory_limit_bytes here or this test stops testing the split");

    // The same read with the pixel mask applied, because the mask buffer and the NaN it writes are
    // per piece too, and a wrong offset there would corrupt every piece but the first.
    std::vector<float> masked_expected(total);
    Require(static_cast<bool>(sky.Read(request, {masked_expected.data(), masked_expected.size() * sizeof(float)})),
            "the masked reference read failed");
    std::vector<float> masked(total, -1.0f);
    carta::zarr::ReadOptions masked_options;
    masked_options.temporary_memory_limit_bytes = 160;
    std::size_t masked_pieces = 0;
    masked_options.progress = [&](std::size_t, std::size_t) {
        ++masked_pieces;
        return true;
    };
    Require(static_cast<bool>(sky.Read(request, {masked.data(), masked.size() * sizeof(float)}, masked_options)),
            "a progressive masked read failed");
    Require(masked_pieces > 1, "the masked read should have been split too");
    for (std::size_t i = 0; i < total; ++i) {
        const bool matches = masked.at(i) == masked_expected.at(i) ||
                             (std::isnan(masked.at(i)) && std::isnan(masked_expected.at(i)));
        Require(matches, "a progressive masked read should return what an ordinary one returns");
    }

    std::vector<float> abandoned(total, -1.0f);
    auto cancelling = Unmasked();
    cancelling.temporary_memory_limit_bytes = 160;
    std::size_t calls = 0;
    cancelling.progress = [&](std::size_t, std::size_t) {
        ++calls;
        return false;
    };
    const auto cancelled = sky.Read(request, {abandoned.data(), abandoned.size() * sizeof(float)}, cancelling);
    Require(!cancelled, "a progress callback returning false should cancel the read");
    Require(cancelled.error().code == carta::zarr::ErrorCode::cancelled, "cancelling should report cancelled");
    Require(calls == 1, "a cancelled read should stop at the piece that refused");
}

}  // namespace

int main() {
    try {
        const auto sky = OpenSky();
        TestAxesAndGeometry(sky);
        TestWholeImage(sky);
        TestSubsetAndStride(sky);
        TestPixelMask(sky);
        TestMaskFusion(sky);
        TestRejectedRequests(sky);
        TestReadControls(sky);
        TestProgressiveRead(sky);
        TestConcurrentReads(sky);
    } catch (const std::exception& error) {
        std::cerr << "pixel read test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
