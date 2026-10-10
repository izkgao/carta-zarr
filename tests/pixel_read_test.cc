/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

// Pixel reads against a fixture whose every value spells its own logical coordinates. The point of
// that encoding is that the axis permutation cannot pass by accident: the fixture's axes all have
// different lengths, so a wrong order changes the shape, and a right shape with a wrong order
// changes the values.

#include "support/check.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <carta-zarr/carta_zarr.h>
#include <unistd.h>

namespace {

// The same image written both ways round: one store puts m last, the other l. Every test below runs
// against both, because a reader that decides anything from where an axis sits rather than from
// what it is named gets a different answer for one of the two.
const char* const kFixtures[]{CARTA_ZARR_PIXEL_FIXTURE, CARTA_ZARR_PIXEL_FIXTURE_L_FASTEST};

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
bool ExpectedGood(std::uint64_t l, std::uint64_t m) {
    return ((l + m) % 3) != 0;
}

using carta::zarr::testing::Require;

carta::zarr::Image OpenSky(const char* kFixture) {
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
    const auto read = sky.Read(request, {pixels.data(), pixels.size()}, Unmasked());
    Require(static_cast<bool>(read), "reading the whole image failed");
    Require(read.value() == pixels.size(), "the whole-image read reported the wrong element count");

    std::size_t missing = 0;
    for (std::uint64_t p = 0; p < kPolarization; ++p) {
        for (std::uint64_t f = 0; f < kFrequency; ++f) {
            for (std::uint64_t m = 0; m < kM; ++m) {
                for (std::uint64_t l = 0; l < kL; ++l) {
                    const float value = pixels.at(LogicalOffset(l, m, f, p));
                    const std::string where = "at l=" + std::to_string(l) + " m=" + std::to_string(m) +
                                              " frequency=" + std::to_string(f) + " polarization=" + std::to_string(p);
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
    const auto read = sky.Read(request, {pixels.data(), pixels.size()}, Unmasked());
    Require(static_cast<bool>(read), "a strided read failed");
    Require(read.value() == pixels.size(), "a strided read reported the wrong element count");

    const std::vector<std::pair<std::uint64_t, std::uint64_t>> selected{{0, 1}, {2, 1}, {0, 3}, {2, 3}};
    for (std::size_t i = 0; i < selected.size(); ++i) {
        Require(pixels.at(i) == ExpectedValue(selected.at(i).first, selected.at(i).second, 0, 1, 0),
                "a strided read returned the wrong element at offset " + std::to_string(i));
    }
}

// A stride says how far apart the elements an axis selects are, so on an axis selecting one it says
// nothing, and any positive one is accepted. TensorStore takes a signed stride, and one of 2^63 or
// more went through as a negative number: the single pixel that read at stride 1 failed at 2^63.
void TestAStrideOverOneElementIsNoStride(const carta::zarr::Image& sky) {
    for (const std::uint64_t stride : {std::uint64_t{1} << 63U, std::numeric_limits<std::uint64_t>::max()}) {
        carta::zarr::ReadRequest request;
        request.axes = {{1, 1, stride}, {3, 1, stride}, {0, 1, stride}, {1, 1, stride}, {0, 1, stride}};
        float pixel = 0.0F;
        const auto read = sky.Read(request, {&pixel, 1}, Unmasked());
        Require(static_cast<bool>(read), "one pixel at a stride of " + std::to_string(stride) +
                                             " was not read: " + (read ? std::string{} : read.error().message));
        Require(pixel == ExpectedValue(1, 3, 0, 1, 0),
                "one pixel at a stride of " + std::to_string(stride) + " read the wrong value");
    }
}

// The flag is not read on its own: it reaches a caller as NaN in the pixels it marks, which is how
// CARTA reports a masked pixel anyway. The descriptor still says there is one, and which it is.
void TestMaskFusion(const carta::zarr::Image& sky) {
    Require(sky.descriptor().has_pixel_mask, "the fixture image declares a flag variable");
    Require(sky.descriptor().pixel_mask_id == "FLAG", "the pixel mask should name the flag variable");

    const auto request = WholeImage(sky.descriptor());
    std::vector<float> pixels(kL * kM * kFrequency * kPolarization * kTime);
    // Masking is the default, so this is the plain two-argument read.
    const auto read = sky.Read(request, {pixels.data(), pixels.size()});
    Require(static_cast<bool>(read), "a masked read failed");

    for (std::uint64_t p = 0; p < kPolarization; ++p) {
        for (std::uint64_t f = 0; f < kFrequency; ++f) {
            for (std::uint64_t m = 0; m < kM; ++m) {
                for (std::uint64_t l = 0; l < kL; ++l) {
                    const float value = pixels.at(LogicalOffset(l, m, f, p));
                    const std::string where = "at l=" + std::to_string(l) + " m=" + std::to_string(m);
                    if (!ExpectedGood(l, m) || InMissingChunk(l, f, p)) {
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
    const carta::zarr::BufferView<float> buffer{pixels.data(), pixels.size()};

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

    // The span of a strided selection is (count - 1) * stride, and in 64 bits these two multiply to
    // exactly 2^64: the span wraps to zero, so a selection running far past the axis looks like it
    // ends at its first element. The element count and TensorStore both refuse it anyway, so this
    // is here to keep the answer a rejection if either of them ever stops refusing.
    auto overflowing_span = whole;
    overflowing_span.axes.at(0) = {0, (std::uint64_t{1} << 32U) + 1, std::uint64_t{1} << 32U};
    expect_rejected(overflowing_span, "a count and stride whose span overflows");

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
    const auto short_buffer = sky.Read(whole, {small.data(), small.size()});
    Require(!short_buffer, "a destination that is too small should be rejected");
    Require(short_buffer.error().code == carta::zarr::ErrorCode::invalid_argument,
            "a short destination is an invalid argument");
}

void TestReadControls(const carta::zarr::Image& sky) {
    const auto request = WholeImage(sky.descriptor());
    const std::size_t elements = kL * kM * kFrequency * kPolarization * kTime;
    std::vector<float> pixels(elements, 123.0F);

    carta::zarr::ReadOptions cancelled;
    cancelled.control.cancellation_requested = [] { return true; };
    const auto cancelled_read = sky.Read(request, {pixels.data(), pixels.size()}, cancelled);
    Require(!cancelled_read && cancelled_read.error().code == carta::zarr::ErrorCode::cancelled,
            "a cancelled read was not rejected");
    Require(std::all_of(pixels.begin(), pixels.end(), [](float value) { return value == 123.0F; }),
            "a cancelled read modified its destination");

    carta::zarr::ReadOptions expired;
    expired.control.deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    const auto expired_read = sky.Read(request, {pixels.data(), pixels.size()}, expired);
    Require(!expired_read && expired_read.error().code == carta::zarr::ErrorCode::cancelled,
            "a read past its deadline was not rejected");

    // A stated ceiling splits the read to fit rather than refusing it, whether or not anyone asked
    // to watch: a caller who says how much memory the read can have is not told no.
    std::vector<float> reference(elements, 0.0F);
    const auto reference_read = sky.Read(request, {reference.data(), reference.size()});
    Require(static_cast<bool>(reference_read), "the unrestricted read this compares against failed");

    carta::zarr::ReadOptions budgeted;
    // Less than one piece covering everything needs, and more than one polarization's worth -- so it
    // has to be split, and splitting is enough.
    budgeted.read_budget_bytes = elements - 1;
    std::vector<float> budgeted_pixels(elements, 0.0F);
    const auto budgeted_read = sky.Read(request, {budgeted_pixels.data(), budgeted_pixels.size()}, budgeted);
    Require(static_cast<bool>(budgeted_read),
            "a read with a memory ceiling was refused instead of split" +
                (budgeted_read ? std::string{} : ": " + budgeted_read.error().message));
    for (std::size_t i = 0; i < elements; ++i) {
        const bool both_nan = std::isnan(budgeted_pixels.at(i)) && std::isnan(reference.at(i));
        Require(both_nan || budgeted_pixels.at(i) == reference.at(i),
                "splitting to fit a memory ceiling changed the pixel at offset " + std::to_string(i));
    }

    // A ceiling no amount of splitting gets under is read a chunk at a time rather than refused:
    // asking for less than a chunk decodes the whole chunk anyway, so one chunk is what a read holds
    // however small its budget, and a caller whose data is chunked that way still gets its pixels.
    carta::zarr::ReadOptions unreachable;
    unreachable.read_budget_bytes = 1;
    std::vector<float> chunk_at_a_time(elements, 0.0F);
    const auto one_chunk = sky.Read(request, {chunk_at_a_time.data(), chunk_at_a_time.size()}, unreachable);
    Require(static_cast<bool>(one_chunk), "a read under a ceiling smaller than one chunk was refused" +
                                              (one_chunk ? std::string{} : ": " + one_chunk.error().message));
    for (std::size_t i = 0; i < elements; ++i) {
        const bool both_nan = std::isnan(chunk_at_a_time.at(i)) && std::isnan(reference.at(i));
        Require(both_nan || chunk_at_a_time.at(i) == reference.at(i),
                "reading a chunk at a time changed the pixel at offset " + std::to_string(i));
    }
}

// A read through a pool of its own returns what a read through the session's does: the pool decides
// what is kept, never what is read. Read twice, so that the second is answered from what the first
// kept, and at zero bytes too, which is the pool that keeps nothing.
void TestReadsThroughAPoolOfTheirOwn(const char* fixture) {
    const auto context = carta::zarr::Context::Create();
    Require(static_cast<bool>(context), "Context::Create failed");
    const auto dataset = carta::zarr::Dataset::Open(context.value(), fixture);
    Require(static_cast<bool>(dataset), "Dataset::Open failed on the pixel fixture");
    const auto sky = dataset.value().OpenImage("SKY");
    Require(static_cast<bool>(sky), "SKY could not be opened");

    const auto request = WholeImage(sky->descriptor());
    const std::size_t elements = kL * kM * kFrequency * kPolarization * kTime;
    std::vector<float> reference(elements, 0.0F);
    Require(static_cast<bool>(sky->Read(request, {reference.data(), reference.size()})),
            "the read through the session's pool failed");

    for (const std::size_t bytes : {std::size_t{0}, std::size_t{64} << 20}) {
        const auto pool = context->NewCachePool(bytes);
        Require(static_cast<bool>(pool), "NewCachePool(" + std::to_string(bytes) + ") failed");
        Require(pool->bytes() == bytes, "a pool of " + std::to_string(bytes) + " bytes says it is another size");
        carta::zarr::ReadOptions options;
        options.control.cache_pool = *pool;
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<float> pixels(elements, 123.0F);
            const auto read = sky->Read(request, {pixels.data(), pixels.size()}, options);
            Require(read && *read == elements, "a read through a pool of " + std::to_string(bytes) + " bytes failed" +
                                                   (read ? std::string{} : ": " + read.error().message));
            for (std::size_t i = 0; i < elements; ++i) {
                const bool both_nan = std::isnan(pixels.at(i)) && std::isnan(reference.at(i));
                Require(both_nan || pixels.at(i) == reference.at(i),
                        "a pool of " + std::to_string(bytes) + " bytes changed the pixel at offset " +
                            std::to_string(i) + " on pass " + std::to_string(pass));
            }
        }
    }
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
                const auto read = sky.Read(request, {pixels.data(), pixels.size()}, Unmasked());
                if (!read || read.value() != elements) {
                    ++failures;
                    return;
                }
                for (std::uint64_t p = 0; p < kPolarization; ++p) {
                    for (std::uint64_t f = 0; f < kFrequency; ++f) {
                        for (std::uint64_t m = 0; m < kM; ++m) {
                            for (std::uint64_t l = 0; l < kL; ++l) {
                                const float value = pixels.at(LogicalOffset(l, m, f, p));
                                const bool ok =
                                    InMissingChunk(l, f, p) ? std::isnan(value) : value == ExpectedValue(l, m, f, p, 0);
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
    Require(static_cast<bool>(sky.Read(request, {expected.data(), expected.size()}, Unmasked())),
            "the reference read failed");

    std::vector<float> pixels(total, -1.0f);
    std::vector<std::size_t> reported;
    auto options = Unmasked();
    // A piece is sized by how much chunk data it decodes, so a small limit is what makes this
    // fixture -- forty bytes per chunk -- produce more than one. Without it the whole image fits in
    // a single piece and the split below is never exercised.
    options.read_budget_bytes = 160;
    const carta::zarr::ProgressCallback progress = [&](std::size_t written, std::size_t elements_total) {
        Require(elements_total == total, "progress should report the request's own element count");
        Require(written > 0 && written <= total, "progress should report a prefix of the destination");
        Require(reported.empty() || written > reported.back(), "the finished prefix should only grow");
        // The point of a prefix rather than a count: what has been reported is already readable.
        for (std::size_t i = 0; i < written; ++i) {
            const bool matches =
                pixels.at(i) == expected.at(i) || (std::isnan(pixels.at(i)) && std::isnan(expected.at(i)));
            Require(matches, "a reported prefix should already hold its final values");
        }
        reported.push_back(written);
        return true;
    };
    const auto read = sky.Read(request, {pixels.data(), pixels.size()}, options, progress);
    Require(static_cast<bool>(read), "a progressive read failed");
    Require(read.value() == total, "a progressive read reported the wrong element count");
    Require(!reported.empty() && reported.back() == total, "the last progress report should cover everything");
    for (std::size_t i = 0; i < total; ++i) {
        const bool matches = pixels.at(i) == expected.at(i) || (std::isnan(pixels.at(i)) && std::isnan(expected.at(i)));
        Require(matches, "a progressive read should return what an ordinary one returns");
    }

    Require(reported.size() > 1,
            "the read should have been split; if kDecodedBytesPerRead or the fixture's chunk shape "
            "changed, raise read_budget_bytes here or this test stops testing the split");

    // The same read with the pixel mask applied, because the mask buffer and the NaN it writes are
    // per piece too, and a wrong offset there would corrupt every piece but the first.
    std::vector<float> masked_expected(total);
    Require(static_cast<bool>(sky.Read(request, {masked_expected.data(), masked_expected.size()})),
            "the masked reference read failed");
    std::vector<float> masked(total, -1.0f);
    carta::zarr::ReadOptions masked_options;
    masked_options.read_budget_bytes = 160;
    std::size_t masked_pieces = 0;
    const carta::zarr::ProgressCallback masked_progress = [&](std::size_t, std::size_t) {
        ++masked_pieces;
        return true;
    };
    Require(static_cast<bool>(sky.Read(request, {masked.data(), masked.size()}, masked_options, masked_progress)),
            "a progressive masked read failed");
    Require(masked_pieces > 1, "the masked read should have been split too");
    for (std::size_t i = 0; i < total; ++i) {
        const bool matches =
            masked.at(i) == masked_expected.at(i) || (std::isnan(masked.at(i)) && std::isnan(masked_expected.at(i)));
        Require(matches, "a progressive masked read should return what an ordinary one returns");
    }

    std::vector<float> abandoned(total, -1.0f);
    auto cancelling = Unmasked();
    cancelling.read_budget_bytes = 160;
    std::size_t calls = 0;
    const carta::zarr::ProgressCallback refusing = [&](std::size_t, std::size_t) {
        ++calls;
        return false;
    };
    const auto cancelled = sky.Read(request, {abandoned.data(), abandoned.size()}, cancelling, refusing);
    Require(!cancelled, "a progress callback returning false should cancel the read");
    Require(cancelled.error().code == carta::zarr::ErrorCode::cancelled, "cancelling should report cancelled");
    Require(calls == 1, "a cancelled read should stop at the piece that refused");
}

// A location is whatever the consumer typed, and a relative one is legal. The store keeps it as
// given -- the filesystem transport holds the root it was handed -- so the one place that makes a
// path absolute, Store::ResolveArrayDirectory, is what stops an opened image from depending on the
// process staying in the directory it was opened from. TensorStore's file kvstore would otherwise
// resolve the array against whatever the working directory has since become.
//
// Metadata is read and cached while the image is being opened, so it is the pixel read that is
// exposed, and this reads pixels after moving away.
void TestAnOpenImageOutlivesTheWorkingDirectory(const char* fixture) {
    const std::filesystem::path located(fixture);
    const auto previous = std::filesystem::current_path();
    struct Restore {
        std::filesystem::path path;
        ~Restore() {
            std::error_code ignored;
            std::filesystem::current_path(path, ignored);
        }
    } const restore{previous};
    std::filesystem::current_path(located.parent_path());

    const auto context = carta::zarr::Context::Create();
    Require(static_cast<bool>(context), "Context::Create failed");
    const auto dataset = carta::zarr::Dataset::Open(context.value(), located.filename().string());
    Require(static_cast<bool>(dataset), "Dataset::Open failed for a relative location" +
                                            (dataset ? std::string{} : ": " + dataset.error().message));
    const auto image = dataset.value().OpenImage("SKY");
    Require(static_cast<bool>(image),
            "OpenImage failed for a relative location" + (image ? std::string{} : ": " + image.error().message));
    Require(image.value().descriptor().spectral.has_value(),
            "the spectral coordinate was not read from a relatively located store");

    std::filesystem::current_path(previous);

    const auto& sky = image.value();
    std::vector<float> pixels(static_cast<std::size_t>(kL * kM * kFrequency * kPolarization * kTime), 0.0F);
    const auto read = sky.Read(WholeImage(sky.descriptor()), {pixels.data(), pixels.size()}, Unmasked());
    Require(static_cast<bool>(read), "an image opened relatively could not be read from another directory" +
                                         (read ? std::string{} : ": " + read.error().message));
    Require(pixels.at(LogicalOffset(1, 2, 1, 2)) == ExpectedValue(1, 2, 1, 2, 0),
            "an image opened relatively read different pixels");
}

// What a prefetch decoded, a read through the same pool finds without going to storage: after it the
// chunks are emptied on disk, and the plane it covered still reads as it did, while the next plane,
// which it did not cover, no longer reads at all. The fixture's flag is emptied too, so the plane
// reading back masked shows that its chunks were decoded as well.
//
// `expected_chunks` counts the pixels' chunks and the flag's, each by its own layout: a flag may be
// chunked finer than its image, and then sampling it where the pixels are sampled leaves some of its
// chunks to be read from storage after all.
void TestAPrefetchedPlaneIsReadFromThePool(const char* fixture, std::uint64_t expected_chunks) {
    const auto copy = std::filesystem::temp_directory_path() / ("carta-zarr-prefetch-" + std::to_string(getpid()));
    std::filesystem::remove_all(copy);
    std::filesystem::copy(fixture, copy, std::filesystem::copy_options::recursive);
    struct Remove {
        std::filesystem::path path;
        ~Remove() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } const remove{copy};

    const auto plane = [](std::uint64_t frequency) {
        carta::zarr::ReadRequest request;
        request.axes = {{0, kL, 1}, {0, kM, 1}, {frequency, 1, 1}, {0, 1, 1}, {0, 1, 1}};
        return request;
    };
    const auto open = [&] {
        const auto context = carta::zarr::Context::Create();
        Require(static_cast<bool>(context), "Context::Create failed");
        const auto dataset = carta::zarr::Dataset::Open(context.value(), copy.string());
        Require(static_cast<bool>(dataset), "Dataset::Open failed on the copy of the fixture");
        const auto image = dataset->OpenImage("SKY");
        Require(static_cast<bool>(image), "SKY could not be opened in the copy");
        return std::make_pair(context.value(), image.value());
    };
    std::vector<float> reference(kL * kM, 0.0F);
    Require(static_cast<bool>(open().second.Read(plane(0), {reference.data(), reference.size()})),
            "the plane could not be read before anything was emptied");

    const auto [context, sky] = open();
    const auto pool = context.NewCachePool(std::size_t{64} << 20);
    Require(static_cast<bool>(pool), "NewCachePool failed");
    carta::zarr::ReadOptions options;
    options.control.cache_pool = *pool;
    const auto chunks = sky.Prefetch(plane(0), options);
    Require(chunks && *chunks == expected_chunks,
            "a prefetch of a plane did not say it decoded the plane's chunks and its flag's" +
                (chunks ? ", it said " + std::to_string(*chunks) : ": " + chunks.error().message));

    for (const char* array : {"SKY", "FLAG"}) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(copy / array / "c")) {
            if (entry.is_regular_file()) {
                std::filesystem::resize_file(entry.path(), 0);
            }
        }
    }

    std::vector<float> pixels(kL * kM, 123.0F);
    const auto read = sky.Read(plane(0), {pixels.data(), pixels.size()}, options);
    Require(static_cast<bool>(read),
            "a prefetched plane went to storage for its chunks" + (read ? std::string{} : ": " + read.error().message));
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        Require((std::isnan(pixels.at(i)) && std::isnan(reference.at(i))) || pixels.at(i) == reference.at(i),
                "a prefetched plane read back differently at offset " + std::to_string(i));
    }
    Require(!sky.Read(plane(1), {pixels.data(), pixels.size()}, options),
            "a plane that was not prefetched still read from emptied chunks, so this shows nothing about the prefetch");
}

// A pool outlives the datasets read through it, and what it keeps of one is not the next one's.
// Here a dataset is read through a pool, closed, rewritten -- every pixel chunk removed, so every
// pixel is the fill value -- and opened again through the same pool: it reads the fill value, as a
// read through the session's pool does. An array handle kept by path would answer with the chunks
// it had decoded before, since a chunk cached after a handle was opened is never asked about again
// (ADR 0015).
void TestAPoolDoesNotCarryOneDatasetsArraysIntoTheNext(const char* fixture) {
    const auto copy = std::filesystem::temp_directory_path() / ("carta-zarr-pool-reopen-" + std::to_string(getpid()));
    std::filesystem::remove_all(copy);
    std::filesystem::copy(fixture, copy, std::filesystem::copy_options::recursive);
    struct Remove {
        std::filesystem::path path;
        ~Remove() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } const remove{copy};

    carta::zarr::ReadRequest plane;
    plane.axes = {{0, kL, 1}, {0, kM, 1}, {0, 1, 1}, {0, 1, 1}, {0, 1, 1}};
    const auto context = carta::zarr::Context::Create();
    Require(static_cast<bool>(context), "Context::Create failed");
    const auto pool = context->NewCachePool(std::size_t{64} << 20);
    Require(static_cast<bool>(pool), "NewCachePool failed");
    auto options = Unmasked();
    options.control.cache_pool = *pool;
    const auto read = [&](const carta::zarr::ReadOptions& with) {
        const auto dataset = carta::zarr::Dataset::Open(context.value(), copy.string());
        Require(static_cast<bool>(dataset), "Dataset::Open failed on the copy of the fixture");
        const auto sky = dataset->OpenImage("SKY");
        Require(static_cast<bool>(sky), "SKY could not be opened in the copy");
        std::vector<float> pixels(kL * kM, 123.0F);
        const auto outcome = sky->Read(plane, {pixels.data(), pixels.size()}, with);
        Require(static_cast<bool>(outcome),
                "the plane did not read" + (outcome ? std::string{} : ": " + outcome.error().message));
        return pixels;
    };
    const auto before = read(options);
    Require(std::none_of(before.begin(), before.end(), [](float value) { return std::isnan(value); }),
            "the plane read is all fill value already, so removing its chunks shows nothing");

    std::filesystem::remove_all(copy / "SKY" / "c");
    const auto after = read(options);
    Require(std::all_of(after.begin(), after.end(), [](float value) { return std::isnan(value); }),
            "a dataset opened again through the same pool read what the pool kept of the one before it");
    const auto session = read(Unmasked());
    Require(std::all_of(session.begin(), session.end(), [](float value) { return std::isnan(value); }),
            "the session's pool did not read the rewritten dataset as rewritten");
}

std::string ReadText(const std::filesystem::path& path) {
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void WriteText(const std::filesystem::path& path, const std::string& text) {
    std::ofstream(path) << text;
}

// Replace the first `from` after `after` in a document's text. The fixtures are written by
// zarr-python and this test links nothing that parses JSON, so a document is edited as the text it is.
std::string Replaced(std::string text, const std::string& after, const std::string& from, const std::string& to) {
    const auto anchor = text.find(after);
    const auto at = anchor == std::string::npos ? anchor : text.find(from, anchor);
    Require(at != std::string::npos, "no '" + from + "' after '" + after + "' to replace");
    return text.replace(at, from.size(), to);
}

// What a chunk keeps in a cache once decoded: its elements at the type they are stored as, and with
// the mask applied the flag chunks it brings at a byte an element -- not the float a read hands back.
// A consumer sizing a cache for the chunks a walk comes back to is sizing it in these; counted in
// floats, a float64 image's cache held half the chunks, and a flagged one's less again.
void TestAChunkSaysWhatItDecodesTo(const char* fixture) {
    const auto bytes = [](const std::string& location, bool masked) {
        const auto context = carta::zarr::Context::Create();
        Require(static_cast<bool>(context), "Context::Create failed");
        const auto dataset = carta::zarr::Dataset::Open(context.value(), location);
        Require(static_cast<bool>(dataset), "Dataset::Open failed on " + location);
        const auto sky = dataset->OpenImage("SKY");
        Require(static_cast<bool>(sky), "SKY could not be opened in " + location);
        carta::zarr::ReadOptions options;
        options.apply_pixel_mask = masked;
        return sky->DecodedChunkBytes(options);
    };
    // Chunks of 1 x 1 x 1 x 2 x 5 float32, and a flag chunked alike.
    Require(bytes(fixture, false) == 10 * 4, "an unmasked float32 chunk of ten is not 40 bytes");
    Require(bytes(fixture, true) == (10 * 4) + 10, "a masked one is not 40 bytes and its flag's 10");
    // The whole flag in one chunk of 120, beside every pixel chunk.
    Require(bytes(CARTA_ZARR_PIXEL_FIXTURE_COARSE_FLAG, true) == (10 * 4) + 120,
            "a chunk did not count the coarse flag chunk it brings");

    const auto copy = std::filesystem::temp_directory_path() / ("carta-zarr-float64-" + std::to_string(getpid()));
    std::filesystem::remove_all(copy);
    std::filesystem::copy(fixture, copy, std::filesystem::copy_options::recursive);
    struct Remove {
        std::filesystem::path path;
        ~Remove() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } const remove{copy};
    const auto sky = copy / "SKY" / "zarr.json";
    WriteText(sky, Replaced(ReadText(sky), "\"data_type\"", "float32", "float64"));
    Require(bytes(copy.string(), true) == (10 * 8) + 10, "a float64 chunk was not counted at eight bytes an element");
}

using Rewrite = std::function<std::string(std::string)>;

// A copy of a fixture with the root's consolidated metadata written in, as zarr-python writes it --
// every child's own document copied into the root -- and removed again with this object. `copies`
// rewrites what the root's copy says about a node and leaves the node's own document as it was;
// rewriting a node's own document afterwards makes the two disagree the other way round.
class ConsolidatedCopy {
public:
    ConsolidatedCopy(const char* fixture, const std::string& name,
                     const std::vector<std::pair<std::string, Rewrite>>& copies = {})
        : _path(std::filesystem::temp_directory_path() / ("carta-zarr-" + name + "-" + std::to_string(getpid()))) {
        std::filesystem::remove_all(_path);
        std::filesystem::copy(fixture, _path, std::filesystem::copy_options::recursive);
        std::string consolidated;
        for (const auto& entry : std::filesystem::directory_iterator(_path)) {
            if (entry.is_directory() && std::filesystem::exists(entry.path() / "zarr.json")) {
                const auto node = entry.path().filename().string();
                auto document = ReadText(entry.path() / "zarr.json");
                for (const auto& [rewritten, rewrite] : copies) {
                    if (rewritten == node) {
                        document = rewrite(std::move(document));
                    }
                }
                consolidated += (consolidated.empty() ? "\"" : ",\"") + node + "\":" + document;
            }
        }
        auto root = ReadText(_path / "zarr.json");
        const auto end = root.find_last_of('}');
        Require(end != std::string::npos, "the fixture's root metadata is not an object");
        root.insert(end, ",\"consolidated_metadata\":{\"kind\":\"inline\",\"must_understand\":false,\"metadata\":{" +
                             consolidated + "}}");
        WriteText(_path / "zarr.json", root);
    }
    ConsolidatedCopy(const ConsolidatedCopy&) = delete;
    ConsolidatedCopy& operator=(const ConsolidatedCopy&) = delete;
    ConsolidatedCopy(ConsolidatedCopy&&) = delete;
    ConsolidatedCopy& operator=(ConsolidatedCopy&&) = delete;
    ~ConsolidatedCopy() {
        std::error_code ignored;
        std::filesystem::remove_all(_path, ignored);
    }

    const std::filesystem::path& path() const { return _path; }

    // Rewrite a node's own document, leaving the root's copy of it as it was.
    void RewriteOwn(const std::string& node, const Rewrite& rewrite) const {
        const auto document = _path / node / "zarr.json";
        WriteText(document, rewrite(ReadText(document)));
    }

    carta::zarr::Result<carta::zarr::Image> OpenSky() const {
        const auto context = carta::zarr::Context::Create();
        Require(static_cast<bool>(context), "Context::Create failed");
        const auto dataset = carta::zarr::Dataset::Open(context.value(), _path.string());
        Require(static_cast<bool>(dataset), "Dataset::Open failed on " + _path.filename().string());
        return dataset->OpenImage("SKY");
    }

private:
    std::filesystem::path _path;
};

void RequireRefusedAtOpen(const carta::zarr::Result<carta::zarr::Image>& image, const std::string& node,
                          const std::string& what) {
    Require(!image, "SKY opened with " + what);
    Require(image.error().code == carta::zarr::ErrorCode::invalid_metadata,
            what + " was refused as something else: " + image.error().message);
    Require(image.error().node_path == node, what + " was refused naming '" + image.error().node_path + "'");
}

// A coordinate is the image's only when it is the array the image was qualified against. With
// consolidated metadata the store qualifies an image on the root's copy, while the coordinate's
// values come from the array's own document; here the copy says two channels, as the image has, and
// the array's own says one. Rather than opening described as two channels with one frequency, it is
// refused as the pixels would be, for an array that is not what the store said.
void TestACoordinateDisagreeingWithItsConsolidatedCopyIsRefused(const char* fixture) {
    const ConsolidatedCopy copy(fixture, "coordinate-copy");
    copy.RewriteOwn("frequency", [](std::string text) { return Replaced(std::move(text), "\"shape\"", "2", "1"); });
    RequireRefusedAtOpen(copy.OpenSky(), "frequency", "a frequency coordinate disagreeing with its copy");
}

// What a coordinate's values mean is read from its attributes -- the unit, the frame, the reference
// and rest frequencies -- and those were read from the root's copy while the values came from the
// array's own document. A copy left behind by a rewrite of the attributes alone described the values
// in the old terms: here the own document gives the rest frequency in GHz, and the image opened
// reporting the copy's figure in Hz beside them. Refused, as a disagreement on the extent is.
void TestACoordinateWhoseAttributesDisagreeWithItsCopyIsRefused(const char* fixture) {
    const ConsolidatedCopy copy(fixture, "attribute-copy");
    copy.RewriteOwn("frequency", [](std::string text) {
        return Replaced(std::move(text), "\"rest_frequency\"", "1420405751.0", "1.420405751");
    });
    RequireRefusedAtOpen(copy.OpenSky(), "frequency", "a frequency coordinate whose attributes disagree with its copy");
}

// An image that opens can be read. Its own array and its flag disagreeing with the root's copy used
// to open, and then fail every read -- and for the flag, only every masked one. Both are refused
// when the image is opened, naming the array that disagrees.
void TestAnImageWhoseArraysDisagreeWithTheirCopiesDoesNotOpen(const char* fixture) {
    const ConsolidatedCopy sky(fixture, "sky-copy");
    sky.RewriteOwn("SKY",
                   [](std::string text) { return Replaced(std::move(text), "\"data_type\"", "float32", "float64"); });
    RequireRefusedAtOpen(sky.OpenSky(), "SKY", "a SKY whose own document holds another data type");

    // Chunked otherwise, every pixel still reads -- TensorStore decodes with the array's own
    // document -- but the chunk geometry the image reports, and every read planned from it, is the
    // copy's.
    const ConsolidatedCopy chunks(fixture, "chunk-copy");
    chunks.RewriteOwn("SKY", [](std::string text) { return Replaced(std::move(text), "\"chunk_shape\"", "2", "1"); });
    RequireRefusedAtOpen(chunks.OpenSky(), "SKY", "a SKY chunked otherwise than its copy says");

    const ConsolidatedCopy flag(fixture, "flag-copy");
    flag.RewriteOwn("FLAG",
                    [](std::string text) { return Replaced(std::move(text), "\"data_type\"", "bool", "uint8"); });
    RequireRefusedAtOpen(flag.OpenSky(), "FLAG", "a flag whose own document holds another data type");
}

// Labels are decoded by this library rather than by TensorStore, and were held to nothing. Decoded
// with the root's copy of their document, a copy saying one label a chunk over chunks of three read
// the first label and then the fill value for the rest, and a label array whose directory was gone
// read as nothing but fill. Both opened an image with labels that were not the store's.
void TestLabelsAreHeldToTheirOwnDocument(const char* fixture) {
    const ConsolidatedCopy rechunked(
        fixture, "label-copy",
        {{"polarization", [](std::string text) { return Replaced(std::move(text), "\"chunk_shape\"", "3", "1"); }}});
    RequireRefusedAtOpen(rechunked.OpenSky(), "polarization", "labels chunked otherwise than their copy says");

    const ConsolidatedCopy missing(fixture, "label-gone");
    std::filesystem::remove_all(missing.path() / "polarization");
    RequireRefusedAtOpen(missing.OpenSky(), "polarization", "a label array whose own document is gone");

    // How the chunks are compressed is not something the copy is held to: the labels are decoded with
    // the document that wrote them, so a copy naming a compressor they were never written with
    // reads them as they are.
    const ConsolidatedCopy recompressed(
        fixture, "label-codec",
        {{"polarization", [](std::string text) {
              const auto codecs = text.find("\"codecs\"");
              const auto end = text.find(']', codecs);
              Require(codecs != std::string::npos && end != std::string::npos, "the labels name no codecs");
              return text.replace(codecs, end + 1 - codecs,
                                  R"("codecs":[{"name":"bytes","configuration":{"endian":"little"}},)"
                                  R"({"name":"zstd","configuration":{"level":1,"checksum":false}}])");
          }}});
    const auto image = recompressed.OpenSky();
    Require(static_cast<bool>(image), "labels whose copy names another compressor did not open: " +
                                          (image ? std::string{} : image.error().message));
    Require(image->descriptor().polarization &&
                image->descriptor().polarization->labels == OpenSky(fixture).descriptor().polarization->labels,
            "labels whose copy names another compressor were not read as written");
}

// What a backslash in a variable's name costs. On POSIX it is a character like any other in a file
// name, and the store accepts the name, so the variable is listed and opens. TensorStore's file
// kvstore reads it as a separator, so its pixels cannot be reached, and a read says so rather than
// looking for the array somewhere else.
void TestAVariableNamedWithABackslashSaysWhyItIsNotRead(const char* fixture) {
    const auto copy = std::filesystem::temp_directory_path() / ("carta-zarr-backslash-" + std::to_string(getpid()));
    std::filesystem::remove_all(copy);
    std::filesystem::copy(fixture, copy, std::filesystem::copy_options::recursive);
    struct Remove {
        std::filesystem::path path;
        ~Remove() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } const remove{copy};
    std::filesystem::copy(copy / "SKY", copy / "SKY\\2", std::filesystem::copy_options::recursive);

    const auto context = carta::zarr::Context::Create();
    Require(static_cast<bool>(context), "Context::Create failed");
    const auto dataset = carta::zarr::Dataset::Open(context.value(), copy.string());
    Require(static_cast<bool>(dataset), "Dataset::Open failed on the copy with a backslash in a name");
    const auto image = dataset->OpenImage("SKY\\2");
    Require(static_cast<bool>(image),
            "the variable named with a backslash did not open: " + (image ? std::string{} : image.error().message));
    std::vector<float> pixels(kL * kM * kFrequency * kPolarization * kTime);
    const auto read = image->Read(WholeImage(image->descriptor()), {pixels.data(), pixels.size()}, Unmasked());
    Require(!read, "the variable named with a backslash was read, from wherever TensorStore found it");
    Require(read.error().code == carta::zarr::ErrorCode::invalid_argument &&
                read.error().message.find("backslash") != std::string::npos,
            "a read of the variable named with a backslash did not say why it was refused: " + read.error().message);
}

}  // namespace

// The bytes of a large dataset are as often as not elsewhere, its arrays links to them. Unconsolidated,
// as this fixture is, the store finds its nodes by walking the directory, and it named a linked array
// by where the link resolved -- ../elsewhere/SKY, which it then refused -- so the dataset opened only
// once it was consolidated.
void TestAnImageWhoseArraysAreLinkedInReads(const char* fixture) {
    const auto root = std::filesystem::temp_directory_path() / ("carta-zarr-linked-" + std::to_string(getpid()));
    std::filesystem::remove_all(root);
    struct Remove {
        std::filesystem::path path;
        ~Remove() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } const remove{root};
    const auto store = root / "store";
    std::filesystem::create_directories(root / "elsewhere");
    std::filesystem::copy(fixture, store, std::filesystem::copy_options::recursive);
    for (const char* const array : {"SKY", "FLAG"}) {
        std::filesystem::rename(store / array, root / "elsewhere" / array);
        std::filesystem::create_directory_symlink(root / "elsewhere" / array, store / array);
    }
    Require(!std::filesystem::exists(store / "zarr.json") ||
                ReadText(store / "zarr.json").find("consolidated_metadata") == std::string::npos,
            "the fixture is consolidated, so its listing is not walked and this asks nothing");

    const auto sky = OpenSky(store.string().c_str());
    TestWholeImage(sky);
    carta::zarr::ReadRequest plane;
    plane.axes = {{0, kL, 1}, {0, kM, 1}, {0, 1, 1}, {0, 1, 1}, {0, 1, 1}};
    std::vector<float> masked(kL * kM);
    Require(static_cast<bool>(sky.Read(plane, {masked.data(), masked.size()})),
            "a masked plane of linked arrays did not read");
}

// What a caller's callback throws is the caller's business, and need not be a std::exception: an
// int thrown from a progress callback escaped Image::Read, which promises a Result whatever happens.
void TestACallbackThrowingAnythingIsAnError(const carta::zarr::Image& sky) {
    const auto request = WholeImage(sky.descriptor());
    std::vector<float> pixels(kL * kM * kFrequency * kPolarization * kTime);
    bool escaped = false;
    try {
        const auto read = sky.Read(request, {pixels.data(), pixels.size()}, Unmasked(),
                                   [](std::size_t, std::size_t) -> bool { throw 42; });
        Require(!read, "a read whose progress callback threw succeeded");
        auto cancelling = Unmasked();
        cancelling.control.cancellation_requested = []() -> bool { throw 42; };
        const auto cancelled = sky.Read(request, {pixels.data(), pixels.size()}, cancelling);
        Require(!cancelled, "a read whose cancellation callback threw succeeded");
    } catch (int) {
        escaped = true;
    }
    Require(!escaped, "an int a callback threw escaped Image::Read");
}

int main() {
    std::vector<carta::zarr::AxisRole> fast_axes;
    for (const char* const fixture : kFixtures) {
        try {
            const auto sky = OpenSky(fixture);
            fast_axes.push_back(sky.chunk_geometry().fastest_spatial_axis);
            TestAxesAndGeometry(sky);
            TestWholeImage(sky);
            TestSubsetAndStride(sky);
            TestAStrideOverOneElementIsNoStride(sky);
            TestMaskFusion(sky);
            TestRejectedRequests(sky);
            TestReadControls(sky);
            TestProgressiveRead(sky);
            TestConcurrentReads(sky);
            TestReadsThroughAPoolOfTheirOwn(fixture);
        } catch (const std::exception& error) {
            std::cerr << "pixel read test failed on " << fixture << ": " << error.what() << "\n";
            return 1;
        }
    }
    try {
        TestAnOpenImageOutlivesTheWorkingDirectory(kFixtures[0]);
        // A plane of the pixels is two chunks along l; its flag is two more, or four when it is
        // chunked half as long.
        TestAPrefetchedPlaneIsReadFromThePool(kFixtures[0], 4);
        TestAPrefetchedPlaneIsReadFromThePool(CARTA_ZARR_PIXEL_FIXTURE_FINE_FLAG, 6);
        TestAPoolDoesNotCarryOneDatasetsArraysIntoTheNext(kFixtures[0]);
        TestAChunkSaysWhatItDecodesTo(kFixtures[0]);
        TestACoordinateDisagreeingWithItsConsolidatedCopyIsRefused(kFixtures[0]);
        TestACoordinateWhoseAttributesDisagreeWithItsCopyIsRefused(kFixtures[0]);
        TestAnImageWhoseArraysDisagreeWithTheirCopiesDoesNotOpen(kFixtures[0]);
        TestLabelsAreHeldToTheirOwnDocument(kFixtures[0]);
        TestAVariableNamedWithABackslashSaysWhyItIsNotRead(kFixtures[0]);
        TestAnImageWhoseArraysAreLinkedInReads(kFixtures[0]);
        TestACallbackThrowingAnythingIsAnError(OpenSky(kFixtures[0]));
        Require(fast_axes.size() == 2 && fast_axes.at(0) != fast_axes.at(1),
                "the two fixtures should disagree about which spatial axis the store varies fastest; "
                "if they agree, one of them was regenerated wrongly and half of this is untested");
    } catch (const std::exception& error) {
        std::cerr << "pixel read test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
