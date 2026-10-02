/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// What a cube histogram counts into, asked directly.
//
// Until this had a seam of its own, the rule for how finely to bin, the per-task accumulators and the
// collection over them could be reached only by a whole walk over a fixture, checked against a
// two-pass answer to a tolerance. Here the pixels are a few rows written out in the test, so every
// answer is exact: the values are chosen so that each provisional bin falls wholly inside one target
// bin, and nothing is split between two.

#include "reduce/provisional_histograms.h"

#include "reducible_image.h"
#include "support/check.h"
#include "support/synthetic_pixel_source.h"
#include "work_pool.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace {

using carta::zarr::AxisRole;
using carta::zarr::ChunkGeometry;
using carta::zarr::CubeHistogramRequest;
using carta::zarr::CubeHistogramResult;
using carta::zarr::ImageDescriptor;
using carta::zarr::internal::ProvisionalHistograms;
using carta::zarr::internal::ReducibleImage;
using carta::zarr::internal::Slab;
using carta::zarr::internal::WorkPool;
using carta::zarr::testing::Require;
using carta::zarr::testing::SyntheticPixelSource;

// A ReducibleImage is what hands out the split; nothing here reads its pixels, so the image behind it
// only has to be one whose axes map.
ImageDescriptor MakeImage() {
    ImageDescriptor descriptor;
    descriptor.id = "SKY";
    descriptor.stored_type = carta::zarr::DataType::float32;
    const struct {
        const char* name;
        AxisRole role;
    } axes[]{{"l", AxisRole::spatial_x},
             {"m", AxisRole::spatial_y},
             {"frequency", AxisRole::spectral},
             {"polarization", AxisRole::polarization},
             {"time", AxisRole::time}};
    const std::size_t storage[]{3, 4, 1, 2, 0};
    for (std::size_t i = 0; i < 5; ++i) {
        carta::zarr::AxisDescriptor axis;
        axis.name = axes[i].name;
        axis.role = axes[i].role;
        axis.length = 1;
        axis.storage_index = storage[i];
        descriptor.axes.push_back(axis);
    }
    return descriptor;
}

ChunkGeometry MakeGeometry() {
    ChunkGeometry geometry;
    geometry.fastest_spatial_axis = AxisRole::spatial_y;
    geometry.chunk_shape = {1, 1, 1, 1, 1};
    return geometry;
}

float Unused(const std::vector<std::uint64_t>&) {
    return 0.0F;
}

// Everything a ProvisionalHistograms needs to be made, held together so that it outlives the test.
struct Fixture {
    explicit Fixture(unsigned int workers)
        : image(MakeImage()), geometry(MakeGeometry()), source(image, geometry, Unused), pool(workers) {}

    ReducibleImage Reducible() {
        auto reducible = ReducibleImage::Of(source, image, geometry, pool);
        Require(static_cast<bool>(reducible), "the image's axes could not be mapped");
        return reducible.value();
    }

    ImageDescriptor image;
    ChunkGeometry geometry;
    SyntheticPixelSource source;
    WorkPool pool;
};

// One plane of `rows` rows of `width` pixels, packed.
Slab PlaneOf(const std::vector<float>& pixels, std::uint64_t width, std::uint64_t rows) {
    Slab slab;
    slab.channel_count = 1;
    slab.pixels = pixels.data();
    slab.stride_u = 1;
    slab.stride_v = width;
    slab.stride_z = width * rows;
    slab.u_count = width;
    slab.v_count = rows;
    return slab;
}

std::string Show(const std::vector<std::uint64_t>& counts) {
    std::string text;
    for (const auto count : counts) {
        text += " " + std::to_string(count);
    }
    return text;
}

// How finely the walk bins. The default is sixteen times what was asked, held to the range tuning.h
// measured; a stated value is taken as given up to kMaxHistogramBins, so a caller can ask for
// coarser as well as finer; and either is rounded up to a power of two, and to at least two.
void TestTheResolutionRule() {
    const auto resolution = [](std::uint32_t bins, std::uint32_t stated, std::size_t expected, const char* what) {
        const auto got = ProvisionalHistograms::Resolution(bins, stated);
        Require(got == expected,
                std::string(what) + ": " + std::to_string(got) + " rather than " + std::to_string(expected));
    };
    resolution(1000, 0, 16384, "a thousand bins by default is sixteen thousand, rounded to a power of two");
    resolution(16, 0, 4096, "a few bins by default is held up to 4,096");
    resolution(0, 0, 4096, "and so is none");
    resolution(8192, 0, 65536, "many bins by default is held down to 65,536");
    resolution(4, 8, 8, "a stated value below the default range is taken as given");
    resolution(4, 65537, 131072, "and so is one above it, rounded up");
    resolution(4, 5, 8, "a stated value is rounded up to a power of two");
    resolution(4, 1, 2, "and to at least two");
    resolution(4, 3000000, 1u << 20, "and held to kMaxHistogramBins");
}

// Four rows, each of one value, read as one slab across four tasks: each task's provisional
// histogram sees one value and seeds its range around it, so the four ranges have nothing in common.
// What comes back is still exact, because the answer is not a merge of the four but each of them
// re-aggregated onto one grid taken from the exact extremes.
//
// A row of 16,384 pixels is the least a task takes, so four of them is four tasks on a pool of four
// -- and if a build's tuning makes it fewer, every number below still holds; only the drifting stops
// being exercised.
void TestTasksWhoseRangesDriftedApartAddUpExactly() {
    constexpr std::uint64_t kWidth = 1U << 14;
    constexpr std::uint64_t kRows = 4;
    std::vector<float> pixels(kWidth * kRows);
    for (std::uint64_t row = 0; row < kRows; ++row) {
        for (std::uint64_t u = 0; u < kWidth; ++u) {
            pixels.at((row * kWidth) + u) = static_cast<float>(row + 1);
        }
    }
    // Ten NaNs in the first row and five infinities in the third: neither is a finite pixel, so both
    // are counted as absent, and neither may move the range.
    for (std::uint64_t u = 0; u < 10; ++u) {
        pixels.at(u) = std::numeric_limits<float>::quiet_NaN();
    }
    for (std::uint64_t u = 0; u < 5; ++u) {
        pixels.at((2 * kWidth) + u) = std::numeric_limits<float>::infinity();
    }

    Fixture fixture(4);
    CubeHistogramRequest request;
    request.bins = 4;
    ProvisionalHistograms histograms(fixture.Reducible(), request);
    histograms.Add(PlaneOf(pixels, kWidth, kRows));
    const auto result = histograms.Collect();

    // Over [1, 4] in four bins, each value lands in a bin of its own.
    const std::vector<std::uint64_t> expected{kWidth - 10, kWidth, kWidth - 5, kWidth};
    Require(result.counts == expected, "counted" + Show(result.counts) + " rather than" + Show(expected));
    Require(result.totals.num_pixels == static_cast<double>((4 * kWidth) - 15), "every finite pixel, once");
    Require(result.totals.nan_count == 15.0, "and the fifteen that are not, as absent");
    Require(result.totals.min == 1.0 && result.totals.max == 4.0, "the extremes are exact");
    const double sum = (1.0 * (kWidth - 10)) + (2.0 * kWidth) + (3.0 * (kWidth - 5)) + (4.0 * kWidth);
    const double sum_sq = (1.0 * (kWidth - 10)) + (4.0 * kWidth) + (9.0 * (kWidth - 5)) + (16.0 * kWidth);
    Require(result.totals.sum == sum, "the sum, exactly, since every term is a small integer");
    Require(result.totals.sum_sq == sum_sq, "and the sum of squares");
    Require(!result.sampled, "a request that did not sample does not say it did");
}

// What is counted is what the slab's strides select, across its planes. The buffer around the
// selected pixels holds a value that would be the maximum if any of it were read.
void TestAReadIsCountedAtItsStrides() {
    constexpr std::uint64_t kPlanes = 2;
    constexpr std::uint64_t kRows = 3;
    constexpr std::uint64_t kColumns = 5;
    // Every other element along a row, rows eleven apart, planes thirty-seven apart.
    constexpr std::uint64_t kStrideU = 2;
    constexpr std::uint64_t kStrideV = 11;
    constexpr std::uint64_t kStrideZ = 37;
    std::vector<float> buffer(kStrideZ * kPlanes, 1000.0F);
    float next = 1.0F;
    for (std::uint64_t z = 0; z < kPlanes; ++z) {
        for (std::uint64_t v = 0; v < kRows; ++v) {
            for (std::uint64_t u = 0; u < kColumns; ++u) {
                buffer.at((z * kStrideZ) + (v * kStrideV) + (u * kStrideU)) = next;
                next += 1.0F;
            }
        }
    }
    Slab slab;
    slab.channel_count = kPlanes;
    slab.pixels = buffer.data();
    slab.stride_u = kStrideU;
    slab.stride_v = kStrideV;
    slab.stride_z = kStrideZ;
    slab.u_count = kColumns;
    slab.v_count = kRows;

    Fixture fixture(1);
    CubeHistogramRequest request;
    request.bins = 3;
    ProvisionalHistograms histograms(fixture.Reducible(), request);
    histograms.Add(slab);
    const auto result = histograms.Collect();

    Require(result.totals.num_pixels == 30.0,
            "thirty selected pixels, not " + std::to_string(result.totals.num_pixels));
    Require(result.totals.min == 1.0 && result.totals.max == 30.0, "and none of the buffer around them");
    Require(result.totals.sum == 465.0, "one to thirty adds up to 465");
    std::uint64_t total = 0;
    for (const auto count : result.counts) {
        total += count;
    }
    Require(total == 30, "and every one of them is in some bin");
}

// Two reads add up, and the answer part of the way through is an answer over what was read so far --
// which is what a walk hands a caller as a snapshot.
void TestASnapshotIsTheAnswerSoFar() {
    std::vector<float> first(8, 2.0F);
    std::vector<float> second(8, 6.0F);
    Fixture fixture(1);
    CubeHistogramRequest request;
    request.bins = 2;
    ProvisionalHistograms histograms(fixture.Reducible(), request);

    histograms.Add(PlaneOf(first, 4, 2));
    const auto part = histograms.Collect();
    Require(part.totals.num_pixels == 8.0 && part.totals.min == 2.0 && part.totals.max == 2.0,
            "the snapshot is over the first read");
    Require(part.counts == std::vector<std::uint64_t>{8, 0}, "a single value is all in the first bin");

    histograms.Add(PlaneOf(second, 4, 2));
    const auto whole = histograms.Collect();
    Require(whole.totals.num_pixels == 16.0 && whole.totals.min == 2.0 && whole.totals.max == 6.0,
            "and the answer at the end is over both");
    Require(whole.counts == std::vector<std::uint64_t>{8, 8}, "one read in each bin, not" + Show(whole.counts));
}

// A walk that read no finite pixel -- or nothing at all -- found no range: the extremes are NaN, the
// counts are zero and there are as many of them as were asked for.
void TestNothingFiniteFoundNoRange() {
    const auto check = [](const CubeHistogramResult& result, double absent, const char* what) {
        Require(result.totals.num_pixels == 0.0 && result.totals.nan_count == absent,
                std::string(what) + ": the wrong counts");
        Require(std::isnan(result.totals.min) && std::isnan(result.totals.max),
                std::string(what) + ": extremes where there is no range");
        Require(result.counts == std::vector<std::uint64_t>(5, 0), std::string(what) + ": five empty bins");
        Require(result.sampled, std::string(what) + ": a sampled request should say so");
    };

    Fixture fixture(2);
    CubeHistogramRequest request;
    request.bins = 5;
    request.spatial_sample = 3;
    ProvisionalHistograms histograms(fixture.Reducible(), request);
    check(histograms.Collect(), 0.0, "before any read");

    const std::vector<float> absent(6, std::numeric_limits<float>::quiet_NaN());
    histograms.Add(PlaneOf(absent, 3, 2));
    check(histograms.Collect(), 6.0, "after a read of nothing finite");
}

}  // namespace

int main() {
    try {
        TestTheResolutionRule();
        TestTasksWhoseRangesDriftedApartAddUpExactly();
        TestAReadIsCountedAtItsStrides();
        TestASnapshotIsTheAnswerSoFar();
        TestNothingFiniteFoundNoRange();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "provisional histograms test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
