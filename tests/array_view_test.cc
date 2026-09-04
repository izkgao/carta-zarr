/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// ArrayView is where a Zarr array's stored order stops being the caller's problem. The cases that
// matter are the ones a schema profile used to get wrong silently: a dimension it does not have, an
// index past the end of one it does, and a buffer shorter than the shape claims.

#include "zarr/array_view.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using carta::zarr::ErrorCode;
using carta::zarr::internal::zarr::ArrayMetadata;
using carta::zarr::internal::zarr::ArrayView;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// (time, frequency, polarization) = (2, 3, 2), values numbered in C order.
ArrayMetadata BeamShapedMetadata() {
    ArrayMetadata metadata;
    metadata.shape = {2, 3, 2};
    metadata.dimension_names = {"time", "frequency", "polarization"};
    return metadata;
}

std::vector<double> CountingValues(std::size_t count) {
    std::vector<double> values;
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        values.push_back(static_cast<double>(index));
    }
    return values;
}

void TestAddressesByName() {
    const auto metadata = BeamShapedMetadata();
    const auto values = CountingValues(12);
    const ArrayView view(metadata, values);

    // C order: polarization varies fastest, then frequency, then time.
    Require(view.At({{"time", 0}, {"frequency", 0}, {"polarization", 0}}).value() == 0.0, "origin was misaddressed");
    Require(view.At({{"time", 0}, {"frequency", 0}, {"polarization", 1}}).value() == 1.0,
            "the fastest dimension was not the last one");
    Require(view.At({{"time", 0}, {"frequency", 1}, {"polarization", 0}}).value() == 2.0,
            "the middle dimension used the wrong stride");
    Require(view.At({{"time", 1}, {"frequency", 0}, {"polarization", 0}}).value() == 6.0,
            "the slowest dimension used the wrong stride");
    Require(view.At({{"time", 1}, {"frequency", 2}, {"polarization", 1}}).value() == 11.0,
            "the last element was misaddressed");
}

// Naming the dimensions in a different order must not change the element addressed. This is the
// property that lets a profile stop tracking a store's stored order.
void TestOrderOfNamesDoesNotMatter() {
    const auto metadata = BeamShapedMetadata();
    const auto values = CountingValues(12);
    const ArrayView view(metadata, values);

    Require(view.At({{"time", 1}, {"frequency", 2}, {"polarization", 1}}).value() ==
                view.At({{"polarization", 1}, {"time", 1}, {"frequency", 2}}).value(),
            "the order the dimensions were named changed the element addressed");
}

void TestUnnamedDimensionsAreZero() {
    const auto metadata = BeamShapedMetadata();
    const auto values = CountingValues(12);
    const ArrayView view(metadata, values);

    Require(
        view.At({{"frequency", 1}}).value() == view.At({{"time", 0}, {"frequency", 1}, {"polarization", 0}}).value(),
        "an unnamed dimension was not taken at index 0");
}

void TestUnknownDimensionIsAnError() {
    const auto metadata = BeamShapedMetadata();
    const auto values = CountingValues(12);
    const ArrayView view(metadata, values);

    const auto missing = view.At({{"beam_params_label", 0}});
    Require(!missing && missing.error().code == ErrorCode::invalid_slice,
            "a dimension the array does not have was not reported as an invalid slice");
}

void TestIndexPastTheEndIsAnError() {
    const auto metadata = BeamShapedMetadata();
    const auto values = CountingValues(12);
    const ArrayView view(metadata, values);

    const auto past = view.At({{"frequency", 3}});
    Require(!past && past.error().code == ErrorCode::invalid_slice,
            "an index past the end of a dimension was not reported as an invalid slice");
}

// The case the old hand-rolled indexing turned into a silent 0.0: metadata promising more elements
// than the store actually holds.
void TestTruncatedBufferIsAnError() {
    const auto metadata = BeamShapedMetadata();
    const auto values = CountingValues(4);
    const ArrayView view(metadata, values);

    Require(view.At({{"time", 0}, {"frequency", 1}, {"polarization", 0}}).value() == 2.0,
            "an element inside a short buffer was not readable");
    const auto truncated = view.At({{"time", 1}, {"frequency", 2}, {"polarization", 1}});
    Require(!truncated && truncated.error().code == ErrorCode::invalid_metadata,
            "reading past a truncated buffer did not report the store as truncated");
}

void TestMissingDimensionNamesIsAnError() {
    ArrayMetadata metadata;
    metadata.shape = {2, 3};
    const auto values = CountingValues(6);
    const ArrayView view(metadata, values);

    const auto unnamed = view.At({{"time", 0}});
    Require(!unnamed && unnamed.error().code == ErrorCode::invalid_metadata,
            "an array without dimension names was addressed by name anyway");
}

}  // namespace

int main() {
    try {
        TestAddressesByName();
        TestOrderOfNamesDoesNotMatter();
        TestUnnamedDimensionsAreZero();
        TestUnknownDimensionIsAnError();
        TestIndexPastTheEndIsAnError();
        TestTruncatedBufferIsAnError();
        TestMissingDimensionNamesIsAnError();
        std::cout << "carta-zarr array view tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carta-zarr array view tests failed: " << error.what() << '\n';
        return 1;
    }
}
