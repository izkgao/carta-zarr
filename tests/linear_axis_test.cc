/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// The linear axis fit takes a coordinate's samples and reports the linear description they support.
// It needs no store, so the cases that matter most -- unevenly spaced samples, a reference value no
// sample lands on, degenerate axes -- cost a vector literal each.

#include "schema/xradio/linear_axis.h"

#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using carta::zarr::internal::xradio::FitLinearAxis;
using carta::zarr::internal::xradio::LinearAxisFit;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool Near(double left, double right) {
    return std::abs(left - right) <= 1.0e-9 * std::max({1.0, std::abs(left), std::abs(right)});
}

bool HasDiagnostic(const LinearAxisFit& fit, const std::string& code) {
    for (const auto& diagnostic : fit.diagnostics) {
        if (diagnostic.code == code) {
            return true;
        }
    }
    return false;
}

// Evenly spaced samples with one sitting exactly on the reference value: the reference pixel is that
// sample's 1-based index, and nothing is diagnosed.
void TestExactReferencePixel() {
    const auto fit = FitLinearAxis({100.0, 102.0, 104.0}, 100.0, "frequency");
    Require(fit.uniform, "evenly spaced samples were not reported as uniform");
    Require(fit.increment && Near(*fit.increment, 2.0), "increment was not the sample spacing");
    Require(fit.reference_value && Near(*fit.reference_value, 100.0), "reference value was not carried through");
    Require(fit.reference_pixel && Near(*fit.reference_pixel, 1.0), "reference pixel was not the exact sample index");
    Require(fit.diagnostics.empty(), "an exact fit raised a diagnostic");

    const auto middle = FitLinearAxis({100.0, 102.0, 104.0}, 102.0, "frequency");
    Require(middle.reference_pixel && Near(*middle.reference_pixel, 2.0), "reference pixel was not 1-based");
}

// A direction cosine axis is measured from the tangent point, so a nullopt reference means 0.0.
void TestTangentPointReference() {
    const auto fit = FitLinearAxis({-0.003, -0.002, -0.001, 0.0}, std::nullopt, "l");
    Require(fit.reference_pixel && Near(*fit.reference_pixel, 4.0), "the tangent point was not located");
    Require(fit.diagnostics.empty(), "locating the tangent point exactly raised a diagnostic");
}

// No sample lands on the reference value, so the reference pixel is extrapolated and said to be.
void TestInexactReferencePixel() {
    const auto fit = FitLinearAxis({-0.003, -0.002, -0.001, -0.0005}, std::nullopt, "l");
    Require(HasDiagnostic(fit, "inexact_reference_pixel"), "an extrapolated reference pixel was not diagnosed");
    // Spacing is 0.001 from the first pair, and 0.0 lies three increments past -0.003.
    Require(fit.reference_pixel && Near(*fit.reference_pixel, 4.0), "reference pixel was not linearly extrapolated");
    Require(fit.uniform == false, "samples with a changing spacing were reported as uniform");
}

// Unevenly spaced samples still yield an increment -- a direction axis needs one -- but say so.
void TestNonUniformIsReported() {
    const auto fit = FitLinearAxis({1.4e9, 1.401e9, 1.403e9}, 1.4e9, "frequency");
    Require(!fit.uniform, "unevenly spaced samples were reported as uniform");
    Require(HasDiagnostic(fit, "nonuniform_axis"), "unevenly spaced samples were not diagnosed");
    Require(fit.increment && Near(*fit.increment, 1.0e6), "increment was not taken from the first pair");
    Require(fit.reference_pixel && Near(*fit.reference_pixel, 1.0), "reference pixel was not located");
}

// Two samples are evenly spaced by definition; there is no second spacing to disagree with.
void TestTwoSamplesAreUniform() {
    const auto fit = FitLinearAxis({10.0, 20.0}, 10.0, "frequency");
    Require(fit.uniform, "a two-sample axis was not reported as uniform");
    Require(!HasDiagnostic(fit, "nonuniform_axis"), "a two-sample axis was diagnosed as uneven");
}

// Degenerate axes are not the unevenly sampled axis the diagnostic is about, so they stay silent.
void TestDegenerateAxesAreSilent() {
    const auto single = FitLinearAxis({0.5}, std::nullopt, "l");
    Require(!single.increment, "a single-sample axis reported an increment");
    Require(!single.reference_pixel, "a single-sample axis reported a reference pixel");
    Require(single.diagnostics.empty(), "a single-sample axis raised a diagnostic");
    Require(!single.uniform, "a single-sample axis claimed to be uniform");

    const auto empty = FitLinearAxis({}, std::nullopt, "l");
    Require(!empty.increment && empty.diagnostics.empty(), "an empty axis was not silent");

    // Repeated samples have a zero increment: an increment exists but no pixel can be located from
    // it, and dividing by it would be the only way to try.
    const auto repeated = FitLinearAxis({7.0, 7.0, 7.0}, 9.0, "frequency");
    Require(repeated.increment && Near(*repeated.increment, 0.0), "a zero increment was not reported");
    Require(!repeated.reference_pixel, "a zero increment produced a reference pixel");
    Require(repeated.diagnostics.empty(), "a zero-increment axis raised a diagnostic");
}

// A descending axis is ordinary: increments are signed.
void TestDescendingAxis() {
    const auto fit = FitLinearAxis({104.0, 102.0, 100.0}, 100.0, "frequency");
    Require(fit.uniform, "a descending axis was not reported as uniform");
    Require(fit.increment && Near(*fit.increment, -2.0), "a descending axis lost the sign of its increment");
    Require(fit.reference_pixel && Near(*fit.reference_pixel, 3.0), "a descending axis mislocated its reference pixel");
}

}  // namespace

int main() {
    try {
        TestExactReferencePixel();
        TestTangentPointReference();
        TestInexactReferencePixel();
        TestNonUniformIsReported();
        TestTwoSamplesAreUniform();
        TestDegenerateAxesAreSilent();
        TestDescendingAxis();
        std::cout << "carta-zarr linear axis tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carta-zarr linear axis tests failed: " << error.what() << '\n';
        return 1;
    }
}
