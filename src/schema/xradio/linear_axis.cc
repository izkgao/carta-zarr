/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "linear_axis.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
#include <utility>

namespace carta::zarr::internal::xradio {
namespace {

// How close a sample must sit to the reference world value to be treated as landing on it. Scaled
// by magnitude so that it means the same thing for direction cosines and for hertz.
double ExactnessTolerance(const std::vector<double>& values) {
    double maximum = 1.0;
    for (const double value : values) {
        maximum = std::max(maximum, std::abs(value));
    }
    return 1.0e-12 * maximum;
}

std::size_t ClosestIndex(const std::vector<double>& values, double reference) {
    return static_cast<std::size_t>(std::distance(
        values.begin(), std::min_element(values.begin(), values.end(), [reference](double left, double right) {
            return std::abs(left - reference) < std::abs(right - reference);
        })));
}

// Whether the samples are evenly spaced. This is a looser test than exactness above: it asks whether
// consecutive spacings agree, not whether a value lands on a target.
bool IsEvenlySpaced(const std::vector<double>& values, double increment) {
    for (std::size_t index = 2; index < values.size(); ++index) {
        const double spacing = values.at(index) - values.at(index - 1);
        const double tolerance = 1.0e-9 * std::max({1.0, std::abs(increment), std::abs(spacing)});
        if (std::abs(spacing - increment) > tolerance) {
            return false;
        }
    }
    return true;
}

Diagnostic MakeDiagnostic(std::string code, std::string message, std::string_view axis_name) {
    return Diagnostic{std::move(code), std::move(message), std::string(axis_name)};
}

}  // namespace

LinearAxisFit FitLinearAxis(const std::vector<double>& values, std::optional<double> reference,
                            std::string_view axis_name) {
    LinearAxisFit fit;
    if (values.size() < 2) {
        return fit;
    }

    const double increment = values.at(1) - values.at(0);
    fit.increment = increment;
    if (increment == 0.0) {
        return fit;
    }

    fit.uniform = IsEvenlySpaced(values, increment);
    if (!fit.uniform) {
        fit.diagnostics.push_back(
            MakeDiagnostic("nonuniform_axis",
                           "The " + std::string(axis_name) +
                               " coordinate is not evenly spaced; any linear description of it is an approximation",
                           axis_name));
    }

    const double reference_value = reference.value_or(0.0);
    fit.reference_value = reference_value;

    const std::size_t closest = ClosestIndex(values, reference_value);
    if (std::abs(values.at(closest) - reference_value) <= ExactnessTolerance(values)) {
        fit.reference_pixel = static_cast<double>(closest + 1);
    } else {
        fit.reference_pixel = ((reference_value - values.front()) / increment) + 1.0;
        fit.diagnostics.push_back(MakeDiagnostic("inexact_reference_pixel",
                                                 "The " + std::string(axis_name) +
                                                     " coordinate has no sample at its reference world "
                                                     "value; CRPIX was linearly extrapolated",
                                                 axis_name));
    }
    return fit;
}

}  // namespace carta::zarr::internal::xradio
