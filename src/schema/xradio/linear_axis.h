/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_SCHEMA_XRADIO_LINEAR_AXIS_H_
#define CARTA_ZARR_SRC_SCHEMA_XRADIO_LINEAR_AXIS_H_

#include "carta-zarr/carta_zarr.h"

#include <optional>
#include <string_view>
#include <vector>

namespace carta::zarr::internal::xradio {

// The linear description of a sampled coordinate: the reference pixel, reference value, and
// increment a consumer needs to build a linear axis. Absent fields mean the samples do not support
// one; the consumer must build a tabular axis from the values instead.
struct LinearAxisFit {
    std::optional<double> reference_pixel;  // CRPIX, 1-based
    std::optional<double> reference_value;  // CRVAL, in the coordinate's own unit
    std::optional<double> increment;        // CDELT, in the coordinate's own unit
    bool uniform = false;
    std::vector<Diagnostic> diagnostics;
};

/**
 * Fit a linear description to a coordinate's samples.
 *
 * `reference` is the world value the reference pixel should land on; nullopt asks for the tangent
 * point at 0.0, which is what a direction cosine axis is measured from. `axis_name` names the axis
 * in any diagnostic raised.
 *
 * The fit reports rather than decides. A direction axis is linear by construction, so its caller
 * uses the increment even when `uniform` is false; a spectral axis need not be evenly spaced, so its
 * caller withholds the whole linear description in that case. Both surface the diagnostics.
 *
 * Values are returned in the unit they arrived in. A caller reporting a different unit scales the
 * increment afterwards, which is safe because the reference pixel is derived from the raw samples.
 *
 * A degenerate axis -- fewer than two samples, or a zero increment -- yields the increment where one
 * exists, no reference pixel, and no diagnostic. It is not evenly spaced, but neither is it the
 * unevenly sampled axis the diagnostic is about.
 */
LinearAxisFit FitLinearAxis(const std::vector<double>& values, std::optional<double> reference,
                            std::string_view axis_name);

}  // namespace carta::zarr::internal::xradio

#endif  // CARTA_ZARR_SRC_SCHEMA_XRADIO_LINEAR_AXIS_H_
