/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_REDUCE_AXIS_MAP_H_
#define CARTA_ZARR_SRC_REDUCE_AXIS_MAP_H_

#include "carta-zarr/types.h"
#include "carta-zarr/result.h"

#include <cstddef>
#include <string>

namespace carta::zarr::internal {

// Where each role sits among the logical axes. Shared by every walk that reads planes, because they
// all have to find x, y and the spectrum without assuming they are the first three.
struct AxisMap {
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t spectral = 0;
    bool has_polarization = false;
    std::size_t polarization = 0;
    bool has_time = false;
    std::size_t time = 0;
};

inline Result<AxisMap> MapAxes(const ImageDescriptor& descriptor) {
    AxisMap map;
    bool has_x = false;
    bool has_y = false;
    bool has_spectral = false;
    for (std::size_t i = 0; i < descriptor.axes.size(); ++i) {
        const auto& axis = descriptor.axes.at(i);
        switch (axis.role) {
            case AxisRole::spatial_x: map.x = i; has_x = true; break;
            case AxisRole::spatial_y: map.y = i; has_y = true; break;
            case AxisRole::spectral: map.spectral = i; has_spectral = true; break;
            case AxisRole::polarization: map.polarization = i; map.has_polarization = true; break;
            case AxisRole::time: map.time = i; map.has_time = true; break;
            case AxisRole::other:
                // Taking index 0 of an axis nobody named would report a number for a plane the
                // caller never asked about.
                if (axis.length != 1) {
                    return Error{ErrorCode::not_implemented,
                                 "Axis '" + axis.name + "' has no known role and is not degenerate",
                                 descriptor.id};
                }
                break;
        }
    }
    if (!has_x || !has_y || !has_spectral) {
        return Error{ErrorCode::not_implemented, "A plane walk needs both spatial axes and a spectral axis",
                     descriptor.id};
    }
    return map;
}

// Of the two spatial axes, the one the store varies fastest -- the last one written. A walk takes
// this as its own inner axis so that a plane arrives without being transposed; see
// ChunkGeometry::fastest_spatial_axis.
inline bool SpatialYIsFastest(const ChunkGeometry& geometry) {
    return geometry.fastest_spatial_axis == AxisRole::spatial_y;
}

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_AXIS_MAP_H_
