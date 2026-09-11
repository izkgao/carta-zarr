/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_ZARR_PIXEL_READER_H_
#define CARTA_ZARR_SRC_ZARR_PIXEL_READER_H_

#include "store.h"

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace carta::zarr::internal::zarr {

/**
 * One hyperslab of an array, addressed in the array's own stored axis order.
 *
 * The permutation is carried alongside rather than applied by the caller because TensorStore can
 * fold it into the same copy that moves the decoded chunk into the destination: transposing here
 * costs a strided write, transposing afterwards costs a second full pass over the data.
 */
struct PixelSelection {
    // All in stored axis order, one entry per stored dimension.
    std::vector<std::uint64_t> start;
    std::vector<std::uint64_t> count;
    std::vector<std::uint64_t> stride;
    // logical_to_stored[i] is the stored dimension that logical axis i names. The destination is
    // written densely in logical order with axis 0 fastest-varying.
    std::vector<std::size_t> logical_to_stored;
};

// Element count the selection produces, or zero when it is malformed.
std::uint64_t SelectionElementCount(const PixelSelection& selection);

/**
 * Read pixels as float32, converting from the stored type during the read.
 *
 * Missing chunks resolve to the array's fill value, which is what a Zarr reader is required to do
 * and is the only definition of "absent pixel" the format offers.
 */
Result<void> ReadFloat32(const std::filesystem::path& array_directory, const StoreContextPtr& context,
                         std::string_view node, const PixelSelection& selection, float* destination,
                         std::size_t destination_elements);

// Read a boolean array as one byte per element, true meaning a good pixel.
Result<void> ReadMaskBytes(const std::filesystem::path& array_directory, const StoreContextPtr& context,
                           std::string_view node, const PixelSelection& selection, std::uint8_t* destination,
                           std::size_t destination_elements);

}  // namespace carta::zarr::internal::zarr

#endif  // CARTA_ZARR_SRC_ZARR_PIXEL_READER_H_
