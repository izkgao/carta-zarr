/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_ZARR_STRING_ARRAY_H_
#define CARTA_ZARR_SRC_ZARR_STRING_ARRAY_H_

#include "carta-zarr/carta_zarr.h"

#include "array_metadata.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace carta::zarr::internal::zarr {

/**
 * Decode a 1-D Zarr v3 "fixed_length_utf32" string array by hand.
 *
 * TensorStore's zarr3 driver does not support string data types, so the chunk is read and decoded
 * directly. Only single-chunk 1-D arrays using the default or v2 chunk key encoding are supported,
 * with a codec chain of: an optional identity transpose, bytes, at most one of zstd/gzip/blosc, and
 * any number of crc32c codecs (checksums are verified). This covers the layouts produced by XRADIO
 * for coordinate label arrays. A missing chunk yields the empty fill value for every element.
 */
Result<std::vector<std::string>> ReadFixedLengthUtf32StringArray(const std::filesystem::path& array_directory,
                                                                 const ArrayMetadata& array_metadata,
                                                                 const nlohmann::json& metadata, std::string_view node);

}  // namespace carta::zarr::internal::zarr

#endif  // CARTA_ZARR_SRC_ZARR_STRING_ARRAY_H_
