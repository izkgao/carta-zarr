/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_ZARR_VALUE_READER_H_
#define CARTA_ZARR_SRC_ZARR_VALUE_READER_H_

#include "store.h"

#include <filesystem>
#include <string_view>
#include <vector>

namespace carta::zarr::internal::zarr {

/**
 * Read a numeric Zarr array as doubles through TensorStore.
 *
 * This is the only translation unit that reads array data through TensorStore, so a consumer that
 * needs metadata but never coordinate values -- a schema profile and its tests -- can link without
 * it. A null context uses TensorStore's own default resources.
 */
Result<std::vector<double>> ReadNumericValues(const std::filesystem::path& array_directory,
                                              const StoreContextPtr& context,
                                              std::string_view node);

}  // namespace carta::zarr::internal::zarr

#endif  // CARTA_ZARR_SRC_ZARR_VALUE_READER_H_
