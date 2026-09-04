/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "zarr/value_reader.h"

#include <string>

namespace carta::zarr::internal::zarr {

// The schema profile tests link without TensorStore. This is not a shortcut around the real reader:
// an in-memory transport holds no array data, so a coordinate value read has no answer to give and
// unsupported_transport is the honest one. Probing and discovery read metadata only, so nothing
// under test reaches this.
Result<std::vector<double>> ReadNumericValues(const std::filesystem::path&, const StoreContextPtr&,
                                              std::string_view node) {
    return Error{ErrorCode::unsupported_transport, "This build reads no array values", std::string(node)};
}

}  // namespace carta::zarr::internal::zarr
