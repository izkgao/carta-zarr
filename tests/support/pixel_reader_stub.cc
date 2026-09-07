/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "zarr/pixel_reader.h"

#include <limits>
#include <string>

namespace carta::zarr::internal::zarr {

// Companion to value_reader_stub.cc: the schema profile tests link without TensorStore. The
// in-memory transport holds no array data, so a pixel read has no answer to give and
// unsupported_transport is the honest one. Only SelectionElementCount is real, because it is
// arithmetic over the request rather than a read.

std::uint64_t SelectionElementCount(const PixelSelection& selection) {
    if (selection.count.empty()) {
        return 0;
    }
    std::uint64_t elements = 1;
    for (const auto value : selection.count) {
        if (value == 0 || elements > std::numeric_limits<std::uint64_t>::max() / value) {
            return 0;
        }
        elements *= value;
    }
    return elements;
}

Result<void> ReadFloat32(const std::filesystem::path&, const StoreContextPtr&, std::string_view node,
                         const PixelSelection&, float*, std::size_t) {
    return Error{ErrorCode::unsupported_transport, "This build reads no pixels", std::string(node)};
}

Result<void> ReadMaskBytes(const std::filesystem::path&, const StoreContextPtr&, std::string_view node,
                           const PixelSelection&, std::uint8_t*, std::size_t) {
    return Error{ErrorCode::unsupported_transport, "This build reads no pixels", std::string(node)};
}

}  // namespace carta::zarr::internal::zarr
