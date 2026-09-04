/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "array_view.h"

#include <string>

namespace carta::zarr::internal::zarr {
namespace {

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

}  // namespace

Result<double> ArrayView::At(const std::vector<NamedIndex>& indices) const {
    const std::size_t rank = _metadata.shape.size();
    if (_metadata.dimension_names.size() != rank) {
        return MakeError(ErrorCode::invalid_metadata, "Array cannot be addressed by name without dimension names");
    }

    // C order: the last dimension varies fastest.
    std::vector<std::uint64_t> strides(rank, 1);
    for (std::size_t dimension = rank; dimension-- > 1;) {
        strides[dimension - 1] = strides[dimension] * _metadata.shape[dimension];
    }

    std::uint64_t offset = 0;
    for (const auto& [name, index] : indices) {
        const auto dimension = FindDimensionIndex(_metadata, name);
        if (!dimension) {
            return MakeError(ErrorCode::invalid_slice, "Array has no dimension named " + std::string(name));
        }
        if (index >= _metadata.shape[*dimension]) {
            return MakeError(ErrorCode::invalid_slice,
                             "Index " + std::to_string(index) + " is past the end of dimension " + std::string(name));
        }
        offset += index * strides[*dimension];
    }

    if (offset >= _values.size()) {
        return MakeError(ErrorCode::invalid_metadata,
                         "Array holds fewer values than its shape declares; the store is truncated");
    }
    return _values[offset];
}

}  // namespace carta::zarr::internal::zarr
