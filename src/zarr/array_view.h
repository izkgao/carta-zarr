/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_ZARR_ARRAY_VIEW_H_
#define CARTA_ZARR_SRC_ZARR_ARRAY_VIEW_H_

#include "carta-zarr/carta_zarr.h"

#include "array_metadata.h"

#include <string_view>
#include <utility>
#include <vector>

namespace carta::zarr::internal::zarr {

/**
 * Addresses a Zarr array's values by dimension name rather than by offset.
 *
 * Numeric arrays are read as one flat buffer in C order, so a caller wanting one element of a
 * multi-dimensional array had to derive the strides itself -- which means knowing the array's stored
 * order, the very thing a schema profile should not have to track. This view keeps that knowledge
 * with the metadata that describes it.
 *
 * A dimension the caller does not name is taken at index 0. Naming a dimension the array does not
 * have, or an index past the end of one it does, is an error rather than a value: reading past the
 * buffer means the array on disk is shorter than its metadata claims, and a store in that state
 * should say so rather than yield a number.
 *
 * The view owns nothing. It must not outlive the metadata or the values it was built from, which is
 * why it cannot be copied or stored.
 */
class ArrayView {
public:
    using NamedIndex = std::pair<std::string_view, std::uint64_t>;

    ArrayView(const ArrayMetadata& metadata, const std::vector<double>& values)
        : _metadata(metadata), _values(values) {}
    ArrayView(const ArrayView&) = delete;
    ArrayView& operator=(const ArrayView&) = delete;
    ArrayView(ArrayView&&) = delete;
    ArrayView& operator=(ArrayView&&) = delete;
    ~ArrayView() = default;

    // Takes a vector so a caller can name a dimension conditionally; a braced list still works.
    Result<double> At(const std::vector<NamedIndex>& indices) const;

private:
    const ArrayMetadata& _metadata;
    const std::vector<double>& _values;
};

}  // namespace carta::zarr::internal::zarr

#endif  // CARTA_ZARR_SRC_ZARR_ARRAY_VIEW_H_
