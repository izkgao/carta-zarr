/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pixel_reader.h"

#include "store_context.h"

#include <tensorstore/array.h>
#include <tensorstore/cast.h>
#include <tensorstore/context.h>
#include <tensorstore/data_type.h>
#include <tensorstore/index.h>
#include <tensorstore/index_space/dim_expression.h>
#include <tensorstore/internal/unowned_to_shared.h>
#include <tensorstore/tensorstore.h>
#include <tensorstore/util/result.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <string>
#include <utility>

namespace carta::zarr::internal::zarr {
namespace {

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

Result<void> CheckReadControl(const ReadOptions& options, std::string_view node) {
    if (options.cancellation_requested && options.cancellation_requested()) {
        return MakeError(ErrorCode::cancelled, "Pixel read was cancelled", std::string(node));
    }
    if (std::chrono::steady_clock::now() >= options.deadline) {
        return MakeError(ErrorCode::cancelled, "Pixel read deadline expired", std::string(node));
    }
    return {};
}

bool SelectionIsWellFormed(const PixelSelection& selection) {
    const auto rank = selection.start.size();
    if (rank == 0 || selection.count.size() != rank || selection.stride.size() != rank ||
        selection.logical_to_stored.size() != rank) {
        return false;
    }
    std::vector<bool> seen(rank, false);
    for (const auto stored : selection.logical_to_stored) {
        if (stored >= rank || seen.at(stored)) {
            return false;
        }
        seen.at(stored) = true;
    }
    return std::all_of(selection.stride.begin(), selection.stride.end(),
                       [](std::uint64_t value) { return value > 0; });
}

template <typename Element>
Result<void> ReadInto(const std::filesystem::path& array_path, const StoreContextPtr& context,
                      std::string_view node, const PixelSelection& selection, tensorstore::DataType target_dtype,
                      Element* destination, std::size_t destination_elements, const ReadOptions& options) {
    if (destination == nullptr) {
        return MakeError(ErrorCode::invalid_argument, "Destination buffer is null", std::string(node));
    }
    if (!SelectionIsWellFormed(selection)) {
        return MakeError(ErrorCode::invalid_argument, "Malformed pixel selection", std::string(node));
    }
    const auto elements = SelectionElementCount(selection);
    if (elements == 0) {
        return MakeError(ErrorCode::invalid_argument, "Pixel selection is empty", std::string(node));
    }
    if (elements > destination_elements) {
        return MakeError(ErrorCode::invalid_argument, "Destination buffer is too small", std::string(node));
    }

    if (!context) {
        return MakeError(ErrorCode::invalid_argument, "Pixel reads require a context", std::string(node));
    }

    auto control = CheckReadControl(options, node);
    if (!control) {
        return control.error();
    }

    try {
        // Reused across calls. A slice read happens once per casacore cursor step, so opening here
        // would make a fixed cost a per-call one.
        auto opened = context->OpenArray(array_path, node);
        if (!opened) {
            return opened.error();
        }
        control = CheckReadControl(options, node);
        if (!control) {
            return control.error();
        }
        auto const store = std::move(opened).value();
        const auto rank = selection.start.size();
        if (static_cast<std::size_t>(store.rank()) != rank) {
            return MakeError(ErrorCode::invalid_argument, "Selection rank does not match the array rank",
                             std::string(node));
        }

        std::vector<tensorstore::Index> start(rank);
        std::vector<tensorstore::Index> count(rank);
        std::vector<tensorstore::Index> stride(rank);
        std::vector<tensorstore::DimensionIndex> order(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            start.at(i) = static_cast<tensorstore::Index>(selection.start.at(i));
            count.at(i) = static_cast<tensorstore::Index>(selection.count.at(i));
            stride.at(i) = static_cast<tensorstore::Index>(selection.stride.at(i));
            order.at(i) = static_cast<tensorstore::DimensionIndex>(selection.logical_to_stored.at(i));
        }

        // Slice in stored order, then move the stored dimensions into logical order. Both are index
        // transforms, so TensorStore composes them into the one copy the read already performs.
        auto sliced = store | tensorstore::AllDims().TranslateSizedInterval(start, count, stride);
        if (!sliced.ok()) {
            return MakeError(ErrorCode::invalid_argument,
                             "Requested region is outside the array: " + sliced.status().ToString(),
                             std::string(node));
        }
        auto transposed = std::move(sliced).value() | tensorstore::Dims(order).Transpose();
        if (!transposed.ok()) {
            return MakeError(ErrorCode::invalid_argument,
                             "Failed to reorder axes: " + transposed.status().ToString(), std::string(node));
        }

        // Conversion rides the same copy, so a float64 or integer array is never materialized in
        // its stored type first.
        auto converted = tensorstore::Cast(std::move(transposed).value(), target_dtype);
        if (!converted.ok()) {
            return MakeError(ErrorCode::unsupported_data_type,
                             "Array cannot be converted to the requested output type: " +
                                 converted.status().ToString(),
                             std::string(node));
        }

        std::vector<tensorstore::Index> shape(rank);
        for (std::size_t i = 0; i < rank; ++i) {
            shape.at(i) = count.at(order.at(i));
        }
        // Axis 0 is the fastest-varying destination dimension, so a logical-order shape over a
        // densely packed buffer is Fortran-ordered. TensorStore requires a shared array here; the
        // caller owns this buffer and the read below is awaited before returning, so a non-owning
        // shared pointer is what the ownership actually is rather than a way around the check.
        auto target = tensorstore::Array(tensorstore::internal::UnownedToShared(destination), shape,
                                         tensorstore::fortran_order);

        auto const read_result = tensorstore::Read(std::move(converted).value(), target).result();
        if (!read_result.ok()) {
            return MakeError(ErrorCode::io_error, "TensorStore read failed: " + read_result.status().ToString(),
                             std::string(node));
        }
        return CheckReadControl(options, node);
    } catch (const std::exception& error) {
        return MakeError(ErrorCode::io_error, error.what(), std::string(node));
    }
}

}  // namespace

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

Result<void> ReadFloat32(const std::filesystem::path& array_path, const StoreContextPtr& context,
                         std::string_view node, const PixelSelection& selection, float* destination,
                         std::size_t destination_elements, const ReadOptions& options) {
    return ReadInto(array_path, context, node, selection, tensorstore::dtype_v<float>, destination,
                    destination_elements, options);
}

Result<void> ReadMaskBytes(const std::filesystem::path& array_path, const StoreContextPtr& context,
                           std::string_view node, const PixelSelection& selection, std::uint8_t* destination,
                           std::size_t destination_elements, const ReadOptions& options) {
    return ReadInto(array_path, context, node, selection, tensorstore::dtype_v<bool>,
                    reinterpret_cast<bool*>(destination), destination_elements, options);
}

}  // namespace carta::zarr::internal::zarr
