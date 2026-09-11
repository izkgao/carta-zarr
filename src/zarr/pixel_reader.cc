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

bool MatchesDataType(std::string_view expected, tensorstore::DataType actual) {
    if (expected == "bool") return actual == tensorstore::dtype_v<bool>;
    if (expected == "int8") return actual == tensorstore::dtype_v<std::int8_t>;
    if (expected == "uint8") return actual == tensorstore::dtype_v<std::uint8_t>;
    if (expected == "int16") return actual == tensorstore::dtype_v<std::int16_t>;
    if (expected == "uint16") return actual == tensorstore::dtype_v<std::uint16_t>;
    if (expected == "int32") return actual == tensorstore::dtype_v<std::int32_t>;
    if (expected == "uint32") return actual == tensorstore::dtype_v<std::uint32_t>;
    if (expected == "int64") return actual == tensorstore::dtype_v<std::int64_t>;
    if (expected == "uint64") return actual == tensorstore::dtype_v<std::uint64_t>;
    if (expected == "float16") return actual == tensorstore::dtype_v<tensorstore::dtypes::float16_t>;
    if (expected == "float32") return actual == tensorstore::dtype_v<float>;
    if (expected == "float64") return actual == tensorstore::dtype_v<double>;
    return false;
}

bool SelectionIsWellFormed(const PixelSelection& selection) {
    const auto rank = selection.start.size();
    if (rank == 0 || selection.count.size() != rank || selection.stride.size() != rank || selection.shape.size() != rank ||
        selection.dimension_names.size() != rank ||
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
                      std::string_view node, std::string_view expected_data_type, const PixelSelection& selection,
                      tensorstore::DataType target_dtype,
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
        const auto actual_shape = store.domain().shape();
        if (actual_shape.size() != rank) {
            return MakeError(ErrorCode::invalid_metadata, "Array rank differs between metadata sources",
                             std::string(node));
        }
        for (std::size_t axis = 0; axis < rank; ++axis) {
            if (actual_shape[axis] != static_cast<tensorstore::Index>(selection.shape.at(axis))) {
                return MakeError(ErrorCode::invalid_metadata,
                                 "Array shape differs between canonical metadata and the array store",
                                 std::string(node));
            }
        }
        const auto actual_dimension_names = store.domain().labels();
        if (actual_dimension_names.size() != rank) {
            return MakeError(ErrorCode::invalid_metadata, "Array dimension names differ between metadata sources",
                             std::string(node));
        }
        for (std::size_t axis = 0; axis < rank; ++axis) {
            if (actual_dimension_names[axis] != selection.dimension_names.at(axis)) {
                return MakeError(ErrorCode::invalid_metadata,
                                 "Array dimension names differ between canonical metadata and the array store",
                                 std::string(node));
            }
        }
        if (!MatchesDataType(expected_data_type, store.dtype())) {
            return MakeError(ErrorCode::invalid_metadata,
                             "Array data type differs between canonical metadata and the array store",
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

Result<PixelSelection> BuildSelection(const ImageDescriptor& descriptor, const ReadRequest& request) {
    const auto rank = descriptor.axes.size();
    if (request.axes.size() != rank) {
        return MakeError(ErrorCode::invalid_argument,
                         "Request has " + std::to_string(request.axes.size()) + " axes but the image has " +
                             std::to_string(rank),
                         descriptor.id);
    }

    PixelSelection selection;
    selection.start.assign(rank, 0);
    selection.count.assign(rank, 0);
    selection.stride.assign(rank, 1);
    selection.shape.assign(rank, 0);
    selection.dimension_names.assign(rank, {});
    selection.logical_to_stored.resize(rank);

    for (std::size_t logical = 0; logical < rank; ++logical) {
        const auto& axis = descriptor.axes.at(logical);
        const auto& range = request.axes.at(logical);
        if (range.stride == 0) {
            return MakeError(ErrorCode::invalid_argument, "Axis '" + axis.name + "' has a zero stride",
                             descriptor.id);
        }
        if (range.count == 0) {
            return MakeError(ErrorCode::invalid_argument, "Axis '" + axis.name + "' selects no elements",
                             descriptor.id);
        }
        // The last selected index, which is what has to fall inside the axis.
        const std::uint64_t span = (range.count - 1) * range.stride;
        if (range.start >= axis.length || span > axis.length - 1 - range.start) {
            return MakeError(ErrorCode::invalid_argument,
                             "Axis '" + axis.name + "' request exceeds its length of " +
                                 std::to_string(axis.length),
                             descriptor.id);
        }
        const auto stored = axis.storage_index;
        if (stored >= rank) {
            return MakeError(ErrorCode::invalid_metadata, "Axis '" + axis.name + "' has an out-of-range storage index",
                             descriptor.id);
        }
        selection.start.at(stored) = range.start;
        selection.count.at(stored) = range.count;
        selection.stride.at(stored) = range.stride;
        selection.shape.at(stored) = axis.length;
        selection.dimension_names.at(stored) = axis.name;
        selection.logical_to_stored.at(logical) = stored;
    }
    return selection;
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
                         std::string_view node, std::string_view expected_data_type, const PixelSelection& selection, float* destination,
                         std::size_t destination_elements, const ReadOptions& options) {
    return ReadInto(array_path, context, node, expected_data_type, selection, tensorstore::dtype_v<float>, destination,
                    destination_elements, options);
}

Result<void> ReadMaskBytes(const std::filesystem::path& array_path, const StoreContextPtr& context,
                           std::string_view node, std::string_view expected_data_type, const PixelSelection& selection, std::uint8_t* destination,
                           std::size_t destination_elements, const ReadOptions& options) {
    return ReadInto(array_path, context, node, expected_data_type, selection, tensorstore::dtype_v<bool>,
                    reinterpret_cast<bool*>(destination), destination_elements, options);
}

}  // namespace carta::zarr::internal::zarr
