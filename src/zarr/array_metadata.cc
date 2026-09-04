/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "array_metadata.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <set>
#include <utility>

namespace carta::zarr::internal::zarr {
namespace {

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

bool IsNumeric(const nlohmann::json& value) {
    return value.is_number();
}

} // namespace

bool IsNonNegativeInteger(const nlohmann::json& value) {
    return value.is_number_unsigned() || (value.is_number_integer() && value.get<std::int64_t>() >= 0);
}

bool IsPositiveInteger(const nlohmann::json& value) {
    return IsNonNegativeInteger(value) && value.get<std::uint64_t>() > 0;
}

bool IsNumericVector(const nlohmann::json& value, std::size_t length) {
    if (!value.is_array() || value.size() != length) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), IsNumeric);
}

bool IsNumericMatrix(const nlohmann::json& value, std::size_t rows, std::size_t columns) {
    if (!value.is_array() || value.size() != rows) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [&](const auto& row) { return IsNumericVector(row, columns); });
}

bool IsRealDataType(std::string_view data_type) {
    static constexpr std::array<std::string_view, 11> types{"int8",  "uint8",  "int16",   "uint16",  "int32",  "uint32",
                                                            "int64", "uint64", "float16", "float32", "float64"};
    return std::find(types.begin(), types.end(), data_type) != types.end();
}

DataType ParseDataType(std::string_view data_type) {
    static const std::map<std::string_view, DataType> types{
        {"int8", DataType::int8},       {"uint8", DataType::uint8},     {"int16", DataType::int16},
        {"uint16", DataType::uint16},   {"int32", DataType::int32},     {"uint32", DataType::uint32},
        {"int64", DataType::int64},     {"uint64", DataType::uint64},   {"float16", DataType::float16},
        {"float32", DataType::float32}, {"float64", DataType::float64}, {"bool", DataType::boolean},
    };
    const auto found = types.find(data_type);
    return found == types.end() ? DataType::unknown : found->second;
}

bool IsFixedLengthUtf32(const ArrayMetadata& metadata) {
    if (metadata.data_type != "fixed_length_utf32" || !metadata.data_type_configuration.is_object() ||
        !metadata.data_type_configuration.contains("length_bytes")) {
        return false;
    }
    return IsPositiveInteger(metadata.data_type_configuration.at("length_bytes")) &&
           metadata.data_type_configuration.at("length_bytes").get<std::uint64_t>() % 4 == 0;
}

std::optional<std::size_t> FindDimensionIndex(const ArrayMetadata& metadata, std::string_view name) {
    const auto found = std::find(metadata.dimension_names.begin(), metadata.dimension_names.end(), name);
    if (found == metadata.dimension_names.end()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(std::distance(metadata.dimension_names.begin(), found));
}

const char* ErrorCodeName(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::not_found: return "not_found";
    case ErrorCode::not_zarr: return "not_zarr";
    case ErrorCode::unsupported_transport: return "unsupported_transport";
    case ErrorCode::unsupported_zarr_version: return "unsupported_zarr_version";
    case ErrorCode::unsupported_schema: return "unsupported_schema";
    case ErrorCode::unsupported_schema_version: return "unsupported_schema_version";
    case ErrorCode::ambiguous_schema: return "ambiguous_schema";
    case ErrorCode::invalid_argument: return "invalid_argument";
    case ErrorCode::invalid_metadata: return "invalid_metadata";
    case ErrorCode::unsupported_data_type: return "unsupported_data_type";
    case ErrorCode::unsupported_codec: return "unsupported_codec";
    case ErrorCode::invalid_slice: return "invalid_slice";
    case ErrorCode::buffer_too_small: return "buffer_too_small";
    case ErrorCode::io_error: return "io_error";
    case ErrorCode::decode_error: return "decode_error";
    case ErrorCode::cancelled: return "cancelled";
    case ErrorCode::not_implemented: return "not_implemented";
    }
    return "unknown";
}

Result<ArrayMetadata> ParseArrayMetadata(const nlohmann::json& metadata, std::string_view node) {
    const std::string node_path(node);
    if (!metadata.is_object()) {
        return MakeError(ErrorCode::invalid_metadata, "Zarr array metadata must be a JSON object", node_path);
    }
    if (!metadata.contains("node_type") || !metadata.at("node_type").is_string() ||
        metadata.at("node_type").get<std::string>() != "array") {
        return MakeError(ErrorCode::invalid_metadata, "Zarr node is not an array", node_path);
    }
    if (!metadata.contains("zarr_format") || !IsNonNegativeInteger(metadata.at("zarr_format"))) {
        return MakeError(ErrorCode::invalid_metadata, "Zarr array metadata has no valid zarr_format", node_path);
    }
    if (metadata.at("zarr_format").get<std::uint64_t>() != 3) {
        return MakeError(ErrorCode::unsupported_zarr_version, "Only Zarr format 3 arrays are supported", node_path);
    }
    if (!metadata.contains("shape") || !metadata.at("shape").is_array()) {
        return MakeError(ErrorCode::invalid_metadata, "Zarr array metadata requires a shape", node_path);
    }

    ArrayMetadata result;
    for (const auto& dimension : metadata.at("shape")) {
        if (!IsNonNegativeInteger(dimension)) {
            return MakeError(ErrorCode::invalid_metadata, "Zarr array shape must contain non-negative integers",
                             node_path);
        }
        result.shape.push_back(dimension.get<std::uint64_t>());
    }

    const nlohmann::json* dimensions = nullptr;
    if (metadata.contains("dimension_names")) {
        dimensions = &metadata.at("dimension_names");
    } else if (metadata.contains("attributes") && metadata.at("attributes").is_object() &&
               metadata.at("attributes").contains("dimension_names")) {
        // Some XRADIO v1.2 coordinate fixtures store this field in attributes.
        dimensions = &metadata.at("attributes").at("dimension_names");
    }
    if (dimensions != nullptr) {
        if (!dimensions->is_array() || dimensions->size() != result.shape.size()) {
            return MakeError(ErrorCode::invalid_metadata, "Array dimension_names must match shape rank", node_path);
        }
        for (const auto& dimension : *dimensions) {
            if (!dimension.is_string()) {
                return MakeError(ErrorCode::invalid_metadata, "Array dimension names must be strings", node_path);
            }
            result.dimension_names.push_back(dimension.get<std::string>());
        }
        std::set<std::string> const unique_dimensions(result.dimension_names.begin(), result.dimension_names.end());
        if (unique_dimensions.size() != result.dimension_names.size()) {
            return MakeError(ErrorCode::invalid_metadata, "Array dimension names must be unique", node_path);
        }
    }

    if (!metadata.contains("data_type")) {
        return MakeError(ErrorCode::invalid_metadata, "Zarr array metadata requires data_type", node_path);
    }
    const auto& data_type = metadata.at("data_type");
    if (data_type.is_string()) {
        result.data_type = data_type.get<std::string>();
    } else if (data_type.is_object() && data_type.contains("name") && data_type.at("name").is_string()) {
        result.data_type = data_type.at("name").get<std::string>();
        if (data_type.contains("configuration")) {
            if (!data_type.at("configuration").is_object()) {
                return MakeError(ErrorCode::invalid_metadata, "Zarr data_type configuration must be an object",
                                 node_path);
            }
            result.data_type_configuration = data_type.at("configuration");
        }
    } else {
        return MakeError(ErrorCode::unsupported_data_type, "Zarr array data_type is not recognized", node_path);
    }

    if (metadata.contains("attributes")) {
        if (!metadata.at("attributes").is_object()) {
            return MakeError(ErrorCode::invalid_metadata, "Zarr array attributes must be an object", node_path);
        }
        result.attributes = metadata.at("attributes");
    } else {
        result.attributes = nlohmann::json::object();
    }

    if (!metadata.contains("chunk_grid") || !metadata.at("chunk_grid").is_object() ||
        !metadata.at("chunk_grid").contains("name") || !metadata.at("chunk_grid").at("name").is_string() ||
        metadata.at("chunk_grid").at("name").get<std::string>() != "regular" ||
        !metadata.at("chunk_grid").contains("configuration") ||
        !metadata.at("chunk_grid").at("configuration").is_object() ||
        !metadata.at("chunk_grid").at("configuration").contains("chunk_shape")) {
        return MakeError(ErrorCode::unsupported_codec, "Only regular Zarr chunk grids are supported", node_path);
    }
    const auto& chunks = metadata.at("chunk_grid").at("configuration").at("chunk_shape");
    if (!chunks.is_array() || chunks.size() != result.shape.size() ||
        !std::all_of(chunks.begin(), chunks.end(), IsPositiveInteger)) {
        return MakeError(ErrorCode::invalid_metadata, "Zarr chunk_shape must be positive and match shape rank",
                         node_path);
    }
    for (const auto& chunk : chunks) {
        result.chunk_shape.push_back(chunk.get<std::uint64_t>());
    }
    return result;
}

} // namespace carta::zarr::internal::zarr
