/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "array_metadata.h"

#include "data_type.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <set>
#include <utility>

namespace carta::zarr::internal::zarr {
namespace {

// Return the named codec from a Zarr v3 codec chain, or nullptr when it is absent.
const nlohmann::json* FindCodec(const nlohmann::json* codecs, std::string_view name) {
    if (codecs == nullptr) {
        return nullptr;
    }
    for (const auto& codec : *codecs) {
        if (codec.is_object() && codec.value("name", "") == name) {
            return &codec;
        }
    }
    return nullptr;
}

// Whether every codec in a chain that names itself does so with a string. FindCodec matches codecs
// by name, and a name of any other type throws out of that lookup rather than failing to match --
// so a chain is held to this before anything looks a codec up in it.
bool CodecNamesAreStrings(const nlohmann::json& codecs) {
    return std::all_of(codecs.begin(), codecs.end(), [](const auto& codec) {
        return !codec.is_object() || !codec.contains("name") || codec.at("name").is_string();
    });
}

// Return the bytes-to-bytes compressor in a codec chain, or an empty string when the chain stores
// raw bytes. Checksum and array-to-bytes codecs are not compressors and are ignored here.
std::string FindCompressor(const nlohmann::json* codecs) {
    static constexpr std::array<std::string_view, 3> compressors{"zstd", "gzip", "blosc"};
    for (const auto compressor : compressors) {
        if (FindCodec(codecs, compressor) != nullptr) {
            return std::string(compressor);
        }
    }
    return {};
}

// Reads the inner chunk shape and the compressor that applies to it out of a sharding codec that
// ParseArrayMetadata has already accepted: the configuration is an object, its chunk_shape is an
// array of positive integers, and it has the array's rank. Nothing here checks that again, because
// the only way to reach this function is through metadata that parse turned away otherwise -- and
// both live in this file, so there is no seam for a second caller to arrive through.
void ApplyShardingLayout(const nlohmann::json& sharding, StorageLayout& layout) {
    layout.sharded = true;
    layout.shard_shape = std::move(layout.chunk_shape);
    layout.chunk_shape.clear();

    const auto& configuration = sharding.at("configuration");
    for (const auto& dimension : configuration.at("chunk_shape")) {
        layout.chunk_shape.push_back(dimension.get<std::uint64_t>());
    }
    // The compressor applies to the inner chunks, so look inside the sharding codec first.
    if (configuration.contains("codecs") && configuration.at("codecs").is_array()) {
        layout.compressor = FindCompressor(&configuration.at("codecs"));
    }
}

}  // namespace

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
    return std::all_of(value.begin(), value.end(), [](const auto& element) { return element.is_number(); });
}

bool IsNumericMatrix(const nlohmann::json& value, std::size_t rows, std::size_t columns) {
    if (!value.is_array() || value.size() != rows) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [&](const auto& row) { return IsNumericVector(row, columns); });
}

bool IsRealDataType(std::string_view data_type) {
    const auto* const info = FindDataType(data_type);
    return info != nullptr && info->real;
}

DataType ParseDataType(std::string_view data_type) {
    const auto* const info = FindDataType(data_type);
    return info == nullptr ? DataType::unknown : info->kind;
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

StorageLayout ParseStorageLayout(const ArrayMetadata& metadata) {
    StorageLayout layout;
    layout.chunk_shape = metadata.chunk_shape;

    const nlohmann::json* const codecs = &metadata.codecs;

    // A sharded array's chunk_grid describes the shard; the chunks that are actually decoded are
    // inside the sharding codec, and so is the compressor that applies to them.
    if (const nlohmann::json* sharding = FindCodec(codecs, "sharding_indexed"); sharding != nullptr) {
        ApplyShardingLayout(*sharding, layout);
    }

    if (layout.compressor.empty()) {
        layout.compressor = FindCompressor(codecs);
    }
    return layout;
}

Result<ArrayMetadata> ParseArrayMetadata(const nlohmann::json& metadata, std::string_view node) {
    const std::string node_path(node);
    if (!metadata.is_object()) {
        return Error{ErrorCode::invalid_metadata, "Zarr array metadata must be a JSON object", node_path};
    }
    if (!metadata.contains("node_type") || !metadata.at("node_type").is_string() ||
        metadata.at("node_type").get<std::string>() != "array") {
        return Error{ErrorCode::invalid_metadata, "Zarr node is not an array", node_path};
    }
    if (!metadata.contains("zarr_format") || !IsNonNegativeInteger(metadata.at("zarr_format"))) {
        return Error{ErrorCode::invalid_metadata, "Zarr array metadata has no valid zarr_format", node_path};
    }
    if (metadata.at("zarr_format").get<std::uint64_t>() != 3) {
        return Error{ErrorCode::unsupported_zarr_version, "Only Zarr format 3 arrays are supported", node_path};
    }
    if (!metadata.contains("shape") || !metadata.at("shape").is_array()) {
        return Error{ErrorCode::invalid_metadata, "Zarr array metadata requires a shape", node_path};
    }

    ArrayMetadata result;
    for (const auto& dimension : metadata.at("shape")) {
        if (!IsNonNegativeInteger(dimension)) {
            return Error{ErrorCode::invalid_metadata, "Zarr array shape must contain non-negative integers",
                         node_path};
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
            return Error{ErrorCode::invalid_metadata, "Array dimension_names must match shape rank", node_path};
        }
        for (const auto& dimension : *dimensions) {
            if (!dimension.is_string()) {
                return Error{ErrorCode::invalid_metadata, "Array dimension names must be strings", node_path};
            }
            result.dimension_names.push_back(dimension.get<std::string>());
        }
        std::set<std::string> const unique_dimensions(result.dimension_names.begin(), result.dimension_names.end());
        if (unique_dimensions.size() != result.dimension_names.size()) {
            return Error{ErrorCode::invalid_metadata, "Array dimension names must be unique", node_path};
        }
    }

    if (!metadata.contains("data_type")) {
        return Error{ErrorCode::invalid_metadata, "Zarr array metadata requires data_type", node_path};
    }
    const auto& data_type = metadata.at("data_type");
    if (data_type.is_string()) {
        result.data_type = data_type.get<std::string>();
    } else if (data_type.is_object() && data_type.contains("name") && data_type.at("name").is_string()) {
        result.data_type = data_type.at("name").get<std::string>();
        if (data_type.contains("configuration")) {
            if (!data_type.at("configuration").is_object()) {
                return Error{ErrorCode::invalid_metadata, "Zarr data_type configuration must be an object",
                             node_path};
            }
            result.data_type_configuration = data_type.at("configuration");
        }
    } else {
        return Error{ErrorCode::unsupported_data_type, "Zarr array data_type is not recognized", node_path};
    }

    if (metadata.contains("attributes")) {
        if (!metadata.at("attributes").is_object()) {
            return Error{ErrorCode::invalid_metadata, "Zarr array attributes must be an object", node_path};
        }
        result.attributes = metadata.at("attributes");
    } else {
        result.attributes = nlohmann::json::object();
    }

    if (metadata.contains("codecs") && metadata.at("codecs").is_array()) {
        result.codecs = metadata.at("codecs");
        if (!CodecNamesAreStrings(result.codecs)) {
            return Error{ErrorCode::invalid_metadata, "Zarr codec names must be strings", node_path};
        }
    }
    if (metadata.contains("chunk_key_encoding") && metadata.at("chunk_key_encoding").is_object()) {
        result.chunk_key_encoding = metadata.at("chunk_key_encoding");
    }

    if (!metadata.contains("chunk_grid") || !metadata.at("chunk_grid").is_object() ||
        !metadata.at("chunk_grid").contains("name") || !metadata.at("chunk_grid").at("name").is_string() ||
        metadata.at("chunk_grid").at("name").get<std::string>() != "regular" ||
        !metadata.at("chunk_grid").contains("configuration") ||
        !metadata.at("chunk_grid").at("configuration").is_object() ||
        !metadata.at("chunk_grid").at("configuration").contains("chunk_shape")) {
        return Error{ErrorCode::unsupported_codec, "Only regular Zarr chunk grids are supported", node_path};
    }
    const auto& chunks = metadata.at("chunk_grid").at("configuration").at("chunk_shape");
    if (!chunks.is_array() || chunks.size() != result.shape.size() ||
        !std::all_of(chunks.begin(), chunks.end(), IsPositiveInteger)) {
        return Error{ErrorCode::invalid_metadata, "Zarr chunk_shape must be positive and match shape rank",
                     node_path};
    }
    for (const auto& chunk : chunks) {
        result.chunk_shape.push_back(chunk.get<std::uint64_t>());
    }

    // A sharded array's chunk_grid describes the shard; the chunks that are actually decoded are
    // named inside the sharding codec, and they are checked here for the same reason the outer ones
    // are -- this is where an array that cannot be read is turned away, and an inner chunk shape
    // that is not a positive extent of the array's rank describes nothing.
    //
    // Checked and not kept: ParseStorageLayout reads it out when someone asks how the array is
    // stored. That is a second walk over a handful of integers, once per array, and the alternative
    // is a field here that one caller wants.
    if (const nlohmann::json* const sharding = FindCodec(&result.codecs, "sharding_indexed");
        sharding != nullptr) {
        const auto* inner = sharding->contains("configuration") && sharding->at("configuration").is_object() &&
                                    sharding->at("configuration").contains("chunk_shape")
                                ? &sharding->at("configuration").at("chunk_shape")
                                : nullptr;
        if (inner == nullptr || !inner->is_array() || inner->size() != result.shape.size() ||
            !std::all_of(inner->begin(), inner->end(), IsPositiveInteger)) {
            return Error{ErrorCode::invalid_metadata,
                         "Sharding codec chunk_shape must be positive and match the array rank", node_path};
        }
        const auto& configuration = sharding->at("configuration");
        if (configuration.contains("codecs") && configuration.at("codecs").is_array() &&
            !CodecNamesAreStrings(configuration.at("codecs"))) {
            return Error{ErrorCode::invalid_metadata, "Zarr codec names must be strings", node_path};
        }
    }
    return result;
}

} // namespace carta::zarr::internal::zarr
