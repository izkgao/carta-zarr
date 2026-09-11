/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "store.h"

#include "zarr/array_metadata.h"
#include "zarr/pixel_reader.h"
#include "zarr/string_array.h"
#include "zarr/transport.h"
#include "zarr/value_reader.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace carta::zarr::internal {
namespace {

namespace zarr_metadata = ::carta::zarr::internal::zarr;

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

// Parse metadata bytes handed up by a Transport. Parsing lives above the seam so that every
// Transport reports a malformed node the same way.
Result<nlohmann::json> ParseNodeMetadata(const std::string& bytes, std::string_view node_path) {
    try {
        return nlohmann::json::parse(bytes);
    } catch (const nlohmann::json::parse_error& error) {
        return MakeError(ErrorCode::invalid_metadata, "Invalid JSON in Zarr metadata: " + std::string(error.what()),
                         std::string(node_path));
    } catch (const std::exception& error) {
        return MakeError(ErrorCode::io_error, "Unable to read Zarr metadata: " + std::string(error.what()),
                         std::string(node_path));
    }
}

// A node name is a relative path carrying no ".." component. Validating it here rather than in a
// Transport holds every Transport to the same rule, and yields the key the caches are stored under.
Result<std::string> NormalizeNodeName(std::string_view node) {
    const std::filesystem::path relative(node);
    if (relative.empty() || relative.is_absolute() || relative.has_root_path()) {
        return MakeError(ErrorCode::invalid_argument, "Invalid Zarr node path", std::string(node));
    }
    for (const auto& part : relative) {
        if (part == "..") {
            return MakeError(ErrorCode::invalid_argument, "Invalid Zarr node path", std::string(node));
        }
    }
    return relative.generic_string();
}

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

std::string NormalizeMetadataKey(std::string key) {
    while (!key.empty() && key.front() == '/') {
        key.erase(key.begin());
    }
    constexpr std::string_view suffix = "/zarr.json";
    if (key.size() >= suffix.size() && key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
        key.erase(key.size() - suffix.size());
    }
    if (key == "zarr.json") {
        key.clear();
    }
    return key;
}

void CollectConsolidatedMetadata(const nlohmann::json& metadata, const std::string& prefix,
                                 std::map<std::string, nlohmann::json>& output) {
    if (!metadata.is_object()) {
        return;
    }
    for (const auto& [key, value] : metadata.items()) {
        std::string path = prefix.empty() ? key : prefix + "/" + key;
        path = NormalizeMetadataKey(std::move(path));
        if (value.is_object() && value.contains("node_type")) {
            if (!path.empty()) {
                output[path] = value;
            }
        } else if (value.is_object()) {
            CollectConsolidatedMetadata(value, path, output);
        }
    }
}

Result<void> ApplyShardingLayout(const nlohmann::json& sharding, StorageLayout& layout, std::string_view node) {
    layout.sharded = true;
    layout.shard_shape = std::move(layout.chunk_shape);
    layout.chunk_shape.clear();

    if (sharding.contains("configuration") && sharding.at("configuration").is_object()) {
        const auto& configuration = sharding.at("configuration");
        if (configuration.contains("chunk_shape") && configuration.at("chunk_shape").is_array()) {
            for (const auto& dimension : configuration.at("chunk_shape")) {
                if (!zarr_metadata::IsPositiveInteger(dimension)) {
                    return MakeError(ErrorCode::invalid_metadata,
                                     "Sharding codec chunk_shape must be positive integers", std::string(node));
                }
                layout.chunk_shape.push_back(dimension.get<std::uint64_t>());
            }
        }
        // The compressor applies to the inner chunks, so look inside the sharding codec first.
        if (configuration.contains("codecs") && configuration.at("codecs").is_array()) {
            layout.compressor = FindCompressor(&configuration.at("codecs"));
        }
    }

    if (layout.chunk_shape.size() != layout.shard_shape.size()) {
        return MakeError(ErrorCode::invalid_metadata, "Sharding codec chunk_shape must match the shard rank",
                         std::string(node));
    }
    return {};
}

Result<std::uint64_t> ElementSizeBytes(const zarr::ArrayMetadata& metadata, std::string_view node) {
    static const std::map<std::string_view, std::uint64_t> element_sizes{
        {"bool", 1},      {"int8", 1},       {"uint8", 1},      {"int16", 2},
        {"uint16", 2},    {"int32", 4},      {"uint32", 4},     {"int64", 8},
        {"uint64", 8},    {"float16", 2},    {"float32", 4},    {"float64", 8},
        {"complex64", 8}, {"complex128", 16},
    };
    if (const auto found = element_sizes.find(metadata.data_type); found != element_sizes.end()) {
        return found->second;
    }

    // XRADIO coordinate labels use the fixed_length_utf32 extension data type. Other fixed-length
    // extension types can be sized the same way when they declare length_bytes.
    if (metadata.data_type_configuration.is_object() &&
        metadata.data_type_configuration.contains("length_bytes")) {
        const auto& length_bytes = metadata.data_type_configuration.at("length_bytes");
        if (zarr::IsPositiveInteger(length_bytes)) {
            return length_bytes.get<std::uint64_t>();
        }
    }

    return MakeError(ErrorCode::unsupported_data_type,
                     "Array " + std::string(node) + " has unsupported data_type " + metadata.data_type,
                     std::string(node));
}

}  // namespace

Store::Store(TransportPtr transport, nlohmann::json root_attributes,
             std::map<std::string, nlohmann::json> consolidated_metadata, bool has_consolidated_metadata,
             StoreContextPtr context)
    : _transport(std::move(transport)),
      _root_attributes(std::move(root_attributes)),
      _consolidated_metadata(std::move(consolidated_metadata)),
      _has_consolidated_metadata(has_consolidated_metadata),
      _context(std::move(context)),
      _caches(std::make_shared<StoreCaches>()) {}

Result<Store> OpenStore(std::string_view location, StoreContextPtr context) {
    auto transport = OpenFilesystemTransport(location);
    if (!transport) {
        return transport.error();
    }
    return OpenStore(std::move(transport.value()), std::move(context));
}

Result<Store> OpenStore(TransportPtr transport, StoreContextPtr context) {
    if (!transport) {
        return MakeError(ErrorCode::invalid_argument, "Zarr transport must not be null");
    }

    auto bytes = transport->ReadNodeBytes({});
    if (!bytes) {
        // A transport with no root node is not a Zarr store at all, whatever else it holds.
        if (bytes.error().code == ErrorCode::not_found) {
            return MakeError(ErrorCode::not_zarr, "Zarr store is missing zarr.json", bytes.error().node_path);
        }
        return bytes.error();
    }

    auto metadata_result = ParseNodeMetadata(bytes.value(), "zarr.json");
    if (!metadata_result) {
        return metadata_result.error();
    }
    const nlohmann::json& metadata = metadata_result.value();
    if (!metadata.is_object()) {
        return MakeError(ErrorCode::invalid_metadata, "Root Zarr metadata must be a JSON object", "zarr.json");
    }
    if (!metadata.contains("zarr_format") ||
        !::carta::zarr::internal::zarr::IsNonNegativeInteger(metadata.at("zarr_format"))) {
        return MakeError(ErrorCode::invalid_metadata, "Root Zarr metadata has no valid zarr_format", "zarr.json");
    }
    if (metadata.at("zarr_format").get<std::uint64_t>() != 3) {
        return MakeError(ErrorCode::unsupported_zarr_version, "Only Zarr format 3 is supported", "zarr.json");
    }
    if (metadata.value("node_type", "") != "group") {
        return MakeError(ErrorCode::not_zarr, "The Zarr root must be a group", "zarr.json");
    }

    std::map<std::string, nlohmann::json> consolidated;
    bool has_consolidated = false;
    if (metadata.contains("consolidated_metadata")) {
        const auto& block = metadata.at("consolidated_metadata");
        if (!block.is_object() || !block.contains("metadata") || !block.at("metadata").is_object()) {
            return MakeError(ErrorCode::invalid_metadata,
                             "Zarr consolidated_metadata must contain an object metadata member", "zarr.json");
        }
        has_consolidated = true;
        CollectConsolidatedMetadata(block.at("metadata"), {}, consolidated);
    }

    return Store{std::move(transport), metadata.value("attributes", nlohmann::json::object()), std::move(consolidated),
                 has_consolidated, std::move(context)};
}

const nlohmann::json& Store::RootAttributes() const noexcept {
    return _root_attributes;
}

Result<nlohmann::json> Store::ReadNodeMetadata(std::string_view node) const {
    auto node_name_result = NormalizeNodeName(node);
    if (!node_name_result) {
        return node_name_result.error();
    }
    const std::string node_name = std::move(node_name_result.value());

    return _caches->node_metadata.GetOrCompute(node_name, [&]() -> Result<nlohmann::json> {
        if (_has_consolidated_metadata) {
            const auto found = _consolidated_metadata.find(node_name);
            if (found != _consolidated_metadata.end()) {
                return found->second;
            }
        }

        auto bytes = _transport->ReadNodeBytes(node_name);
        if (!bytes) {
            return bytes.error();
        }
        return ParseNodeMetadata(bytes.value(), node);
    });
}

Result<zarr::ArrayMetadata> Store::ReadArrayMetadata(std::string_view node) const {
    return _caches->array_metadata.GetOrCompute(std::string(node), [&]() -> Result<zarr::ArrayMetadata> {
        auto metadata_result = ReadNodeMetadata(node);
        if (!metadata_result) {
            return metadata_result.error();
        }
        return zarr_metadata::ParseArrayMetadata(metadata_result.value(), node);
    });
}

Result<std::vector<std::pair<std::string, nlohmann::json>>> Store::ListNodeMetadata() const {
    using Listing = Result<std::vector<std::pair<std::string, nlohmann::json>>>;
    return _caches->listed_metadata.GetOrCompute([&]() -> Listing {
        std::vector<std::string> node_names;
        if (_has_consolidated_metadata) {
            node_names.reserve(_consolidated_metadata.size());
            for (const auto& [node, _] : _consolidated_metadata) {
                node_names.push_back(node);
            }
        } else {
            auto listed = _transport->ListNodes();
            if (!listed) {
                return listed.error();
            }
            node_names = std::move(listed.value());
        }

        std::sort(node_names.begin(), node_names.end());
        node_names.erase(std::unique(node_names.begin(), node_names.end()), node_names.end());

        std::vector<std::pair<std::string, nlohmann::json>> result;
        result.reserve(node_names.size());
        for (const auto& node : node_names) {
            auto metadata = ReadNodeMetadata(node);
            if (!metadata) {
                return metadata.error();
            }
            result.emplace_back(node, std::move(metadata.value()));
        }
        return result;
    });
}

Result<std::uint64_t> Store::ComputeTotalArraySizeBytes() const {
    auto nodes_result = ListNodeMetadata();
    if (!nodes_result) {
        return nodes_result.error();
    }

    std::uint64_t total_bytes = 0;
    std::size_t array_count = 0;
    for (const auto& [node, metadata] : nodes_result.value()) {
        if (!metadata.is_object() || metadata.value("node_type", "") != "array") {
            continue;
        }
        ++array_count;

        auto array_metadata_result = ReadArrayMetadata(node);
        if (!array_metadata_result) {
            return array_metadata_result.error();
        }
        const auto& array_metadata = array_metadata_result.value();
        auto element_size_result = ElementSizeBytes(array_metadata, node);
        if (!element_size_result) {
            return element_size_result.error();
        }

        std::uint64_t array_bytes = element_size_result.value();
        for (const auto dimension : array_metadata.shape) {
            if (dimension == 0) {
                array_bytes = 0;
                break;
            }
            if (array_bytes > std::numeric_limits<std::uint64_t>::max() / dimension) {
                return MakeError(ErrorCode::invalid_metadata,
                                 "Array " + node + " byte size overflows uint64_t", node);
            }
            array_bytes *= dimension;
        }
        if (total_bytes > std::numeric_limits<std::uint64_t>::max() - array_bytes) {
            return MakeError(ErrorCode::invalid_metadata, "Total Zarr array byte size overflows uint64_t");
        }
        total_bytes += array_bytes;
    }

    if (array_count == 0) {
        return MakeError(ErrorCode::invalid_metadata, "Zarr store contains no arrays");
    }
    return total_bytes;
}

Result<std::vector<double>> Store::ReadNumericArray(std::string_view node) const {
    return _caches->double_arrays.GetOrCompute(std::string(node), [&] { return ReadNumericArrayUncached(node); });
}

Result<std::vector<double>> Store::ReadNumericArrayUncached(std::string_view node) const {
    auto array_path = ResolveArrayDirectory(node);
    if (!array_path) {
        return array_path.error();
    }
    try {
        const std::filesystem::path target_path =
            std::filesystem::weakly_canonical(std::filesystem::absolute(array_path.value()));
        return zarr_metadata::ReadNumericValues(target_path, _context, node);
    } catch (const std::exception& e) {
        return MakeError(ErrorCode::io_error, e.what(), std::string(node));
    }
}

Result<std::filesystem::path> Store::ResolveArrayDirectory(std::string_view node) const {
    auto array_path = _transport->ArrayDirectory(node);
    if (!array_path) {
        return array_path.error();
    }
    return std::filesystem::weakly_canonical(std::filesystem::absolute(array_path.value()));
}

Result<void> Store::ReadPixelsFloat32(std::string_view node, const zarr::PixelSelection& selection,
                                      float* destination, std::size_t destination_elements,
                                      const ReadOptions& options) const {
    try {
        auto metadata = ReadArrayMetadata(node);
        if (!metadata) {
            return metadata.error();
        }
        auto target_path = ResolveArrayDirectory(node);
        if (!target_path) {
            return target_path.error();
        }
        return zarr_metadata::ReadFloat32(target_path.value(), _context, node, metadata.value().data_type, selection, destination,
                                           destination_elements, options);
    } catch (const std::exception& e) {
        return MakeError(ErrorCode::io_error, e.what(), std::string(node));
    }
}

Result<void> Store::ReadPixelMaskBytes(std::string_view node, const zarr::PixelSelection& selection,
                                       std::uint8_t* destination, std::size_t destination_elements,
                                       const ReadOptions& options) const {
    try {
        auto metadata = ReadArrayMetadata(node);
        if (!metadata) {
            return metadata.error();
        }
        auto target_path = ResolveArrayDirectory(node);
        if (!target_path) {
            return target_path.error();
        }
        return zarr_metadata::ReadMaskBytes(target_path.value(), _context, node, metadata.value().data_type, selection, destination,
                                            destination_elements, options);
    } catch (const std::exception& e) {
        return MakeError(ErrorCode::io_error, e.what(), std::string(node));
    }
}

Result<std::vector<std::string>> Store::ReadStringArray1D(std::string_view node) const {
    return _caches->string_arrays.GetOrCompute(std::string(node), [&] { return ReadStringArray1DUncached(node); });
}

Result<std::vector<std::string>> Store::ReadStringArray1DUncached(std::string_view node) const {
    auto meta_res = ReadNodeMetadata(node);
    if (!meta_res) {
        return meta_res.error();
    }
    const auto& metadata = meta_res.value();
    auto array_meta_res = ReadArrayMetadata(node);
    if (!array_meta_res) {
        return array_meta_res.error();
    }
    auto array_path = ResolveArrayDirectory(node);
    if (!array_path) {
        return array_path.error();
    }
    try {
        return zarr_metadata::ReadFixedLengthUtf32StringArray(array_path.value(), array_meta_res.value(), metadata,
                                                              node);
    } catch (const std::exception& e) {
        return MakeError(ErrorCode::invalid_argument, e.what(), std::string(node));
    }
}

Result<StorageLayout> Store::ReadStorageLayout(std::string_view node) const {
    auto meta_res = ReadNodeMetadata(node);
    if (!meta_res) {
        return meta_res.error();
    }
    const auto& metadata = meta_res.value();
    auto array_meta_res = ReadArrayMetadata(node);
    if (!array_meta_res) {
        return array_meta_res.error();
    }

    StorageLayout layout;
    layout.chunk_shape = array_meta_res.value().chunk_shape;

    const nlohmann::json* codecs = nullptr;
    if (metadata.contains("codecs") && metadata.at("codecs").is_array()) {
        codecs = &metadata.at("codecs");
    }

    if (const nlohmann::json* sharding = FindCodec(codecs, "sharding_indexed"); sharding != nullptr) {
        auto sharding_result = ApplyShardingLayout(*sharding, layout, node);
        if (!sharding_result) {
            return sharding_result.error();
        }
    }

    if (layout.compressor.empty()) {
        layout.compressor = FindCompressor(codecs);
    }
    return layout;
}

}  // namespace carta::zarr::internal
