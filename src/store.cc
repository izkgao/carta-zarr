/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "store.h"

#include "zarr/array_metadata.h"
#include "zarr/store_context.h"
#include "zarr/string_array.h"

#include <tensorstore/array.h>
#include <tensorstore/context.h>
#include <tensorstore/open.h>
#include <tensorstore/open_mode.h>
#include <tensorstore/spec.h>
#include <tensorstore/static_cast.h>
#include <tensorstore/tensorstore.h>
#include <tensorstore/util/result.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
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

Result<std::filesystem::path> NormalizeLocation(std::string_view location) {
    if (location.empty()) {
        return MakeError(ErrorCode::invalid_argument, "Zarr location must not be empty");
    }

    std::string const location_string(location);
    std::filesystem::path path;
    if (location_string.rfind("file://", 0) == 0) {
        path = std::filesystem::path(location_string.substr(7));
    } else if (location_string.find("://") != std::string::npos) {
        return MakeError(ErrorCode::unsupported_transport,
                         "Only local filesystem and file:// Zarr stores are supported");
    } else {
        path = std::filesystem::path(location_string);
    }

    if (path.empty()) {
        return MakeError(ErrorCode::invalid_argument, "Zarr location must not be empty");
    }
    return path;
}

Result<nlohmann::json> ReadJsonFile(const std::filesystem::path& path, std::string_view node_path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        return MakeError(ErrorCode::io_error, "Unable to read Zarr metadata", std::string(node_path));
    }

    try {
        return nlohmann::json::parse(input);
    } catch (const nlohmann::json::parse_error& error) {
        return MakeError(ErrorCode::invalid_metadata, "Invalid JSON in Zarr metadata: " + std::string(error.what()),
                         std::string(node_path));
    } catch (const std::exception& error) {
        return MakeError(ErrorCode::io_error, "Unable to read Zarr metadata: " + std::string(error.what()),
                         std::string(node_path));
    }
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

std::filesystem::path ResolveArrayPath(const std::filesystem::path& root_path, std::string_view array_name) {
    const std::filesystem::path relative_path(array_name);
    if (relative_path.is_absolute() || relative_path.has_root_name() ||
        array_name.find('\\') != std::string_view::npos) {
        throw std::runtime_error("Invalid Zarr array path " + std::string(array_name));
    }
    for (const auto& component : relative_path) {
        if (component == "." || component == "..") {
            throw std::runtime_error("Invalid Zarr array path " + std::string(array_name));
        }
    }
    return root_path / relative_path;
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

}  // namespace

Result<Store> OpenStore(std::string_view location, StoreContextPtr context) {
    auto path_result = NormalizeLocation(location);
    if (!path_result) {
        return path_result.error();
    }
    const std::filesystem::path& path = path_result.value();

    std::error_code error;
    if (!std::filesystem::exists(path, error)) {
        if (error) {
            return MakeError(ErrorCode::io_error, "Unable to inspect Zarr location: " + error.message());
        }
        return MakeError(ErrorCode::not_found, "Zarr location does not exist", path.string());
    }
    if (error || !std::filesystem::is_directory(path, error)) {
        return MakeError(ErrorCode::not_zarr, "Zarr location is not a directory", path.string());
    }

    const std::filesystem::path metadata_path = path / "zarr.json";
    if (!std::filesystem::exists(metadata_path, error)) {
        return MakeError(ErrorCode::not_zarr, "Zarr store is missing zarr.json", metadata_path.string());
    }
    if (error) {
        return MakeError(ErrorCode::io_error, "Unable to inspect Zarr metadata: " + error.message(),
                         metadata_path.string());
    }

    auto metadata_result = ReadJsonFile(metadata_path, metadata_path.string());
    if (!metadata_result) {
        return metadata_result.error();
    }
    const nlohmann::json& metadata = metadata_result.value();
    if (!metadata.is_object()) {
        return MakeError(ErrorCode::invalid_metadata, "Root Zarr metadata must be a JSON object",
                         metadata_path.string());
    }
    if (!metadata.contains("zarr_format") ||
        !::carta::zarr::internal::zarr::IsNonNegativeInteger(metadata["zarr_format"])) {
        return MakeError(ErrorCode::invalid_metadata, "Root Zarr metadata has no valid zarr_format",
                         metadata_path.string());
    }
    if (metadata["zarr_format"].get<std::uint64_t>() != 3) {
        return MakeError(ErrorCode::unsupported_zarr_version, "Only Zarr format 3 is supported",
                         metadata_path.string());
    }
    if (metadata.value("node_type", "") != "group") {
        return MakeError(ErrorCode::not_zarr, "The Zarr root must be a group", metadata_path.string());
    }

    Store store{path, metadata, std::move(context)};
    store.coordinate_cache = std::make_shared<SharedCoordinateCache>();
    store.metadata_cache = std::make_shared<SharedMetadataCache>();
    if (metadata.contains("consolidated_metadata")) {
        const auto& consolidated = metadata.at("consolidated_metadata");
        if (!consolidated.is_object() || !consolidated.contains("metadata") ||
            !consolidated.at("metadata").is_object()) {
            return MakeError(ErrorCode::invalid_metadata,
                             "Zarr consolidated_metadata must contain an object metadata member",
                             metadata_path.string());
        }
        store.has_consolidated_metadata = true;
        CollectConsolidatedMetadata(consolidated.at("metadata"), {}, store.consolidated_metadata);
    }
    return store;
}

Result<nlohmann::json> Store::ReadNodeMetadata(std::string_view node) const {
    const std::filesystem::path relative(node);
    if (relative.empty() || relative.is_absolute() || relative.has_root_path()) {
        return MakeError(ErrorCode::invalid_argument, "Invalid Zarr node path", std::string(node));
    }
    for (const auto& part : relative) {
        if (part == "..") {
            return MakeError(ErrorCode::invalid_argument, "Invalid Zarr node path", std::string(node));
        }
    }

    const std::string node_name = relative.generic_string();

    const auto read_metadata = [&]() -> Result<nlohmann::json> {
        if (has_consolidated_metadata) {
            const auto found = consolidated_metadata.find(node_name);
            if (found != consolidated_metadata.end()) {
                return found->second;
            }
        }

        const std::filesystem::path metadata_path = root / relative / "zarr.json";
        std::error_code error;
        if (!std::filesystem::exists(metadata_path, error)) {
            if (error) {
                return MakeError(ErrorCode::io_error, "Unable to inspect Zarr node metadata: " + error.message(),
                                 std::string(node));
            }
            return MakeError(ErrorCode::not_found, "Zarr node is missing zarr.json", std::string(node));
        }
        return ReadJsonFile(metadata_path, node);
    };

    if (metadata_cache) {
        // Hold the cache lock while loading a node. Metadata files are small, and this prevents
        // concurrent schema operations from reading and parsing the same file more than once.
        std::scoped_lock const lock(metadata_cache->mutex);
        const auto found = metadata_cache->node_metadata.find(node_name);
        if (found != metadata_cache->node_metadata.end()) {
            return found->second;
        }
        auto result = read_metadata();
        const auto insertion = metadata_cache->node_metadata.emplace(node_name, std::move(result));
        return insertion.first->second;
    }
    return read_metadata();
}

Result<zarr::ArrayMetadata> Store::ReadArrayMetadata(std::string_view node) const {
    const std::string node_name(node);
    if (metadata_cache) {
        std::scoped_lock const lock(metadata_cache->mutex);
        const auto found = metadata_cache->array_metadata.find(node_name);
        if (found != metadata_cache->array_metadata.end()) {
            return found->second;
        }
    }

    auto metadata_result = ReadNodeMetadata(node);
    if (!metadata_result) {
        return metadata_result.error();
    }
    auto result = zarr_metadata::ParseArrayMetadata(metadata_result.value(), node);
    if (metadata_cache) {
        std::scoped_lock const lock(metadata_cache->mutex);
        const auto insertion = metadata_cache->array_metadata.emplace(node_name, std::move(result));
        return insertion.first->second;
    }
    return result;
}

Result<std::vector<std::pair<std::string, nlohmann::json>>> Store::ListNodeMetadata() const {
    if (metadata_cache) {
        std::scoped_lock const lock(metadata_cache->mutex);
        if (metadata_cache->listed_metadata.has_value()) {
            return *metadata_cache->listed_metadata;
        }
    }

    std::vector<std::string> node_names;
    if (has_consolidated_metadata) {
        node_names.reserve(consolidated_metadata.size());
        for (const auto& [node, _] : consolidated_metadata) {
            node_names.push_back(node);
        }
    } else {
        std::error_code error;
        std::filesystem::recursive_directory_iterator iterator(
            root, std::filesystem::directory_options::skip_permission_denied, error);
        const std::filesystem::recursive_directory_iterator end;
        for (; iterator != end; iterator.increment(error)) {
            if (error) {
                return MakeError(ErrorCode::io_error, "Unable to enumerate Zarr metadata: " + error.message(),
                                 root.string());
            }
            if (!iterator->is_regular_file(error) || error || iterator->path().filename() != "zarr.json") {
                continue;
            }
            const auto relative_parent = std::filesystem::relative(iterator->path().parent_path(), root, error);
            if (error) {
                return MakeError(ErrorCode::io_error, "Unable to enumerate Zarr metadata: " + error.message(),
                                 iterator->path().string());
            }
            if (relative_parent.empty() || relative_parent == ".") {
                continue;
            }
            node_names.push_back(relative_parent.generic_string());
        }
        if (error) {
            return MakeError(ErrorCode::io_error, "Unable to enumerate Zarr metadata: " + error.message(),
                             root.string());
        }
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

    if (metadata_cache) {
        std::scoped_lock const lock(metadata_cache->mutex);
        if (!metadata_cache->listed_metadata.has_value()) {
            metadata_cache->listed_metadata = std::move(result);
        }
        return *metadata_cache->listed_metadata;
    }
    return result;
}

Result<std::vector<double>> Store::ReadDoubleArray1D(std::string_view node) const {
    if (!coordinate_cache) {
        return ReadDoubleArray1DUncached(node);
    }
    std::scoped_lock const lock(coordinate_cache->mutex);
    const auto found = coordinate_cache->double_arrays.find(std::string(node));
    if (found != coordinate_cache->double_arrays.end()) {
        return *found->second;
    }
    auto result = std::make_shared<Result<std::vector<double>>>(ReadDoubleArray1DUncached(node));
    coordinate_cache->double_arrays.emplace(std::string(node), result);
    return *result;
}

Result<std::vector<double>> Store::ReadDoubleArray1DUncached(std::string_view node) const {
    try {
        const std::filesystem::path target_path =
            std::filesystem::weakly_canonical(std::filesystem::absolute(ResolveArrayPath(root, node)));
        auto spec_result = tensorstore::Spec::FromJson({
            {"driver", "zarr3"},
            {"kvstore", {{"driver", "file"}, {"path", target_path.string()}}},
        });
        if (!spec_result.ok()) {
            return MakeError(ErrorCode::io_error,
                             "Failed to create TensorStore spec: " + spec_result.status().ToString(),
                             std::string(node));
        }

        auto open_result =
            tensorstore::Open(spec_result.value(), context ? context->context : tensorstore::Context::Default(),
                              tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read)
                .result();
        if (!open_result.ok()) {
            return MakeError(ErrorCode::io_error, "Failed to open TensorStore: " + open_result.status().ToString(),
                             std::string(node));
        }

        auto typed_store_result = tensorstore::StaticCast<tensorstore::TensorStore<double>>(open_result.value());
        if (!typed_store_result.ok()) {
            return MakeError(ErrorCode::unsupported_data_type, "Array is not readable as double", std::string(node));
        }

        auto read_result = tensorstore::Read(typed_store_result.value()).result();
        if (!read_result.ok()) {
            return MakeError(ErrorCode::io_error, "TensorStore read failed: " + read_result.status().ToString(),
                             std::string(node));
        }

        const auto& array = read_result.value();
        std::vector<double> result;
        result.reserve(array.num_elements());
        tensorstore::IterateOverArrays([&result](const double* val) { result.push_back(*val); }, tensorstore::c_order,
                                       array);
        return result;
    } catch (const std::exception& e) {
        return MakeError(ErrorCode::io_error, e.what(), std::string(node));
    }
}

Result<std::vector<std::string>> Store::ReadStringArray1D(std::string_view node) const {
    if (!coordinate_cache) {
        return ReadStringArray1DUncached(node);
    }
    std::scoped_lock const lock(coordinate_cache->mutex);
    const auto found = coordinate_cache->string_arrays.find(std::string(node));
    if (found != coordinate_cache->string_arrays.end()) {
        return *found->second;
    }
    auto result = std::make_shared<Result<std::vector<std::string>>>(ReadStringArray1DUncached(node));
    coordinate_cache->string_arrays.emplace(std::string(node), result);
    return *result;
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
    try {
        return zarr_metadata::ReadFixedLengthUtf32StringArray(ResolveArrayPath(root, node), array_meta_res.value(),
                                                              metadata, node);
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

    // In a sharded array the chunk grid describes the shard, and the sharding codec carries the
    // inner chunk shape that reads actually address.
    if (const nlohmann::json* sharding = FindCodec(codecs, "sharding_indexed"); sharding != nullptr) {
        layout.sharded = true;
        layout.shard_shape = std::move(layout.chunk_shape);
        layout.chunk_shape.clear();
        if (sharding->contains("configuration") && sharding->at("configuration").is_object()) {
            const auto& configuration = sharding->at("configuration");
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
    }

    if (layout.compressor.empty()) {
        layout.compressor = FindCompressor(codecs);
    }
    return layout;
}

}  // namespace carta::zarr::internal
