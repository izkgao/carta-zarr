/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "transport.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <utility>

namespace carta::zarr::internal {
namespace {

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

class FilesystemTransport final : public Transport {
public:
    explicit FilesystemTransport(std::filesystem::path root) : _root(std::move(root)) {}

    Result<std::string> ReadNodeBytes(std::string_view node) const override {
        const std::filesystem::path metadata_path = node.empty() ? _root / "zarr.json" : _root / node / "zarr.json";
        std::error_code error;
        if (!std::filesystem::exists(metadata_path, error)) {
            if (error) {
                return MakeError(ErrorCode::io_error, "Unable to inspect Zarr node metadata: " + error.message(),
                                 metadata_path.string());
            }
            return MakeError(ErrorCode::not_found, "Zarr node is missing zarr.json", metadata_path.string());
        }

        std::ifstream const input(metadata_path, std::ios::binary);
        if (!input.is_open()) {
            return MakeError(ErrorCode::io_error, "Unable to read Zarr metadata", metadata_path.string());
        }
        std::ostringstream buffer;
        buffer << input.rdbuf();
        // An empty file leaves failbit set but is not an I/O failure; it reaches the JSON parser
        // above the seam and is reported as invalid metadata, which is what it is.
        if (input.bad()) {
            return MakeError(ErrorCode::io_error, "Unable to read Zarr metadata", metadata_path.string());
        }
        return buffer.str();
    }

    Result<std::vector<std::string>> ListNodes() const override {
        std::vector<std::string> nodes;
        std::error_code error;
        std::filesystem::recursive_directory_iterator iterator(
            _root, std::filesystem::directory_options::skip_permission_denied, error);
        const std::filesystem::recursive_directory_iterator end;
        for (; iterator != end; iterator.increment(error)) {
            if (error) {
                return MakeError(ErrorCode::io_error, "Unable to enumerate Zarr metadata: " + error.message(),
                                 _root.string());
            }

            const auto path = iterator->path();
            if (!iterator->is_directory(error)) {
                if (error) {
                    continue;
                }
                if (!iterator->is_regular_file(error) || error || path.filename() != "zarr.json") {
                    continue;
                }
            } else if (error) {
                continue;
            }

            const auto metadata_path = iterator->is_directory() ? path / "zarr.json" : path;
            std::error_code metadata_error;
            if (iterator->is_directory() && !std::filesystem::is_regular_file(metadata_path, metadata_error)) {
                if (metadata_error) {
                    continue;
                }
                continue;
            }

            const auto relative_parent = std::filesystem::relative(
                iterator->is_directory() ? path : path.parent_path(), _root, error);
            if (error) {
                return MakeError(ErrorCode::io_error, "Unable to enumerate Zarr metadata: " + error.message(),
                                 path.string());
            }
            if (relative_parent.empty() || relative_parent == ".") {
                if (iterator->is_directory()) {
                    iterator.disable_recursion_pending();
                }
                continue;
            }
            nodes.push_back(relative_parent.generic_string());

            // Array chunks are descendants of the array node, but are not Zarr nodes. Inspect only
            // the small node header to avoid walking millions of chunk files. Malformed metadata is
            // left for Store::ReadNodeMetadata to diagnose; in that case traversal remains conservative.
            if (iterator->is_directory()) {
                std::ifstream input(metadata_path);
                nlohmann::json metadata;
                if (input.is_open()) {
                    try {
                        input >> metadata;
                        if (metadata.is_object() && metadata.value("node_type", "") == "array") {
                            iterator.disable_recursion_pending();
                        }
                    } catch (const std::exception&) {
                        // The metadata parser above the transport seam reports the definitive error.
                    }
                }
            }
        }
        if (error) {
            return MakeError(ErrorCode::io_error, "Unable to enumerate Zarr metadata: " + error.message(),
                             _root.string());
        }
        return nodes;
    }

    Result<std::filesystem::path> ArrayPath(std::string_view node) const override {
        const std::filesystem::path relative(node);
        if (relative.empty() || relative.is_absolute() || relative.has_root_name() ||
            node.find('\\') != std::string_view::npos) {
            return MakeError(ErrorCode::invalid_argument, "Invalid Zarr array path " + std::string(node),
                             std::string(node));
        }
        for (const auto& component : relative) {
            if (component == "." || component == "..") {
                return MakeError(ErrorCode::invalid_argument, "Invalid Zarr array path " + std::string(node),
                                 std::string(node));
            }
        }
        return _root / relative;
    }

private:
    std::filesystem::path _root;
};

}  // namespace

Result<TransportPtr> OpenFilesystemTransport(std::string_view location) {
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

    return TransportPtr(std::make_shared<const FilesystemTransport>(path));
}

}  // namespace carta::zarr::internal
