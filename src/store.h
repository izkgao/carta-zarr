/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_STORE_H_
#define CARTA_ZARR_SRC_STORE_H_

#include "carta-zarr/carta_zarr.h"

#include "zarr/array_metadata.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace carta::zarr::internal {

struct SharedCoordinateCache {
    std::mutex mutex;
    std::map<std::string, std::shared_ptr<Result<std::vector<double>>>> double_arrays;
    std::map<std::string, std::shared_ptr<Result<std::vector<std::string>>>> string_arrays;
};

// A Store is a read-only view of a Zarr hierarchy. Keep metadata stable for the lifetime of that
// view so schema probing, image discovery, and descriptor construction share the same parsed data.
struct SharedMetadataCache {
    std::mutex mutex;
    std::map<std::string, Result<nlohmann::json>> node_metadata;
    std::map<std::string, Result<zarr::ArrayMetadata>> array_metadata;
    std::optional<std::vector<std::pair<std::string, nlohmann::json>>> listed_metadata;
};

// Defined in zarr/store_context.h. Kept opaque here so that including store.h does not pull in
// TensorStore; a null pointer means "use TensorStore's default resources".
class StoreContext;
using StoreContextPtr = std::shared_ptr<const StoreContext>;

struct Store {
    Store(std::filesystem::path root, nlohmann::json root_metadata, StoreContextPtr context)
        : root(std::move(root)), root_metadata(std::move(root_metadata)), context(std::move(context)) {}

    std::filesystem::path root;
    nlohmann::json root_metadata;
    StoreContextPtr context;
    std::map<std::string, nlohmann::json> consolidated_metadata;
    bool has_consolidated_metadata = false;
    std::shared_ptr<SharedCoordinateCache> coordinate_cache;
    std::shared_ptr<SharedMetadataCache> metadata_cache;

    Result<nlohmann::json> ReadNodeMetadata(std::string_view node) const;
    Result<zarr::ArrayMetadata> ReadArrayMetadata(std::string_view node) const;
    Result<std::vector<std::pair<std::string, nlohmann::json>>> ListNodeMetadata() const;
    Result<std::vector<double>> ReadDoubleArray1D(std::string_view node) const;
    Result<std::vector<std::string>> ReadStringArray1D(std::string_view node) const;
    Result<StorageLayout> ReadStorageLayout(std::string_view node) const;

private:
    Result<std::vector<double>> ReadDoubleArray1DUncached(std::string_view node) const;
    Result<std::vector<std::string>> ReadStringArray1DUncached(std::string_view node) const;
};

Result<Store> OpenStore(std::string_view location, StoreContextPtr context = {});

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_STORE_H_
