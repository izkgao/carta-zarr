/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_STORE_H_
#define CARTA_ZARR_SRC_STORE_H_

#include "carta-zarr/carta_zarr.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>

namespace carta::zarr::internal {

struct SharedCoordinateCache {
    std::mutex mutex;
    std::map<std::string, std::shared_ptr<Result<std::vector<double>>>> double_arrays;
    std::map<std::string, std::shared_ptr<Result<std::vector<std::string>>>> string_arrays;
};

// Defined in zarr/store_context.h. Kept opaque here so that including store.h does not pull in
// TensorStore; a null pointer means "use TensorStore's default resources".
class StoreContext;
using StoreContextPtr = std::shared_ptr<const StoreContext>;

struct Store {
    std::filesystem::path root;
    nlohmann::json root_metadata;
    StoreContextPtr context;
    std::map<std::string, nlohmann::json> consolidated_metadata;
    bool has_consolidated_metadata = false;
    std::shared_ptr<SharedCoordinateCache> coordinate_cache;

    Result<nlohmann::json> ReadNodeMetadata(std::string_view node) const;
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
