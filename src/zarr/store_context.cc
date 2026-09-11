/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "store_context.h"

#include <tensorstore/open.h>
#include <tensorstore/open_mode.h>
#include <tensorstore/spec.h>

#include <nlohmann/json.hpp>

#include <memory>
#include <utility>

namespace carta::zarr::internal {

StoreContextPtr StoreContext::CloneForStore() const {
    return std::make_shared<const StoreContext>(context);
}

Result<tensorstore::TensorStore<>> StoreContext::OpenArray(const std::filesystem::path& array_path,
                                                           std::string_view node) const {
    const std::string key = array_path.string();
    {
        const std::scoped_lock lock(_arrays_mutex);
        if (const auto found = _arrays.find(key); found != _arrays.end()) {
            return found->second;
        }
    }

    // Opened outside the lock so that a slow open of one array does not stall reads of another.
    auto spec = tensorstore::Spec::FromJson({
        {"driver", "zarr3"},
        {"kvstore", {{"driver", "file"}, {"path", key}}},
    });
    if (!spec.ok()) {
        return Error{ErrorCode::io_error, "Failed to create TensorStore spec: " + spec.status().ToString(),
                     std::string(node)};
    }
    auto opened = tensorstore::Open(spec.value(), context, tensorstore::OpenMode::open,
                                    tensorstore::ReadWriteMode::read)
                      .result();
    if (!opened.ok()) {
        return Error{ErrorCode::io_error, "Failed to open TensorStore: " + opened.status().ToString(),
                     std::string(node)};
    }

    const std::scoped_lock lock(_arrays_mutex);
    // Another thread may have opened the same array first; either handle is equivalent, so keep
    // whichever landed in the table.
    return _arrays.emplace(key, std::move(opened).value()).first->second;
}

StoreContextPtr StoreContext::WithoutCache() const {
    std::scoped_lock lock(_without_cache_mutex);
    if (_without_cache) {
        return _without_cache;
    }
    nlohmann::json spec = nlohmann::json::object();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    spec["cache_pool"] = {{"total_bytes_limit", 0}};
    auto child = tensorstore::Context::FromJson(std::move(spec), context);
    if (!child.ok()) {
        // Nothing here can fail that is not a programming error, and a read that cannot bypass the
        // cache is better served with it than refused.
        _without_cache = shared_from_this();
        return _without_cache;
    }
    _without_cache = std::make_shared<const StoreContext>(std::move(child.value()));
    return _without_cache;
}

Result<StoreContextPtr> MakeStoreContext(const OpenOptions& options) {
    nlohmann::json spec = nlohmann::json::object();
    if (options.cache_bytes > 0) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        spec["cache_pool"] = {{"total_bytes_limit", options.cache_bytes}};
    } else if (options.disable_cache) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        spec["cache_pool"] = {{"total_bytes_limit", 0}};
    }
    if (options.io_threads > 0) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        spec["file_io_concurrency"] = {{"limit", options.io_threads}};
    }
    if (options.decode_threads > 0) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        spec["data_copy_concurrency"] = {{"limit", options.decode_threads}};
    }

    if (spec.empty()) {
        return std::make_shared<const StoreContext>(tensorstore::Context::Default());
    }

    auto context = tensorstore::Context::FromJson(std::move(spec));
    if (!context.ok()) {
        return Error{ErrorCode::invalid_argument,
                     "Unable to apply the requested resource limits: " + context.status().ToString(), {}};
    }
    return std::make_shared<const StoreContext>(std::move(context.value()));
}

}  // namespace carta::zarr::internal
