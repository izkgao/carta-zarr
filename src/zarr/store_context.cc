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

#include <cstddef>
#include <memory>
#include <utility>

namespace carta::zarr::internal {
namespace {

// What a carta-zarr array is, stated once. Every array this library reads -- pixels, flags and
// coordinate values alike -- is opened through here.
Result<tensorstore::TensorStore<>> OpenZarr3File(const std::string& path, const tensorstore::Context& context,
                                                 std::string_view node) {
    auto spec = tensorstore::Spec::FromJson({
        {"driver", "zarr3"},
        {"kvstore", {{"driver", "file"}, {"path", path}}},
        // A chunk cached after the array was opened is used without asking storage whether it has
        // changed. TensorStore's default asks on every read, which on a parallel file system is a
        // metadata round trip per chunk even when the cache holds it. ADR 0015.
        {"recheck_cached_data", "open"},
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
    return std::move(opened).value();
}

// A pool of `bytes`, said once for the session's pool and for a read's own.
nlohmann::json CachePoolSpec(std::size_t bytes) {
    return {{"total_bytes_limit", bytes}};
}

}  // namespace

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
    auto opened = OpenZarr3File(key, context, node);
    if (!opened) {
        return opened.error();
    }

    const std::scoped_lock lock(_arrays_mutex);
    // Another thread may have opened the same array first; either handle is equivalent, so keep
    // whichever landed in the table.
    return _arrays.emplace(key, std::move(opened.value())).first->second;
}

Result<tensorstore::TensorStore<>> OpenZarrArray(const std::filesystem::path& array_directory,
                                                 const StoreContextPtr& context, std::string_view node) {
    if (context) {
        return context->OpenArray(array_directory, node);
    }
    return OpenZarr3File(array_directory.string(), tensorstore::Context::Default(), node);
}

Result<StoreContextPtr> StoreContext::WithCachePool(std::size_t bytes) const {
    nlohmann::json spec = nlohmann::json::object();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
    spec["cache_pool"] = CachePoolSpec(bytes);
    auto child = tensorstore::Context::FromJson(std::move(spec), context);
    if (!child.ok()) {
        return Error{ErrorCode::invalid_argument, "Unable to make a cache pool: " + child.status().ToString(), {}};
    }
    return std::make_shared<const StoreContext>(std::move(child.value()));
}

Result<StoreContextPtr> MakeStoreContext(const ContextOptions& options) {
    nlohmann::json spec = nlohmann::json::object();
    if (options.cache_bytes) {
        // Zero is a size like any other here: it is the pool that holds nothing, which is what a
        // caller declining the cache asks for. No value at all is the only thing that leaves
        // TensorStore's default in place.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-avoid-unchecked-container-access)
        spec["cache_pool"] = CachePoolSpec(*options.cache_bytes);
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
