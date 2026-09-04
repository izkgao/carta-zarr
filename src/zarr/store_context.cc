/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "store_context.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <utility>

namespace carta::zarr::internal {

Result<StoreContextPtr> MakeStoreContext(const OpenOptions& options) {
    nlohmann::json spec = nlohmann::json::object();
    if (options.cache_bytes > 0) {
        spec["cache_pool"] = {{"total_bytes_limit", options.cache_bytes}};
    }
    if (options.io_threads > 0) {
        spec["file_io_concurrency"] = {{"limit", options.io_threads}};
    }
    if (options.decode_threads > 0) {
        spec["data_copy_concurrency"] = {{"limit", options.decode_threads}};
    }

    if (spec.empty()) {
        return std::make_shared<const StoreContext>(tensorstore::Context::Default());
    }

    auto context = tensorstore::Context::FromJson(std::move(spec));
    if (!context.ok()) {
        return Error{ErrorCode::invalid_argument,
                     "Unable to apply the requested resource limits: " + context.status().ToString()};
    }
    return std::make_shared<const StoreContext>(std::move(context.value()));
}

}  // namespace carta::zarr::internal
