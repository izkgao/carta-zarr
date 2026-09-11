/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_ZARR_STORE_CONTEXT_H_
#define CARTA_ZARR_SRC_ZARR_STORE_CONTEXT_H_

#include "store.h"

#include <tensorstore/context.h>
#include <tensorstore/tensorstore.h>

#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace carta::zarr::internal {

// The resources shared by every read made through one public carta::zarr::Context. Array handles
// are cloned into a per-Store context, so this object retains only shared TensorStore resources.
// Only translation units that talk to TensorStore include this header; store.h forward declares the
// type so that the schema layer never sees TensorStore.
class StoreContext : public std::enable_shared_from_this<StoreContext> {
public:
    explicit StoreContext(tensorstore::Context context) : context(std::move(context)) {}

    // Make a context for one Dataset/Store. The TensorStore resource handles remain shared, while
    // the array-handle table has the same lifetime as that store rather than the public Context.
    StoreContextPtr CloneForStore() const;

    /**
     * Open an array once and hand back the same handle afterwards.
     *
     * Opening is not cheap: it reads and parses the array's metadata and builds a driver. A pixel
     * read is issued once per casacore cursor step, so paying that per call turns a fixed cost into
     * a per-call one and dominates everything else -- measured at several milliseconds a call,
     * against roughly a hundred milliseconds for a whole 4096-square plane read in one go.
     *
     * The handle is cheap to copy and safe to use from several threads, so callers get a copy and
     * the table is only locked around the lookup.
     */
    Result<tensorstore::TensorStore<>> OpenArray(const std::filesystem::path& array_path,
                                                 std::string_view node) const;

    /**
     * The same resources with a cache pool of zero bytes.
     *
     * A child context, so the thread pools are the shared ones -- a scan that ran on its own
     * threads would compete with the session rather than take its turn.
     *
     * It keeps its own array table because a handle carries the pool it was opened against, so a
     * bypassed read cannot reuse one opened with the shared pool. Built once and kept, since the
     * scans that want it are long and there are few of them.
     */
    StoreContextPtr WithoutCache() const;

    tensorstore::Context context;

private:
    mutable std::mutex _arrays_mutex;
    mutable std::map<std::string, tensorstore::TensorStore<>> _arrays;
    mutable std::mutex _without_cache_mutex;
    mutable StoreContextPtr _without_cache;
};

// Translate the public options into TensorStore context resources. Positive limits are written
// into the spec; zero limits keep TensorStore's defaults unless disable_cache explicitly requests
// a zero-byte cache pool.
Result<StoreContextPtr> MakeStoreContext(const OpenOptions& options);

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_ZARR_STORE_CONTEXT_H_
