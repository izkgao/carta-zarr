/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_ZARR_STORE_CONTEXT_H_
#define CARTA_ZARR_SRC_ZARR_STORE_CONTEXT_H_

#include "store.h"

#include <tensorstore/context.h>

namespace carta::zarr::internal {

// The resources shared by every read made through one public carta::zarr::Context. Only translation
// units that talk to TensorStore include this header; store.h forward declares the type so that the
// schema layer never sees TensorStore.
class StoreContext {
public:
    explicit StoreContext(tensorstore::Context context) : context(std::move(context)) {}

    tensorstore::Context context;
};

// Translate the public options into TensorStore context resources. Positive limits are written
// into the spec; zero limits keep TensorStore's defaults unless disable_cache explicitly requests
// a zero-byte cache pool.
Result<StoreContextPtr> MakeStoreContext(const OpenOptions& options);

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_ZARR_STORE_CONTEXT_H_
