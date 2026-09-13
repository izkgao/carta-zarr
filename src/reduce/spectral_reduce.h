/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_REDUCE_SPECTRAL_REDUCE_H_
#define CARTA_ZARR_SRC_REDUCE_SPECTRAL_REDUCE_H_

#include "carta-zarr/carta_zarr.h"

#include "store.h"
#include "work_pool.h"

namespace carta::zarr::internal {

/**
 * Reduce N regions over a run of channels in one pass over the pixels.
 *
 * Kept out of carta_zarr.cc because it is a different kind of code: the public entry points
 * translate a request and hand it to the store, while this walks a chunk grid and owns an inner
 * loop whose shape is the entire reason the API takes N regions at once.
 */
Result<void> ReduceSpectral(const Store& store, const ImageDescriptor& descriptor,
                            const ChunkGeometry& geometry, const SpectralReduceRequest& request,
                            const SpectralSink& sink, const ReadOptions& options, WorkPool& workers);

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_SPECTRAL_REDUCE_H_
