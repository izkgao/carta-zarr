/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_REDUCE_PLANE_HISTOGRAM_H_
#define CARTA_ZARR_SRC_REDUCE_PLANE_HISTOGRAM_H_

#include "carta-zarr/carta_zarr.h"

#include "store.h"

namespace carta::zarr::internal {

/**
 * Bin every pixel of each plane over a fixed range, one pass over the pixels.
 *
 * Much simpler than ReduceSpectral because the region is always the whole plane: there is nothing
 * to index, nothing to clip, and no mask to consult, so the walk is a band of chunk rows at a time
 * and the binning runs over the read buffer end to end.
 *
 * Like the reduction it reads in the store's own axis order -- binning does not care what order it
 * sees the pixels in, so there is no reason to pay a transpose for them.
 */
Result<void> ComputeHistogram(const Store& store, const ImageDescriptor& descriptor,
                              const ChunkGeometry& geometry, const HistogramRequest& request,
                              const HistogramSink& sink, const ReadOptions& options);

/**
 * One histogram for the whole selection in a single pass. See CubeHistogramRequest.
 */
Result<CubeHistogramResult> ComputeCubeHistogram(const Store& store, const ImageDescriptor& descriptor,
                                                 const ChunkGeometry& geometry,
                                                 const CubeHistogramRequest& request,
                                                 const ReadOptions& options);

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_PLANE_HISTOGRAM_H_
