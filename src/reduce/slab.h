/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_REDUCE_SLAB_H_
#define CARTA_ZARR_SRC_REDUCE_SLAB_H_

// In a header of its own because more than the walk speaks it: the pass hands one to a visitor, and
// a reduction's accumulation takes one, and neither the accumulation nor its tests should have to
// include the walk -- its pixel source seam and its reader -- to say what a read of pixels is.

#include "reduce/pass_plan.h"

#include <cstdint>

namespace carta::zarr::internal {

/**
 * One read of the pass, handed to the visitor.
 *
 * A pointer and three strides rather than a packed buffer, because the destination comes back in
 * the store's own order and packing it would be the transpose the pass exists to avoid.
 *
 * A read, not a plane: a visitor that splits the work across threads needs a piece big enough to
 * pay for the dispatch, and a plane of a few hundred thousand pixels is not one. A visitor that
 * wants planes loops over `channel_count` itself, which costs it nothing.
 */
struct Slab {
    // Where this slab starts in the channel range its walk was given, which is what a visitor
    // accumulating into a block of its own indexes by. Set by the walk, not by the reader: the
    // reader is given an absolute index and has nothing to measure a relative one against.
    BlockChannel first_channel;
    std::uint64_t channel_count = 0;
    const float* pixels = nullptr;
    std::uint64_t stride_u = 1;
    std::uint64_t stride_v = 1;
    std::uint64_t stride_z = 1;
    std::uint64_t u_count = 0;
    std::uint64_t v_count = 0;
};

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCE_SLAB_H_
