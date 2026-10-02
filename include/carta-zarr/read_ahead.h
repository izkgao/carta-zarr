/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_READ_AHEAD_H_
#define CARTA_ZARR_READ_AHEAD_H_

#include "carta-zarr/read.h"

#include <cstddef>

namespace carta::zarr {

// One plane a frame of an animation shows: which of the images its read-ahead was made for, by
// position, and the read of the plane.
struct AnimatedPlane {
    std::size_t image = 0;
    ReadRequest request;
};

// What reading ahead has done so far.
struct ReadAheadStats {
    // Runs decoded ahead, or started to be: one for each image a prefetch was for.
    unsigned prefetches = 0;
    // Of those, the ones a frame reached before they had finished. That frame waited for the decode
    // under way rather than starting its own, so a prefetch caught up with is not one wasted -- only
    // one that was started too late to hide the whole of its run.
    unsigned caught_up = 0;
    // Whether reading ahead has stopped, for a frame late while a prefetch was under way or because
    // it was cancelled. It does not start again.
    bool stopped = false;
};

}  // namespace carta::zarr

#endif  // CARTA_ZARR_READ_AHEAD_H_
