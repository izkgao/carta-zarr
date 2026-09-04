/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <carta-zarr/carta_zarr.h>

int main() {
    const auto probe = carta::zarr::Probe(".");
    return probe.kind == carta::zarr::ProbeKind::not_zarr ? 0 : 1;
}
