/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_EXPORT_H_
#define CARTA_ZARR_EXPORT_H_

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(CARTA_ZARR_BUILDING_LIBRARY)
#define CARTA_ZARR_EXPORT __declspec(dllexport)
#else
#define CARTA_ZARR_EXPORT __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define CARTA_ZARR_EXPORT __attribute__((visibility("default")))
#else
#define CARTA_ZARR_EXPORT
#endif

#endif // CARTA_ZARR_EXPORT_H_
