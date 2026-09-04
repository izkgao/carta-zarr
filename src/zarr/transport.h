/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_ZARR_TRANSPORT_H_
#define CARTA_ZARR_SRC_ZARR_TRANSPORT_H_

#include "carta-zarr/carta_zarr.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace carta::zarr::internal {

// The seam between a Zarr hierarchy and where its bytes live. A Transport supplies raw access only:
// it never parses JSON and it never decides what a node means. Keeping interpretation above the
// seam is deliberate -- it is what stops a test transport and the filesystem transport from
// disagreeing about the very metadata the tests exist to pin down.
//
// Node names reaching a Transport have already been validated and normalized by Store, so every
// Transport is held to the same rule about what a legal node path is.
class Transport {
public:
    Transport() = default;
    Transport(const Transport&) = delete;
    Transport& operator=(const Transport&) = delete;
    Transport(Transport&&) = delete;
    Transport& operator=(Transport&&) = delete;
    virtual ~Transport() = default;

    // Read one node's zarr.json verbatim. An empty node names the root. Reports not_found when the
    // node carries no metadata, and io_error for every other failure.
    virtual Result<std::string> ReadNodeBytes(std::string_view node) const = 0;

    // Every node in the hierarchy, excluding the root, in any order. Only consulted when the store
    // carries no consolidated metadata.
    virtual Result<std::vector<std::string>> ListNodes() const = 0;

    // Where a node's array data lives, for the coordinate value reads that go through TensorStore.
    // A transport holding no filesystem data reports unsupported_transport, so those reads fail with
    // a reason rather than obscurely; an unusable node name reports invalid_argument.
    virtual Result<std::filesystem::path> ArrayPath(std::string_view node) const = 0;
};

using TransportPtr = std::shared_ptr<const Transport>;

// Normalize a location (a bare path, or file://) and check that it names a directory.
Result<TransportPtr> OpenFilesystemTransport(std::string_view location);

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_ZARR_TRANSPORT_H_
