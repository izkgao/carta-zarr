/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_SCHEMA_REGISTRY_H_
#define CARTA_ZARR_SRC_SCHEMA_REGISTRY_H_

#include "../store.h"

#include <vector>

namespace carta::zarr::internal {

struct ImageDiscovery {
    std::vector<std::string> image_ids;
    std::vector<std::string> openable_image_ids;
    std::vector<Diagnostic> diagnostics;
};

struct SchemaAdapter {
    SchemaId id;
    Result<SchemaProbeResult> (*probe)(const Store&);
    Result<ImageDiscovery> (*discover)(const Store&);
    Result<ImageDescriptor> (*describe)(const Store&, std::string_view image_id);
    Result<std::vector<Beam>> (*read_beams)(const Store&, std::string_view image_id);
};

const std::vector<SchemaAdapter>& SchemaRegistry();
Result<SchemaProbeResult> ProbeSchemaFromRegistry(const Store& store, std::string_view schema_id);
Result<ImageDiscovery> DiscoverImagesFromSchema(const Store& store, std::string_view schema_id);
Result<ImageDescriptor> DescribeSchema(const Store& store, std::string_view schema_id, std::string_view image_id);
Result<std::vector<Beam>> ReadBeamsFromSchema(const Store& store, std::string_view schema_id,
                                              std::string_view image_id);
Result<ProbeResult> ProbeStore(const Store& store);

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_SCHEMA_REGISTRY_H_
