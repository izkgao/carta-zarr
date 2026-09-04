/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "registry.h"

#include "xradio/sky.h"

#include <algorithm>
#include <utility>

namespace carta::zarr::internal {
namespace {

const SchemaAdapter* FindAdapter(std::string_view schema_id) {
    const auto& adapters = SchemaRegistry();
    const auto adapter = std::find_if(adapters.begin(), adapters.end(),
                                      [&](const SchemaAdapter& candidate) { return candidate.id == schema_id; });
    return adapter == adapters.end() ? nullptr : &*adapter;
}

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

}  // namespace

const std::vector<SchemaAdapter>& SchemaRegistry() {
    static const std::vector<SchemaAdapter> adapters{
        {SchemaAdapter{SchemaId(kXradioImageSchema), &xradio::ProbeSky, &xradio::DiscoverSkyImages,
                       &xradio::DescribeSky, &xradio::ReadBeamsSky}}};
    return adapters;
}

Result<SchemaProbeResult> ProbeSchemaFromRegistry(const Store& store, std::string_view schema_id) {
    const SchemaAdapter* adapter = FindAdapter(schema_id);
    if (adapter == nullptr) {
        return MakeError(ErrorCode::unsupported_schema,
                         "No built-in adapter exists for schema " + std::string(schema_id));
    }
    return adapter->probe(store);
}

Result<ImageDiscovery> DiscoverImagesFromSchema(const Store& store, std::string_view schema_id) {
    const SchemaAdapter* adapter = FindAdapter(schema_id);
    if (adapter == nullptr) {
        return MakeError(ErrorCode::unsupported_schema,
                         "No built-in adapter exists for schema " + std::string(schema_id));
    }
    return adapter->discover(store);
}

Result<ImageDescriptor> DescribeSchema(const Store& store, std::string_view schema_id, std::string_view image_id) {
    const SchemaAdapter* adapter = FindAdapter(schema_id);
    if (adapter == nullptr) {
        return MakeError(ErrorCode::unsupported_schema,
                         "No built-in adapter exists for schema " + std::string(schema_id));
    }
    auto discovery = adapter->discover(store);
    if (!discovery) {
        return discovery.error();
    }
    if (std::find(discovery.value().openable_image_ids.begin(), discovery.value().openable_image_ids.end(), image_id) ==
        discovery.value().openable_image_ids.end()) {
        if (std::find(discovery.value().image_ids.begin(), discovery.value().image_ids.end(), image_id) !=
            discovery.value().image_ids.end()) {
            return MakeError(ErrorCode::unsupported_data_type, "Image variable is not openable by this profile",
                             std::string(image_id));
        }
        return MakeError(ErrorCode::not_found, "Image variable was not found", std::string(image_id));
    }
    return adapter->describe(store, image_id);
}

Result<std::vector<Beam>> ReadBeamsFromSchema(const Store& store, std::string_view schema_id,
                                              std::string_view image_id) {
    const SchemaAdapter* adapter = FindAdapter(schema_id);
    if (adapter == nullptr) {
        return MakeError(ErrorCode::unsupported_schema,
                         "No built-in adapter exists for schema " + std::string(schema_id));
    }
    auto discovery = adapter->discover(store);
    if (!discovery) {
        return discovery.error();
    }
    if (std::find(discovery.value().openable_image_ids.begin(), discovery.value().openable_image_ids.end(), image_id) ==
        discovery.value().openable_image_ids.end()) {
        return MakeError(ErrorCode::not_found, "Image variable was not found", std::string(image_id));
    }
    return adapter->read_beams(store, image_id);
}

Result<ProbeResult> ProbeStore(const Store& store) {
    struct AdapterResult {
        const SchemaAdapter* adapter;
        SchemaProbeResult result;
    };

    ProbeResult result;
    std::vector<AdapterResult> matches;
    std::vector<SchemaProbeResult> invalid;
    for (const auto& adapter : SchemaRegistry()) {
        auto probe = adapter.probe(store);
        if (!probe) {
            return probe.error();
        }
        if (probe.value().kind == SchemaMatchKind::match) {
            matches.push_back(AdapterResult{&adapter, probe.value()});
        } else if (probe.value().kind == SchemaMatchKind::invalid) {
            invalid.push_back(probe.value());
        }
    }

    if (matches.size() > 1) {
        result.kind = ProbeKind::invalid_dataset;
        result.diagnostics.push_back(
            Diagnostic{"ambiguous_schema", "More than one built-in schema adapter matched the Zarr store", {}});
    } else if (matches.size() == 1) {
        const auto& match = matches.front();
        auto discovery = match.adapter->discover(store);
        if (!discovery) {
            return discovery.error();
        }
        result.kind = ProbeKind::supported_dataset;
        result.schema_id = match.result.schema_id;
        result.schema_version = match.result.schema_version;
        result.image_ids = std::move(discovery.value().image_ids);
        result.diagnostics = match.result.diagnostics;
    } else if (!invalid.empty()) {
        const auto& invalid_result = invalid.front();
        result.kind = ProbeKind::invalid_dataset;
        result.schema_id = invalid_result.schema_id;
        result.schema_version = invalid_result.schema_version;
        result.diagnostics = invalid_result.diagnostics;
    } else {
        result.kind = ProbeKind::zarr_without_supported_schema;
    }
    return result;
}

}  // namespace carta::zarr::internal
