/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "profile.h"

#include "xradio/image.h"

#include <algorithm>
#include <utility>

namespace carta::zarr::internal {
namespace {

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

const ImageEntry* FindImage(const ImageDiscovery& discovery, std::string_view image_id) {
    const auto found = std::find_if(discovery.images.begin(), discovery.images.end(),
                                    [&](const ImageEntry& image) { return image.id == image_id; });
    return found == discovery.images.end() ? nullptr : &*found;
}

}  // namespace

const std::vector<SchemaProfile::Entry>& SchemaProfile::BuiltIn() {
    static const std::vector<SchemaProfile::Entry> profiles{
        {SchemaProfile::Entry{SchemaId(kXradioImageSchema), &xradio::ProbeImage, &xradio::DiscoverImages,
                              &xradio::DescribeImage, &xradio::ReadBeams}}};
    return profiles;
}

Result<SchemaProfile> SchemaProfile::For(std::string_view schema_id) {
    const auto& profiles = BuiltIn();
    const auto found = std::find_if(profiles.begin(), profiles.end(),
                                    [&](const Entry& candidate) { return candidate.id == schema_id; });
    if (found == profiles.end()) {
        return MakeError(ErrorCode::unsupported_schema,
                         "No built-in profile exists for schema " + std::string(schema_id));
    }
    return SchemaProfile{*found};
}

const SchemaId& SchemaProfile::id() const noexcept {
    return _entry->id;
}

Result<SchemaProbeResult> SchemaProfile::Probe(const Store& store) const {
    return _entry->probe(store);
}

Result<ImageDiscovery> SchemaProfile::Discover(const Store& store) const {
    return store.CachedImageDiscovery(_entry->id, [&] { return _entry->discover(store); });
}

Result<void> SchemaProfile::RequireOpenable(const Store& store, std::string_view image_id) const {
    auto discovery = Discover(store);
    if (!discovery) {
        return discovery.error();
    }
    const auto* image = FindImage(discovery.value(), image_id);
    if (image != nullptr && image->readable) {
        return {};
    }
    // A variable this profile listed but will not open is a different answer from one it never saw,
    // and the caller can act on the difference.
    if (image != nullptr) {
        return MakeError(ErrorCode::unsupported_data_type, "Image variable is not openable by this profile",
                         std::string(image_id));
    }
    return MakeError(ErrorCode::not_found, "Image variable was not found", std::string(image_id));
}

Result<ImageDescriptor> SchemaProfile::Describe(const Store& store, std::string_view image_id) const {
    auto openable = RequireOpenable(store, image_id);
    if (!openable) {
        return openable.error();
    }
    return _entry->describe(store, image_id);
}

Result<ImageDescriptor> SchemaProfile::DescribeVerified(const Store& store, std::string_view image_id) const {
    return _entry->describe(store, image_id);
}

Result<std::vector<Beam>> SchemaProfile::ReadBeams(const Store& store, std::string_view image_id) const {
    auto openable = RequireOpenable(store, image_id);
    if (!openable) {
        return openable.error();
    }
    return _entry->read_beams(store, image_id);
}

Result<ProbeResult> ProbeStore(const Store& store) {
    struct Match {
        const SchemaProfile::Entry* entry;
        SchemaProbeResult result;
    };

    ProbeResult result;
    std::vector<Match> matches;
    std::vector<SchemaProbeResult> invalid;
    for (const auto& entry : SchemaProfile::BuiltIn()) {
        auto probe = entry.probe(store);
        if (!probe) {
            return probe.error();
        }
        if (probe.value().kind == SchemaMatchKind::match) {
            matches.push_back(Match{&entry, probe.value()});
        } else if (probe.value().kind == SchemaMatchKind::invalid) {
            invalid.push_back(probe.value());
        }
    }

    if (matches.size() > 1) {
        result.kind = ProbeKind::invalid_dataset;
        result.diagnostics.push_back(
            Diagnostic{"ambiguous_schema", "More than one built-in schema profile matched the Zarr store", {}});
    } else if (matches.size() == 1) {
        const auto& match = matches.front();
        auto discovery = SchemaProfile{*match.entry}.Discover(store);
        if (!discovery) {
            return discovery.error();
        }
        result.kind = ProbeKind::supported_dataset;
        result.schema_id = match.result.schema_id;
        result.schema_version = match.result.schema_version;
        result.images = std::move(discovery.value().images);
        result.default_image_id = std::move(discovery.value().default_image_id);
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
