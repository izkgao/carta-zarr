/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_SCHEMA_PROFILE_H_
#define CARTA_ZARR_SRC_SCHEMA_PROFILE_H_

#include "../store.h"

#include <string_view>
#include <vector>

namespace carta::zarr::internal {

struct ImageDiscovery {
    std::vector<std::string> image_ids;
    std::vector<std::string> openable_image_ids;
    std::vector<Diagnostic> diagnostics;
};

/**
 * A named, versioned description of how an image dataset is laid out, bound to its identifier.
 *
 * A profile is looked up once and then asked questions, rather than every question carrying an
 * identifier to be resolved again. It binds the identifier only: the built-in table outlives every
 * store, so binding a store here would invent a lifetime problem that does not exist.
 *
 * "Profile" rather than "adapter" deliberately -- CONTEXT.md names this concept the schema profile,
 * and an adapter in this codebase is the thing satisfying an interface at a seam, as the filesystem
 * and in-memory transports do.
 */
class SchemaProfile {
public:
    // Reports unsupported_schema when no built-in profile carries this identifier.
    static Result<SchemaProfile> For(std::string_view schema_id);

    const SchemaId& id() const noexcept;

    Result<SchemaProbeResult> Probe(const Store& store) const;
    Result<ImageDiscovery> Discover(const Store& store) const;

    // Both of these first ask whether the profile will open this image at all; ADR-0001 decides what
    // counts as one.
    Result<ImageDescriptor> Describe(const Store& store, std::string_view image_id) const;
    Result<std::vector<Beam>> ReadBeams(const Store& store, std::string_view image_id) const;

private:
    struct Entry {
        SchemaId id;
        Result<SchemaProbeResult> (*probe)(const Store&);
        Result<ImageDiscovery> (*discover)(const Store&);
        Result<ImageDescriptor> (*describe)(const Store&, std::string_view);
        Result<std::vector<Beam>> (*read_beams)(const Store&, std::string_view);
    };

    static const std::vector<Entry>& BuiltIn();

    explicit SchemaProfile(const Entry& entry) : _entry(&entry) {}
    Result<void> RequireOpenable(const Store& store, std::string_view image_id) const;

    const Entry* _entry;

    friend Result<ProbeResult> ProbeStore(const Store& store);
};

// Ask every built-in profile about a store. More than one match is reported as ambiguous rather
// than resolved silently.
Result<ProbeResult> ProbeStore(const Store& store);

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_SCHEMA_PROFILE_H_
