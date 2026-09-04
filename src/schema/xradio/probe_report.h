/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_SCHEMA_XRADIO_PROBE_REPORT_H_
#define CARTA_ZARR_SRC_SCHEMA_XRADIO_PROBE_REPORT_H_

#include "../../store.h"
#include "../../zarr/array_metadata.h"

#include <string>
#include <string_view>
#include <vector>

namespace carta::zarr::internal::xradio {

enum class CoordinateKind {
    numeric,  // a real-valued coordinate such as frequency, l, or m
    labels,   // a fixed-length UTF-32 coordinate such as polarization
};

/**
 * What a probe found, accumulated as it goes.
 *
 * Each requirement has three outcomes -- the read failed, the check failed, or it passed -- and
 * having every call site unpack all three was most of the length of a probe. The report latches
 * instead: once a requirement is unmet or a read errors, later requirements are no-ops, so a probe
 * states what it needs in sequence and resolves the whole thing once at the end.
 *
 * Only the first unmet requirement is diagnosed, which is what a probe reported before. Reporting
 * every problem at once would be a different contract, and a deliberate change rather than a
 * consequence of this one.
 */
class ProbeReport {
public:
    ProbeReport(const Store& store, std::string profile_name);

    // True while every requirement so far has been met and no read has failed.
    bool ok() const noexcept;

    // The image's coordinate array for one axis must exist, match the image's length for that axis,
    // and hold the right kind of data. An axis the image does not carry is not required.
    bool RequireCoordinateOf(const zarr::ArrayMetadata& image, std::string_view axis, CoordinateKind kind);

    // The root attributes must describe a direction coordinate casacore can be built from.
    bool RequireCoordinateSystem(const nlohmann::json& root_attributes);

    // A node's metadata must have parsed.
    bool RequireArrayMetadata(const Result<zarr::ArrayMetadata>& metadata, std::string_view node);

    // A requirement the caller checked itself. Diagnosing without latching is the one way to get
    // this class wrong -- a probe that says what is broken and then reports a match -- so there is
    // no way to add a diagnostic that does not also stop the probe.
    bool RequireThat(bool condition, std::string code, std::string message, std::string node_path);

    // Findings that are not failures: what discovery observed while listing a store's variables.
    void SetDiagnostics(std::vector<Diagnostic> diagnostics);
    void ClearDiagnostics();

    const std::vector<Diagnostic>& diagnostics() const noexcept;

    // The message for a caller reporting this probe's failure as an error rather than a diagnostic.
    std::string InvalidMessage() const;

    // Yields the probe result, or the read error if one stopped the probe.
    Result<SchemaProbeResult> Finish(SchemaMatchKind kind, std::string schema_version) const;

private:
    void AddDiagnostic(std::string code, std::string message, std::string node_path = {});
    bool Fail(std::string code, std::string message, std::string node_path);

    // ProbeReport borrows the store for its short-lived probing operation.
    const Store* _store;
    std::string _profile_name;
    std::vector<Diagnostic> _diagnostics;
    std::optional<Error> _error;
    bool _met = true;
};

}  // namespace carta::zarr::internal::xradio

#endif  // CARTA_ZARR_SRC_SCHEMA_XRADIO_PROBE_REPORT_H_
