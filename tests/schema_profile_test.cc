/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// The schema profile's structural decisions -- does this store match, which variables are images,
// which of them are openable -- read metadata and nothing else. They are exercised here through an
// in-memory transport, so a case costs a map entry rather than a directory tree, and the build links
// no TensorStore.
//
// Descriptors are deliberately absent: they read coordinate values, which an in-memory transport has
// none of. Descriptor behaviour is covered in tests/schema_probe_test.cc against fixtures on disk.

#include "schema/profile.h"
#include "store.h"
#include "support/in_memory_transport.h"

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using carta::zarr::ErrorCode;
using carta::zarr::ProbeKind;
using carta::zarr::SchemaMatchKind;
using carta::zarr::testing::MakeInMemoryTransport;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool HasDiagnostic(const carta::zarr::SchemaProbeResult& probe, const std::string& code);

bool HasDiagnostic(const std::vector<carta::zarr::Diagnostic>& diagnostics, const std::string& code) {
    for (const auto& diagnostic : diagnostics) {
        if (diagnostic.code == code) {
            return true;
        }
    }
    return false;
}

bool HasDiagnostic(const carta::zarr::SchemaProbeResult& probe, const std::string& code) {
    return HasDiagnostic(probe.diagnostics, code);
}

std::string RootGroup(bool coordinate_system = true, bool declared = true) {
    std::string attributes = declared ? R"("type": "image_dataset")" : "";
    if (coordinate_system) {
        if (!attributes.empty()) {
            attributes += ",";
        }
        attributes += R"("coordinate_system_info": {
      "projection": "SIN",
      "reference_direction": {"data": [1.0, 0.5]},
      "native_pole_direction": {"data": [0.0, 1.5707963267948966]},
      "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]]
    })";
    }
    return "{\"attributes\":{" + attributes + "},\"zarr_format\":3,\"node_type\":\"group\"}";
}

std::string NumericArray(const std::string& shape, const std::string& dimensions,
                         const std::string& data_type = "float64", const std::string& attributes = "{}") {
    return "{\"shape\":" + shape + ",\"data_type\":\"" + data_type +
           "\",\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":" + shape +
           "}},\"attributes\":" + attributes + ",\"dimension_names\":" + dimensions +
           ",\"zarr_format\":3,\"node_type\":\"array\"}";
}

std::string SkyArray(const std::string& data_type = "float32", const std::string& attributes = R"({"units":"Jy/beam"})",
                     const std::string& dimensions = R"(["time","frequency","polarization","l","m"])") {
    return NumericArray("[1,3,2,4,5]", dimensions, data_type, attributes);
}

std::string PolarizationArray(const std::string& data_type = R"({"name": "fixed_length_utf32",
                                    "configuration": {"length_bytes": 4}})") {
    return "{\"shape\":[2],\"data_type\":" + data_type +
           ",\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":[2]}},"
           "\"attributes\":{},\"dimension_names\":[\"polarization\"],\"zarr_format\":3,\"node_type\":\"array\"}";
}

// A declared image dataset carries a coordinate array for every axis its image uses, time included.
std::map<std::string, std::string> DeclaredStore() {
    return {
        {"", RootGroup()},
        {"SKY", SkyArray()},
        {"time", NumericArray("[1]", R"(["time"])")},
        {"frequency", NumericArray("[3]", R"(["frequency"])")},
        {"polarization", PolarizationArray()},
        {"l", NumericArray("[4]", R"(["l"])")},
        {"m", NumericArray("[5]", R"(["m"])")},
    };
}

// A store predating the root type marker, found by structural detection instead.
std::map<std::string, std::string> LegacyStore() {
    auto nodes = DeclaredStore();
    nodes[""] = RootGroup(true, false);
    nodes.erase("time");
    return nodes;
}

// A store whose child metadata lives only in the root's consolidated_metadata, with no node of its
// own. Probing must find SKY there: consulting the node listing alone silently rejects a store that
// plainly declares itself. See docs/adr/0004-store-seam-at-the-transport.md.
std::map<std::string, std::string> ConsolidatedOnlyLegacyStore() {
    auto children = LegacyStore();
    children.erase("");

    std::string metadata;
    for (const auto& child : children) {
        if (!metadata.empty()) {
            metadata += ",";
        }
        metadata += "\"" + child.first + "\":" + child.second;
    }
    return {{"",
             "{\"attributes\":{\"coordinate_system_info\":{"
             "\"projection\":\"SIN\","
             "\"reference_direction\":{\"data\":[1.0,0.5]},"
             "\"native_pole_direction\":{\"data\":[0.0,1.5707963267948966]},"
             "\"pixel_coordinate_transformation_matrix\":[[1.0,0.0],[0.0,1.0]]}},"
             "\"zarr_format\":3,\"node_type\":\"group\","
             "\"consolidated_metadata\":{\"metadata\":{" +
                 metadata + "}}}"}};
}

carta::zarr::internal::SchemaProfile XradioProfile() {
    auto profile = carta::zarr::internal::SchemaProfile::For(carta::zarr::kXradioImageSchema);
    Require(static_cast<bool>(profile), "the built-in XRADIO profile was not found");
    return profile.value();
}

carta::zarr::Result<carta::zarr::internal::Store> Open(std::map<std::string, std::string> nodes) {
    return carta::zarr::internal::OpenStore(MakeInMemoryTransport(std::move(nodes)));
}

carta::zarr::SchemaProbeResult Probe(std::map<std::string, std::string> nodes) {
    auto store = Open(std::move(nodes));
    Require(static_cast<bool>(store), "OpenStore rejected the in-memory store");
    auto probe = XradioProfile().Probe(store.value());
    Require(static_cast<bool>(probe), "the profile probe reported an error");
    return probe.value();
}

void TestDeclaredAndLegacyMatch() {
    const auto declared = Probe(DeclaredStore());
    Require(declared.kind == SchemaMatchKind::match, "a declared image dataset did not match");
    Require(declared.schema_version == "1.2", "unexpected schema version");

    const auto legacy = Probe(LegacyStore());
    Require(legacy.kind == SchemaMatchKind::match, "a legacy store was not found by structural detection");
}

void TestNonMatch() {
    auto store = Open({{"", RootGroup(true, false)}, {"OTHER", NumericArray("[2]", R"(["x"])")}});
    Require(static_cast<bool>(store), "a valid non-XRADIO group failed to open");
    const auto probe = carta::zarr::internal::ProbeStore(store.value());
    Require(static_cast<bool>(probe), "ProbeStore reported an error for a valid non-XRADIO group");
    Require(probe.value().kind == ProbeKind::zarr_without_supported_schema,
            "a valid non-XRADIO group was not reported as an unsupported schema");
}

void TestMissingCoordinateIsInvalid() {
    auto nodes = DeclaredStore();
    nodes.erase("time");
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid,
            "a declared dataset missing its time coordinate was not reported as invalid");
    Require(HasDiagnostic(probe.diagnostics, "invalid_metadata"), "the missing coordinate produced no diagnostic");
}

void TestCoordinateShapeMismatchIsInvalid() {
    auto nodes = DeclaredStore();
    nodes["frequency"] = NumericArray("[7]", R"(["frequency"])");
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "a coordinate whose length disagrees with the image was accepted");
}

void TestCoordinateDataTypeIsChecked() {
    auto nodes = DeclaredStore();
    // Polarization labels are strings; a numeric polarization coordinate is malformed.
    nodes["polarization"] = PolarizationArray("\"float64\"");
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "a numeric polarization coordinate was accepted");
    Require(HasDiagnostic(probe.diagnostics, "unsupported_data_type"),
            "the polarization data type produced no diagnostic");
}

void TestLegacyStoreNeedsCoordinateSystem() {
    auto nodes = LegacyStore();
    nodes[""] = RootGroup(false, false);
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "a legacy store without coordinate_system_info was accepted");
}

// A legacy store whose SKY does not carry the five sky axes is malformed, not merely old. The axis
// check diagnoses it, and that diagnosis has to stop the probe: without it the probe reaches the
// structural-match path, clears the diagnostics, and calls the store a match.
void TestLegacyStoreNeedsTheSkyAxes() {
    auto missing_axis = LegacyStore();
    missing_axis["SKY"] =
        SkyArray("float32", R"({"units":"Jy/beam"})", R"(["time","frequency","polarization","l","x"])");
    const auto renamed = Probe(missing_axis);
    Require(renamed.kind == SchemaMatchKind::invalid, "a legacy SKY missing a required axis was accepted");
    Require(HasDiagnostic(renamed, "invalid_metadata"), "the missing axis produced no diagnostic");

    auto wrong_rank = LegacyStore();
    wrong_rank["SKY"] =
        NumericArray("[1,3,2,4]", R"(["time","frequency","polarization","l"])", "float32", R"({"units":"Jy/beam"})");
    const auto four = Probe(wrong_rank);
    Require(four.kind == SchemaMatchKind::invalid, "a legacy SKY with four dimensions was accepted");
}

// Discovery reports every variable carrying a whole plane's axes, and says why the ones it cannot
// open are closed. Flags and the optional (l, m) coordinates are never images at all.
void TestDiscoveryClassifiesVariables() {
    auto nodes = DeclaredStore();
    nodes["MODEL"] = SkyArray();
    nodes["COMPLEX"] = SkyArray("complex64");
    nodes["APERTURE"] = SkyArray("float32", "{}", R"(["time","frequency","polarization","u","v"])");
    nodes["MASK_0"] = SkyArray("bool", R"({"type":"flag"})");
    nodes["right_ascension"] = NumericArray("[4,5]", R"(["l","m"])");
    nodes["u"] = NumericArray("[4]", R"(["u"])");
    nodes["v"] = NumericArray("[5]", R"(["v"])");

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the discovery store failed to open");
    auto discovery = XradioProfile().Discover(store.value());
    Require(static_cast<bool>(discovery), "profile discovery reported an error");

    Require(discovery.value().image_ids == std::vector<std::string>{"SKY", "MODEL", "APERTURE", "COMPLEX"},
            "discovery did not enumerate the images in display order");
    Require(discovery.value().openable_image_ids == std::vector<std::string>{"SKY", "MODEL"},
            "discovery did not restrict the openable images to real sky-plane variables");
    Require(HasDiagnostic(discovery.value().diagnostics, "unsupported_coordinate_plane"),
            "the aperture-plane variable produced no diagnostic");
    Require(HasDiagnostic(discovery.value().diagnostics, "unsupported_data_type"),
            "the complex variable produced no diagnostic");
}

// The openable gate sits on the profile handle, ahead of any descriptor work.
void TestOpenableGate() {
    auto nodes = DeclaredStore();
    nodes["COMPLEX"] = SkyArray("complex64");
    nodes["MASK_0"] = SkyArray("bool", R"({"type":"flag"})");

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the openable-gate store failed to open");
    auto profile = carta::zarr::internal::SchemaProfile::For(carta::zarr::kXradioImageSchema);
    Require(static_cast<bool>(profile), "the built-in XRADIO profile was not found");

    const auto complex = profile.value().Describe(store.value(), "COMPLEX");
    Require(!complex && complex.error().code == ErrorCode::unsupported_data_type,
            "a complex variable was not reported as unopenable");

    const auto flag = profile.value().Describe(store.value(), "MASK_0");
    Require(!flag && flag.error().code == ErrorCode::not_found, "a flag variable was exposed as an image");

    const auto absent = profile.value().Describe(store.value(), "NOPE");
    Require(!absent && absent.error().code == ErrorCode::not_found, "an absent variable was not reported as missing");
}

// Regression for the legacy SKY probe. It used to stat the filesystem directly, which cannot see
// consolidated metadata, so a store declaring SKY only there was reported as a non-match rather than
// as the XRADIO image dataset it plainly is. See docs/adr/0004-store-seam-at-the-transport.md.
void TestConsolidatedMetadataDeclaresSky() {
    auto nodes = ConsolidatedOnlyLegacyStore();
    Require(nodes.size() == 1, "the consolidated store must carry no node entries of its own");

    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::match,
            "a store declaring SKY only in consolidated metadata was not matched");

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the consolidated store failed to open");
    auto discovery = XradioProfile().Discover(store.value());
    Require(static_cast<bool>(discovery), "discovery over consolidated metadata reported an error");
    Require(discovery.value().openable_image_ids == std::vector<std::string>{"SKY"},
            "discovery did not find SKY through consolidated metadata");
}

// The report latches: once a requirement is unmet, later ones are no-ops. A store with two faults
// is therefore diagnosed once, by the first fault reached -- the behaviour a probe had when every
// check returned early, now stated somewhere rather than emerging from the control flow.
void TestFirstFaultIsTheOnlyDiagnostic() {
    auto nodes = LegacyStore();
    nodes["frequency"] = NumericArray("[7]", R"(["frequency"])");  // wrong length
    nodes["polarization"] = PolarizationArray("\"float64\"");      // wrong data type

    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "a store with two faults was not reported as invalid");
    Require(probe.diagnostics.size() == 1, "the report diagnosed more than the first unmet requirement");
    Require(probe.diagnostics.front().node_path == "frequency",
            "the diagnostic did not come from the first requirement checked");
    Require(!HasDiagnostic(probe, "unsupported_data_type"), "a requirement after the first failure was still checked");
}

// Store-level rejections, reported before any profile is consulted.
void TestStoreRejections() {
    const auto no_root = Open({{"SKY", SkyArray()}});
    Require(!no_root && no_root.error().code == ErrorCode::not_zarr,
            "a transport with no root node was not rejected as not_zarr");

    const auto version = Open({{"", R"({"zarr_format": 2, "node_type": "group"})"}});
    Require(!version && version.error().code == ErrorCode::unsupported_zarr_version, "Zarr format 2 was not rejected");

    const auto array_root = Open({{"", R"({"zarr_format": 3, "node_type": "array"})"}});
    Require(!array_root && array_root.error().code == ErrorCode::not_zarr, "an array root was not rejected");

    const auto malformed = Open({{"", "{not valid json"}});
    Require(!malformed && malformed.error().code == ErrorCode::invalid_metadata,
            "malformed root metadata was not reported as invalid");

    auto store = Open(DeclaredStore());
    Require(static_cast<bool>(store), "the declared store failed to open");
    const auto unknown = carta::zarr::internal::SchemaProfile::For("future.schema");
    Require(!unknown && unknown.error().code == ErrorCode::unsupported_schema, "an unknown schema id was accepted");
}

}  // namespace

int main() {
    try {
        TestDeclaredAndLegacyMatch();
        TestNonMatch();
        TestMissingCoordinateIsInvalid();
        TestCoordinateShapeMismatchIsInvalid();
        TestCoordinateDataTypeIsChecked();
        TestLegacyStoreNeedsCoordinateSystem();
        TestFirstFaultIsTheOnlyDiagnostic();
        TestLegacyStoreNeedsTheSkyAxes();
        TestDiscoveryClassifiesVariables();
        TestOpenableGate();
        TestConsolidatedMetadataDeclaresSky();
        TestStoreRejections();
        std::cout << "carta-zarr schema profile tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carta-zarr schema profile tests failed: " << error.what() << '\n';
        return 1;
    }
}
