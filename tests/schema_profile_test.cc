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

std::vector<std::string> ImageIds(const std::vector<carta::zarr::ImageEntry>& images) {
    std::vector<std::string> ids;
    ids.reserve(images.size());
    for (const auto& image : images) {
        ids.push_back(image.id);
    }
    return ids;
}

std::vector<std::string> ReadableImageIds(const std::vector<carta::zarr::ImageEntry>& images) {
    std::vector<std::string> ids;
    for (const auto& image : images) {
        if (image.readable) {
            ids.push_back(image.id);
        }
    }
    return ids;
}

std::string RootGroup(bool coordinate_system = true) {
    std::string attributes;
    if (coordinate_system) {
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

// A complete image dataset carries a coordinate array for every axis its image uses, time included.
std::map<std::string, std::string> CompleteStore() {
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

// A store whose child metadata lives only in the root's consolidated_metadata, with no node of its
// own. Discovery must find its images there rather than relying on filesystem children. See
// The profile must use the Store abstraction for both filesystem and consolidated metadata.
std::map<std::string, std::string> ConsolidatedOnlyStore() {
    auto children = CompleteStore();
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

void TestCompleteStoreMatches() {
    const auto probe = Probe(CompleteStore());
    Require(probe.kind == SchemaMatchKind::match, "a complete image dataset did not match");
    Require(probe.schema_version == "1.2", "unexpected schema version");
}

void TestDatasetWithoutSkyMatches() {
    auto nodes = CompleteStore();
    nodes.erase("SKY");
    nodes["RESIDUAL"] = SkyArray();
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::match,
            "an image dataset with only RESIDUAL was incorrectly rejected");
}

void TestNonMatch() {
    auto store = Open({{"", RootGroup()}, {"OTHER", NumericArray("[2]", R"(["x"])")}});
    Require(static_cast<bool>(store), "a valid non-XRADIO group failed to open");
    const auto probe = carta::zarr::internal::ProbeStore(store.value());
    Require(static_cast<bool>(probe), "ProbeStore reported an error for a valid non-XRADIO group");
    Require(probe.value().kind == ProbeKind::zarr_without_supported_schema,
            "a valid non-XRADIO group was not reported as an unsupported schema");
}

void TestMissingCoordinateIsInvalid() {
    auto nodes = CompleteStore();
    nodes.erase("time");
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid,
            "an image dataset missing its time coordinate was not reported as invalid");
    Require(HasDiagnostic(probe.diagnostics, "invalid_metadata"), "the missing coordinate produced no diagnostic");
}

void TestCoordinateShapeMismatchIsInvalid() {
    auto nodes = CompleteStore();
    nodes["frequency"] = NumericArray("[7]", R"(["frequency"])");
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "a coordinate whose length disagrees with the image was accepted");
}

void TestCoordinateDataTypeIsChecked() {
    auto nodes = CompleteStore();
    // Polarization labels are strings; a numeric polarization coordinate is malformed.
    nodes["polarization"] = PolarizationArray("\"float64\"");
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "a numeric polarization coordinate was accepted");
    Require(HasDiagnostic(probe.diagnostics, "unsupported_data_type"),
            "the polarization data type produced no diagnostic");
}

void TestMalformedCoordinateSystemIsInvalid() {
    auto nodes = CompleteStore();
    nodes[""] = R"({"attributes":{"coordinate_system_info":{}},"zarr_format":3,"node_type":"group"})";
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "malformed coordinate_system_info was accepted");
}

// Discovery only recognizes complete image planes. An incomplete SKY-only store is simply a
// non-match because it offers no complete image plane.
void TestIncompleteImageIsNotMatch() {
    auto missing_axis = CompleteStore();
    missing_axis["SKY"] =
        SkyArray("float32", R"({"units":"Jy/beam"})", R"(["time","frequency","polarization","l","x"])");
    const auto renamed = Probe(missing_axis);
    Require(renamed.kind == SchemaMatchKind::no_match, "an incomplete SKY was treated as an image dataset");

    auto wrong_rank = CompleteStore();
    wrong_rank["SKY"] =
        NumericArray("[1,3,2,4]", R"(["time","frequency","polarization","l"])", "float32", R"({"units":"Jy/beam"})");
    const auto four = Probe(wrong_rank);
    Require(four.kind == SchemaMatchKind::no_match, "a four-dimensional SKY was treated as an image dataset");
}

// Discovery reports every variable carrying a whole plane's axes, and says why the ones it cannot
// open are closed. Flags and the optional (l, m) coordinates are never images at all.
void TestDiscoveryClassifiesVariables() {
    auto nodes = CompleteStore();
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

    Require(ImageIds(discovery.value().images) == std::vector<std::string>{"SKY", "MODEL", "APERTURE", "COMPLEX"},
            "discovery did not enumerate the images in display order");
    Require(ReadableImageIds(discovery.value().images) == std::vector<std::string>{"SKY", "MODEL"},
            "discovery did not restrict the openable images to real sky-plane variables");
    Require(discovery.value().default_image_id == "SKY", "discovery did not select the default readable image");
    Require(HasDiagnostic(discovery.value().diagnostics, "unsupported_coordinate_plane"),
            "the aperture-plane variable produced no diagnostic");
    Require(HasDiagnostic(discovery.value().diagnostics, "unsupported_data_type"),
            "the complex variable produced no diagnostic");
}

// The openable gate sits on the profile handle, ahead of any descriptor work.
void TestOpenableGate() {
    auto nodes = CompleteStore();
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

void TestDefaultImageSkipsUnreadablePreferredImage() {
    auto nodes = CompleteStore();
    nodes["SKY"] = SkyArray("complex64");
    nodes["RESIDUAL"] = SkyArray();

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the default-image store failed to open");
    auto discovery = XradioProfile().Discover(store.value());
    Require(static_cast<bool>(discovery), "default-image discovery reported an error");
    Require(discovery.value().default_image_id == "RESIDUAL",
            "discovery selected an unreadable preferred image as the default");

    const auto sky = std::find_if(discovery.value().images.begin(), discovery.value().images.end(),
                                  [](const auto& image) { return image.id == "SKY"; });
    Require(sky != discovery.value().images.end() && !sky->readable &&
                HasDiagnostic(sky->diagnostics, "unsupported_data_type"),
            "the unreadable image did not carry its capability diagnostic");
}

// Regression for discovery through consolidated metadata. It used to stat the filesystem directly,
// which cannot see consolidated metadata.
void TestConsolidatedMetadataDiscovery() {
    auto nodes = ConsolidatedOnlyStore();
    Require(nodes.size() == 1, "the consolidated store must carry no node entries of its own");

    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::match, "a store using consolidated metadata was not matched");

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the consolidated store failed to open");
    auto discovery = XradioProfile().Discover(store.value());
    Require(static_cast<bool>(discovery), "discovery over consolidated metadata reported an error");
    Require(ReadableImageIds(discovery.value().images) == std::vector<std::string>{"SKY"},
            "discovery did not find SKY through consolidated metadata");
}

// The report latches: once a requirement is unmet, later ones are no-ops. A store with two faults
// is therefore diagnosed once, by the first fault reached -- the behaviour a probe had when every
// check returned early, now stated somewhere rather than emerging from the control flow.
void TestFirstFaultIsTheOnlyDiagnostic() {
    auto nodes = CompleteStore();
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

    auto store = Open(CompleteStore());
    Require(static_cast<bool>(store), "the complete store failed to open");
    const auto unknown = carta::zarr::internal::SchemaProfile::For("future.schema");
    Require(!unknown && unknown.error().code == ErrorCode::unsupported_schema, "an unknown schema id was accepted");
}

}  // namespace

int main() {
    try {
        TestCompleteStoreMatches();
        TestDatasetWithoutSkyMatches();
        TestNonMatch();
        TestMissingCoordinateIsInvalid();
        TestCoordinateShapeMismatchIsInvalid();
        TestCoordinateDataTypeIsChecked();
        TestMalformedCoordinateSystemIsInvalid();
        TestFirstFaultIsTheOnlyDiagnostic();
        TestIncompleteImageIsNotMatch();
        TestDiscoveryClassifiesVariables();
        TestOpenableGate();
        TestDefaultImageSkipsUnreadablePreferredImage();
        TestConsolidatedMetadataDiscovery();
        TestStoreRejections();
        std::cout << "carta-zarr schema profile tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carta-zarr schema profile tests failed: " << error.what() << '\n';
        return 1;
    }
}
