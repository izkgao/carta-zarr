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
// Whole descriptors are deliberately absent: they read coordinate values, which an in-memory
// transport has none of. Descriptor behaviour is covered in tests/schema_probe_test.cc against
// fixtures on disk. The parts of describing an image that read only metadata do belong here, which
// is why choosing a flag is below: what a flag has to be is checked in tests/flag_test.cc without a
// store at all, and what happens when the store is consulted is checked here.

#include "schema/profile.h"
#include "schema/xradio/flag.h"
#include "schema/xradio/image.h"
#include "store.h"

#include "support/check.h"

#include "support/in_memory_transport.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using carta::zarr::ErrorCode;
using carta::zarr::internal::ProbeKind;
using carta::zarr::SchemaMatchKind;
using carta::zarr::internal::xradio::DetermineFlag;
using carta::zarr::testing::MakeInMemoryTransport;

using carta::zarr::testing::Require;

bool HasDiagnostic(const carta::zarr::SchemaProbeResult& probe, carta::zarr::DiagnosticCode code);

bool HasDiagnostic(const std::vector<carta::zarr::Diagnostic>& diagnostics, carta::zarr::DiagnosticCode code) {
    for (const auto& diagnostic : diagnostics) {
        if (diagnostic.code == code) {
            return true;
        }
    }
    return false;
}

bool HasDiagnostic(const carta::zarr::SchemaProbeResult& probe, carta::zarr::DiagnosticCode code) {
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

std::vector<std::string> OpenableImageIds(const std::vector<carta::zarr::ImageEntry>& images) {
    std::vector<std::string> ids;
    for (const auto& image : images) {
        if (image.openable) {
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

// The same store with a copy of every child's metadata in the root, which is what zarr-python
// writes when a dataset is consolidated. The children keep their own metadata: consolidated
// metadata is a copy that saves reading them, carrying `must_understand: false` precisely so that a
// reader which ignores it still reads the same hierarchy. A store whose children exist only in the
// root is malformed, and no reader but this one could open it -- the array data behind those names
// is read by TensorStore, which needs each array's own metadata.
std::map<std::string, std::string> ConsolidatedStore() {
    auto nodes = CompleteStore();

    std::string metadata;
    for (const auto& child : nodes) {
        if (child.first.empty()) {
            continue;
        }
        if (!metadata.empty()) {
            metadata += ",";
        }
        metadata += "\"" + child.first + "\":" + child.second;
    }
    nodes[""] = "{\"attributes\":{\"coordinate_system_info\":{"
                "\"projection\":\"SIN\","
                "\"reference_direction\":{\"data\":[1.0,0.5]},"
                "\"native_pole_direction\":{\"data\":[0.0,1.5707963267948966]},"
                "\"pixel_coordinate_transformation_matrix\":[[1.0,0.0],[0.0,1.0]]}},"
                "\"zarr_format\":3,\"node_type\":\"group\","
                "\"consolidated_metadata\":{\"kind\":\"inline\",\"must_understand\":false,\"metadata\":{" +
                metadata + "}}}";
    return nodes;
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
    // Nothing in it looked like an image, so the profile has nothing to say about it either.
    Require(probe.value().diagnostics.empty() && probe.value().schema_id.empty(),
            "a store with nothing the profile recognised was reported with the profile's say-so");
}

// A store whose only images are ones this library does not open is still one no profile matched,
// and the reason is the profile's own: it recognised the images and refused them. That reason used
// to be dropped on the way out of ProbeStore, so the refusal a consumer saw was "no profile matched"
// with nothing to say which variable, or why.
void TestNothingOpenableSaysWhy() {
    using carta::zarr::internal::ProbeStore;
    using carta::zarr::internal::RequireOpenableDataset;

    const auto refused = [](std::map<std::string, std::string> nodes, carta::zarr::DiagnosticCode code,
                            const std::string& message) {
        auto store = Open(std::move(nodes));
        Require(static_cast<bool>(store), "the unopenable store failed to open");
        const auto probe = ProbeStore(store.value());
        Require(static_cast<bool>(probe), "ProbeStore reported an error for a well-formed store");
        Require(probe.value().kind == ProbeKind::zarr_without_supported_schema,
                "a store of variables this library does not open was not reported as unsupported");
        // Diagnostics alone: naming the schema would say the store is one.
        Require(probe.value().schema_id.empty(), "an unmatched store was attributed to a schema");
        Require(HasDiagnostic(probe.value().diagnostics, code) && probe.value().diagnostics.front().node_path == "SKY",
                "an unmatched store lost the reason its images were refused");

        const auto openable = RequireOpenableDataset(probe.value(), "/tmp/store");
        Require(!openable && openable.error().code == ErrorCode::unsupported_schema,
                "a store of variables this library does not open was not refused as an unsupported schema");
        Require(openable.error().message == message,
                "the refusal did not carry the profile's reason: " + openable.error().message);
    };

    auto complex = CompleteStore();
    complex["SKY"] = SkyArray("complex64");
    refused(complex, carta::zarr::DiagnosticCode::unsupported_data_type,
            "Complex sky-plane variables are not openable");

    auto aperture = CompleteStore();
    aperture["SKY"] = SkyArray("float32", "{}", R"(["time","frequency","polarization","u","v"])");
    refused(aperture, carta::zarr::DiagnosticCode::unsupported_coordinate_plane,
            "Aperture-plane variables are not openable");
}

void TestMissingCoordinateIsInvalid() {
    auto nodes = CompleteStore();
    nodes.erase("time");
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid,
            "an image dataset missing its time coordinate was not reported as invalid");
    Require(HasDiagnostic(probe.diagnostics, carta::zarr::DiagnosticCode::invalid_metadata),
            "the missing coordinate produced no diagnostic");
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
    Require(HasDiagnostic(probe.diagnostics, carta::zarr::DiagnosticCode::unsupported_data_type),
            "the polarization data type produced no diagnostic");
}

void TestMalformedCoordinateSystemIsInvalid() {
    auto nodes = CompleteStore();
    nodes[""] = R"({"attributes":{"coordinate_system_info":{}},"zarr_format":3,"node_type":"group"})";
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "malformed coordinate_system_info was accepted");
}

// Every XRADIO writes coordinate_system_info, so a store without it is malformed too. The probe
// used to look only when it was there, and such a store opened with a direction of "" at (0, 0).
void TestAMissingCoordinateSystemIsInvalid() {
    auto nodes = CompleteStore();
    nodes[""] = RootGroup(false);
    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "a store with no coordinate_system_info was accepted");
    Require(!probe.diagnostics.empty() &&
                probe.diagnostics.front().node_path == "/attributes/coordinate_system_info",
            "the refusal should name the attribute that is missing");
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
    Require(OpenableImageIds(discovery.value().images) == std::vector<std::string>{"SKY", "MODEL"},
            "discovery did not restrict the openable images to real sky-plane variables");
    Require(discovery.value().default_image_id == "SKY", "discovery did not select the default openable image");
    Require(HasDiagnostic(discovery.value().diagnostics, carta::zarr::DiagnosticCode::unsupported_coordinate_plane),
            "the aperture-plane variable produced no diagnostic");
    Require(HasDiagnostic(discovery.value().diagnostics, carta::zarr::DiagnosticCode::unsupported_data_type),
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

// Coordinates belong to the dataset and every image references them by dimension name, so an image
// whose own extent disagrees with the coordinate it names cannot be described with it: it would be
// reported with three channels' worth of coordinates over seven channels of pixels. It used to be
// listed as openable and refused when a consumer tried to open it.
void TestAnImageDisagreeingWithACoordinateIsNotOpenable() {
    auto nodes = CompleteStore();
    // Seven channels of pixels where the dataset's frequency coordinate has three.
    nodes["MODEL"] = NumericArray("[1,7,2,4,5]", R"(["time","frequency","polarization","l","m"])", "float32",
                                  R"({"units":"Jy/beam"})");

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the disagreeing-image store failed to open");
    const auto profile = XradioProfile();
    const auto discovery = profile.Discover(store.value());
    Require(static_cast<bool>(discovery), "discovery failed on the disagreeing-image store");
    Require(ImageIds(discovery.value().images) == std::vector<std::string>{"SKY", "MODEL"},
            "the disagreeing image was dropped from the listing rather than listed with its reason");
    Require(OpenableImageIds(discovery.value().images) == std::vector<std::string>{"SKY"},
            "an image that will not open was listed as openable");
    Require(discovery.value().default_image_id == "SKY", "the default image was not the first openable one");

    const auto model = profile.Describe(store.value(), "MODEL");
    Require(!model && model.error().code == ErrorCode::invalid_metadata,
            "a store that is wrong about an image was refused as a library limitation");
}

// Nothing openable is two different answers, and a consumer acts on the difference. A store of
// well-formed variables this library does not open is a store it is not for; a store whose only
// image disagrees with its coordinates is one this profile recognised and found broken, and
// reporting that as "no profile matched" names nothing anyone can act on.
void TestNothingOpenableIsInvalidOnlyWhenSomethingIsMalformed() {
    auto capability = CompleteStore();
    capability["SKY"] = SkyArray("complex64");
    Require(Probe(capability).kind == SchemaMatchKind::no_match,
            "a store of variables this library does not open was reported as broken");

    auto malformed = CompleteStore();
    malformed["SKY"] = NumericArray("[1,7,2,4,5]", R"(["time","frequency","polarization","l","m"])", "float32",
                                    R"({"units":"Jy/beam"})");
    // Enumerated before SKY, and a limitation rather than a defect: without the reason for the
    // refusal being put first, this is the diagnostic a consumer would be shown.
    malformed["APERTURE"] = NumericArray("[1,3,2,4,5]", R"(["time","frequency","polarization","u","v"])", "float32");

    const auto broken = Probe(malformed);
    Require(broken.kind == SchemaMatchKind::invalid,
            "a store whose only image disagrees with a coordinate was reported as unrecognised");
    Require(broken.diagnostics.front().node_path == "SKY",
            "the refusal did not lead with the reason the store was refused");
}

// A group is a node in the hierarchy, and a store is free to hold one. It was being read as an
// array, failing, and reported as a node that could not be read -- so every store with a nested
// group carried a diagnostic about a node that was perfectly well formed, and a promotion rule that
// counted such a diagnostic would close a store over it.
void TestANestedGroupIsNotABrokenArray() {
    auto nodes = CompleteStore();
    nodes["SUBDIR"] = RootGroup(false);

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the nested-group store failed to open");
    const auto profile = XradioProfile();
    const auto discovery = profile.Discover(store.value());
    Require(static_cast<bool>(discovery), "discovery failed on the nested-group store");
    Require(!HasDiagnostic(discovery.value().diagnostics, carta::zarr::DiagnosticCode::unreadable_array),
            "a valid group was diagnosed as an array that could not be read");
    Require(ImageIds(discovery.value().images) == std::vector<std::string>{"SKY"},
            "a group was listed among the dataset's images");

    // Naming one is naming something that is not an image, which is the answer a flag and a
    // coordinate get as well.
    const auto group = profile.Describe(store.value(), "SUBDIR");
    Require(!group && group.error().code == ErrorCode::not_found,
            "a group was refused as a broken array rather than as something that is not an image");
}

// A refusal names the variable's own reason rather than answering generically.
//
// This used to be what two copies of the openability rule disagreed about: the facade passed the
// listing's diagnostic through and the profile answered "not openable by this profile", so a
// consumer got a worse explanation depending on which way it had arrived. The copies are one
// decision now, and what is asserted here is the half that a single rule does not guarantee on its
// own -- that the reason survives the trip from the listing to the refusal.
void TestARefusalCarriesTheVariablesOwnReason() {
    auto nodes = CompleteStore();
    nodes["COMPLEX"] = SkyArray("complex64");
    // Valid JSON, and not array metadata: node_type says array and there is no shape.
    nodes["BROKEN"] = R"({"zarr_format":3,"node_type":"array","data_type":"float32"})";

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the refusal store failed to open");
    const auto profile = XradioProfile();
    const auto discovery = profile.Discover(store.value());
    Require(static_cast<bool>(discovery), "discovery failed on the refusal store");
    const auto& images = discovery.value().images;

    const auto complex = profile.Describe(store.value(), "COMPLEX");
    Require(!complex && complex.error().code == ErrorCode::unsupported_data_type,
            "a listed but unopenable variable should be refused as an unsupported data type");

    const auto listed = std::find_if(images.begin(), images.end(),
                                     [](const auto& image) { return image.id == "COMPLEX"; });
    Require(listed != images.end() && !listed->diagnostics.empty(),
            "the complex variable was not listed with a diagnostic to pass on");
    Require(complex.error().message == listed->diagnostics.front().message,
            "the refusal did not carry the variable's own diagnostic");

    const auto absent = profile.Describe(store.value(), "NOPE");
    Require(!absent && absent.error().code == ErrorCode::not_found,
            "a variable that is not there should be reported as missing, not as unopenable");

    // A node that is there and will not parse is neither of those. It used to answer "not found",
    // because the rule ran against a listing the node had already dropped out of, which is a
    // different thing from a name the dataset does not have.
    const auto broken = profile.Describe(store.value(), "BROKEN");
    Require(!broken && broken.error().code == ErrorCode::invalid_metadata,
            "a node that is present and will not parse was reported as a missing variable");
}

// The dataset-level counterpart, and the three answers it has to keep apart. This used to be
// written out in Dataset::Open, so a probe's refusal meant whatever the facade decided it meant --
// and none of it was reachable without a store on disk shaped to produce each kind.
void TestOneRuleDecidesWhatDatasetIsOpenable() {
    using carta::zarr::internal::RequireOpenableDataset;

    // Nothing matched. Not an error about the file's contents: this library is not for it.
    {
        carta::zarr::internal::ProbeResult probe;
        probe.kind = ProbeKind::zarr_without_supported_schema;
        const auto openable = RequireOpenableDataset(probe, "/tmp/store");
        Require(!openable && openable.error().code == ErrorCode::unsupported_schema,
                "an unmatched store should be refused as an unsupported schema");
    }

    // Something matched and was malformed. A different answer, and a consumer acts on it.
    {
        carta::zarr::internal::ProbeResult probe;
        probe.kind = ProbeKind::invalid_dataset;
        probe.diagnostics.push_back(
            carta::zarr::Diagnostic{carta::zarr::DiagnosticCode::invalid_metadata, "frequency is missing", "SKY"});
        const auto openable = RequireOpenableDataset(probe, "/tmp/store");
        Require(!openable && openable.error().code == ErrorCode::invalid_metadata,
                "a malformed store should be refused as invalid metadata");
        Require(openable.error().message == "frequency is missing",
                "and it should carry what the probe worked out, not a generic refusal");
    }

    // A probe that had nothing to say falls back rather than reporting an empty message.
    {
        carta::zarr::internal::ProbeResult probe;
        probe.kind = ProbeKind::zarr_without_supported_schema;
        const auto openable = RequireOpenableDataset(probe, "/tmp/store");
        Require(!openable && !openable.error().message.empty(), "a silent probe still needs a message");
    }

    // A profile matched a store with nothing openable in it. The profile is not refusing, so there
    // are no diagnostics; opening it would hand back a dataset a consumer can do nothing with.
    {
        carta::zarr::internal::ProbeResult probe;
        probe.kind = ProbeKind::supported_dataset;
        const auto openable = RequireOpenableDataset(probe, "/tmp/store");
        Require(!openable && openable.error().code == ErrorCode::invalid_metadata,
                "a matched store with no images should be refused");
    }

    // And the one case that opens.
    {
        carta::zarr::internal::ProbeResult probe;
        probe.kind = ProbeKind::supported_dataset;
        probe.images.push_back(carta::zarr::ImageEntry{});
        Require(static_cast<bool>(RequireOpenableDataset(probe, "/tmp/store")),
                "a matched store with an image should open");
    }
}

void TestDefaultImageSkipsUnopenablePreferredImage() {
    auto nodes = CompleteStore();
    nodes["SKY"] = SkyArray("complex64");
    nodes["RESIDUAL"] = SkyArray();

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the default-image store failed to open");
    auto discovery = XradioProfile().Discover(store.value());
    Require(static_cast<bool>(discovery), "default-image discovery reported an error");
    Require(discovery.value().default_image_id == "RESIDUAL",
            "discovery selected an unopenable preferred image as the default");

    const auto sky = std::find_if(discovery.value().images.begin(), discovery.value().images.end(),
                                  [](const auto& image) { return image.id == "SKY"; });
    Require(sky != discovery.value().images.end() && !sky->openable &&
                HasDiagnostic(sky->diagnostics, carta::zarr::DiagnosticCode::unsupported_data_type),
            "the unopenable image did not carry its capability diagnostic");
}

// What consolidated metadata is for: the root's copy answers for every child, so discovery reads
// one node instead of one per variable. That saving is the whole reason the block exists -- on a
// store reached over a network each of those reads is a round trip -- so it is asserted as reads
// not taken, which is the only thing that tells a store that used the copy from one that ignored
// it. The children are present throughout: this is a cache, and the test would pass either way if
// it only checked the answer.
void TestConsolidatedMetadataDiscovery() {
    auto nodes = ConsolidatedStore();
    Require(nodes.size() > 1, "the consolidated store must carry its child nodes as well as the root copy");

    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::match, "a store using consolidated metadata was not matched");

    auto transport = MakeInMemoryTransport(nodes);
    auto store = carta::zarr::internal::OpenStore(transport);
    Require(static_cast<bool>(store), "the consolidated store failed to open");
    auto discovery = XradioProfile().Discover(store.value());
    Require(static_cast<bool>(discovery), "discovery over consolidated metadata reported an error");
    Require(OpenableImageIds(discovery.value().images) == std::vector<std::string>{"SKY"},
            "discovery did not find SKY through consolidated metadata");
    Require(transport->nodes_read() == std::set<std::string>{""},
            "discovery read a child node that the root's consolidated copy already answered for");
    Require(transport->listings() == 0,
            "discovery listed the hierarchy although consolidated metadata names every node");

    // The control: the same store without the root's copy reads every child instead, and reaches
    // the same answer. Both are supported; only one of them costs a read per variable.
    auto plain = MakeInMemoryTransport(CompleteStore());
    auto plain_store = carta::zarr::internal::OpenStore(plain);
    Require(static_cast<bool>(plain_store), "the unconsolidated store failed to open");
    auto plain_discovery = XradioProfile().Discover(plain_store.value());
    Require(static_cast<bool>(plain_discovery), "discovery without consolidated metadata reported an error");
    Require(OpenableImageIds(plain_discovery.value().images) == OpenableImageIds(discovery.value().images),
            "the two metadata layouts did not describe the same images");
    Require(plain->nodes_read().size() > 1,
            "a store without consolidated metadata has to read its children");
}

// A consolidated key is filed under the canonical name of the node it describes. A block is entitled
// to spell a node "./SKY" -- it names the node SKY -- and filed under that spelling the copy was
// never found by a read of "SKY": the node was read from the transport regardless, cached twice,
// and listed as "./SKY". That is an image id no reader asks for, and since openability is decided
// against the listing, SKY could not be opened by its own name.
void TestAConsolidatedKeyIsFiledUnderItsCanonicalName() {
    auto nodes = ConsolidatedStore();
    auto& root = nodes[""];
    const auto key = root.find("\"SKY\":");
    Require(key != std::string::npos, "the consolidated fixture has no SKY entry to respell");
    root.replace(key, 6, "\"./SKY\":");

    auto transport = MakeInMemoryTransport(nodes);
    auto store = carta::zarr::internal::OpenStore(transport);
    Require(static_cast<bool>(store), "a consolidated key spelled with ./ was refused");
    const auto discovery = XradioProfile().Discover(store.value());
    Require(static_cast<bool>(discovery), "discovery failed over a respelled consolidated key");
    Require(ImageIds(discovery.value().images) == std::vector<std::string>{"SKY"},
            "the image was listed under the spelling the block used rather than under its name");
    Require(transport->nodes_read() == std::set<std::string>{""},
            "the node was read again although the consolidated copy described it");
}

// Two ways a consolidated block can be wrong about itself, and both refuse the store. For a store
// that consolidated its metadata the block is the whole listing, so a key passed over would be a
// variable missing from the dataset without a word said about it.
void TestAConsolidatedBlockThatMisnamesItsNodesIsRefused() {
    // A key that no node path can be.
    auto escaping = ConsolidatedStore();
    auto& escaping_root = escaping[""];
    escaping_root.replace(escaping_root.find("\"SKY\":"), 6, "\"../SKY\":");
    const auto outside = Open(escaping);
    Require(!outside && outside.error().code == ErrorCode::invalid_metadata,
            "a consolidated key naming a path outside the store was accepted");

    // One node under two spellings. Which document won used to depend on the order the object was
    // walked in.
    auto doubled = ConsolidatedStore();
    auto& doubled_root = doubled[""];
    doubled_root.insert(doubled_root.find("\"SKY\":"), "\"./SKY\":" + SkyArray() + ",");
    const auto twice = Open(doubled);
    Require(!twice && twice.error().code == ErrorCode::invalid_metadata,
            "a consolidated block listing one node twice was accepted");
}

// The inventory decides once what each node is. Its callers used to decide it three ways -- one read
// node_type out of the document, one parsed every node and looked again when a parse failed, one
// parsed every node and ignored the answer -- and agreed only where the three happened to coincide.
void TestTheInventorySaysWhatEachNodeIs() {
    using carta::zarr::internal::NodeKind;
    auto nodes = CompleteStore();
    nodes["SUBDIR"] = RootGroup(false);
    // Says it is an array, and is not one: no shape.
    nodes["BROKEN"] = R"({"zarr_format":3,"node_type":"array","data_type":"float32"})";
    nodes["ODD"] = R"({"zarr_format":3,"node_type":"manifest"})";
    // Present and not a string. Asked for as a string with a default, it threw rather than answered,
    // and the whole discovery went with it.
    nodes["NUMBERED"] = R"({"zarr_format":3,"node_type":3})";

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the inventory store failed to open");
    const auto& inventory = store.value().Inventory();
    Require(static_cast<bool>(inventory), "the inventory could not be taken");

    const auto find = [&](const std::string& name) {
        const auto* entry = store.value().FindNode(name);
        Require(entry != nullptr, "the inventory has no entry for " + name);
        return entry;
    };

    const auto* sky = find("SKY");
    Require(sky->kind == NodeKind::array && sky->array != nullptr && static_cast<bool>(*sky->array) && !sky->reason,
            "an array was not carried with its parsed metadata");
    const auto* group = find("SUBDIR");
    Require(group->kind == NodeKind::group && group->array == nullptr && !group->reason,
            "a group was parsed as an array, or given a reason to be something else");
    const auto* broken = find("BROKEN");
    Require(broken->kind == NodeKind::array && broken->array != nullptr && !*broken->array,
            "an array whose metadata does not parse was not carried with its refusal");
    for (const std::string name : {"ODD", "NUMBERED"}) {
        const auto* entry = find(name);
        Require(entry->kind == NodeKind::unrecognised && entry->array == nullptr && entry->reason.has_value(),
                "a node that does not say it is a group or an array was not unrecognised: " + name);
    }

    // Sorted by name, which is the order discovery's diagnostics come out in.
    Require(std::is_sorted(inventory.value().begin(), inventory.value().end(),
                           [](const auto& left, const auto& right) { return left.name < right.name; }),
            "the inventory was not sorted by name");
    // A name is looked up as it was handed out. Another spelling of it is a name nobody was given.
    Require(store.value().FindNode("./SKY") == nullptr, "a respelled name was looked up as the node");
    Require(store.value().FindNode("NOPE") == nullptr, "a node the store does not hold was found");

    Require(static_cast<bool>(XradioProfile().Discover(store.value())),
            "discovery could not walk a store holding a node whose node_type is not a string");
}

// A node whose document will not parse is diagnosed, not refused. CONTEXT.md says so of a node whose
// metadata would not parse -- the rest of the dataset is still readable -- and the listing used to
// refuse the whole hierarchy over one, so a stray broken document beside a perfectly good image
// closed the dataset.
void TestANodeThatWillNotParseIsDiagnosedNotRefused() {
    auto nodes = CompleteStore();
    nodes["JUNK"] = "{not valid json";
    nodes["ODD"] = R"({"zarr_format":3,"node_type":"manifest"})";

    Require(Probe(nodes).kind == SchemaMatchKind::match,
            "a document that will not parse refused a store whose image is fine");

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the store holding an unparseable node failed to open");
    const auto discovery = XradioProfile().Discover(store.value());
    Require(static_cast<bool>(discovery), "discovery refused a store holding an unparseable node");
    Require(OpenableImageIds(discovery.value().images) == std::vector<std::string>{"SKY"},
            "the image beside an unparseable node was not openable");

    // Neither said what it was, and both say so under the one code for that. Neither is called an
    // array, because neither claimed to be one.
    const auto& said = discovery.value().diagnostics;
    const auto said_of = [&](const std::string& node, carta::zarr::DiagnosticCode code) {
        return std::any_of(said.begin(), said.end(), [&](const auto& diagnostic) {
            return diagnostic.node_path == node && diagnostic.code == code;
        });
    };
    Require(said_of("JUNK", carta::zarr::DiagnosticCode::unrecognised_node),
            "a document that will not parse was not diagnosed as such");
    Require(said_of("ODD", carta::zarr::DiagnosticCode::unrecognised_node),
            "a node_type Zarr does not define was not diagnosed as such");
    Require(!HasDiagnostic(said, carta::zarr::DiagnosticCode::unreadable_array),
            "a node that never said it was an array was called one");

    // The declared size counts the arrays it can see, and a node it cannot read is not one of them.
    const auto size = carta::zarr::internal::DatasetSizeBytes(store.value(),
                                                              std::chrono::steady_clock::now() +
                                                                  std::chrono::seconds(5));
    Require(size && size.value().bytes == 592, "an unparseable node left the dataset without a declared size");
}

// The report latches: once a requirement is unmet, later ones are no-ops. A store with two faults
// is therefore diagnosed once, by the first fault reached -- the behaviour a probe had when every
// check returned early, now stated somewhere rather than emerging from the control flow.
//
// Both faults have to be ones the probe still reaches. A coordinate of the wrong length is not:
// agreeing with it is part of being an image this profile opens, so a store with one has no
// openable image and never gets as far as the requirements below.
void TestFirstFaultIsTheOnlyDiagnostic() {
    auto nodes = CompleteStore();
    nodes["frequency"] = NumericArray("[3]", R"(["x"])");     // right length, names another axis
    nodes["polarization"] = PolarizationArray("\"float64\"");  // wrong data type

    const auto probe = Probe(nodes);
    Require(probe.kind == SchemaMatchKind::invalid, "a store with two faults was not reported as invalid");
    Require(probe.diagnostics.size() == 1, "the report diagnosed more than the first unmet requirement");
    Require(probe.diagnostics.front().node_path == "frequency",
            "the diagnostic did not come from the first requirement checked");
    Require(!HasDiagnostic(probe, carta::zarr::DiagnosticCode::unsupported_data_type),
            "a requirement after the first failure was still checked");
}

// Store-level rejections, reported before any profile is consulted.
// Choosing a flag is the half of it that consults the store: what a flag has to be once it is found
// takes no store at all and is checked in tests/flag_test.cc.
//
// A declared flag is the image's own statement that its pixels need a mask, so an image naming one
// that cannot be read is closed rather than opened unmasked. Reads apply the mask by default, and an
// unusable mask reported as no mask would show flagged pixels as valid -- the one failure a consumer
// has no way to notice.
void TestADeclaredFlagIsBinding() {
    const std::string sky_dimensions = R"(["time","frequency","polarization","l","m"])";
    const auto with_declared_flag = [&](bool write_the_flag) {
        auto nodes = CompleteStore();
        nodes["SKY"] = SkyArray("float32", R"({"units":"Jy/beam","flag":"MASK_0"})");
        if (write_the_flag) {
            nodes["MASK_0"] = NumericArray("[1,3,2,4,5]", sky_dimensions, "bool", R"({"type":"flag"})");
        }
        return nodes;
    };

    auto present = Open(with_declared_flag(true));
    Require(static_cast<bool>(present), "the declared-flag store did not open");
    const auto& present_image = present.value().ReadArrayMetadata("SKY");
    Require(static_cast<bool>(present_image), "SKY was not readable in the declared-flag store");
    std::vector<carta::zarr::Diagnostic> diagnostics;
    auto declared = DetermineFlag(present.value(), present_image.value(), "SKY", diagnostics);
    Require(static_cast<bool>(declared), "a well-formed declared flag was refused");
    Require(declared.value() == "MASK_0", "the declared flag was not the one selected");
    Require(diagnostics.empty(), "selecting a declared flag produced a diagnostic");

    auto absent = Open(with_declared_flag(false));
    Require(static_cast<bool>(absent), "the missing-flag store did not open");
    const auto& absent_image = absent.value().ReadArrayMetadata("SKY");
    Require(static_cast<bool>(absent_image), "SKY was not readable in the missing-flag store");
    auto missing = DetermineFlag(absent.value(), absent_image.value(), "SKY", diagnostics);
    Require(!missing, "an image declaring a flag variable that does not exist was opened");
}

// The listing's half of the same rule. An image whose declared flag cannot mask it is closed when it
// is opened, so it has to be closed when it is listed too: it was listed openable and chosen as the
// default, and a consumer offered an image it could not open.
void TestADeclaredFlagThatCannotMaskClosesTheImageInTheListing() {
    const std::string sky_dimensions = R"(["time","frequency","polarization","l","m"])";
    const auto listing = [&](const std::string& flag_node) {
        auto nodes = CompleteStore();
        nodes["SKY"] = SkyArray("float32", R"({"units":"Jy/beam","flag":"MASK_0"})");
        nodes["MODEL"] = SkyArray();
        if (!flag_node.empty()) {
            nodes["MASK_0"] = flag_node;
        }
        auto store = Open(nodes);
        Require(static_cast<bool>(store), "the declared-flag listing store did not open");
        const auto profile = XradioProfile();
        auto discovery = profile.Discover(store.value());
        Require(static_cast<bool>(discovery), "discovery failed on the declared-flag listing store");
        return std::make_pair(discovery.value(), profile.Describe(store.value(), "SKY"));
    };

    const auto usable = listing(NumericArray("[1,3,2,4,5]", sky_dimensions, "bool", R"({"type":"flag"})"));
    Require(OpenableImageIds(usable.first.images) == std::vector<std::string>{"SKY", "MODEL"},
            "an image whose declared flag can mask it was not listed openable");

    const std::vector<std::pair<std::string, std::string>> unusable{
        {"", "a declared flag that does not exist"},
        {NumericArray("[1,3,2,4,4]", sky_dimensions, "bool", R"({"type":"flag"})"), "a flag of another shape"},
        {NumericArray("[1,3,2,4,5]", sky_dimensions, "uint8", R"({"type":"flag"})"), "a flag that is not boolean"},
    };
    for (const auto& [flag_node, what] : unusable) {
        const auto [discovery, sky] = listing(flag_node);
        Require(ImageIds(discovery.images) == std::vector<std::string>{"SKY", "MODEL"},
                "an image with " + what + " was dropped from the listing rather than listed with its reason");
        Require(OpenableImageIds(discovery.images) == std::vector<std::string>{"MODEL"},
                "an image with " + what + " was listed openable");
        Require(discovery.default_image_id == "MODEL", "an image with " + what + " was chosen as the default");
        Require(HasDiagnostic(discovery.images.front().diagnostics, carta::zarr::DiagnosticCode::invalid_metadata),
                "an image with " + what + " was listed without saying why it will not open");
        Require(!sky && sky.error().code == ErrorCode::invalid_metadata,
                "describing an image with " + what + " was not refused as invalid metadata");
    }
}

// An image is described by the five axes this profile knows. One with a sixth used to be listed
// openable and described with five, so every read of it failed on the rank it had not been told.
void TestAnImageWithAnAxisBeyondTheFiveIsNotOpenable() {
    auto nodes = CompleteStore();
    nodes["MODEL"] = NumericArray("[1,3,2,4,5,1]", R"(["time","frequency","polarization","l","m","extra"])",
                                  "float32", R"({"units":"Jy/beam"})");

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the extra-axis store failed to open");
    const auto profile = XradioProfile();
    const auto discovery = profile.Discover(store.value());
    Require(static_cast<bool>(discovery), "discovery failed on the extra-axis store");
    Require(ImageIds(discovery.value().images) == std::vector<std::string>{"SKY", "MODEL"},
            "the extra-axis image was dropped from the listing rather than listed with its reason");
    Require(OpenableImageIds(discovery.value().images) == std::vector<std::string>{"SKY"},
            "an image with an axis this profile cannot describe was listed openable");
    Require(HasDiagnostic(discovery.value().images.back().diagnostics, carta::zarr::DiagnosticCode::invalid_metadata),
            "the extra-axis image was listed without saying why it will not open");

    const auto model = profile.Describe(store.value(), "MODEL");
    Require(!model && model.error().code == ErrorCode::invalid_metadata,
            "an image with an axis this profile cannot describe was not refused as invalid metadata");
}

// With nothing declared the store is inspected instead, and a store offering two equally good
// candidates is refused rather than guessed at. The refusal is a diagnostic on the image: the image
// is still readable, just unmasked.
void TestAmbiguousFlagsSelectNone() {
    const std::string sky_dimensions = R"(["time","frequency","polarization","l","m"])";
    auto nodes = CompleteStore();
    nodes["FLAG_1"] = NumericArray("[1,3,2,4,5]", sky_dimensions, "bool", R"({"type":"flag"})");
    nodes["FLAG_2"] = NumericArray("[1,3,2,4,5]", sky_dimensions, "bool", R"({"type":"flag"})");

    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the ambiguous-flag store did not open");
    const auto& image = store.value().ReadArrayMetadata("SKY");
    Require(static_cast<bool>(image), "SKY was not readable in the ambiguous-flag store");

    std::vector<carta::zarr::Diagnostic> diagnostics;
    auto chosen = DetermineFlag(store.value(), image.value(), "SKY", diagnostics);
    Require(static_cast<bool>(chosen), "ambiguous flags reported an error rather than no mask");
    Require(chosen.value().empty(), "ambiguous flags selected a pixel mask");
    Require(HasDiagnostic(diagnostics, carta::zarr::DiagnosticCode::ambiguous_pixel_mask),
            "ambiguous flags did not produce a diagnostic");

    // One candidate is not ambiguous: the same store with FLAG_2 removed selects FLAG_1.
    nodes.erase("FLAG_2");
    auto single = Open(nodes);
    Require(static_cast<bool>(single), "the single-flag store did not open");
    const auto& single_image = single.value().ReadArrayMetadata("SKY");
    Require(static_cast<bool>(single_image), "SKY was not readable in the single-flag store");
    std::vector<carta::zarr::Diagnostic> single_diagnostics;
    auto only = DetermineFlag(single.value(), single_image.value(), "SKY", single_diagnostics);
    Require(static_cast<bool>(only) && only.value() == "FLAG_1", "one matching flag was not selected");
    Require(single_diagnostics.empty(), "one matching flag produced an ambiguity diagnostic");
}

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


// How large a dataset is, when the transport cannot say how much room it takes.
//
// The measured half of this answer used to be the facade's: it re-parsed the location string and
// walked the directory itself, so the fallback could only be reached from here by pointing at a real
// store and setting the timeout to zero to make the walk give up. An in-memory transport gives up
// honestly instead -- it holds no bytes -- so the branch is reachable with a map.
void TestSizeFallsBackWhenTheStoreCannotBeMeasured() {
    auto store = Open(CompleteStore());
    Require(static_cast<bool>(store), "OpenStore rejected the in-memory store");

    const auto size = carta::zarr::internal::DatasetSizeBytes(store.value(),
                                                              std::chrono::steady_clock::now() +
                                                                  std::chrono::seconds(5));
    Require(static_cast<bool>(size), "sizing an in-memory store failed");
    Require(size.value().basis == carta::zarr::SizeBasis::declared,
            "a size the transport could not measure must be reported as the declared one");
    // SKY is 120 float32 at 480 bytes; time, frequency, l and m are 1, 3, 4 and 5 float64 at 8, 24,
    // 32 and 40; polarization is two four-byte labels at 8. The arrays, not the store.
    Require(size.value().bytes == 592,
            "the logical total was " + std::to_string(size.value().bytes) + ", not 592");

    // A deadline that has already passed reaches the same answer by the same route: this transport
    // refuses whatever the clock says, and every way of failing to measure means the declared size.
    const auto expired = carta::zarr::internal::DatasetSizeBytes(
        store.value(), std::chrono::steady_clock::now() - std::chrono::seconds(1));
    Require(expired && expired.value().basis == carta::zarr::SizeBasis::declared &&
                expired.value().bytes == 592,
            "an expired deadline did not reach the same declared size");
}

// And the fallback does not turn every failure into a number: a store with nothing to add up is
// still an error, not a zero.
void TestSizeRefusesAStoreWithNoArrays() {
    auto store = Open({{"", RootGroup()}});
    Require(static_cast<bool>(store), "OpenStore rejected a bare root group");

    const auto size = carta::zarr::internal::DatasetSizeBytes(store.value(),
                                                              std::chrono::steady_clock::now());
    Require(!size && size.error().code == ErrorCode::invalid_metadata,
            "a store holding no arrays was given a size");
}

// Every rule that turns metadata and coordinate values into a descriptor, reached with the values
// handed in. This build reads no values, so while describing an image read its own, everything
// below was reachable only from a store written to disk -- and the image_type fallback and the time
// axis's scale and format were not reached at all.
void TestADescriptionIsBuiltFromTheValuesItIsGiven() {
    using carta::zarr::internal::xradio::CoordinateValues;
    using carta::zarr::internal::xradio::DescribeImageFrom;

    auto nodes = CompleteStore();
    nodes["frequency"] = NumericArray("[3]", R"(["frequency"])", "float64",
                                      R"({"units":"Hz","reference_frequency":{"data":1.402e9,"attrs":{"observer":"lsrk"}},
                                          "rest_frequency":{"data":1.420405751e9}})");
    nodes["time"] = NumericArray("[1]", R"(["time"])", "float64", R"({"units":"s","scale":"utc","format":"unix"})");
    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the in-memory store failed to open");

    CoordinateValues values;
    values.l = {-0.001, 0.0, 0.001, 0.002};
    values.m = {-0.002, -0.001, 0.0, 0.001, 0.002};
    values.frequency = {1.4e9, 1.401e9, 1.402e9};
    values.time = {1.6e9};
    values.polarization = std::vector<std::string>{"I", "Q"};

    const auto described = DescribeImageFrom(store.value(), "SKY", values);
    Require(static_cast<bool>(described),
            "an image with its values in hand should be described: " + (described ? "" : described.error().message));
    const auto& descriptor = described.value().descriptor;

    Require(descriptor.direction && descriptor.direction->projection == "SIN", "the direction comes from the root");
    Require(descriptor.spectral.has_value(), "three channels make a spectral coordinate");
    const auto& spectral = *descriptor.spectral;
    Require(spectral.channel_frequencies == values.frequency, "the channels are the values handed in");
    Require(spectral.system == "LSRK", "the frame is the reference frequency's observer, uppercased");
    Require(spectral.reference_value && *spectral.reference_value == 1.402e9,
            "the reference is the frequency named, not the first channel");
    Require(spectral.reference_pixel && *spectral.reference_pixel == 3.0,
            "and its pixel is that channel's, counted from one");
    Require(spectral.rest_frequency && *spectral.rest_frequency == 1.420405751e9, "the rest frequency is read");

    Require(descriptor.polarization && descriptor.polarization->labels == *values.polarization,
            "the polarization labels are the ones handed in");
    Require(descriptor.temporal.has_value(), "a time value makes a temporal coordinate");
    Require(descriptor.temporal->values == values.time, "the times are the values handed in");
    Require(descriptor.temporal->scale == "UTC" && descriptor.temporal->format == "UNIX",
            "the time's scale and format come from its attributes, uppercased");
}

// XRADIO writes an image's role on its own "type" attribute. The v2 schema's "image_type" is read
// only when "type" says nothing -- and "flag" says nothing an image can be.
void TestARoleNotSpelledAsTypeFallsBackToImageType() {
    using carta::zarr::internal::xradio::CoordinateValues;
    using carta::zarr::internal::xradio::DescribeImageFrom;

    auto nodes = CompleteStore();
    nodes["SKY"] = SkyArray("float32", R"({"units":"Jy/beam","image_type":"residual"})");
    auto store = Open(nodes);
    Require(static_cast<bool>(store), "the in-memory store failed to open");

    const auto described = DescribeImageFrom(store.value(), "SKY", CoordinateValues{});
    Require(static_cast<bool>(described), "an image with no values handed in is still described");
    Require(described.value().descriptor.image_role == "residual",
            "with no type attribute the role is image_type's, got '" + described.value().descriptor.image_role + "'");
}

}  // namespace

int main() {
    try {
        TestCompleteStoreMatches();
        TestDatasetWithoutSkyMatches();
        TestNonMatch();
        TestNothingOpenableSaysWhy();
        TestMissingCoordinateIsInvalid();
        TestCoordinateShapeMismatchIsInvalid();
        TestCoordinateDataTypeIsChecked();
        TestMalformedCoordinateSystemIsInvalid();
        TestAMissingCoordinateSystemIsInvalid();
        TestFirstFaultIsTheOnlyDiagnostic();
        TestIncompleteImageIsNotMatch();
        TestDiscoveryClassifiesVariables();
        TestOpenableGate();
        TestAnImageDisagreeingWithACoordinateIsNotOpenable();
        TestNothingOpenableIsInvalidOnlyWhenSomethingIsMalformed();
        TestANestedGroupIsNotABrokenArray();
        TestARefusalCarriesTheVariablesOwnReason();
        TestOneRuleDecidesWhatDatasetIsOpenable();
        TestDefaultImageSkipsUnopenablePreferredImage();
        TestConsolidatedMetadataDiscovery();
        TestAConsolidatedKeyIsFiledUnderItsCanonicalName();
        TestAConsolidatedBlockThatMisnamesItsNodesIsRefused();
        TestTheInventorySaysWhatEachNodeIs();
        TestANodeThatWillNotParseIsDiagnosedNotRefused();
        TestADeclaredFlagIsBinding();
        TestADeclaredFlagThatCannotMaskClosesTheImageInTheListing();
        TestAnImageWithAnAxisBeyondTheFiveIsNotOpenable();
        TestAmbiguousFlagsSelectNone();
        TestStoreRejections();
        TestSizeFallsBackWhenTheStoreCannotBeMeasured();
        TestSizeRefusesAStoreWithNoArrays();
        TestADescriptionIsBuiltFromTheValuesItIsGiven();
        TestARoleNotSpelledAsTypeFallsBackToImageType();
        std::cout << "carta-zarr schema profile tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carta-zarr schema profile tests failed: " << error.what() << '\n';
        return 1;
    }
}
