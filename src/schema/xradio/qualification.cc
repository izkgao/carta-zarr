/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "qualification.h"

#include "../../zarr/array_metadata.h"
#include "attributes.h"
#include "flag.h"

#include <algorithm>
#include <cstddef>
#include <string>

namespace carta::zarr::internal::xradio {
namespace {

namespace zarr_metadata = ::carta::zarr::internal::zarr;

// The Fourier-conjugate plane. Variables on it are images in every other respect, which is why they
// are listed with a reason rather than passed over in silence.
constexpr std::array<std::string_view, 5> kApertureAxes{"time", "frequency", "polarization", "u", "v"};

bool HasAllAxes(const zarr_metadata::ArrayMetadata& metadata, const std::array<std::string_view, 5>& axes) {
    return std::all_of(axes.begin(), axes.end(), [&metadata](const auto axis) {
        return zarr_metadata::FindDimensionIndex(metadata, axis).has_value();
    });
}

// A dataset stores each coordinate once and every image references it by dimension name, so an
// image whose own extent disagrees with the coordinate it names cannot be described with it -- it
// would be reported with another image's coordinate vector, three channels' worth over seven
// channels of pixels.
//
// A coordinate the dataset does not carry at all is the probe's business rather than this one's:
// every image references it, so its absence closes the dataset instead of one variable.
std::optional<Diagnostic> DisagreementWithCoordinates(const Store& store,
                                                      const zarr_metadata::ArrayMetadata& image,
                                                      std::string_view node) {
    const auto rank = std::min(image.dimension_names.size(), image.shape.size());
    for (std::size_t axis = 0; axis < rank; ++axis) {
        const auto& name = image.dimension_names.at(axis);
        const auto& coordinate = store.ReadArrayMetadata(name);
        if (!coordinate) {
            continue;
        }
        if (coordinate.value().shape.size() != 1 || coordinate.value().shape.front() != image.shape.at(axis)) {
            return Diagnostic{DiagnosticCode::invalid_metadata,
                              "Image dimension '" + name + "' is not the length of the coordinate of that name",
                              std::string(node)};
        }
    }
    return std::nullopt;
}

// Why the flag an image declares cannot mask it, or nothing when it declares none or one that can.
// What DetermineFlag refuses an image over, asked of the listing so that the two agree; with nothing
// declared there is no refusal to agree with, because an image with no usable candidate opens
// unmasked.
std::optional<Diagnostic> UnusableDeclaredFlag(const Store& store, const zarr_metadata::ArrayMetadata& image,
                                               std::string_view node) {
    const auto declared = AttributeString(image.attributes, "flag");
    if (declared.empty()) {
        return std::nullopt;
    }
    const auto refused = [&](const Error& error) {
        return Diagnostic{DiagnosticCode::invalid_metadata,
                          "Declared flag '" + declared + "' cannot mask this image: " + error.message,
                          std::string(node)};
    };
    const auto& flag = store.ReadArrayMetadata(declared);
    if (!flag) {
        return refused(flag.error());
    }
    if (auto usable = RequireUsableFlag(flag.value(), image, declared); !usable) {
        return refused(usable.error());
    }
    return std::nullopt;
}

}  // namespace

NodeQualification QualifyNode(const Store& store, const NodeEntry& entry) {
    const std::string& node = entry.name;
    // A group is a node in the hierarchy like any other, and nothing is wrong with it. Passed over,
    // the way everything else that is not an image is passed over.
    if (entry.kind == NodeKind::group) {
        return NodeQualification{};
    }

    // Said rather than passed over, because this is where a node leaves the listing: nothing here
    // knows whether it was an image, a coordinate or something else in the directory, so a store
    // whose sky variable lands here reports no images at all. Without this, that reads as a store
    // this profile does not recognise, with nothing to say which node went missing.
    //
    // Not a refusal. A malformed node elsewhere in a store does not stop the images that parsed from
    // opening.
    //
    // Two codes, because they are two different things to say. unreadable_array is a node that said it
    // was an array and whose metadata will not parse as one; unrecognised_node is a node that did not
    // say what it was -- a document that would not parse, or a node_type Zarr does not define -- and
    // calling that an array would be a guess.
    if (entry.kind == NodeKind::unrecognised) {
        return NodeQualification{false, false,
                                 Diagnostic{DiagnosticCode::unrecognised_node, entry.reason->message, node}, true};
    }
    const auto& metadata = *entry.array;
    if (!metadata) {
        return NodeQualification{false, false,
                                 Diagnostic{DiagnosticCode::unreadable_array, metadata.error().message, node}, true};
    }

    const auto& array = metadata.value();
    if (IsFlag(array)) {
        return NodeQualification{};
    }

    if (HasAllAxes(array, kSkyAxes)) {
        if (!zarr_metadata::IsRealDataType(array.data_type)) {
            return NodeQualification{true, false,
                                     Diagnostic{DiagnosticCode::unsupported_data_type,
                                                "Complex sky-plane variables are not openable", std::string(node)},
                                     false};
        }
        // The five axes are the whole of what an image is described by, so a sixth would be read with
        // a selection one rank short -- every read refused, from an image listed as openable.
        if (array.dimension_names.size() != kSkyAxes.size()) {
            return NodeQualification{true, false,
                                     Diagnostic{DiagnosticCode::invalid_metadata,
                                                "Sky-plane variable has dimensions beyond time, frequency, "
                                                "polarization, l and m",
                                                std::string(node)},
                                     true};
        }
        if (auto disagreement = DisagreementWithCoordinates(store, array, node); disagreement) {
            return NodeQualification{true, false, std::move(disagreement), true};
        }
        // A declared flag binds the image to it -- DetermineFlag closes one whose flag cannot mask it --
        // so the listing has to say so too, or it offers an image that will not open.
        if (auto unusable = UnusableDeclaredFlag(store, array, node); unusable) {
            return NodeQualification{true, false, std::move(unusable), true};
        }
        return NodeQualification{true, true, std::nullopt, false};
    }

    if (HasAllAxes(array, kApertureAxes)) {
        return NodeQualification{true, false,
                                 Diagnostic{DiagnosticCode::unsupported_coordinate_plane,
                                            "Aperture-plane variables are not openable", std::string(node)},
                                 false};
    }

    // A coordinate array, a normalization variable, a beam table: not an image, and nothing to say.
    return NodeQualification{};
}

Result<void> RequireQualified(const Store& store, std::string_view image_id) {
    // Against the store's listing rather than against whatever is on disk under that name. A Store
    // is a read-only view, and its listing is the snapshot a dataset's images were enumerated from;
    // a node written beside them afterwards is not one of them, and reading it by name would reach
    // it. What a listing offers and what will open are the same set in both directions.
    const auto& inventory = store.Inventory();
    if (!inventory) {
        return inventory.error();
    }
    const auto* entry = store.FindNode(image_id);
    if (entry == nullptr) {
        return Error{ErrorCode::not_found, "Image variable was not found", std::string(image_id)};
    }

    const auto qualified = QualifyNode(store, *entry);
    if (qualified.openable) {
        return {};
    }
    if (qualified.listed) {
        // The code comes from the refusal rather than from the asking. A store that is wrong about
        // an image and a library that does not open one are different answers, and a consumer that
        // shows one to whoever picked the file acts on the difference.
        return Error{qualified.malformed ? ErrorCode::invalid_metadata : ErrorCode::unsupported_data_type,
                     qualified.diagnostic->message, std::string(image_id)};
    }
    if (qualified.diagnostic) {
        // The node is there and will not parse. It says what is wrong with it rather than being
        // reported as a name the dataset does not have.
        return entry->kind == NodeKind::array ? entry->array->error() : *entry->reason;
    }

    // Not an image, and nothing wrong with it: the flag beside one, the coordinate under it, a beam
    // table, a group. A caller that named one asked for an image this dataset does not have.
    return Error{ErrorCode::not_found, "Image variable was not found", std::string(image_id)};
}

}  // namespace carta::zarr::internal::xradio
