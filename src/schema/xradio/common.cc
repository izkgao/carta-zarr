/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "common.h"

#include <utility>

namespace carta::zarr::internal::xradio {
namespace {

namespace zarr_metadata = ::carta::zarr::internal::zarr;

} // namespace

void AddDiagnostic(SchemaProbeResult& result, std::string code, std::string message, std::string node_path) {
    result.diagnostics.push_back(Diagnostic{std::move(code), std::move(message), std::move(node_path)});
}

Error MakeError(ErrorCode code, std::string message, std::string node_path) {
    return Error{code, std::move(message), std::move(node_path)};
}

bool ValidateCoordinateSystem(const nlohmann::json& root_attributes, SchemaProbeResult& result) {
    const std::string node = "/attributes/coordinate_system_info";
    if (!root_attributes.is_object() || !root_attributes.contains("coordinate_system_info") ||
        !root_attributes.at("coordinate_system_info").is_object()) {
        AddDiagnostic(result, "invalid_metadata", "XRADIO requires coordinate_system_info metadata", node);
        return false;
    }
    const auto& coordinate = root_attributes.at("coordinate_system_info");
    if (!coordinate.contains("projection") || !coordinate.at("projection").is_string() ||
        coordinate.at("projection").get<std::string>().empty()) {
        AddDiagnostic(result, "invalid_metadata", "coordinate_system_info requires a projection", node);
        return false;
    }
    if (!coordinate.contains("reference_direction") || !coordinate.at("reference_direction").is_object() ||
        !coordinate.at("reference_direction").contains("data") ||
        !zarr_metadata::IsNumericVector(coordinate.at("reference_direction").at("data"), 2)) {
        AddDiagnostic(result, "invalid_metadata", "coordinate_system_info requires a two-value reference direction",
                      node);
        return false;
    }
    if (!coordinate.contains("native_pole_direction") || !coordinate.at("native_pole_direction").is_object() ||
        !coordinate.at("native_pole_direction").contains("data") ||
        !zarr_metadata::IsNumericVector(coordinate.at("native_pole_direction").at("data"), 2)) {
        AddDiagnostic(result, "invalid_metadata", "coordinate_system_info requires a two-value native pole direction",
                      node);
        return false;
    }
    if (!coordinate.contains("pixel_coordinate_transformation_matrix") ||
        !zarr_metadata::IsNumericMatrix(coordinate.at("pixel_coordinate_transformation_matrix"), 2, 2)) {
        AddDiagnostic(result, "invalid_metadata", "coordinate_system_info requires a 2x2 pixel transformation matrix",
                      node);
        return false;
    }
    return true;
}

bool AddArrayMetadataError(const Result<::carta::zarr::internal::zarr::ArrayMetadata>& metadata_result,
                           SchemaProbeResult& result, std::string_view node) {
    if (metadata_result) {
        return false;
    }
    AddDiagnostic(result, zarr_metadata::ErrorCodeName(metadata_result.error().code), metadata_result.error().message,
                  std::string(node));
    return true;
}

Result<bool> ValidateCoordinate(const Store& store, std::string_view name, std::uint64_t expected_length,
                                bool string_coordinate, std::string_view profile_name, SchemaProbeResult& result) {
    const std::string node(name);
    auto metadata_result = store.ReadNodeMetadata(name);
    if (!metadata_result) {
        if (metadata_result.error().code == ErrorCode::not_found) {
            AddDiagnostic(result, "invalid_metadata", "Missing required coordinate array", node);
            return false;
        }
        return metadata_result.error();
    }

    auto array_result = zarr_metadata::ParseArrayMetadata(metadata_result.value(), name);
    if (AddArrayMetadataError(array_result, result, node)) {
        return false;
    }
    const auto& metadata = array_result.value();
    if (metadata.shape.size() != 1 || metadata.shape.front() != expected_length ||
        metadata.dimension_names.size() != 1 || metadata.dimension_names.front() != name) {
        AddDiagnostic(result, "invalid_metadata",
                      "Coordinate shape or dimension name does not match " + std::string(profile_name), node);
        return false;
    }
    if (string_coordinate ? !zarr_metadata::IsFixedLengthUtf32(metadata)
                          : !zarr_metadata::IsRealDataType(metadata.data_type)) {
        AddDiagnostic(result, "unsupported_data_type", "Coordinate array has an unsupported data type", node);
        return false;
    }
    return true;
}

std::string DiagnosticMessage(const SchemaProbeResult& result, std::string_view profile_name) {
    std::string message = "XRADIO " + std::string(profile_name) + " metadata is invalid";
    if (!result.diagnostics.empty()) {
        message += ": " + result.diagnostics.front().message;
    }
    return message;
}

} // namespace carta::zarr::internal::xradio
