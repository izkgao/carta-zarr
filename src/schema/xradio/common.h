/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_SCHEMA_XRADIO_COMMON_H_
#define CARTA_ZARR_SRC_SCHEMA_XRADIO_COMMON_H_

#include "../../store.h"
#include "../../zarr/array_metadata.h"

namespace carta::zarr::internal::xradio {

void AddDiagnostic(SchemaProbeResult& result, std::string code, std::string message, std::string node_path = {});

Error MakeError(ErrorCode code, std::string message, std::string node_path = {});

bool ValidateCoordinateSystem(const nlohmann::json& root_attributes, SchemaProbeResult& result);

bool AddArrayMetadataError(const Result<::carta::zarr::internal::zarr::ArrayMetadata>& metadata_result,
                           SchemaProbeResult& result, std::string_view node);

Result<bool> ValidateCoordinate(const Store& store, std::string_view name, std::uint64_t expected_length,
                                bool string_coordinate, std::string_view profile_name, SchemaProbeResult& result);

std::string DiagnosticMessage(const SchemaProbeResult& result, std::string_view profile_name);

} // namespace carta::zarr::internal::xradio

#endif // CARTA_ZARR_SRC_SCHEMA_XRADIO_COMMON_H_
