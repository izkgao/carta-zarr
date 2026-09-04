/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_ZARR_ARRAY_METADATA_H_
#define CARTA_ZARR_SRC_ZARR_ARRAY_METADATA_H_

#include "carta-zarr/carta_zarr.h"

#include <nlohmann/json.hpp>

#include <optional>

namespace carta::zarr::internal::zarr {

struct ArrayMetadata {
    std::vector<std::uint64_t> shape;
    std::vector<std::string> dimension_names;
    std::string data_type;
    nlohmann::json data_type_configuration;
    nlohmann::json attributes;
    std::vector<std::uint64_t> chunk_shape;
};

Result<ArrayMetadata> ParseArrayMetadata(const nlohmann::json& metadata, std::string_view node);

bool IsNonNegativeInteger(const nlohmann::json& value);
bool IsPositiveInteger(const nlohmann::json& value);
bool IsNumericVector(const nlohmann::json& value, std::size_t length);
bool IsNumericMatrix(const nlohmann::json& value, std::size_t rows, std::size_t columns);
bool IsRealDataType(std::string_view data_type);
DataType ParseDataType(std::string_view data_type);
bool IsFixedLengthUtf32(const ArrayMetadata& metadata);
std::optional<std::size_t> FindDimensionIndex(const ArrayMetadata& metadata, std::string_view name);
const char* ErrorCodeName(ErrorCode code) noexcept;

} // namespace carta::zarr::internal::zarr

#endif // CARTA_ZARR_SRC_ZARR_ARRAY_METADATA_H_
