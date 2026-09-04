/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_ERROR_H_
#define CARTA_ZARR_ERROR_H_

#include <string>

namespace carta::zarr {

enum class ErrorCode {
    not_found,
    not_zarr,
    unsupported_transport,
    unsupported_zarr_version,
    unsupported_schema,
    unsupported_schema_version,
    ambiguous_schema,
    invalid_argument,
    invalid_metadata,
    unsupported_data_type,
    unsupported_codec,
    invalid_slice,
    buffer_too_small,
    io_error,
    decode_error,
    cancelled,
    not_implemented,
};

struct Error {
    ErrorCode code;
    std::string message;
    std::string node_path;
};

} // namespace carta::zarr

#endif // CARTA_ZARR_ERROR_H_
