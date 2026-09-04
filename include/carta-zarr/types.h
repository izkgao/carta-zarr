/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_TYPES_H_
#define CARTA_ZARR_TYPES_H_

#include "carta-zarr/error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace carta::zarr {

struct Diagnostic {
    std::string code;
    std::string message;
    std::string node_path;
};

enum class ProbeKind {
    not_zarr,
    zarr_without_supported_schema,
    supported_dataset,
    invalid_dataset,
};

using SchemaId = std::string;

inline constexpr std::string_view kXradioImageSchema = "xradio.image";

enum class SchemaMatchKind {
    no_match,
    match,
    invalid,
};

struct SchemaProbeResult {
    SchemaMatchKind kind = SchemaMatchKind::no_match;
    SchemaId schema_id;
    std::string schema_version;
    std::vector<Diagnostic> diagnostics;
};

struct ProbeOptions {};

struct ProbeResult {
    ProbeKind kind = ProbeKind::not_zarr;
    SchemaId schema_id;
    std::string schema_version;
    std::vector<std::string> image_ids;
    std::vector<Diagnostic> diagnostics;
};

struct OpenOptions {
    std::size_t cache_bytes = 0;
    unsigned int io_threads = 0;
    unsigned int decode_threads = 0;
};

enum class AxisRole {
    spatial_x,
    spatial_y,
    spectral,
    polarization,
    time,
    other,
};

struct AxisDescriptor {
    std::string name;
    AxisRole role = AxisRole::other;
    std::uint64_t length = 0;
    std::string unit;
    std::size_t storage_index = 0;
};

enum class DataType {
    unknown,
    boolean,
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float16,
    float32,
    float64,
    complex64,
    complex128,
};

struct DirectionCoordinate {
    std::string projection;
    std::string reference_frame;  // e.g. "FK5", "ICRS", "GALACTIC"
    std::optional<double> equinox;
    std::array<double, 2> reference_pixel{
        0.0, 0.0};  // CRPIX (0-indexed or 1-indexed convention noted, standard FITS CRPIX is stored)
    std::array<double, 2> reference_value{0.0, 0.0};                                       // CRVAL in degrees
    std::array<double, 2> increment{0.0, 0.0};                                             // CDELT in degrees
    std::array<std::array<double, 2>, 2> transformation_matrix{{{1.0, 0.0}, {0.0, 1.0}}};  // PC matrix
    std::vector<double> projection_parameters;
    std::array<double, 2> native_pole_direction{0.0, 0.0};  // longPole/latPole in degrees
};

struct SpectralCoordinate {
    std::string unit;
    std::string system;                     // SPECSYS, e.g. "LSRK", "BARY", "TOPOCENT"
    std::optional<double> reference_pixel;  // CRPIX3
    std::optional<double> reference_value;  // CRVAL3
    std::optional<double> increment;        // CDELT3
    std::optional<double> rest_frequency;
    std::vector<double> channel_frequencies;
};

struct TemporalCoordinate {
    std::vector<double> values;  // XRADIO unix seconds
    std::string unit;
    std::string scale;
    std::string format;
};

struct PolarizationCoordinate {
    std::vector<std::string> labels;
};

struct ObservationInfo {
    std::string object_name;
    std::string observer;
    std::string telescope_name;
    std::string timesys;
    std::string date_obs;
    std::optional<double> mjd_obs;
    std::optional<std::array<double, 3>> observatory_position;  // OBSGEO-X, Y, Z (meters)
};

struct StorageLayout {
    std::vector<std::uint64_t> chunk_shape;
    std::vector<std::uint64_t> shard_shape;
    std::string compressor;
    bool sharded = false;
};

struct Beam {
    std::size_t channel = 0;
    std::size_t polarization = 0;
    double major = 0.0;
    double minor = 0.0;
    double position_angle = 0.0;
    std::string unit;
};

struct DatasetDescriptor {
    SchemaId schema_id;
    std::string schema_version;
    std::vector<std::string> image_ids;
    std::vector<Diagnostic> diagnostics;
};

struct ImageDescriptor {
    std::string id;
    std::string image_role;
    std::vector<std::string> data_groups;
    DataType stored_type = DataType::unknown;
    std::vector<AxisDescriptor> axes;
    std::string unit;
    bool has_pixel_mask = false;
    std::optional<DirectionCoordinate> direction;
    std::optional<SpectralCoordinate> spectral;
    std::optional<PolarizationCoordinate> polarization;
    std::optional<TemporalCoordinate> temporal;
    std::optional<ObservationInfo> observation;
    std::optional<StorageLayout> storage;
    std::vector<Diagnostic> diagnostics;
};

struct Range {
    std::uint64_t start = 0;
    std::uint64_t count = 0;
    std::uint64_t stride = 1;
};

struct ReadRequest {
    std::vector<Range> axes;
    DataType output_type = DataType::float32;
};

struct MutableBufferView {
    void* data = nullptr;
    std::size_t byte_size = 0;
};

}  // namespace carta::zarr

#endif  // CARTA_ZARR_TYPES_H_
