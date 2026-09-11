/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_TYPES_H_
#define CARTA_ZARR_TYPES_H_

#include "carta-zarr/error.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
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

struct ImageEntry {
    std::string id;
    bool readable = false;
    std::vector<Diagnostic> diagnostics;
};

struct ProbeOptions {};

struct ProbeResult {
    ProbeKind kind = ProbeKind::not_zarr;
    SchemaId schema_id;
    std::string schema_version;
    std::vector<ImageEntry> images;
    std::optional<std::string> default_image_id;
    std::vector<Diagnostic> diagnostics;
};

struct OpenOptions {
    std::size_t cache_bytes = 0;
    unsigned int io_threads = 0;
    unsigned int decode_threads = 0;
    bool disable_cache = false;
};

enum class AxisRole {
    spatial_x,
    spatial_y,
    spectral,
    polarization,
    time,
    other,
};

inline constexpr std::array<AxisRole, 5> kXradioImageAxisOrder{
    AxisRole::spatial_x, AxisRole::spatial_y, AxisRole::spectral, AxisRole::polarization, AxisRole::time};

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

// The read geometry of one image, reported in the logical axis order of ImageDescriptor::axes so
// that a consumer never has to undo the stored order itself.
//
// Two granularities, deliberately separate: an inner chunk is what must be decoded to reach any
// byte inside it, while a shard is what one I/O request fetches. They are equal when the array is
// not sharded, and can differ by a large factor when it is, so a consumer sizing a cache reasons
// about chunk_shape and one predicting request count reasons about shard_shape.
struct ChunkGeometry {
    std::vector<std::uint64_t> chunk_shape;
    std::vector<std::uint64_t> shard_shape;
    // Number of inner chunks along each axis.
    std::vector<std::uint64_t> grid_shape;
    bool sharded = false;
    // True when the logical order differs from the stored order, so every read carries a transpose.
    bool transpose_required = false;
    std::string compressor;
};

struct Beam {
    // The plane this beam was fitted on. Every plane is reported; a consumer that handles one time
    // step selects it rather than being handed it.
    std::size_t time = 0;
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
    std::vector<ImageEntry> images;
    std::optional<std::string> default_image_id;
    std::vector<Diagnostic> diagnostics;
};

struct DatasetSize {
    // The size of the on-disk store when it could be enumerated quickly, or the total logical
    // bytes represented by all arrays when the directory scan timed out.
    std::uint64_t bytes = 0;
    bool is_upper_bound = false;
};

struct ImageDescriptor {
    std::string id;
    std::string image_role;
    std::vector<std::string> data_groups;
    DataType stored_type = DataType::unknown;
    // XRADIO images report axes in kXradioImageAxisOrder; storage_index identifies each stored dimension.
    std::vector<AxisDescriptor> axes;
    std::string unit;
    bool has_pixel_mask = false;
    // The flag variable supplying this image's pixel mask, empty when it has none. Reported for the
    // same reason `id` is: it names a data variable the consumer may want to see in diagnostics.
    std::string pixel_mask_id;
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
    // One range per ImageDescriptor::axes entry, in the same order.
    std::vector<Range> axes;
    DataType output_type = DataType::float32;
};

struct ReadOptions {
    // Write NaN wherever the pixel mask is false, so that one call answers what would otherwise be
    // a pixel read plus a mask read. On by default: masking during the read costs one pass over
    // data already in hand, while a caller doing it afterwards pays for a second traversal.
    bool apply_pixel_mask = true;
    // Cooperative cancellation checked before and after each storage operation. The callback
    // must be safe to invoke from the calling thread.
    std::function<bool()> cancellation_requested;
    // A steady-clock deadline checked at the same storage-operation boundaries. An in-flight
    // TensorStore operation is not interrupted, but a request never starts another operation once
    // this deadline has passed.
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    // Called as the read advances, with the number of destination elements that are final and the
    // number the request will produce in total. Returning false cancels the read, which then
    // reports cancelled.
    //
    // Supplying this changes how the read is issued: it is split into chunk-aligned pieces along
    // the slowest-varying selected axis, so that there is somewhere to report from and somewhere to
    // stop. The destination is dense in logical order with axis 0 fastest, which is what makes the
    // finished part a prefix rather than a scatter -- a caller can render or forward it as it
    // arrives. Leave it unset and the read is issued exactly as it was before, in one piece.
    //
    // A read that nothing interrupts is not made slower by this: the pieces are sized to hold
    // enough chunks to decode in parallel, and at that size a split read measures the same as an
    // unsplit one.
    std::function<bool(std::size_t elements_written, std::size_t elements_total)> progress;
    // Maximum temporary memory one piece of the read may use. Zero means the library's own budget.
    //
    // This bounds the pixel mask buffer, and it is also what a progressive read sizes its pieces
    // by -- both are "how much this read may hold at once", and splitting to fit is a better answer
    // than refusing. A read that cannot be split still reports buffer_too_small rather than
    // allocating past the limit.
    std::size_t temporary_memory_limit_bytes = 0;
};

struct MutableBufferView {
    void* data = nullptr;
    std::size_t byte_size = 0;
};

// One statistic a spectral reduction can produce. The enumerators are bit flags so that a request
// names a set in one field.
//
// There is deliberately no output-type option here. Every one of these is an N:1 reduction over as
// many as 5 x 10^11 values, accumulation is always double, and a float32 sum at that scale silently
// stops adding: offering the choice would only let a caller ask for a wrong answer quickly.
//
// num_pixels and nan_count partition the region: num_pixels counts the pixels that sum, sum_sq, min
// and max were taken over -- finite, and not rejected by the image's pixel mask -- while nan_count
// counts every other pixel the mask selected. Their total is the number of pixels the mask
// selected, so a caller needing the denominator of a mean asks for num_pixels rather than deriving
// it from the mask.
enum class Statistic : std::uint32_t {
    num_pixels = 1u << 0,
    nan_count = 1u << 1,
    sum = 1u << 2,
    sum_sq = 1u << 3,
    min = 1u << 4,
    max = 1u << 5,
};

using StatisticSet = std::uint32_t;

inline constexpr StatisticSet operator|(Statistic a, Statistic b) noexcept {
    return static_cast<StatisticSet>(a) | static_cast<StatisticSet>(b);
}
inline constexpr StatisticSet operator|(StatisticSet a, Statistic b) noexcept {
    return a | static_cast<StatisticSet>(b);
}
inline constexpr bool Contains(StatisticSet set, Statistic statistic) noexcept {
    return (set & static_cast<StatisticSet>(statistic)) != 0;
}

// Every statistic, in the order a SpectralBlock lays them out.
inline constexpr std::array<Statistic, 6> kStatisticOrder{Statistic::num_pixels, Statistic::nan_count,
                                                          Statistic::sum, Statistic::sum_sq,
                                                          Statistic::min, Statistic::max};

// A 2D (x, y) mask in logical image coordinates, addressed row-major with x fastest.
//
// This is a borrowed view: the pointer must stay valid until ReduceSpectral returns, and the
// library never retains it. The shape is the region's bounding box rather than the image, which is
// what makes handing over tens of thousands of regions at once affordable -- 5,792 PV boxes on a
// 4096^2 image describe themselves in 2.5 MB.
//
// A null mask selects the whole bounding box. That branch is required, not a convenience: an
// unrotated rectangle reaches a caller as a box with no raster mask at all, and it is the most
// common region shape there is.
struct RegionMask {
    std::uint64_t x_start = 0;
    std::uint64_t y_start = 0;
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    const std::uint8_t* mask = nullptr;
};

// The largest number of regions one reduction accepts.
//
// This is a structural guard, not a capacity estimate: the largest legitimate request is one box
// per pixel along an image diagonal, which is 46,341 for a 32768^2 image. The bound exists so that
// a caller passing an uninitialised count gets invalid_argument instead of a 16 GB allocation.
inline constexpr std::size_t kMaxSpectralRegions = 1u << 20;

// The memory one emitted block may occupy. emit_every_channels is reduced to fit it; see
// SpectralReduceRequest::emit_every_channels.
inline constexpr std::size_t kSpectralEmitBudgetBytes = 64u << 20;

struct SpectralReduceRequest {
    // The channels to reduce, over the image's spectral axis. Strides are honoured.
    Range spectral;
    std::uint64_t polarization = 0;
    std::uint64_t time = 0;
    // The regions, all reduced in a single pass over the pixels.
    const RegionMask* regions = nullptr;
    std::size_t region_count = 0;
    StatisticSet statistics = 0;
    // How often to hand results back, as a hint rather than a contract. Zero asks for one block at
    // the end. The library lowers it to fit kSpectralEmitBudgetBytes and then to a whole number of
    // spectral chunks, because a block boundary inside a chunk would split one decode's results
    // across two blocks. The value actually used is reported as SpectralBlock::channel_count, which
    // a caller has to read anyway.
    std::uint32_t emit_every_channels = 0;
};

// One contiguous run of channels, for every region and every requested statistic.
//
// values is laid out [region][statistic][channel] and is owned by the library; it is valid only for
// the duration of the call. statistics lists the statistics in the order they appear, so a caller
// never has to reconstruct the slot order from the requested set.
//
// min and max are reported as NaN for a channel whose region contributed no finite pixel, since
// there is no such thing as the smallest value of nothing. sum and sum_sq are zero in that case,
// which is what they are worth, and num_pixels is zero -- so a caller deriving a mean sees the
// division it must not perform.
struct SpectralBlock {
    // Index into the request's spectral selection, not an image channel: the image channel is
    // spectral.start + (first_channel + i) * spectral.stride.
    std::uint64_t first_channel = 0;
    std::uint64_t channel_count = 0;
    const double* values = nullptr;
    std::size_t value_count = 0;
    std::size_t region_stride = 0;
    std::size_t statistic_stride = 0;
    const Statistic* statistics = nullptr;
    std::size_t statistic_count = 0;
};

// Called once per block, on the thread that called ReduceSpectral. Returning false cancels the
// reduction, which then reports cancelled.
using SpectralSink = std::function<bool(const SpectralBlock&)>;

}  // namespace carta::zarr

#endif  // CARTA_ZARR_TYPES_H_
