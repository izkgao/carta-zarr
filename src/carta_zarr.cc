/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "carta-zarr/carta_zarr.h"

#include "chunk_blocks.h"
#include "reduce/plane_histogram.h"
#include "reduce/spectral_reduce.h"
#include "schema/profile.h"
#include "store.h"
#include "work_pool.h"
#include "zarr/array_metadata.h"
#include "zarr/pixel_reader.h"
#include "zarr/store_context.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>
#include <unordered_map>
#include <utility>

namespace carta::zarr {
namespace {

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

std::string SchemaErrorMessage(const SchemaProbeResult& result) {
    std::string message = "The requested schema did not match";
    if (!result.diagnostics.empty()) {
        message = result.diagnostics.front().message;
    }
    return message;
}

bool TryComputeDirectorySize(std::string_view location, std::chrono::milliseconds timeout, std::uint64_t& size) {
    const std::string location_string(location);
    std::filesystem::path root_path;
    if (location_string.rfind("file://", 0) == 0) {
        root_path = std::filesystem::path(location_string.substr(7));
    } else if (location_string.find("://") != std::string::npos) {
        return false;
    } else {
        root_path = std::filesystem::path(location_string);
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::uint64_t directory_size = 0;
    std::error_code error;
    std::filesystem::recursive_directory_iterator iterator(
        root_path, std::filesystem::directory_options::skip_permission_denied, error);
    if (error) {
        return false;
    }
    const std::filesystem::recursive_directory_iterator end;
    while (iterator != end) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }

        std::error_code entry_error;
        if (iterator->is_regular_file(entry_error)) {
            if (entry_error) {
                return false;
            }
            const auto file_size = iterator->file_size(entry_error);
            if (entry_error || file_size > std::numeric_limits<std::uint64_t>::max() - directory_size) {
                return false;
            }
            directory_size += file_size;
        } else if (entry_error) {
            return false;
        }

        iterator.increment(error);
        if (error) {
            return false;
        }
    }

    size = directory_size;
    return true;
}

}  // namespace

class Context::Impl {
public:
    Impl(OpenOptions options, internal::StoreContextPtr store_context)
        : options(options),
          store_context(std::move(store_context)),
          // decode_threads is the consumer's statement of how much of this machine the library may
          // use, so it sizes both pools rather than only TensorStore's. The two are busy at
          // different moments -- a slab is read and then visited -- so sizing each at the whole
          // budget does not double the demand. Zero means one worker per hardware thread, which is
          // what TensorStore's own default does with the same number.
          workers(std::make_shared<internal::WorkPool>(options.decode_threads)) {}

    OpenOptions options;
    // Shared by every dataset and image opened through this context, so that its cache and
    // concurrency limits apply to all reads rather than being rebuilt per read.
    internal::StoreContextPtr store_context;
    // The per-pixel work of a reduction. See WorkPool.
    std::shared_ptr<internal::WorkPool> workers;
};

Context::Context(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}
Context::~Context() = default;

Result<Context> Context::Create(const OpenOptions& options) {
    auto store_context = internal::MakeStoreContext(options);
    if (!store_context) {
        return store_context.error();
    }
    return Context{std::make_shared<Impl>(options, std::move(store_context.value()))};
}

class Image::Impl {
public:
    Impl(std::shared_ptr<Context::Impl> context, std::string location, std::string schema_id,
         std::shared_ptr<internal::Store> store, ImageDescriptor descriptor, ChunkGeometry geometry)
        : context(std::move(context)),
          location(std::move(location)),
          schema_id(std::move(schema_id)),
          store(std::move(store)),
          descriptor(std::move(descriptor)),
          geometry(std::move(geometry)) {}

    std::shared_ptr<Context::Impl> context;
    std::string location;
    std::string schema_id;
    std::shared_ptr<internal::Store> store;
    ImageDescriptor descriptor;
    ChunkGeometry geometry;
};

namespace {

// The axis a progressive read is split along: the slowest-varying one that selects more than a
// single element. The destination is dense in logical order with axis 0 fastest, so splitting there
// and nowhere else is what makes each finished piece extend a prefix instead of leaving holes.
std::optional<std::size_t> SlowestSelectedAxis(const ReadRequest& request) {
    for (std::size_t i = request.axes.size(); i-- > 0;) {
        if (request.axes.at(i).count > 1) {
            return i;
        }
    }
    return std::nullopt;
}

// How many elements of the split axis one piece should cover, so that the piece pulls roughly the
// budgeted amount of decompressed chunk data through. The other axes already contribute whatever
// they span, so a plane read needs far fewer rows per piece than a single-pixel column needs
// channels -- and an image with very large chunks gets pieces of one chunk rather than pieces it
// could never afford.
std::uint64_t ElementsPerPiece(const ImageDescriptor& descriptor, const ReadRequest& request,
                               const ChunkGeometry& geometry, std::size_t axis, std::size_t budget_bytes) {
    std::uint64_t other_chunks = 1;
    for (std::size_t i = 0; i < request.axes.size(); ++i) {
        if (i == axis) {
            continue;
        }
        const auto chunk = i < geometry.chunk_shape.size() ? geometry.chunk_shape.at(i) : 0;
        const auto& range = request.axes.at(i);
        other_chunks *= internal::ChunksSpanned(range.start, range.count, range.stride, chunk);
    }
    const auto row_bytes = internal::DecodedChunkBytes(descriptor, geometry) * other_chunks;
    // At least one chunk: a piece smaller than that would decode the same chunk twice.
    const auto chunks = std::max<std::uint64_t>(1, budget_bytes / std::max<std::uint64_t>(1, row_bytes));
    const auto chunk = axis < geometry.chunk_shape.size() ? geometry.chunk_shape.at(axis) : 0;
    const auto stride = std::max<std::uint64_t>(1, request.axes.at(axis).stride);
    // AlignedBlockEnd rounds this out to a whole chunk, so a low estimate costs nothing.
    return std::max<std::uint64_t>(1, (chunks * std::max<std::uint64_t>(1, chunk)) / stride);
}

ChunkGeometry BuildChunkGeometry(const ImageDescriptor& descriptor, const StorageLayout& layout) {
    ChunkGeometry geometry;
    geometry.sharded = layout.sharded;
    geometry.compressor = layout.compressor;

    const auto rank = descriptor.axes.size();
    geometry.chunk_shape.resize(rank);
    geometry.shard_shape.resize(rank);
    geometry.grid_shape.resize(rank);
    for (std::size_t logical = 0; logical < rank; ++logical) {
        const auto& axis = descriptor.axes.at(logical);
        const auto stored = axis.storage_index;
        const auto chunk =
            stored < layout.chunk_shape.size() ? layout.chunk_shape.at(stored) : axis.length;
        const auto shard =
            stored < layout.shard_shape.size() ? layout.shard_shape.at(stored) : chunk;
        geometry.chunk_shape.at(logical) = chunk;
        geometry.shard_shape.at(logical) = shard == 0 ? chunk : shard;
        geometry.grid_shape.at(logical) = chunk == 0 ? 0 : (axis.length + chunk - 1) / chunk;
        if (stored != logical) {
            geometry.transpose_required = true;
        }
    }

    // The last stored dimension varies fastest, so of the two spatial axes the one with the larger
    // storage index is the one a plane is contiguous along.
    std::size_t x_stored = 0;
    std::size_t y_stored = 0;
    for (const auto& axis : descriptor.axes) {
        if (axis.role == AxisRole::spatial_x) {
            x_stored = axis.storage_index;
        } else if (axis.role == AxisRole::spatial_y) {
            y_stored = axis.storage_index;
        }
    }
    geometry.fastest_spatial_axis = y_stored > x_stored ? AxisRole::spatial_y : AxisRole::spatial_x;
    return geometry;
}

}  // namespace

Image::Image(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}
Image::~Image() = default;

const ImageDescriptor& Image::descriptor() const noexcept {
    static const ImageDescriptor empty_descriptor;
    return _impl ? _impl->descriptor : empty_descriptor;
}

const ChunkGeometry& Image::chunk_geometry() const noexcept {
    static const ChunkGeometry empty_geometry;
    return _impl ? _impl->geometry : empty_geometry;
}

Result<std::size_t> Image::Read(const ReadRequest& request, MutableBufferView destination) const {
    return Read(request, destination, ReadOptions{});
}

Result<std::size_t> Image::Read(const ReadRequest& request, MutableBufferView destination,
                                const ReadOptions& options) const {
    if (!_impl) {
        return MakeError(ErrorCode::invalid_argument, "Image handle is empty");
    }
    if (request.output_type != DataType::float32) {
        return MakeError(ErrorCode::unsupported_data_type, "Only float32 output is implemented",
                         _impl->descriptor.id);
    }

    auto selection = internal::zarr::BuildSelection(_impl->descriptor, request);
    if (!selection) {
        return selection.error();
    }
    const auto elements = internal::zarr::SelectionElementCount(selection.value());
    if (elements == 0 || elements > destination.byte_size / sizeof(float)) {
        return MakeError(ErrorCode::invalid_argument, "Destination buffer is too small for the request",
                         _impl->descriptor.id);
    }

    // Do this before allocating a mask or starting any storage work. A cancelled request must not
    // consume temporary memory just to discover that it cannot proceed.
    auto control = internal::zarr::CheckReadControl(options, _impl->descriptor.id);
    if (!control) {
        return control.error();
    }

    const bool apply_mask = options.apply_pixel_mask && _impl->descriptor.has_pixel_mask;

    // One piece unless the caller asked to hear about progress, in which case the request is split
    // along its slowest-varying selected axis. Splitting anywhere else, or without aligning to the
    // chunk grid, would decode chunks twice and report a destination that is finished in patches
    // rather than as a prefix.
    const auto slab_axis = SlowestSelectedAxis(request);
    const bool progressive = static_cast<bool>(options.progress) && slab_axis.has_value();

    // Not splitting is the same loop with one piece covering everything, so there is one path to
    // read rather than two to keep in agreement.
    std::uint64_t slab_total = 1;
    std::uint64_t slab_stride = elements;
    std::uint64_t slab_step = 1;
    std::uint64_t slab_chunk = 0;
    if (progressive) {
        const auto axis = slab_axis.value();
        slab_total = request.axes.at(axis).count;
        slab_stride = 1;
        for (std::size_t i = 0; i < axis; ++i) {
            slab_stride *= request.axes.at(i).count;
        }
        slab_chunk = axis < _impl->geometry.chunk_shape.size() ? _impl->geometry.chunk_shape.at(axis) : 0;
        const auto budget =
            options.temporary_memory_limit_bytes != 0
                ? options.temporary_memory_limit_bytes
                : internal::DefaultReadBytes(internal::DecodedChunkBytes(_impl->descriptor, _impl->geometry));
        slab_step = ElementsPerPiece(_impl->descriptor, request, _impl->geometry, axis, budget);
    }

    auto* pixels = static_cast<float*>(destination.data);
    std::vector<std::uint8_t> mask;

    for (std::uint64_t begin = 0; begin < slab_total;) {
        const std::uint64_t end =
            progressive ? internal::AlignedBlockEnd(begin, slab_step, slab_total, request.axes.at(slab_axis.value()).start,
                                                    request.axes.at(slab_axis.value()).stride, slab_chunk)
                        : slab_total;

        ReadRequest piece = request;
        if (progressive) {
            auto& range = piece.axes.at(slab_axis.value());
            range.start = request.axes.at(slab_axis.value()).start + (begin * range.stride);
            range.count = end - begin;
        }
        auto piece_selection = internal::zarr::BuildSelection(_impl->descriptor, piece);
        if (!piece_selection) {
            return piece_selection.error();
        }
        const auto piece_elements = static_cast<std::size_t>((end - begin) * slab_stride);
        float* piece_pixels = pixels + (begin * slab_stride);

        if (apply_mask) {
            // The limit bounds a piece, and a read that is not split is one piece, so an
            // unsplittable request that exceeds it still has to say so rather than allocate.
            if (options.temporary_memory_limit_bytes != 0 && piece_elements > options.temporary_memory_limit_bytes) {
                return MakeError(ErrorCode::buffer_too_small,
                                 "Pixel mask temporary buffer exceeds the configured memory limit",
                                 _impl->descriptor.id);
            }
            mask.assign(piece_elements, 0);
            // The mask is read first so that an unavailable or cancelled mask cannot leave this
            // piece of the destination updated. TensorStore still owns the pixel operation's
            // in-flight completion before it returns, so the destination remains valid for the
            // next read.
            auto mask_read = _impl->store->ReadPixelMaskBytes(_impl->descriptor.pixel_mask_id,
                                                              piece_selection.value(), mask.data(), mask.size(), options);
            if (!mask_read) {
                return mask_read.error();
            }
        }
        auto read = _impl->store->ReadPixelsFloat32(_impl->descriptor.id, piece_selection.value(), piece_pixels,
                                                    piece_elements, options);
        if (!read) {
            return read.error();
        }
        if (apply_mask) {
            // XRADIO stores flags with true meaning a good pixel.
            for (std::size_t i = 0; i < piece_elements; ++i) {
                if (mask[i] == 0) {
                    piece_pixels[i] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }

        begin = end;
        if (options.progress && !options.progress(static_cast<std::size_t>(begin * slab_stride),
                                                  static_cast<std::size_t>(elements))) {
            return MakeError(ErrorCode::cancelled, "The read was cancelled by its progress callback",
                             _impl->descriptor.id);
        }
    }
    return static_cast<std::size_t>(elements);
}

Result<std::size_t> Image::ReadPixelMask(const ReadRequest& request, MutableBufferView destination) const {
    if (!_impl) {
        return MakeError(ErrorCode::invalid_argument, "Image handle is empty");
    }
    if (!_impl->descriptor.has_pixel_mask) {
        return MakeError(ErrorCode::not_found, "This image has no pixel mask", _impl->descriptor.id);
    }

    auto selection = internal::zarr::BuildSelection(_impl->descriptor, request);
    if (!selection) {
        return selection.error();
    }
    const auto elements = internal::zarr::SelectionElementCount(selection.value());
    if (elements == 0 || elements > destination.byte_size) {
        return MakeError(ErrorCode::invalid_argument, "Destination buffer is too small for the request",
                         _impl->descriptor.id);
    }

    auto read = _impl->store->ReadPixelMaskBytes(_impl->descriptor.pixel_mask_id, selection.value(),
                                                 static_cast<std::uint8_t*>(destination.data),
                                                 static_cast<std::size_t>(elements), ReadOptions{});
    if (!read) {
        return read.error();
    }
    return static_cast<std::size_t>(elements);
}

Result<void> Image::ReduceSpectral(const SpectralReduceRequest& request, const SpectralSink& sink) const {
    return ReduceSpectral(request, sink, ReadOptions{});
}

Result<void> Image::ReduceSpectral(const SpectralReduceRequest& request, const SpectralSink& sink,
                                   const ReadOptions& options) const {
    if (!_impl || !_impl->store) {
        return MakeError(ErrorCode::invalid_argument, "Image handle is empty");
    }
    return internal::ReduceSpectral(*_impl->store, _impl->descriptor, _impl->geometry, request, sink, options,
                                    *_impl->context->workers);
}

Result<void> Image::ComputeHistogram(const HistogramRequest& request, const HistogramSink& sink) const {
    return ComputeHistogram(request, sink, ReadOptions{});
}

Result<void> Image::ComputeHistogram(const HistogramRequest& request, const HistogramSink& sink,
                                     const ReadOptions& options) const {
    if (!_impl || !_impl->store) {
        return MakeError(ErrorCode::invalid_argument, "Image handle is empty");
    }
    return internal::ComputeHistogram(*_impl->store, _impl->descriptor, _impl->geometry, request, sink, options,
                                      *_impl->context->workers);
}

Result<CubeHistogramResult> Image::ComputeCubeHistogram(const CubeHistogramRequest& request) const {
    return ComputeCubeHistogram(request, ReadOptions{});
}

Result<CubeHistogramResult> Image::ComputeCubeHistogram(const CubeHistogramRequest& request,
                                                        const ReadOptions& options) const {
    if (!_impl || !_impl->store) {
        return MakeError(ErrorCode::invalid_argument, "Image handle is empty");
    }
    return internal::ComputeCubeHistogram(*_impl->store, _impl->descriptor, _impl->geometry, request, options,
                                          *_impl->context->workers);
}

Result<std::vector<Beam>> Image::ReadBeams() const {
    if (!_impl) {
        return MakeError(ErrorCode::invalid_argument, "Image handle is empty");
    }
    if (!_impl->store) {
        return MakeError(ErrorCode::invalid_argument, "Image store is unavailable");
    }
    auto profile = internal::SchemaProfile::For(_impl->schema_id);
    if (!profile) {
        return profile.error();
    }
    return profile.value().ReadBeams(*_impl->store, _impl->descriptor.id);
}

class Dataset::Impl {
public:
    Impl(std::shared_ptr<Context::Impl> context, std::string location, DatasetDescriptor descriptor,
         internal::Store store)
        : context(std::move(context)),
          location(std::move(location)),
          descriptor(std::move(descriptor)),
          store(std::make_shared<internal::Store>(std::move(store))) {}

    std::shared_ptr<Context::Impl> context;
    std::string location;
    DatasetDescriptor descriptor;
    std::shared_ptr<internal::Store> store;
    mutable std::mutex mutex;
    mutable std::unordered_map<std::string, ImageDescriptor> image_descriptors;
};

Dataset::Dataset(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}
Dataset::~Dataset() = default;

Result<Dataset> Dataset::Open(const Context& context, std::string_view location) {
    try {
        if (!context._impl) {
            return MakeError(ErrorCode::invalid_argument, "Context handle is empty");
        }

        // TensorStore resources are shared by Context, while array handles are scoped to this
        // Dataset and the Images that retain its Store.
        auto store_context = context._impl->store_context->CloneForStore();
        auto store_result = internal::OpenStore(location, std::move(store_context));
        if (!store_result) {
            return store_result.error();
        }
        auto probe_result = internal::ProbeStore(store_result.value());
        if (!probe_result) {
            return probe_result.error();
        }
        const auto& probe = probe_result.value();
        if (probe.kind != ProbeKind::supported_dataset) {
            const ErrorCode code =
                probe.kind == ProbeKind::invalid_dataset ? ErrorCode::invalid_metadata : ErrorCode::unsupported_schema;
            return MakeError(code, "The Zarr store is not a supported XRADIO image dataset", std::string(location));
        }

        if (probe.images.empty()) {
            return MakeError(ErrorCode::invalid_metadata, "Supported schema has no image variables",
                             std::string(location));
        }
        DatasetDescriptor descriptor;
        descriptor.schema_id = probe.schema_id;
        descriptor.schema_version = probe.schema_version;
        descriptor.images = probe.images;
        descriptor.default_image_id = probe.default_image_id;
        descriptor.diagnostics = probe.diagnostics;
        return Dataset{std::make_shared<Impl>(context._impl, std::string(location), std::move(descriptor),
                                              std::move(store_result.value()))};
    } catch (const std::exception& error) {
        return MakeError(ErrorCode::invalid_metadata, error.what(), std::string(location));
    }
}

const DatasetDescriptor& Dataset::descriptor() const noexcept {
    static const DatasetDescriptor empty_descriptor;
    return _impl ? _impl->descriptor : empty_descriptor;
}

Result<DatasetSize> Dataset::Size(std::chrono::milliseconds directory_size_timeout) const {
    if (!_impl) {
        return MakeError(ErrorCode::invalid_argument, "Dataset handle is empty");
    }

    std::uint64_t physical_size = 0;
    if (TryComputeDirectorySize(_impl->location, directory_size_timeout, physical_size)) {
        return DatasetSize{physical_size, false};
    }

    auto logical_size = _impl->store->ComputeTotalArraySizeBytes();
    if (!logical_size) {
        return logical_size.error();
    }
    return DatasetSize{logical_size.value(), true};
}

Result<Image> Dataset::OpenImage(std::string_view image_id) const {
    if (!_impl) {
        return MakeError(ErrorCode::invalid_argument, "Dataset handle is empty");
    }
    std::scoped_lock const lock(_impl->mutex);
    const std::string image_name(image_id);
    const auto entry = std::find_if(_impl->descriptor.images.begin(), _impl->descriptor.images.end(),
                                    [&](const ImageEntry& image) { return image.id == image_name; });
    if (entry == _impl->descriptor.images.end()) {
        return MakeError(ErrorCode::not_found, "Image variable was not found", image_name);
    }
    if (!entry->readable) {
        const auto message = entry->diagnostics.empty() ? "Image variable is not openable by this profile"
                                                        : entry->diagnostics.front().message;
        return MakeError(ErrorCode::unsupported_data_type, message, image_name);
    }
    const auto make_image = [&](const ImageDescriptor& descriptor) {
        // The descriptor already carries the stored layout; the geometry is that layout permuted
        // into logical order, so it is derived here rather than read again.
        const StorageLayout layout = descriptor.storage ? *descriptor.storage : StorageLayout{};
        return Image{std::make_shared<Image::Impl>(_impl->context, _impl->location, _impl->descriptor.schema_id,
                                                   _impl->store, descriptor,
                                                   BuildChunkGeometry(descriptor, layout))};
    };
    const auto cached = _impl->image_descriptors.find(image_name);
    if (cached != _impl->image_descriptors.end()) {
        return make_image(cached->second);
    }
    auto profile = internal::SchemaProfile::For(_impl->descriptor.schema_id);
    if (!profile) {
        return profile.error();
    }
    auto image_descriptor = profile.value().DescribeVerified(*_impl->store, image_id);
    if (!image_descriptor) {
        return image_descriptor.error();
    }
    auto [inserted, _] = _impl->image_descriptors.emplace(image_name, std::move(image_descriptor.value()));
    return make_image(inserted->second);
}

ProbeResult Probe(std::string_view location, const ProbeOptions&) {
    try {
        auto store_result = internal::OpenStore(location);
        if (!store_result) {
            ProbeResult result;
            result.kind = ProbeKind::not_zarr;
            if (store_result.error().code == ErrorCode::invalid_metadata ||
                store_result.error().code == ErrorCode::io_error) {
                result.kind = ProbeKind::invalid_dataset;
            }
            result.diagnostics.push_back(Diagnostic{internal::zarr::ErrorCodeName(store_result.error().code),
                                                    store_result.error().message, store_result.error().node_path});
            return result;
        }
        auto probe_result = internal::ProbeStore(store_result.value());
        if (!probe_result) {
            ProbeResult result;
            result.kind = ProbeKind::invalid_dataset;
            result.diagnostics.push_back(Diagnostic{internal::zarr::ErrorCodeName(probe_result.error().code),
                                                    probe_result.error().message, probe_result.error().node_path});
            return result;
        }
        return probe_result.value();
    } catch (const std::exception& error) {
        ProbeResult result;
        result.kind = ProbeKind::invalid_dataset;
        result.diagnostics.push_back(Diagnostic{"exception", error.what(), std::string(location)});
        return result;
    }
}

Result<SchemaProbeResult> ProbeSchema(std::string_view location, std::string_view schema_id) {
    try {
        // Resolve the profile before touching the store, so an unknown schema reports itself rather
        // than whatever happens to be wrong with the path.
        auto profile = internal::SchemaProfile::For(schema_id);
        if (!profile) {
            return profile.error();
        }
        auto store_result = internal::OpenStore(location);
        if (!store_result) {
            return store_result.error();
        }
        return profile.value().Probe(store_result.value());
    } catch (const std::exception& error) {
        return MakeError(ErrorCode::invalid_metadata, error.what(), std::string(location));
    }
}

Result<bool> IsXradioImage(std::string_view location) {
    auto result = ProbeSchema(location, kXradioImageSchema);
    if (!result) {
        return result.error();
    }
    if (result.value().kind == SchemaMatchKind::match) {
        return true;
    }
    if (result.value().kind == SchemaMatchKind::invalid) {
        return MakeError(ErrorCode::invalid_metadata, SchemaErrorMessage(result.value()), std::string(location));
    }
    return false;
}

}  // namespace carta::zarr
