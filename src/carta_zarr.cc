/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "carta-zarr/carta_zarr.h"

#include "schema/profile.h"
#include "store.h"
#include "zarr/array_metadata.h"
#include "zarr/pixel_reader.h"
#include "zarr/store_context.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdint>
#include <limits>
#include <mutex>
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
        : options(options), store_context(std::move(store_context)) {}

    OpenOptions options;
    // Shared by every dataset and image opened through this context, so that its cache and
    // concurrency limits apply to all reads rather than being rebuilt per read.
    internal::StoreContextPtr store_context;
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

// Translate a request expressed over the logical axes into the stored axis order the array is
// written in, checking it against the descriptor on the way. Ranges are validated here rather than
// left to TensorStore so that an out-of-range request is an invalid_argument naming the axis,
// instead of an I/O error naming a domain.
Result<internal::zarr::PixelSelection> BuildSelection(const ImageDescriptor& descriptor,
                                                      const ReadRequest& request) {
    const auto rank = descriptor.axes.size();
    if (request.axes.size() != rank) {
        return MakeError(ErrorCode::invalid_argument,
                         "Request has " + std::to_string(request.axes.size()) + " axes but the image has " +
                             std::to_string(rank),
                         descriptor.id);
    }

    internal::zarr::PixelSelection selection;
    selection.start.assign(rank, 0);
    selection.count.assign(rank, 0);
    selection.stride.assign(rank, 1);
    selection.shape.assign(rank, 0);
    selection.dimension_names.assign(rank, {});
    selection.logical_to_stored.resize(rank);

    for (std::size_t logical = 0; logical < rank; ++logical) {
        const auto& axis = descriptor.axes.at(logical);
        const auto& range = request.axes.at(logical);
        if (range.stride == 0) {
            return MakeError(ErrorCode::invalid_argument, "Axis '" + axis.name + "' has a zero stride",
                             descriptor.id);
        }
        if (range.count == 0) {
            return MakeError(ErrorCode::invalid_argument, "Axis '" + axis.name + "' selects no elements",
                             descriptor.id);
        }
        // The last selected index, which is what has to fall inside the axis.
        const std::uint64_t span = (range.count - 1) * range.stride;
        if (range.start >= axis.length || span > axis.length - 1 - range.start) {
            return MakeError(ErrorCode::invalid_argument,
                             "Axis '" + axis.name + "' request exceeds its length of " +
                                 std::to_string(axis.length),
                             descriptor.id);
        }
        const auto stored = axis.storage_index;
        if (stored >= rank) {
            return MakeError(ErrorCode::invalid_metadata, "Axis '" + axis.name + "' has an out-of-range storage index",
                             descriptor.id);
        }
        selection.start.at(stored) = range.start;
        selection.count.at(stored) = range.count;
        selection.stride.at(stored) = range.stride;
        selection.shape.at(stored) = axis.length;
        selection.dimension_names.at(stored) = axis.name;
        selection.logical_to_stored.at(logical) = stored;
    }
    return selection;
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

    auto selection = BuildSelection(_impl->descriptor, request);
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

    auto* pixels = static_cast<float*>(destination.data);
    if (options.apply_pixel_mask && _impl->descriptor.has_pixel_mask) {
        if (options.temporary_memory_limit_bytes != 0 &&
            elements > options.temporary_memory_limit_bytes) {
            return MakeError(ErrorCode::buffer_too_small,
                             "Pixel mask temporary buffer exceeds the configured memory limit",
                             _impl->descriptor.id);
        }
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(elements));
        auto mask_read = _impl->store->ReadPixelMaskBytes(_impl->descriptor.pixel_mask_id, selection.value(),
                                                          mask.data(), mask.size(), options);
        if (!mask_read) {
            return mask_read.error();
        }

        // Read the mask first so an unavailable or cancelled mask cannot leave a partially updated
        // destination. TensorStore still owns the pixel operation's in-flight completion before it
        // returns, so the destination remains valid for the next read.
        auto read = _impl->store->ReadPixelsFloat32(_impl->descriptor.id, selection.value(), pixels,
                                                    static_cast<std::size_t>(elements), options);
        if (!read) {
            return read.error();
        }
        // XRADIO stores flags with true meaning a good pixel.
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask.at(i) == 0) {
                pixels[i] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    } else {
        // Read as one request. Splitting it into chunk-aligned slabs was tried and measured slightly
        // slower on real data (360 ms against 348 for a 7763 x 4742 plane), so the region is handed to
        // TensorStore whole and it decides how to fetch the chunks.
        auto read = _impl->store->ReadPixelsFloat32(_impl->descriptor.id, selection.value(), pixels,
                                                    static_cast<std::size_t>(elements), options);
        if (!read) {
            return read.error();
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

    auto selection = BuildSelection(_impl->descriptor, request);
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
