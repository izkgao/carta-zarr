/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "carta-zarr/carta_zarr.h"

#include "schema/profile.h"
#include "store.h"
#include "zarr/array_metadata.h"
#include "zarr/store_context.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <mutex>
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
         std::shared_ptr<internal::Store> store, ImageDescriptor descriptor)
        : context(std::move(context)),
          location(std::move(location)),
          schema_id(std::move(schema_id)),
          store(std::move(store)),
          descriptor(std::move(descriptor)) {}

    std::shared_ptr<Context::Impl> context;
    std::string location;
    std::string schema_id;
    std::shared_ptr<internal::Store> store;
    ImageDescriptor descriptor;
};

Image::Image(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}
Image::~Image() = default;

const ImageDescriptor& Image::descriptor() const noexcept {
    static const ImageDescriptor empty_descriptor;
    return _impl ? _impl->descriptor : empty_descriptor;
}

Result<std::size_t> Image::Read(const ReadRequest&, MutableBufferView) const {
    if (!_impl) {
        return MakeError(ErrorCode::invalid_argument, "Image handle is empty");
    }
    return MakeError(ErrorCode::not_implemented, "Pixel reads are not implemented in PR 1");
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

        auto store_result = internal::OpenStore(location, context._impl->store_context);
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

        if (probe.image_ids.empty()) {
            return MakeError(ErrorCode::invalid_metadata, "Supported schema has no image variables",
                             std::string(location));
        }
        DatasetDescriptor descriptor;
        descriptor.schema_id = probe.schema_id;
        descriptor.schema_version = probe.schema_version;
        descriptor.image_ids = probe.image_ids;
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
    const auto make_image = [&](const ImageDescriptor& descriptor) {
        return Image{std::make_shared<Image::Impl>(_impl->context, _impl->location, _impl->descriptor.schema_id,
                                                   _impl->store, descriptor)};
    };
    const auto cached = _impl->image_descriptors.find(image_name);
    if (cached != _impl->image_descriptors.end()) {
        return make_image(cached->second);
    }
    auto profile = internal::SchemaProfile::For(_impl->descriptor.schema_id);
    if (!profile) {
        return profile.error();
    }
    auto image_descriptor = profile.value().Describe(*_impl->store, image_id);
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
