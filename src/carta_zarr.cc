/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "carta-zarr/carta_zarr.h"

#include "schema/registry.h"
#include "store.h"
#include "zarr/array_metadata.h"
#include "zarr/store_context.h"

#include <algorithm>
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
    return internal::ReadBeamsFromSchema(*_impl->store, _impl->schema_id, _impl->descriptor.id);
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
        if (probe_result.value().kind != ProbeKind::supported_dataset) {
            const auto& probe = probe_result.value();
            const ErrorCode code =
                probe.kind == ProbeKind::invalid_dataset ? ErrorCode::invalid_metadata : ErrorCode::unsupported_schema;
            return MakeError(code, "The Zarr store is not a supported XRADIO image dataset", std::string(location));
        }

        const auto& supported_probe = probe_result.value();
        if (supported_probe.image_ids.empty()) {
            return MakeError(ErrorCode::invalid_metadata, "Supported schema has no image variables",
                             std::string(location));
        }
        DatasetDescriptor descriptor;
        descriptor.schema_id = supported_probe.schema_id;
        descriptor.schema_version = supported_probe.schema_version;
        descriptor.image_ids = supported_probe.image_ids;
        descriptor.diagnostics = supported_probe.diagnostics;
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

Result<Image> Dataset::OpenImage(std::string_view image_id) const {
    if (!_impl) {
        return MakeError(ErrorCode::invalid_argument, "Dataset handle is empty");
    }
    std::scoped_lock const lock(_impl->mutex);
    const auto cached = _impl->image_descriptors.find(std::string(image_id));
    if (cached != _impl->image_descriptors.end()) {
        return Image{std::make_shared<Image::Impl>(_impl->context, _impl->location, _impl->descriptor.schema_id,
                                                   _impl->store, cached->second)};
    }
    auto image_descriptor = internal::DescribeSchema(*_impl->store, _impl->descriptor.schema_id, image_id);
    if (!image_descriptor) {
        return image_descriptor.error();
    }
    auto [inserted, _] = _impl->image_descriptors.emplace(std::string(image_id), std::move(image_descriptor.value()));
    return Image{std::make_shared<Image::Impl>(_impl->context, _impl->location, _impl->descriptor.schema_id,
                                               _impl->store, inserted->second)};
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
        const auto& adapters = internal::SchemaRegistry();
        const auto known_schema =
            std::find_if(adapters.begin(), adapters.end(),
                         [&](const internal::SchemaAdapter& adapter) { return adapter.id == schema_id; });
        if (known_schema == adapters.end()) {
            return MakeError(ErrorCode::unsupported_schema,
                             "No built-in adapter exists for schema " + std::string(schema_id));
        }
        auto store_result = internal::OpenStore(location);
        if (!store_result) {
            return store_result.error();
        }
        return internal::ProbeSchemaFromRegistry(store_result.value(), schema_id);
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
