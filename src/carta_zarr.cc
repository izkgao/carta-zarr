/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "carta-zarr/carta_zarr.h"

#include "carta-zarr/read_ahead.h"

#include "chunk_blocks.h"
#include "pixel_mask.h"
#include "read/pieces.h"
#include "read_ahead.h"
#include "reduce/plane_histogram.h"
#include "reduce/spectral_reduce.h"
#include "reducible_image.h"
#include "schema/profile.h"
#include "store.h"
#include "store_pixel_source.h"
#include "work_pool.h"
#include "zarr/array_metadata.h"
#include "zarr/pixel_reader.h"
#include "zarr/store_context.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace carta::zarr {
namespace {

// Every public entry point reports its failures as a Result, and a consumer that checks one should
// never have to catch as well. Underneath, though, metadata comes from a file and buffers are sized
// from what it says: nlohmann throws on a value that is not the type it is read as, and the standard
// library throws on a length it cannot allocate. This is where that becomes a report.
//
// It is the only try in this file, so that whether an entry point catches is never a question.
//
// The node is a view, and it becomes a string only in the handler. Taken as a string it would be
// copied at the call, before the try it is meant to be reported from, and an allocation that failed
// there would escape.
// The report is a string too, and under the same shortage it may not be possible to make one: then
// the code is what reaches the caller, with nothing to say, which is still a Result.
template <typename Function>
auto Guarded(ErrorCode code, std::string_view node, Function&& function) -> decltype(function()) {
    try {
        return function();
    } catch (const std::exception& error) {
        try {
            return Error{code, error.what(), std::string(node)};
        } catch (const std::bad_alloc&) {
            return Error{code, {}, {}};
        }
    } catch (...) {
        // What a caller's callback throws need not be a std::exception, and it is still not to leave
        // an entry point that promises a Result. There is nothing in it to report but that it was thrown.
        try {
            return Error{code, "A callback threw something other than a std::exception", std::string(node)};
        } catch (const std::bad_alloc&) {
            return Error{code, {}, {}};
        }
    }
}

// Everything a reduction's entry point does before it has something to walk: name the node so a
// failure says which image it was about, build the ReducibleImage, and guard the lot. Image::Read
// does the first and the last and not the middle, because a read is not a reduction.
//
// There is no empty handle to refuse here: a handle's _impl is never null; see carta_zarr.h.
//
// The handle is a template parameter because Image::Impl is private to Image and this is not.
// Deducing the type asks nothing of access control, where naming it would.
template <typename ImplPtr, typename Function>
auto WithReducibleImage(const ImplPtr& impl, Function&& function)
    -> decltype(function(std::declval<const internal::ReducibleImage&>())) {
    using Answer = decltype(function(std::declval<const internal::ReducibleImage&>()));
    return Guarded(ErrorCode::io_error, impl->descriptor.id, [&]() -> Answer {
        auto image = impl->Reducible();
        if (!image) {
            return image.error();
        }
        return function(image.value());
    });
}

}  // namespace

class Context::Impl {
public:
    Impl(ContextOptions options, internal::StoreContextPtr store_context)
        : options(options),
          store_context(std::move(store_context)),
          // decode_threads is the consumer's statement of how much of this machine the library may
          // use, so it sizes both pools rather than only TensorStore's. The two are busy at
          // different moments -- a slab is read and then visited -- so sizing each at the whole
          // budget does not double the demand. Zero means one worker per hardware thread, which is
          // what TensorStore's own default does with the same number.
          workers(std::make_shared<internal::WorkPool>(options.decode_threads)) {}

    ContextOptions options;
    // Shared by every dataset and image opened through this context, so that its cache and
    // concurrency limits apply to all reads rather than being rebuilt per read.
    internal::StoreContextPtr store_context;
    // The per-pixel work of a reduction. See WorkPool.
    std::shared_ptr<internal::WorkPool> workers;
};

Context::Context(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}
Context::~Context() = default;

Result<Context> Context::Create(const ContextOptions& options) {
    // Guarded like every other public entry point, and for a reason the others do not have: building
    // an Impl starts the worker threads, and a system that refuses one throws std::system_error.
    // Without this, the one call a consumer makes before it can do anything else is also the only
    // one that could throw at it. io_error is the nearest existing code for "the machine would not
    // give us what we asked for"; the message says which resource it was.
    return Guarded(ErrorCode::io_error, {}, [&]() -> Result<Context> {
        auto store_context = internal::MakeStoreContext(options);
        if (!store_context) {
            return store_context.error();
        }
        return Context{std::make_shared<Impl>(options, std::move(store_context.value()))};
    });
}

Result<CachePool> Context::NewCachePool(std::size_t bytes) const {
    // Guarded for the reason Create is: the pool's spec and its handle are both allocations, and
    // one that fails is the machine refusing a resource, which a caller is told as a Result.
    return Guarded(ErrorCode::io_error, {}, [&]() -> Result<CachePool> {
        auto store_context = _impl->store_context->WithCachePool(bytes);
        if (!store_context) {
            return store_context.error();
        }
        return CachePool{std::make_shared<const CachePool::Impl>(bytes, std::move(store_context.value()))};
    });
}

CachePool::CachePool(std::shared_ptr<const Impl> impl) : _impl(std::move(impl)) {}

std::size_t CachePool::bytes() const noexcept {
    return _impl->bytes;
}

class Image::Impl {
public:
    Impl(std::shared_ptr<Context::Impl> context, std::string location, internal::SchemaProfile profile,
         std::shared_ptr<internal::Store> store, ImageDescriptor descriptor, ChunkGeometry geometry,
         ChunkGeometry flag_geometry)
        : context(std::move(context)),
          location(std::move(location)),
          profile(profile),
          store(std::move(store)),
          descriptor(std::move(descriptor)),
          geometry(std::move(geometry)),
          flag_geometry(std::move(flag_geometry)),
          source(*this->store, this->descriptor) {}

    // How many threads decode this image's chunks, which every read's default budget aims at.
    std::size_t DecodeThreads() const noexcept { return context->workers->size(); }

    std::shared_ptr<Context::Impl> context;
    std::string location;
    // The profile that described this image, rather than the name of one to look up again. It is a
    // pointer into a table that outlives every store, so holding it costs nothing and removes an
    // error path that could only fire if a descriptor named a profile the library does not have.
    internal::SchemaProfile profile;
    std::shared_ptr<internal::Store> store;
    ImageDescriptor descriptor;
    ChunkGeometry geometry;
    // The flag's own, which a prefetch samples the flag by and every read and reduction is sized by
    // beside the pixels'. See DescribedImage.
    ChunkGeometry flag_geometry;
    // Built once, from members rather than from the constructor's arguments, and declared last so
    // that both of those are already initialised. A PixelSource cannot be copied or moved, which is
    // why it lives here rather than being made at each entry point.
    internal::StorePixelSource source;

    // What every reduction is against. Built per call, so that an image whose axes cannot be mapped
    // fails the reduction that needed them rather than the open -- which is where that failure has
    // always reached the caller.
    Result<internal::ReducibleImage> Reducible() const {
        return internal::ReducibleImage::Of(source, descriptor, geometry, flag_geometry, *context->workers);
    }
};

Image::Image(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}
Image::~Image() = default;

const ImageDescriptor& Image::descriptor() const noexcept {
    return _impl->descriptor;
}

const ChunkGeometry& Image::chunk_geometry() const noexcept {
    return _impl->geometry;
}

Result<std::size_t> Image::Read(const ReadRequest& request, BufferView<float> destination, const ReadOptions& options,
                                const ProgressCallback& progress) const {
    return Guarded(ErrorCode::io_error, _impl->descriptor.id, [&] {
        return internal::ReadInPieces(_impl->source, _impl->descriptor, _impl->geometry, _impl->flag_geometry, request,
                                      destination, options, _impl->DecodeThreads(), progress);
    });
}

std::uint64_t Image::DecodedChunkBytes(const ReadOptions& options) const {
    // What every read and reduction sizes itself against, so that a caller's cache agrees with them.
    return internal::ReadCost::Of(_impl->descriptor, _impl->geometry, _impl->flag_geometry, options,
                                  internal::PixelsHeld::by_caller, _impl->DecodeThreads())
        .chunk_bytes;
}

Result<std::uint64_t> Image::Prefetch(const ReadRequest& request, const ReadOptions& options) const {
    return Guarded(ErrorCode::io_error, _impl->descriptor.id, [&] {
        return internal::PrefetchChunks(_impl->source, _impl->descriptor, _impl->geometry, _impl->flag_geometry,
                                        request, options, _impl->DecodeThreads());
    });
}

Result<void> Image::ReduceSpectral(const SpectralReduceRequest& request, const SpectralSink& sink,
                                   const ReadOptions& options) const {
    return WithReducibleImage(_impl, [&](const internal::ReducibleImage& image) {
        return internal::ReduceSpectral(image, request, sink, options);
    });
}

Result<void> Image::ComputeHistogram(const HistogramRequest& request, const HistogramSink& sink,
                                     const ReadOptions& options) const {
    return WithReducibleImage(_impl, [&](const internal::ReducibleImage& image) {
        return internal::ComputeHistogram(image, request, sink, options);
    });
}

Result<CubeHistogramResult> Image::ComputeCubeHistogram(const CubeHistogramRequest& request, const ReadOptions& options,
                                                        const CubeHistogramProgressCallback& progress) const {
    return WithReducibleImage(_impl, [&](const internal::ReducibleImage& image) {
        return internal::ComputeCubeHistogram(image, request, options, progress);
    });
}

Result<std::vector<Beam>> Image::ReadBeams() const {
    return Guarded(ErrorCode::invalid_metadata, _impl->descriptor.id, [&]() -> Result<std::vector<Beam>> {
        return _impl->profile.ReadBeams(*_impl->store, _impl->descriptor.id);
    });
}

namespace {

// What reading ahead asks of one image, answered by the image: a run from its chunk geometry, what a
// run holds from what its chunks decode to, and the decode itself by Image::Prefetch with the options
// its frames are read with -- so into the cache they will look in.
class ImageRuns final : public internal::RunSource {
public:
    ImageRuns(Image image, ChunkGeometry flag_geometry, ReadOptions options, internal::CacheShare cache)
        : _image(std::move(image)),
          _flag_geometry(std::move(flag_geometry)),
          _options(std::move(options)),
          _cache(cache) {}

    internal::Run RunOf(const ReadRequest& plane) const override {
        return internal::RunOf(_image.chunk_geometry(), plane);
    }
    std::uint64_t PlaneRunBytes() const override {
        const auto& descriptor = _image.descriptor();
        return internal::PlaneRunBytes(descriptor, _image.chunk_geometry(), _flag_geometry,
                                       internal::AppliesPixelMask(_options, descriptor));
    }
    internal::CacheShare Cache() const override { return _cache; }
    std::string Name() const override { return _image.descriptor().id; }
    // Stopped by either reading ahead or the caller's own cancellation, whichever says so first.
    bool Prefetch(const ReadRequest& plane, const std::function<bool()>& cancelled) const override {
        ReadOptions options = _options;
        const auto callers = options.control.cancellation_requested;
        options.control.cancellation_requested = [&cancelled, callers] {
            return cancelled() || (callers && callers());
        };
        return _image.Prefetch(internal::RunRequest(_image.descriptor(), _image.chunk_geometry(), plane), options)
            .has_value();
    }

private:
    Image _image;
    // Not on Image's public face: a consumer asks how the pixels are chunked, and what the flag
    // costs beside them is this class's question.
    ChunkGeometry _flag_geometry;
    ReadOptions _options;
    internal::CacheShare _cache;
};

}  // namespace

class ReadAhead::Impl {
public:
    explicit Impl(std::unique_ptr<internal::ReadingAhead> reading) : reading(std::move(reading)) {}

    std::unique_ptr<internal::ReadingAhead> reading;
};

ReadAhead::ReadAhead(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}
ReadAhead::~ReadAhead() = default;

Result<ReadAhead> ReadAhead::For(const std::vector<std::pair<Image, ReadOptions>>& images) {
    return Guarded(ErrorCode::io_error, {}, [&]() -> Result<ReadAhead> {
        std::vector<std::shared_ptr<const internal::RunSource>> sources;
        sources.reserve(images.size());
        for (const auto& [image, options] : images) {
            // The cache a read keeps its chunks in: a pool of its own when it names one, otherwise the
            // one its context shares among every read -- which holds nothing unless it was sized.
            internal::CacheShare cache;
            if (options.control.cache_pool) {
                cache.identity = internal::CachePoolAccess::StoreContextOf(*options.control.cache_pool).get();
                cache.bytes = options.control.cache_pool->bytes();
            } else {
                cache.identity = image._impl->context.get();
                cache.bytes = image._impl->context->options.cache_bytes.value_or(0);
            }
            sources.push_back(std::make_shared<ImageRuns>(image, image._impl->flag_geometry, options, cache));
        }
        auto reading = internal::ReadingAhead::For(std::move(sources));
        if (!reading) {
            return reading.error();
        }
        return ReadAhead{std::make_shared<Impl>(std::move(reading).value())};
    });
}

void ReadAhead::Served(Clock::time_point began, bool late, const std::vector<AnimatedPlane>& shown,
                       const std::vector<std::vector<AnimatedPlane>>& upcoming) {
    try {
        _impl->reading->Served(began, late, shown, upcoming);
    } catch (...) {
        // Reading ahead is for time to spare, and a machine that cannot find the memory to decide what
        // to read has none.
        _impl->reading->Cancel();
    }
}

void ReadAhead::Cancel() {
    _impl->reading->Cancel();
}

ReadAheadStats ReadAhead::stats() const {
    return _impl->reading->Stats();
}

class Dataset::Impl {
public:
    Impl(std::shared_ptr<Context::Impl> context, std::string location, DatasetDescriptor descriptor,
         internal::SchemaProfile profile, internal::Store store)
        : context(std::move(context)),
          location(std::move(location)),
          descriptor(std::move(descriptor)),
          profile(profile),
          store(std::make_shared<internal::Store>(std::move(store))) {}

    // How many threads decode this image's chunks, which every read's default budget aims at.
    std::size_t DecodeThreads() const noexcept { return context->workers->size(); }

    std::shared_ptr<Context::Impl> context;
    std::string location;
    DatasetDescriptor descriptor;
    // The profile the probe matched. descriptor.schema_id names it for the consumer; this is the
    // one the library asks, resolved where the match happened rather than at each use.
    internal::SchemaProfile profile;
    std::shared_ptr<internal::Store> store;
    mutable std::mutex mutex;
    mutable std::unordered_map<std::string, internal::DescribedImage> image_descriptors;
};

Dataset::Dataset(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {}
Dataset::~Dataset() = default;

Result<Dataset> Dataset::Open(const Context& context, std::string_view location) {
    return Guarded(ErrorCode::invalid_metadata, location, [&]() -> Result<Dataset> {
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
        if (auto openable = internal::RequireOpenableDataset(probe, location); !openable) {
            return openable.error();
        }
        auto profile = internal::SchemaProfile::For(probe.schema_id);
        if (!profile) {
            return profile.error();
        }
        // The descriptor is the probe's answer without the question it was answering, so it is taken
        // rather than rebuilt field by field -- and taken by move, since the probe result dies here.
        DatasetDescriptor descriptor = std::move(static_cast<DatasetDescriptor&>(probe_result.value()));
        return Dataset{std::make_shared<Impl>(context._impl, std::string(location), std::move(descriptor),
                                              profile.value(), std::move(store_result.value()))};
    });
}

const DatasetDescriptor& Dataset::descriptor() const noexcept {
    return _impl->descriptor;
}

Result<DatasetSize> Dataset::Size(std::chrono::milliseconds stored_size_timeout) const {
    return Guarded(ErrorCode::io_error, _impl->location, [&]() -> Result<DatasetSize> {
        // The caller's timeout becomes a deadline here and nowhere lower: a timeout is measured from
        // whenever the caller asked, which is a fact only this end of the call knows. Everything
        // below speaks deadlines, as every other storage operation in this library does.
        //
        // One too far off for the clock is none at all. Added as it came, the milliseconds overflowed
        // the clock's nanoseconds past about 292 years and wrapped into a deadline already past.
        using Clock = std::chrono::steady_clock;
        const auto now = Clock::now();
        const auto room = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - now);
        const auto deadline = stored_size_timeout >= room
                                ? Clock::time_point::max()
                                : now + std::chrono::duration_cast<Clock::duration>(stored_size_timeout);
        return internal::DatasetSizeBytes(*_impl->store, deadline);
    });
}

Result<Image> Dataset::OpenImage(std::string_view image_id) const {
    return Guarded(ErrorCode::invalid_metadata, image_id, [&]() -> Result<Image> {
        std::scoped_lock const lock(_impl->mutex);
        const std::string image_name(image_id);
        const auto make_image = [&](const internal::DescribedImage& described) {
            return Image{std::make_shared<Image::Impl>(_impl->context, _impl->location, _impl->profile, _impl->store,
                                                       described.descriptor, described.geometry,
                                                       described.flag_geometry)};
        };
        const auto cached = _impl->image_descriptors.find(image_name);
        if (cached != _impl->image_descriptors.end()) {
            return make_image(cached->second);
        }
        // Describing asks the profile whether it will open this variable, so there is no gate here
        // ahead of it. The listing this dataset kept would answer the same question, and answering
        // it in both places is what let a listing say one thing and opening another.
        auto image_descriptor = _impl->profile.Describe(*_impl->store, image_id);
        if (!image_descriptor) {
            return image_descriptor.error();
        }
        auto [inserted, _] = _impl->image_descriptors.emplace(image_name, std::move(image_descriptor.value()));
        return make_image(inserted->second);
    });
}

Result<SchemaProbeResult> ProbeSchema(std::string_view location, std::string_view schema_id) {
    return Guarded(ErrorCode::invalid_metadata, location, [&]() -> Result<SchemaProbeResult> {
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
    });
}

}  // namespace carta::zarr
