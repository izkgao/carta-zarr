/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_CARTA_ZARR_H_
#define CARTA_ZARR_CARTA_ZARR_H_

#include "carta-zarr/export.h"
#include "carta-zarr/result.h"
#include "carta-zarr/types.h"

#include <chrono>
#include <memory>

namespace carta::zarr {

class CARTA_ZARR_EXPORT Context final {
public:
    Context(const Context&) = default;
    Context& operator=(const Context&) = default;
    Context(Context&&) noexcept = default;
    Context& operator=(Context&&) noexcept = default;
    ~Context();

    static Result<Context> Create(const OpenOptions& options = {});

private:
    class Impl;
    explicit Context(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> _impl;

    friend class Dataset;
    friend class Image;
};

class CARTA_ZARR_EXPORT Image final {
public:
    Image(const Image&) = default;
    Image& operator=(const Image&) = default;
    Image(Image&&) noexcept = default;
    Image& operator=(Image&&) noexcept = default;
    ~Image();

    const ImageDescriptor& descriptor() const noexcept;

    // The read geometry, in the same axis order as descriptor().axes.
    const ChunkGeometry& chunk_geometry() const noexcept;

    // Reads a densely packed result in logical axis order, axis 0 fastest-varying. Returns the
    // number of elements written. Safe to call concurrently on one handle. On failure, the
    // destination may be unchanged, partially written, or fully written; callers must discard it.
    Result<std::size_t> Read(const ReadRequest& request, MutableBufferView destination) const;
    Result<std::size_t> Read(const ReadRequest& request, MutableBufferView destination,
                             const ReadOptions& options) const;

    // Reads this image's pixel mask over the same region, one byte per pixel, true meaning a good
    // pixel. Reports not_found when the image has no mask.
    Result<std::size_t> ReadPixelMask(const ReadRequest& request, MutableBufferView destination) const;

    Result<std::vector<Beam>> ReadBeams() const;

private:
    class Impl;
    explicit Image(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> _impl;

    friend class Dataset;
};

class CARTA_ZARR_EXPORT Dataset final {
public:
    Dataset(const Dataset&) = default;
    Dataset& operator=(const Dataset&) = default;
    Dataset(Dataset&&) noexcept = default;
    Dataset& operator=(Dataset&&) noexcept = default;
    ~Dataset();

    static Result<Dataset> Open(const Context& context, std::string_view location);

    const DatasetDescriptor& descriptor() const noexcept;
    // Returns the physical store size when directory enumeration completes within the timeout;
    // otherwise returns the logical uncompressed size of all arrays and marks it as an upper bound.
    Result<DatasetSize> Size(
        std::chrono::milliseconds directory_size_timeout = std::chrono::milliseconds(50)) const;
    Result<Image> OpenImage(std::string_view image_id) const;

private:
    class Impl;
    explicit Dataset(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> _impl;
};

CARTA_ZARR_EXPORT ProbeResult Probe(std::string_view location, const ProbeOptions& options = {});

CARTA_ZARR_EXPORT Result<SchemaProbeResult> ProbeSchema(std::string_view location, std::string_view schema_id);

// Returns an error for an unreadable or malformed store; false is a valid non-match.
CARTA_ZARR_EXPORT Result<bool> IsXradioImage(std::string_view location);

}  // namespace carta::zarr

#endif  // CARTA_ZARR_CARTA_ZARR_H_
