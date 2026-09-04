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
    Result<std::size_t> Read(const ReadRequest& request, MutableBufferView destination) const;
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
