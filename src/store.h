/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_STORE_H_
#define CARTA_ZARR_SRC_STORE_H_

#include "carta-zarr/carta_zarr.h"

#include "memo.h"
#include "zarr/array_metadata.h"
#include "zarr/transport.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace carta::zarr::internal {

// Defined in zarr/pixel_reader.h. Forward declared so that store.h stays free of the reader that
// pulls in TensorStore, the same way StoreContext is kept opaque below.
namespace zarr {
struct PixelSelection;
}

struct ImageDiscovery {
    std::vector<ImageEntry> images;
    std::optional<std::string> default_image_id;
    std::vector<Diagnostic> diagnostics;
};

// Everything a Store remembers for the lifetime of its read-only view.
//
// Each table carries its own lock, and that is load-bearing rather than incidental: a Memo holds its
// lock across the computation, and these computations nest -- reading array metadata reads node
// metadata, listing nodes reads each node's metadata. One shared lock would deadlock. The nesting
// forms a DAG, and a reader added later must not add an edge back:
//
//     string_arrays ---> array_metadata ---> node_metadata ---> transport
//     listed_metadata ---------------------> node_metadata ---> transport
//     double_arrays -------------------------------------------> transport
//
// These locks are not the store's concurrency story. Descriptor construction is serialized a level
// up, by the mutex Dataset::OpenImage holds across the whole of DescribeSchema, so today only two
// concurrent Image::ReadBeams calls on one Dataset reach these tables at the same time. When pixel
// reads arrive that changes, and holding a lock across a TensorStore read becomes worth revisiting
// -- in here, rather than in five hand-written places as before.
struct StoreCaches {
    Memo<std::string, Result<nlohmann::json>> node_metadata;
    Memo<std::string, Result<zarr::ArrayMetadata>> array_metadata;
    Lazy<Result<std::vector<std::pair<std::string, nlohmann::json>>>> listed_metadata;
    Memo<std::string, Result<ImageDiscovery>> image_discoveries;
    Memo<std::string, Result<std::vector<double>>> double_arrays;
    Memo<std::string, Result<std::vector<std::string>>> string_arrays;
};

// Defined in zarr/store_context.h. Kept opaque here so that including store.h does not pull in
// TensorStore; a null pointer means "use TensorStore's default resources".
class StoreContext;
using StoreContextPtr = std::shared_ptr<const StoreContext>;

// A Store is the Dataset-scoped storage session that interprets a Zarr hierarchy over a Transport.
// It owns everything that turns bytes into meaning -- node path validation, JSON parsing,
// consolidated metadata, array metadata, storage layout, data locations, and the caches -- so that
// every read uses one consistent session. Nothing above this module learns where the bytes came from.
class Store {
public:
    Store(TransportPtr transport, nlohmann::json root_attributes,
          std::map<std::string, nlohmann::json> consolidated_metadata, bool has_consolidated_metadata,
          StoreContextPtr context);

    // The root group's attributes, or an empty object when it declares none.
    const nlohmann::json& RootAttributes() const noexcept;

    Result<nlohmann::json> ReadNodeMetadata(std::string_view node) const;
    Result<zarr::ArrayMetadata> ReadArrayMetadata(std::string_view node) const;
    Result<std::vector<std::pair<std::string, nlohmann::json>>> ListNodeMetadata() const;
    template <typename Compute>
    Result<ImageDiscovery> CachedImageDiscovery(std::string_view profile_id, Compute compute) const {
        return _caches->image_discoveries.GetOrCompute(std::string(profile_id), compute);
    }
    Result<std::uint64_t> ComputeTotalArraySizeBytes() const;
    // Values in C order, flattened. The rank is in the node's ArrayMetadata; ArrayView addresses
    // them by dimension name rather than by offset.
    Result<std::vector<double>> ReadNumericArray(std::string_view node) const;
    Result<std::vector<std::string>> ReadStringArray1D(std::string_view node) const;
    Result<StorageLayout> ReadStorageLayout(std::string_view node) const;

    // Pixel reads are deliberately uncached here: a slab is requested once and is far larger than
    // anything the metadata tables hold. Reuse belongs in TensorStore's chunk cache, which already
    // works at chunk granularity and is sized by the consumer's Context.
    Result<void> ReadPixelsFloat32(std::string_view node, const zarr::PixelSelection& selection,
                                   float* destination, std::size_t destination_elements,
                                   const ReadOptions& options) const;
    Result<void> ReadPixelMaskBytes(std::string_view node, const zarr::PixelSelection& selection,
                                    std::uint8_t* destination, std::size_t destination_elements,
                                    const ReadOptions& options) const;

private:
    Result<std::filesystem::path> ResolveArrayDirectory(std::string_view node) const;
    Result<std::vector<double>> ReadNumericArrayUncached(std::string_view node) const;
    Result<std::vector<std::string>> ReadStringArray1DUncached(std::string_view node) const;

    TransportPtr _transport;
    nlohmann::json _root_attributes;
    std::map<std::string, nlohmann::json> _consolidated_metadata;
    bool _has_consolidated_metadata = false;
    StoreContextPtr _context;
    // Held indirectly so that Store stays movable: the tables own mutexes and cannot be moved.
    std::shared_ptr<StoreCaches> _caches;
};

// Open a store on the local filesystem.
Result<Store> OpenStore(std::string_view location, StoreContextPtr context = {});

// Open a store over an already-built transport. This is the seam tests enter through.
Result<Store> OpenStore(TransportPtr transport, StoreContextPtr context = {});

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_STORE_H_
