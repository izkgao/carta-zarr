/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_ZARR_CARTA_ZARR_H_
#define CARTA_ZARR_CARTA_ZARR_H_

#include "carta-zarr/descriptor.h"
#include "carta-zarr/export.h"
#include "carta-zarr/read.h"
#include "carta-zarr/reduce.h"
#include "carta-zarr/result.h"

#include <chrono>
#include <cstdint>
#include <memory>

namespace carta::zarr {

// Context, Dataset and Image are shared handles, and a handle always refers to something: the only
// way to get one is from the factory that made what it refers to, and nothing empties it afterwards.
//
// That includes moving from one. Each of the three moves by copying -- one reference-count
// increment -- so the handle left behind still works. A defaulted move would null it, which gave every
// handle an empty state reachable only by std::move: each entry point then had to refuse it, and no
// consumer could construct one or test for it. A consumer that needs a handle it fills in later
// holds a std::optional of one. See ADR 0013.
class CARTA_ZARR_EXPORT Context final {
public:
    Context(const Context&) = default;
    Context& operator=(const Context&) = default;
    // A copy, so that the handle moved from still refers to what it did. See above.
    Context(Context&& other) noexcept : Context(other) {}
    Context& operator=(Context&& other) noexcept { return *this = other; }
    ~Context();

    static Result<Context> Create(const ContextOptions& options = {});

    // A decoded-chunk cache of `bytes`, apart from the one this context shares among every read, for
    // a read to keep what it decodes in through ReadControl::cache_pool. Zero is the pool that keeps
    // nothing. See CachePool.
    Result<CachePool> NewCachePool(std::size_t bytes) const;

private:
    class Impl;
    explicit Context(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> _impl;

    friend class Dataset;
    friend class Image;
};

class ReadAhead;

class CARTA_ZARR_EXPORT Image final {
public:
    Image(const Image&) = default;
    Image& operator=(const Image&) = default;
    // A copy, so that the handle moved from still refers to what it did. See above.
    Image(Image&& other) noexcept : Image(other) {}
    Image& operator=(Image&& other) noexcept { return *this = other; }
    ~Image();

    const ImageDescriptor& descriptor() const noexcept;

    // The read geometry, in the same axis order as descriptor().axes.
    const ChunkGeometry& chunk_geometry() const noexcept;

    // Every operation below takes its ReadOptions, and Read and ComputeCubeHistogram their progress
    // callback, as defaulted trailing arguments rather than as one overload per argument left off.
    // The overloads said nothing but "and the rest are defaults", three times over for Read, and a
    // reader had to open each to see that it forwarded.

    // Reads a densely packed result in logical axis order, axis 0 fastest-varying. Returns the
    // number of elements written. Safe to call concurrently on one handle. On failure, the
    // destination may be unchanged, partially written, or fully written; callers must discard it.
    //
    // float is the only output this library produces. It is said in the destination's type rather
    // than asked for in the request, because a request that could name a type the buffer was not
    // shaped for is a mistake worth making unspellable.
    //
    /// It is read in pieces that keep to ReadOptions::read_budget_bytes, and a progress callback
    /// watches them as they finish. See ProgressCallback.
    Result<std::size_t> Read(const ReadRequest& request, BufferView<float> destination, const ReadOptions& options = {},
                             const ProgressCallback& progress = {}) const;

    // Decodes the chunks a Read of `request` would decode, into the cache that options.control
    // names, and returns how many chunks that was. Nothing is written anywhere the caller can see:
    // the point is that a Read which follows finds them decoded.
    //
    // For reading ahead of somebody -- the next run of chunks of a playing animation, while this run's
    // frames are served from the cache -- where reading the pixels themselves would allocate and fill
    // a destination only to throw it away. An animation need not call it: ReadAhead decides when, and
    // calls it on a thread of its own. It reads one element
    // of each chunk, which decodes the whole chunk, so its cost is the decoding alone.
    //
    // A Read of the same chunks that starts before this has finished waits for the decode already
    // under way rather than starting another, as long as what is cached is not checked against
    // storage again; see ADR 0015. Checked, cancelled and masked as Read is, so with the pixel mask
    // applied its chunks are decoded too. Safe to call concurrently with Read on one handle.
    Result<std::uint64_t> Prefetch(const ReadRequest& request, const ReadOptions& options = {}) const;

    // The bytes one chunk of this image keeps in a cache once decoded: its elements at the type they
    // are stored as -- not the float a Read hands back -- and, when `options` apply the pixel mask,
    // the flag chunks decoding it brings, whole, at a byte an element. What a caller sizing a cache
    // for chunks it will come back to counts in: counted in floats, a float64 image's cache holds half
    // the chunks it was meant to, and a flagged one's fewer again.
    //
    // The most one chunk can bring. A flag chunked coarser than its image is shared by several pixel
    // chunks and counted beside each, which is the side to be wrong on when what is being sized is
    // room to keep things.
    std::uint64_t DecodedChunkBytes(const ReadOptions& options = {}) const;

    // Reduces every region over the same channels in one pass over the pixels, handing results to
    // the sink block by block.
    //
    // The pass visits each covered chunk once and accumulates every region that touches it, which
    // is the whole point of taking N regions instead of being called N times: a position-velocity
    // cut along the diagonal of a 4096^2 image is 5,792 overlapping boxes, and reducing them one at
    // a time decompresses the same chunks thousands of times over.
    //
    // Results stream rather than accumulate: those 5,792 regions over 30,000 channels would be
    // 2.59 GiB returned at once. Each block is valid only inside the sink call.
    //
    // The image's pixel mask is applied when it has one and ReadOptions::apply_pixel_mask is left
    // on, exactly as it is for a read: a flagged pixel reaches the statistics as NaN, and declining
    // the mask means the flag is never read at all. ReadOptions also supplies cancellation, the
    // deadline, and a ceiling on the pixel buffer the pass may hold.
    Result<void> ReduceSpectral(const SpectralReduceRequest& request, const SpectralSink& sink,
                                const ReadOptions& options = {}) const;

    // Bins every pixel of each plane over a fixed range, handing counts to the sink block by block.
    //
    // Separate from ReduceSpectral because a histogram is not one of the statistics that reduction
    // accumulates, and because it needs none of that machinery: the region is always the whole
    // plane, so there is nothing to index and no region raster to consult. The image's own pixel
    // mask is a different thing and still applies -- see below.
    //
    // The image's pixel mask is applied when it has one and ReadOptions::apply_pixel_mask is left
    // on, so a flagged pixel is not counted -- the same thing that happens to a NaN. Declining the
    // mask counts every stored pixel, flagged or not.
    Result<void> ComputeHistogram(const HistogramRequest& request, const HistogramSink& sink,
                                  const ReadOptions& options = {}) const;

    // One histogram for the whole selection in a single pass, when the range is not known in
    // advance. See CubeHistogramRequest for what that costs and what it keeps exact. The pixel mask
    // is applied on the same terms as ComputeHistogram, and on both passes it makes. A progress
    // callback watches it as it advances; see CubeHistogramProgressCallback.
    Result<CubeHistogramResult> ComputeCubeHistogram(const CubeHistogramRequest& request,
                                                     const ReadOptions& options = {},
                                                     const CubeHistogramProgressCallback& progress = {}) const;

    Result<std::vector<Beam>> ReadBeams() const;

private:
    class Impl;
    explicit Image(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> _impl;

    friend class Dataset;
    // Which cache an image's reads go through, which reading ahead has to hold two runs of it.
    friend class ReadAhead;
};

class CARTA_ZARR_EXPORT Dataset final {
public:
    Dataset(const Dataset&) = default;
    Dataset& operator=(const Dataset&) = default;
    // A copy, so that the handle moved from still refers to what it did. See above.
    Dataset(Dataset&& other) noexcept : Dataset(other) {}
    Dataset& operator=(Dataset&& other) noexcept { return *this = other; }
    ~Dataset();

    static Result<Dataset> Open(const Context& context, std::string_view location);

    const DatasetDescriptor& descriptor() const noexcept;
    // How large this dataset is, and which of two questions the answer is to -- see SizeBasis.
    // Within the timeout, the sum of the sizes of the files the store holds; otherwise the
    // uncompressed size the metadata declares for all arrays.
    //
    // The two are not interchangeable and the second does not bound the first, so a consumer that
    // shows the number to someone should show which one it got.
    //
    // The timeout is not named after a directory because a dataset need not live in one: what can
    // be sized, and how quickly, is the transport's affair.
    Result<DatasetSize> Size(std::chrono::milliseconds stored_size_timeout = std::chrono::milliseconds(50)) const;
    Result<Image> OpenImage(std::string_view image_id) const;

private:
    class Impl;
    explicit Dataset(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> _impl;
};

// Ask one named schema profile about a location. The answer is its SchemaMatchKind -- matched, did
// not match, or matched something malformed -- and an Error only when the store could not be read
// at all.
//
// Three answers rather than a yes or no: a store that matched something malformed is neither, and a
// Result<bool> would also make `if (ProbeSchema(...))` compile and mean "did not fail" rather than
// "yes".
CARTA_ZARR_EXPORT Result<SchemaProbeResult> ProbeSchema(std::string_view location, std::string_view schema_id);

}  // namespace carta::zarr

#endif  // CARTA_ZARR_CARTA_ZARR_H_
