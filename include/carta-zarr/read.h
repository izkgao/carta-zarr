/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_ZARR_READ_H_
#define CARTA_ZARR_READ_H_

// Asking for pixels: which of them, into what, and under what limits.

#include "carta-zarr/descriptor.h"
#include "carta-zarr/export.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace carta::zarr {

struct Range {
    std::uint64_t start = 0;
    std::uint64_t count = 0;
    std::uint64_t stride = 1;
};

struct ReadRequest {
    // One range per ImageDescriptor::axes entry, in the same order.
    std::vector<Range> axes;
};

namespace internal {
struct CachePoolAccess;
}  // namespace internal

// A decoded-chunk cache of a read's own, apart from the one its Context shares among every read.
//
// What a read keeps decides what the next one has to decode again. The shared cache holds a session's
// interactive working set -- the planes and profiles being looked at -- and two kinds of read do
// better elsewhere. A scan that touches every chunk once reuses none of them, so caching what it
// decodes evicts that working set to no purpose, and rebuilding it costs decompression, which is the
// resource the scan is already saturating: it reads through a pool of zero bytes. A walk that does
// come back -- a moment, whose neighbouring slabs share the chunks between them -- needs as much as it
// will come back for, which is more than a session should keep for everything else and is wanted only
// while the walk lasts: it reads through a pool of that size, and lets go of it when it ends.
//
// Made by Context::NewCachePool, and a shared handle like Context: copying one shares the pool, and
// it is the last copy let go of that frees what the pool holds. The pool runs on its Context's
// threads, so a read through it takes its turn among the session's reads rather than competing with
// them. Arrays are opened per pool, so the first read of an array through a new one pays to open it
// again -- a few milliseconds, once per array, against a walk that reads all of it.
class CARTA_ZARR_EXPORT CachePool final {
public:
    CachePool(const CachePool&) = default;
    CachePool& operator=(const CachePool&) = default;
    // A copy, so that the handle moved from still refers to what it did. See ADR 0013.
    CachePool(CachePool&& other) noexcept : CachePool(other) {}
    CachePool& operator=(CachePool&& other) noexcept { return *this = other; }
    // Inline, unlike the other handles': ReadControl holds one, and the parts of this library that
    // are built and tested without the rest of it take a ReadControl. A shared_ptr's deleter is fixed
    // where it was made, so nothing here needs Impl to be complete.
    ~CachePool() = default;

    // How much decoded chunk data the pool may hold, in bytes, as it was asked for.
    std::size_t bytes() const noexcept;

private:
    class Impl;
    explicit CachePool(std::shared_ptr<const Impl> impl);

    std::shared_ptr<const Impl> _impl;

    friend class Context;
    friend struct internal::CachePoolAccess;
};

/// Called as a read advances, with the number of destination elements that are final and the number
/// the request will produce in total. Returning false cancels the read, which then reports cancelled.
///
/// A watched read is cut into chunk-aligned pieces along the slowest-varying selected axis, as many as
/// ReadOptions::read_budget_bytes needs, and this is called as each piece is finished. The
/// destination is dense in logical order with axis 0 fastest, which is what makes the finished part
/// a prefix rather than a scatter -- a caller can render or forward it as it arrives. A piece too
/// large to read whole is read in parts and reported when all of them are in, so the prefix holds.
///
/// An argument of Image::Read rather than a field of ReadOptions, because it is the only operation
/// that has anywhere to report from in these terms -- a reduction reports through its sink, and a
/// cube histogram through a CubeHistogramProgressCallback of its own. As a field it would be one the
/// other entry points silently ignored; as an argument it is simply not part of what they take.
///
/// Supplying one can cut a read that would otherwise be issued whole: a read nobody watches is not
/// cut while the chunks decoded at once, no more than one a decode thread, fit its budget, and one that
/// is watched is cut to the chunks the budget affords so that there is a piece to report. On large
/// chunks the two are cut alike; on small ones the watched read is somewhat slower.
using ProgressCallback = std::function<bool(std::size_t elements_written, std::size_t elements_total)>;

// What a read is allowed to do while it runs, whatever it is reading for.
//
// These three are the whole of what every path through this library honours: an ordinary read and
// all three reductions reach the same storage operations underneath and check the same things at
// the same boundaries. Said in its own type so that an operation which honours only these can take
// only these -- everything below the pixel source seam does, because that is all any of it ever
// looked at.
struct ReadControl {
    // Cooperative cancellation checked before and after each storage operation. The callback
    // must be safe to invoke from the calling thread.
    std::function<bool()> cancellation_requested;
    // A steady-clock deadline checked at the same storage-operation boundaries. An in-flight
    // TensorStore operation is not interrupted, but a request never starts another operation once
    // this deadline has passed.
    std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max();
    // The cache this read keeps what it decodes in, and finds what an earlier read kept: its Context's
    // shared one when there is none. See CachePool.
    //
    // One field rather than a policy beside a pool, so that "bypass the cache" and "use this pool"
    // cannot both be said at once: declining the cache is a pool of zero bytes.
    std::optional<CachePool> cache_pool;
};

// Everything a read of pixels takes, on top of what any read takes.
//
// The two fields here are the ones that mean something only when pixels are being read into a
// buffer this library sized: whether a flag is folded in on the way, and how much the library may
// hold at once while doing it. A pixel mask read has neither -- it is the flag, and the destination
// is the caller's -- which is why it takes a ReadControl and this cannot be handed to it.
//
// That is the point of the split. Every field of this type is honoured by every operation that
// takes it, so there is no table of which ones apply where, and setting one where it would have
// been ignored does not compile.
struct ReadOptions {
    ReadControl control;
    // Write NaN wherever the image's flag marks a pixel -- a nonzero flag value, XRADIO's true, which
    // means the pixel is bad -- so that one call answers what would otherwise be a pixel read plus a
    // flag read. On by default: masking during the read costs one pass over data already in hand,
    // while a caller doing it afterwards pays for a second traversal.
    bool apply_pixel_mask = true;
    /// How much memory one read may hold at once beyond the caller's own destination, in bytes. Zero
    /// means the library's own budget: two chunks for every decode thread of the image's context,
    /// between 256 MiB and 2 GiB.
    ///
    /// What a read holds is the chunks it is decoding -- each about three times what it decodes to,
    /// for its compressed bytes and the codec's buffer beside the decoded copy -- and the buffers the
    /// library allocates for them: the folded-in pixel mask, a byte an element, and the pixels
    /// themselves wherever the library rather than the caller holds them. Every operation that takes
    /// these options spends it the same way, and every one keeps to it whether or not anybody watches:
    /// Image::Read and Image::Prefetch cut their request into pieces that fit, and ReduceSpectral,
    /// ComputeHistogram and ComputeCubeHistogram size each read of their walk by it. A read holds no
    /// more chunks than are decoded at once, at most one a decode thread, so one whose budget affords
    /// that many is held to it without being cut; see ProgressCallback.
    ///
    /// A chunk is the smallest thing that can be decoded: asking for part of one decodes all of it.
    /// So a read holds at least one chunk, and under a budget smaller than that it reads one chunk at
    /// a time and holds what that chunk does rather than refusing. Image::DecodedChunkBytes says in
    /// advance what a chunk decodes to.
    ///
    /// The budget is per read. Reads running at once each hold their own.
    std::size_t read_budget_bytes = 0;
};

// A run of elements the caller owns and lends for one call, counted in elements rather than in
// bytes: somewhere for a read to write, or, as BufferView<const T>, something for a reduction to read
// -- its regions and each region's raster -- whose length the library checks rather than assumes.
//
// Typed because the element type is not the caller's to choose: pixels arrive as float, and a pixel
// mask as one byte per pixel. An untyped view with a byte count could express neither fact, so the
// same field meant "bytes" at one entry point and "elements" at the other, and a buffer of the
// wrong kind was a run-time error at best.
//
// The library never holds one past the call it was passed to.
template <typename T>
struct BufferView {
    T* data = nullptr;
    std::size_t size = 0;  // elements, not bytes
};

}  // namespace carta::zarr

#endif  // CARTA_ZARR_READ_H_
