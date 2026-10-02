/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// The half of an ordinary read that reads. See pieces.cc for why the two are apart.
//
// It names no Store: pixels arrive through the PixelSource seam, so everything under src/read/
// compiles without one, the same way src/reduce/ already did.

#include "read/pieces.h"

#include "pixel_mask.h"
#include "zarr/pixel_selection.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace carta::zarr::internal {
namespace {

// What every pixel read of an image asks before it touches storage.
//
// Written out twice until this: once here and once in the facade, where the mask read built its own
// selection, sized its own buffer check and made its own control check on the way to the pixel seam.
// The three are one question -- can this image serve this request into this buffer, now -- and a
// read that answered two of them would be a read that had not checked.
Result<zarr::PixelSelection> CheckRead(const ImageDescriptor& descriptor, const ReadRequest& request,
                                       std::size_t destination_size, const ReadControl& control) {
    auto selection = zarr::BuildSelection(descriptor, request, zarr::DestinationOrder::logical);
    if (!selection) {
        return selection.error();
    }
    if (selection.value().elements() > destination_size) {
        return Error{ErrorCode::invalid_argument, "Destination buffer is too small for the request",
                     descriptor.id};
    }

    // Before allocating a mask or starting any storage work. A cancelled request must not consume
    // temporary memory, or open an array, just to discover that it cannot proceed.
    if (auto allowed = zarr::CheckReadControl(control, descriptor.id); !allowed) {
        return allowed.error();
    }
    return selection;
}

}  // namespace

Result<std::size_t> ReadInPieces(const PixelSource& source, const ImageDescriptor& descriptor,
                                 const ChunkGeometry& geometry, const ReadRequest& request,
                                 BufferView<float> destination, const ReadOptions& options,
                                 const ProgressCallback& progress) {
    const auto checked = CheckRead(descriptor, request, destination.size, options.control);
    if (!checked) {
        return checked.error();
    }
    // The whole read's element count is what a piece's share is measured against; the selection each
    // piece is actually read through is built per piece below.
    const auto elements = checked.value().elements();

    const bool apply_mask = AppliesPixelMask(options, descriptor);
    const auto pieces = PlanPieces(descriptor, geometry, request, options, static_cast<bool>(progress));

    std::vector<std::uint8_t> mask;
    for (const auto& piece : pieces) {
        auto piece_selection = zarr::BuildSelection(descriptor, piece.request, zarr::DestinationOrder::logical);
        if (!piece_selection) {
            return piece_selection.error();
        }
        const auto piece_elements = static_cast<std::size_t>(piece_selection.value().elements());
        // The rest of the caller's buffer from where this piece lands, not the piece's own size: the
        // seam holds the one to the other, and handing it the piece's size would have it compare the
        // piece with itself. first_element is short of the whole read's count, which CheckRead held
        // to the buffer, so this never runs backwards.
        const BufferView<float> piece_destination{destination.data + piece.first_element,
                                                  destination.size - static_cast<std::size_t>(piece.first_element)};

        if (apply_mask) {
            // The budget bounds the flag a piece holds, and a read that is not split is one piece,
            // so a request that cannot be cut any further still has to say so rather than allocate.
            if (options.read_budget_bytes != 0 &&
                piece_elements > options.read_budget_bytes) {
                return Error{ErrorCode::buffer_too_small,
                             "Pixel mask temporary buffer exceeds the read budget",
                             descriptor.id};
            }
            mask.assign(piece_elements, 0);
            // The mask is read first so that an unavailable or cancelled mask cannot leave this
            // piece of the destination updated. TensorStore still owns the pixel operation's
            // in-flight completion before it returns, so the destination remains valid for the next
            // read.
            auto mask_read = source.ReadMask(piece_selection.value(), {mask.data(), mask.size()}, options.control);
            if (!mask_read) {
                return mask_read.error();
            }
        }
        auto read = source.ReadPixels(piece_selection.value(), piece_destination, options.control);
        if (!read) {
            return read.error();
        }
        if (apply_mask) {
            ApplyPixelMask(piece_destination.data, mask.data(), piece_elements);
        }

        const auto finished = static_cast<std::size_t>(piece.first_element + piece_elements);
        if (progress && !progress(finished, static_cast<std::size_t>(elements))) {
            return Error{ErrorCode::cancelled, "The read was cancelled by its progress callback",
                         descriptor.id};
        }
    }
    return static_cast<std::size_t>(elements);
}

Result<std::uint64_t> PrefetchChunks(const PixelSource& source, const ImageDescriptor& descriptor,
                                     const ChunkGeometry& geometry, const ReadRequest& request,
                                     const ReadOptions& options) {
    // Checked as the request the caller made, so that a mistake in it is reported in its own terms
    // rather than in those of the sample made from it.
    if (auto checked = zarr::BuildSelection(descriptor, request, zarr::DestinationOrder::logical); !checked) {
        return checked.error();
    }
    const auto sample = OneElementPerChunk(geometry, request);
    std::uint64_t chunks = 1;
    for (const auto& range : sample.axes) {
        chunks *= range.count;
    }
    std::vector<float> discarded(static_cast<std::size_t>(chunks));
    auto read = ReadInPieces(source, descriptor, geometry, sample, {discarded.data(), discarded.size()}, options, {});
    if (!read) {
        return read.error();
    }
    return chunks;
}

}  // namespace carta::zarr::internal
