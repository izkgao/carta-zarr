/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "plane_histogram.h"

#include "chunk_blocks.h"
#include "reduce/axis_map.h"
#include "zarr/pixel_reader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace carta::zarr::internal {
namespace {

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

Result<void> ValidateRequest(const ImageDescriptor& descriptor, const AxisMap& axes,
                             const HistogramRequest& request) {
    const auto& node = descriptor.id;
    if (request.bins == 0 || request.bins > kMaxHistogramBins) {
        return MakeError(ErrorCode::invalid_argument,
                         "A histogram needs between 1 and " + std::to_string(kMaxHistogramBins) + " bins",
                         node);
    }
    // A zero-width range would divide by zero on every pixel. The caller decides what an image with
    // no finite pixel should look like -- there is more than one defensible answer -- so this says
    // no rather than inventing one.
    if (!(request.lower < request.upper) || !std::isfinite(request.lower) || !std::isfinite(request.upper)) {
        return MakeError(ErrorCode::invalid_argument,
                         "A histogram needs a finite range with a lower bound below its upper bound", node);
    }
    if (!axes.has_polarization && request.polarization != 0) {
        return MakeError(ErrorCode::invalid_argument, "The image has no polarization axis to select", node);
    }
    if (axes.has_polarization && request.polarization >= descriptor.axes.at(axes.polarization).length) {
        return MakeError(ErrorCode::invalid_argument, "The polarization index is outside the image", node);
    }
    if (!axes.has_time && request.time != 0) {
        return MakeError(ErrorCode::invalid_argument, "The image has no time axis to select", node);
    }
    if (axes.has_time && request.time >= descriptor.axes.at(axes.time).length) {
        return MakeError(ErrorCode::invalid_argument, "The time index is outside the image", node);
    }
    return {};
}

}  // namespace

Result<void> ComputeHistogram(const Store& store, const ImageDescriptor& descriptor,
                              const ChunkGeometry& geometry, const HistogramRequest& request,
                              const HistogramSink& sink, const ReadOptions& options) {
    const auto& node = descriptor.id;
    if (!sink) {
        return MakeError(ErrorCode::invalid_argument, "A histogram needs a sink", node);
    }
    auto axes = MapAxes(descriptor);
    if (!axes) {
        return axes.error();
    }
    const auto& map = axes.value();
    if (auto valid = ValidateRequest(descriptor, map, request); !valid) {
        return valid.error();
    }

    const Range spectral = request.spectral;
    const auto channels = descriptor.axes.at(map.spectral).length;
    if (spectral.stride == 0 || spectral.count == 0 || spectral.start >= channels ||
        (spectral.count - 1) * spectral.stride > channels - 1 - spectral.start) {
        return MakeError(ErrorCode::invalid_argument, "The spectral range falls outside the image", node);
    }

    // The same rule the reduction follows: walk the spatial axis the store varies fastest, so that a
    // plane arrives without being transposed. Binning does not care what order it sees pixels in,
    // which makes the transpose pure cost here.
    const bool swap_spatial = SpatialYIsFastest(geometry);
    const auto axis_u = swap_spatial ? map.y : map.x;
    const auto axis_v = swap_spatial ? map.x : map.y;
    const auto u_length = descriptor.axes.at(axis_u).length;
    const auto v_length = descriptor.axes.at(axis_v).length;
    const auto chunk_u = std::max<std::uint64_t>(1, geometry.chunk_shape.at(axis_u));
    const auto chunk_v = std::max<std::uint64_t>(1, geometry.chunk_shape.at(axis_v));
    const auto chunk_depth = std::max<std::uint64_t>(1, geometry.chunk_shape.at(map.spectral));

    const bool apply_mask = options.apply_pixel_mask && descriptor.has_pixel_mask;
    const std::uint64_t chunk_bytes = DecodedChunkBytes(descriptor, geometry) * (apply_mask ? 2 : 1);
    const std::size_t slab_budget_bytes = options.temporary_memory_limit_bytes != 0
                                              ? options.temporary_memory_limit_bytes
                                              : DefaultReadBytes(chunk_bytes);
    const std::uint64_t least_channels = ((chunk_depth + spectral.stride - 1) / spectral.stride);
    const std::uint64_t layer_chunks =
        std::max<std::uint64_t>(1, (((u_length - 1) / chunk_u) + 1) * (((v_length - 1) / chunk_v) + 1));

    // Emit granularity, as in the reduction: without a hint a block costs one read budget, so it is
    // as often as the walk can report without making any read smaller.
    const std::size_t bytes_per_channel = static_cast<std::size_t>(request.bins) * sizeof(std::uint64_t);
    const std::uint64_t budget_channels =
        std::max<std::uint64_t>(1, kSpectralEmitBudgetBytes / std::max<std::size_t>(1, bytes_per_channel));
    const std::uint64_t block_chunks = std::min(
        spectral.count,
        std::max<std::uint64_t>(1, slab_budget_bytes / std::max<std::uint64_t>(1, layer_chunks * chunk_bytes)));
    const std::uint64_t wanted_channels =
        std::min({request.emit_every_channels == 0 ? block_chunks * least_channels
                                                   : static_cast<std::uint64_t>(request.emit_every_channels),
                  budget_channels, spectral.count});

    // How many chunk rows one read may hold, so that a read is a budget's worth of chunk data.
    const std::uint64_t row_chunks = std::max<std::uint64_t>(1, ((u_length - 1) / chunk_u) + 1);
    const std::uint64_t band_rows =
        std::max<std::uint64_t>(1, slab_budget_bytes / std::max<std::uint64_t>(1, row_chunks * chunk_bytes));

    // The caller's own sequence: divide in double, narrow the width, compare against the narrowed
    // bounds. Doing any one of those in the other type moves pixels across bin edges.
    const float width = static_cast<float>((request.upper - request.lower) / request.bins);
    const float lower = static_cast<float>(request.lower);
    const float upper = static_cast<float>(request.upper);
    const auto bins = static_cast<std::size_t>(request.bins);
    const auto rank = descriptor.axes.size();
    std::vector<std::uint64_t> counts;
    std::vector<float> pixels;
    std::vector<std::uint8_t> mask;

    for (std::uint64_t block_begin = 0; block_begin < spectral.count;) {
        const std::uint64_t block_end = AlignedBlockEnd(block_begin, wanted_channels, spectral.count,
                                                        spectral.start, spectral.stride, chunk_depth);
        const auto block_length = static_cast<std::size_t>(block_end - block_begin);
        counts.assign(block_length * bins, 0);

        const std::uint64_t block_spectral_chunks = (block_length + least_channels - 1) / least_channels;
        const std::uint64_t block_chunks_total = std::max<std::uint64_t>(1, layer_chunks * block_spectral_chunks);
        std::uint64_t block_chunks_done = 0;
        std::uint64_t reads_done = 0;

        const auto hand_over = [&](bool complete) -> Result<void> {
            HistogramBlock block;
            block.first_channel = block_begin;
            block.channel_count = block_length;
            block.counts = counts.data();
            block.bin_count = bins;
            block.complete = complete;
            block.completeness =
                complete ? 1.0
                         : static_cast<double>(block_chunks_done) / static_cast<double>(block_chunks_total);
            if (!sink(block)) {
                return MakeError(ErrorCode::cancelled, "The histogram was cancelled by its sink", node);
            }
            return {};
        };

        for (std::uint64_t v_begin = 0; v_begin < v_length;) {
            const std::uint64_t v_end = std::min(v_length, v_begin + (band_rows * chunk_v));
            const std::uint64_t v_count = v_end - v_begin;
            const std::uint64_t band_chunks =
                std::max<std::uint64_t>(1, row_chunks * (((v_count - 1) / chunk_v) + 1));
            const std::uint64_t spectral_chunks =
                std::max<std::uint64_t>(1, slab_budget_bytes / (band_chunks * chunk_bytes));
            const std::uint64_t slab_channels = std::max<std::uint64_t>(1, spectral_chunks * least_channels);

            for (std::uint64_t slab_begin = block_begin; slab_begin < block_end;) {
                const std::uint64_t slab_end =
                    AlignedBlockEnd(slab_begin, std::min(slab_channels, block_end - slab_begin), block_end,
                                    spectral.start, spectral.stride, chunk_depth);
                const std::uint64_t slab_length = slab_end - slab_begin;

                // What is in hand before spending another budget on this block.
                if (reads_done > 0) {
                    if (auto handed = hand_over(false); !handed) {
                        return handed.error();
                    }
                }
                ++reads_done;

                if (auto control = zarr::CheckReadControl(options, node); !control) {
                    return control.error();
                }

                ReadRequest read_request;
                read_request.axes.assign(rank, Range{0, 1, 1});
                read_request.axes.at(axis_u) = Range{0, u_length, 1};
                read_request.axes.at(axis_v) = Range{v_begin, v_count, 1};
                read_request.axes.at(map.spectral) =
                    Range{spectral.start + (slab_begin * spectral.stride), slab_length, spectral.stride};
                if (map.has_polarization) {
                    read_request.axes.at(map.polarization) = Range{request.polarization, 1, 1};
                }
                if (map.has_time) {
                    read_request.axes.at(map.time) = Range{request.time, 1, 1};
                }

                auto selection = zarr::BuildSelection(descriptor, read_request);
                if (!selection) {
                    return selection.error();
                }
                for (std::size_t i = 0; i < rank; ++i) {
                    selection.value().logical_to_stored.at(i) = rank - 1 - i;
                }

                std::vector<std::uint64_t> stored_stride(rank, 1);
                std::uint64_t running = 1;
                for (std::size_t stored = rank; stored-- > 0;) {
                    stored_stride.at(stored) = running;
                    running *= selection.value().count.at(stored);
                }
                const std::uint64_t stride_u = stored_stride.at(descriptor.axes.at(axis_u).storage_index);
                const std::uint64_t stride_v = stored_stride.at(descriptor.axes.at(axis_v).storage_index);
                const std::uint64_t stride_z = stored_stride.at(descriptor.axes.at(map.spectral).storage_index);
                const auto elements = static_cast<std::size_t>(running);

                pixels.resize(elements);
                if (auto read = store.ReadPixelsFloat32(descriptor.id, selection.value(), pixels.data(),
                                                        pixels.size(), options);
                    !read) {
                    return read.error();
                }
                if (apply_mask) {
                    mask.resize(elements);
                    if (auto read = store.ReadPixelMaskBytes(descriptor.pixel_mask_id, selection.value(),
                                                             mask.data(), mask.size(), options);
                        !read) {
                        return read.error();
                    }
                    for (std::size_t i = 0; i < elements; ++i) {
                        if (mask[i] == 0) {
                            pixels[i] = std::numeric_limits<float>::quiet_NaN();
                        }
                    }
                }

                for (std::uint64_t channel = 0; channel < slab_length; ++channel) {
                    std::uint64_t* into =
                        counts.data() + (static_cast<std::size_t>(slab_begin - block_begin + channel) * bins);
                    const float* plane = pixels.data() + (channel * stride_z);
                    for (std::uint64_t v = 0; v < v_count; ++v) {
                        const float* row = plane + (v * stride_v);
                        for (std::uint64_t u = 0; u < u_length; ++u) {
                            const float value = row[u * stride_u];
                            // The caller's own rule, in the caller's own type: a pixel outside the
                            // range is not counted, and NaN fails both comparisons.
                            if (lower <= value && value <= upper) {
                                auto bin = static_cast<std::size_t>((value - lower) / width);
                                if (bin >= bins) {
                                    bin = bins - 1;
                                }
                                ++into[bin];
                            }
                        }
                    }
                }

                block_chunks_done += band_chunks * ((slab_length + least_channels - 1) / least_channels);
                slab_begin = slab_end;
            }
            v_begin = v_end;
        }

        if (auto handed = hand_over(true); !handed) {
            return handed.error();
        }
        block_begin = block_end;
    }
    return {};
}

}  // namespace carta::zarr::internal
