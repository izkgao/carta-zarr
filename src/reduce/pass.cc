/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "reduce/pass_plan.h"
#include "reduce/tuning.h"

namespace carta::zarr::internal {

PassPlan PlanPass(const ImageDescriptor& descriptor, const ChunkGeometry& geometry, const ChunkGeometry& flag_geometry,
                  const AxisMap& map, const CheckedPlanes& planes, std::uint64_t sample, const ReadOptions& options,
                  std::size_t decode_threads) {
    const Range spectral = planes.spectral();
    PassPlan plan;
    plan.descriptor = &descriptor;
    plan.map = map;
    // The spatial axis the store varies fastest is the one to ask for first: reading a plane with
    // the other one fastest means transposing every chunk on the way into the destination.
    const bool swap_spatial = geometry.fastest_spatial_axis == AxisRole::spatial_y;
    plan.axis_u = swap_spatial ? map.y : map.x;
    plan.axis_v = swap_spatial ? map.x : map.y;
    plan.u_length = descriptor.axes.at(plan.axis_u).length;
    plan.v_length = descriptor.axes.at(plan.axis_v).length;
    plan.chunk_u = std::max<std::uint64_t>(1, geometry.chunk_shape.at(plan.axis_u));
    plan.chunk_v = std::max<std::uint64_t>(1, geometry.chunk_shape.at(plan.axis_v));
    plan._chunk_depth = std::max<std::uint64_t>(1, geometry.chunk_shape.at(map.spectral));
    const auto cost =
        ReadCost::Of(descriptor, geometry, flag_geometry, options, PixelsHeld::by_library, decode_threads);
    plan.apply_mask = cost.apply_mask;
    plan.chunk_bytes = cost.chunk_bytes;
    plan.held_bytes = cost.held_bytes;
    plan.slab_budget_bytes = cost.budget_bytes;
    // Rounded up without adding stride - 1, which wraps for a stride near the top of the range.
    plan._least_channels =
        (plan._chunk_depth / spectral.stride) + static_cast<std::uint64_t>(plan._chunk_depth % spectral.stride != 0);
    plan.planes = planes.selection();
    plan.sample = std::max<std::uint64_t>(1, sample);
    // The chunks of a layer the sample has a pixel in, which is every chunk unless it steps over some:
    // what a read decodes, and so what the budget and progress are both counted in.
    const auto touched = [&](std::uint64_t length, std::uint64_t chunk) {
        return length == 0 ? 0 : ChunksTouched(0, ((length - 1) / plan.sample) + 1, plan.sample, chunk);
    };
    const std::uint64_t row_chunks = std::max<std::uint64_t>(1, touched(plan.u_length, plan.chunk_u));
    const std::uint64_t column_chunks = std::max<std::uint64_t>(1, touched(plan.v_length, plan.chunk_v));
    plan.layer_chunks = std::max<std::uint64_t>(1, row_chunks * column_chunks);
    // How many chunk rows one read may hold, so that a read is a budget's worth of chunk data.
    plan.band_rows = plan.UnitsAffordable(row_chunks);
    return plan;
}

std::uint64_t PassPlan::EmitChannels(std::uint64_t layer_chunks, std::size_t bytes_per_channel,
                                     std::uint32_t hint) const {
    const std::uint64_t budget_channels =
        std::max<std::uint64_t>(1, kSpectralEmitBudgetBytes / std::max<std::size_t>(1, bytes_per_channel));
    // A region set that occupies nothing costs nothing per layer, so the spatial walk leaves the whole
    // read budget and nothing bounds the block but the emit budget and the hint. Said here rather than
    // left to UnitsAffordable, whose units are at least one chunk.
    const std::uint64_t spatial_channels =
        layer_chunks == 0 ? planes.spectral.count
                          : std::min(planes.spectral.count, UnitsAffordable(layer_chunks)) * _least_channels;
    return std::min(
        {hint == 0 ? spatial_channels : static_cast<std::uint64_t>(hint), budget_channels, planes.spectral.count});
}

}  // namespace carta::zarr::internal
