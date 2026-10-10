/* This file is part of the CARTA Image Viewer: https://github.com/CARTAvis/carta-zarr
   Copyright 2026- Academia Sinica Institute of Astronomy and Astrophysics (ASIAA),
   Associated Universities, Inc. (AUI) and the Inter-University Institute for Data Intensive Astronomy (IDIA)
   SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef CARTA_ZARR_SRC_REDUCIBLE_IMAGE_H_
#define CARTA_ZARR_SRC_REDUCIBLE_IMAGE_H_

#include "carta-zarr/descriptor.h"
#include "carta-zarr/read.h"
#include "carta-zarr/reduce.h"
#include "carta-zarr/result.h"

#include "axis_map.h"
#include "pixel_source.h"
#include "reduce/pass_plan.h"
#include "reduce/plane_selection.h"
#include "reduce/task_split.h"
#include "work_pool.h"

namespace carta::zarr::internal {

/**
 * One image, opened, with everything a reduction needs before a request arrives.
 *
 * Four things travelled together to every reduction -- where the pixels come from, what the image
 * is, how it is laid out, and what may run the arithmetic -- and the descriptor travelled twice,
 * once on its own and once inside the source. Three entry points took six or seven arguments of
 * which six were the same six in the same order.
 *
 * It is not only the bundle, because a bundle would just move those arguments rather than absorb
 * them: it also answers what each reduction used to work out again for itself. Where the roles sit
 * among the axes is derived once here rather than three times.
 *
 * An ordinary read is not a reduction and does not come through here. It did, while this was called
 * ReadableImage, and so it was handed an axis map and a pool it never asked anything of, and a read
 * of an image whose axes could not be mapped would have failed for a reason that belonged to the
 * walks. No image that opens today is such an image, which is why that was never seen. ReadInPieces
 * takes the three things it uses; see ADR 0005 for why a read stays outside the pass.
 *
 * Checking a plane selection used to live here too, as `ValidateSpectral`. That was one third of
 * the question in the one place the other two thirds were not; `CheckedPlanes::Of` asks all of it,
 * from the descriptor and the map this already holds. What lives here now is `Plan`, which asks it
 * and plans the pass over the answer in one step, because no reduction ever did one without the
 * other.
 *
 * Holds references and an AxisMap by value, so it is cheap to build per call and owns nothing. It
 * must not outlive the source, the descriptor, the geometries or the pool it was built from.
 */
class ReducibleImage {
public:
    // Reports not_implemented when the image's axes cannot be mapped -- an axis with no known role
    // that is not degenerate, or a missing spatial or spectral axis. Built per call, so that
    // failure reaches the caller of the reduction that needed it and no other.
    static Result<ReducibleImage> Of(const PixelSource& source, const ImageDescriptor& descriptor,
                                     const ChunkGeometry& geometry, const ChunkGeometry& flag_geometry,
                                     WorkPool& workers) {
        auto map = MapAxes(descriptor);
        if (!map) {
            return map.error();
        }
        return ReducibleImage(source, descriptor, geometry, flag_geometry, workers, map.value());
    }

    const PixelSource& source() const noexcept { return *_source; }
    const ImageDescriptor& descriptor() const noexcept { return *_descriptor; }
    const AxisMap& map() const noexcept { return _map; }

    // The pass a reduction makes over `planes`, once they are checked against this image: every
    // reduction asked CheckedPlanes::Of and then PlanPass, from the same descriptor, geometry and
    // map, in the same order, and did nothing between them. The plan keeps the checked planes, so
    // the answer is the one value a reduction needs afterwards. `sample` takes every nth pixel along
    // both spatial axes; one reads them all.
    //
    // Reports invalid_argument for planes this image cannot serve; see CheckedPlanes::Of. What a
    // request asks beyond its planes -- bins, bounds, regions, a sink -- stays with the reduction
    // that understands it.
    Result<PassPlan> Plan(const PlaneSelection& planes, std::uint64_t sample, const ReadOptions& options) const {
        const auto checked = CheckedPlanes::Of(*_descriptor, _map, planes);
        if (!checked) {
            return checked.error();
        }
        return PlanPass(*_descriptor, *_geometry, *_flag_geometry, _map, checked.value(), sample, options,
                        _workers->size());
    }

    // How a reduction whose tasks each hold an accumulator of `accumulator_bytes` divides a read
    // among the pool, with `budget_bytes` for all of them together. The only way a reduction reaches
    // the pool, so that the cap ADR 0005 turns on is applied wherever tasks are run. See TaskSplit.
    TaskSplit Split(std::size_t budget_bytes, std::size_t accumulator_bytes) const {
        return TaskSplit(*_workers, budget_bytes, accumulator_bytes);
    }

private:
    ReducibleImage(const PixelSource& source, const ImageDescriptor& descriptor, const ChunkGeometry& geometry,
                   const ChunkGeometry& flag_geometry, WorkPool& workers, const AxisMap& map)
        : _source(&source),
          _descriptor(&descriptor),
          _geometry(&geometry),
          _flag_geometry(&flag_geometry),
          _workers(&workers),
          _map(map) {}

    const PixelSource* _source;
    const ImageDescriptor* _descriptor;
    const ChunkGeometry* _geometry;
    // The flag's, which a masked pass's reads are sized by beside the pixels'. See DescribedImage.
    const ChunkGeometry* _flag_geometry;
    WorkPool* _workers;
    AxisMap _map;
};

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_REDUCIBLE_IMAGE_H_
