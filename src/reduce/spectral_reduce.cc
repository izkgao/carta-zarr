/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "spectral_reduce.h"

#include "chunk_blocks.h"
#include "zarr/pixel_reader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace carta::zarr::internal {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// A region is bucketed once per chunk its bounding box touches. This bound turns a request whose
// regions each cover the whole image -- legal, just not what this API is for -- into an error
// rather than an allocation nothing can serve.
constexpr std::size_t kMaxChunkIncidences = 1u << 26;

Error MakeError(ErrorCode code, std::string message, std::string node_path = {}) {
    return Error{code, std::move(message), std::move(node_path)};
}

// Where each axis role sits in the logical axis order.
struct AxisMap {
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t spectral = 0;
    bool has_polarization = false;
    std::size_t polarization = 0;
    bool has_time = false;
    std::size_t time = 0;
};

Result<AxisMap> MapAxes(const ImageDescriptor& descriptor) {
    AxisMap map;
    bool has_x = false;
    bool has_y = false;
    bool has_spectral = false;
    for (std::size_t i = 0; i < descriptor.axes.size(); ++i) {
        const auto& axis = descriptor.axes.at(i);
        switch (axis.role) {
            case AxisRole::spatial_x: map.x = i; has_x = true; break;
            case AxisRole::spatial_y: map.y = i; has_y = true; break;
            case AxisRole::spectral: map.spectral = i; has_spectral = true; break;
            case AxisRole::polarization: map.polarization = i; map.has_polarization = true; break;
            case AxisRole::time: map.time = i; map.has_time = true; break;
            case AxisRole::other:
                // Taking index 0 of an axis nobody named would report a number for a plane the
                // caller never asked about.
                if (axis.length != 1) {
                    return MakeError(ErrorCode::not_implemented,
                                     "Axis '" + axis.name + "' has no known role and is not degenerate",
                                     descriptor.id);
                }
                break;
        }
    }
    if (!has_x || !has_y || !has_spectral) {
        return MakeError(ErrorCode::not_implemented,
                         "A spectral reduction needs both spatial axes and a spectral axis", descriptor.id);
    }
    return map;
}

// Regions indexed by the chunks they touch, in compressed-row form over the bounding box's chunk
// grid. Without it the walk is chunks x regions intersection tests -- 6.5 x 10^8 for a WSU
// diagonal -- which would put the region count back on the critical path this API exists to clear.
struct ChunkBuckets {
    // The union bounding box of every region, in pixels.
    std::uint64_t x0 = 0;
    std::uint64_t y0 = 0;
    std::uint64_t x1 = 0;
    std::uint64_t y1 = 0;
    // The chunk grid covering that box.
    std::uint64_t chunk_x0 = 0;
    std::uint64_t chunk_y0 = 0;
    std::uint64_t columns = 0;
    std::uint64_t rows = 0;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint32_t> entries;

    std::size_t Cell(std::uint64_t chunk_x, std::uint64_t chunk_y) const {
        return static_cast<std::size_t>(((chunk_y - chunk_y0) * columns) + (chunk_x - chunk_x0));
    }
};

Result<ChunkBuckets> BuildChunkBuckets(const RegionMask* regions, std::size_t region_count,
                                       std::uint64_t chunk_width, std::uint64_t chunk_height,
                                       const std::string& node) {
    ChunkBuckets buckets;
    buckets.x0 = std::numeric_limits<std::uint64_t>::max();
    buckets.y0 = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < region_count; ++i) {
        const auto& region = regions[i];
        buckets.x0 = std::min(buckets.x0, region.x_start);
        buckets.y0 = std::min(buckets.y0, region.y_start);
        buckets.x1 = std::max(buckets.x1, region.x_start + region.width);
        buckets.y1 = std::max(buckets.y1, region.y_start + region.height);
    }

    buckets.chunk_x0 = buckets.x0 / chunk_width;
    buckets.chunk_y0 = buckets.y0 / chunk_height;
    buckets.columns = ((buckets.x1 - 1) / chunk_width) - buckets.chunk_x0 + 1;
    buckets.rows = ((buckets.y1 - 1) / chunk_height) - buckets.chunk_y0 + 1;

    const auto cells = static_cast<std::size_t>(buckets.columns * buckets.rows);
    if (cells > std::numeric_limits<std::uint32_t>::max()) {
        return MakeError(ErrorCode::invalid_argument,
                         "The regions span more chunks than one reduction can index", node);
    }
    buckets.offsets.assign(cells + 1, 0);

    // The chunk span of one region's bounding box.
    const auto span = [&](const RegionMask& region, std::uint64_t& cx0, std::uint64_t& cx1, std::uint64_t& cy0,
                          std::uint64_t& cy1) {
        cx0 = region.x_start / chunk_width;
        cx1 = (region.x_start + region.width - 1) / chunk_width;
        cy0 = region.y_start / chunk_height;
        cy1 = (region.y_start + region.height - 1) / chunk_height;
    };

    // The chunks a region actually occupies, which is not the same as the chunks its bounding box
    // covers. A thin rectangle laid along the diagonal has a bounding box the size of the image and
    // sets a thousandth of it; bucketing by the box would read every chunk to reach the band. The
    // mask decides instead, at the cost of one pass over it -- bytes the caller already holds, and
    // a fraction of what reading the chunks they would otherwise stand for costs.
    //
    // A null mask means the whole bounding box, so there is nothing to narrow and nothing to scan.
    std::vector<std::uint8_t> occupied;
    const auto for_each_cell = [&](const RegionMask& region, auto&& visit) {
        std::uint64_t cx0 = 0, cx1 = 0, cy0 = 0, cy1 = 0;
        span(region, cx0, cx1, cy0, cy1);
        if (region.mask == nullptr) {
            for (auto cy = cy0; cy <= cy1; ++cy) {
                for (auto cx = cx0; cx <= cx1; ++cx) {
                    visit(cx, cy);
                }
            }
            return;
        }

        const std::uint64_t columns = cx1 - cx0 + 1;
        const std::uint64_t x_end = region.x_start + region.width;
        occupied.assign(static_cast<std::size_t>(columns), 0);
        for (auto cy = cy0; cy <= cy1; ++cy) {
            std::fill(occupied.begin(), occupied.end(), std::uint8_t{0});
            std::uint64_t found = 0;
            const std::uint64_t y_first = std::max(region.y_start, cy * chunk_height);
            const std::uint64_t y_last = std::min(region.y_start + region.height, (cy + 1) * chunk_height);
            for (std::uint64_t y = y_first; y < y_last && found < columns; ++y) {
                const std::uint8_t* row = region.mask + ((y - region.y_start) * region.width);
                std::uint64_t x = region.x_start;
                while (x < x_end) {
                    const std::uint64_t column = x / chunk_width;
                    const std::uint64_t boundary = std::min(x_end, (column + 1) * chunk_width);
                    auto& seen = occupied.at(static_cast<std::size_t>(column - cx0));
                    if (seen == 0) {
                        for (std::uint64_t k = x; k < boundary; ++k) {
                            if (row[k - region.x_start] != 0) {
                                seen = 1;
                                ++found;
                                break;
                            }
                        }
                    }
                    x = boundary;
                }
            }
            for (std::uint64_t cx = cx0; cx <= cx1; ++cx) {
                if (occupied.at(static_cast<std::size_t>(cx - cx0)) != 0) {
                    visit(cx, cy);
                }
            }
        }
    };

    // Counting sort needs the counts before it can place anything, but the mask is the most
    // expensive thing here to look at twice -- a region whose box is the image is tens of megabytes
    // of it. So the incidences are recorded once, in region order, and the two passes read that.
    std::vector<std::uint32_t> incidence_cells;
    std::vector<std::uint64_t> region_first(region_count + 1, 0);
    for (std::size_t i = 0; i < region_count; ++i) {
        for_each_cell(regions[i], [&](std::uint64_t cx, std::uint64_t cy) {
            incidence_cells.push_back(static_cast<std::uint32_t>(buckets.Cell(cx, cy)));
        });
        if (incidence_cells.size() > kMaxChunkIncidences) {
            return MakeError(ErrorCode::invalid_argument,
                             "The regions together touch more chunks than one reduction can index", node);
        }
        region_first.at(i + 1) = incidence_cells.size();
    }

    for (const auto cell : incidence_cells) {
        ++buckets.offsets.at(static_cast<std::size_t>(cell) + 1);
    }
    for (std::size_t cell = 0; cell < cells; ++cell) {
        buckets.offsets.at(cell + 1) += buckets.offsets.at(cell);
    }

    buckets.entries.resize(incidence_cells.size());
    std::vector<std::uint64_t> cursor(buckets.offsets.begin(), buckets.offsets.end() - 1);
    for (std::size_t i = 0; i < region_count; ++i) {
        for (auto k = region_first.at(i); k < region_first.at(i + 1); ++k) {
            const auto cell = static_cast<std::size_t>(incidence_cells.at(static_cast<std::size_t>(k)));
            buckets.entries.at(static_cast<std::size_t>(cursor.at(cell)++)) = static_cast<std::uint32_t>(i);
        }
    }
    return buckets;
}

Result<void> ValidateRequest(const ImageDescriptor& descriptor, const AxisMap& axes,
                             const SpectralReduceRequest& request) {
    const auto& node = descriptor.id;
    if (request.region_count == 0 || request.regions == nullptr) {
        return MakeError(ErrorCode::invalid_argument, "A spectral reduction needs at least one region", node);
    }
    if (request.region_count > kMaxSpectralRegions) {
        return MakeError(ErrorCode::invalid_argument,
                         "A spectral reduction accepts at most " + std::to_string(kMaxSpectralRegions) +
                             " regions, not " + std::to_string(request.region_count),
                         node);
    }
    if (request.statistics == 0) {
        return MakeError(ErrorCode::invalid_argument, "A spectral reduction needs at least one statistic", node);
    }

    const auto width = descriptor.axes.at(axes.x).length;
    const auto height = descriptor.axes.at(axes.y).length;
    for (std::size_t i = 0; i < request.region_count; ++i) {
        const auto& region = request.regions[i];
        if (region.width == 0 || region.height == 0) {
            return MakeError(ErrorCode::invalid_argument, "Region " + std::to_string(i) + " is empty", node);
        }
        if (region.x_start >= width || region.width > width - region.x_start || region.y_start >= height ||
            region.height > height - region.y_start) {
            return MakeError(ErrorCode::invalid_argument,
                             "Region " + std::to_string(i) + " falls outside the image", node);
        }
    }

    if (!axes.has_polarization && request.polarization != 0) {
        return MakeError(ErrorCode::invalid_argument, "The image has no polarization axis", node);
    }
    if (axes.has_polarization && request.polarization >= descriptor.axes.at(axes.polarization).length) {
        return MakeError(ErrorCode::invalid_argument, "The requested polarization is outside the image", node);
    }
    if (!axes.has_time && request.time != 0) {
        return MakeError(ErrorCode::invalid_argument, "The image has no time axis", node);
    }
    if (axes.has_time && request.time >= descriptor.axes.at(axes.time).length) {
        return MakeError(ErrorCode::invalid_argument, "The requested time step is outside the image", node);
    }
    return {};
}

// One row of one region inside one chunk, accumulated in registers so that the per-pixel loop never
// asks which statistics were requested. The merge into the output happens once per row.
struct RowTotals {
    std::uint64_t good = 0;
    std::uint64_t bad = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double smallest = kInfinity;
    double largest = -kInfinity;
};

// The per-pixel loop, written so that a compiler can vectorise it.
//
// Both template parameters are loop invariants that used to be runtime tests, and each one on its
// own was enough to stop vectorisation: the x stride is 1 for every image whose fastest logical
// axis is x, which is every XRADIO image, but the compiler cannot know that and pays a multiply per
// pixel for the possibility; and a mask that most regions do not have cost a branch per pixel.
//
// The finiteness test is branchless for the same reason. A non-finite value contributes zero to the
// sums and its own identity to the extrema, which is exactly what excluding it means, so there is
// nothing an if would do that arithmetic does not.
template <bool kUnitStride, bool kMasked>
void AccumulateRow(const float* row, std::uint64_t stride, std::uint64_t count, const std::uint8_t* selected,
                   RowTotals& totals) {
    std::uint64_t good = 0;
    std::uint64_t selected_count = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double smallest = kInfinity;
    double largest = -kInfinity;

    for (std::uint64_t i = 0; i < count; ++i) {
        if (kMasked && selected[i] == 0) {
            continue;
        }
        ++selected_count;
        const float value = kUnitStride ? row[i] : row[i * stride];
        const bool finite = std::isfinite(value);
        const double clean = finite ? static_cast<double>(value) : 0.0;
        good += static_cast<std::uint64_t>(finite);
        sum += clean;
        sum_sq += clean * clean;
        smallest = std::min(smallest, finite ? clean : kInfinity);
        largest = std::max(largest, finite ? clean : -kInfinity);
    }

    totals.good = good;
    totals.bad = selected_count - good;
    totals.sum = sum;
    totals.sum_sq = sum_sq;
    totals.smallest = smallest;
    totals.largest = largest;
}

// A maximal run of consecutive chunk columns that at least one region touches, in bounding-box
// grid coordinates.
struct ColumnRun {
    std::uint64_t first = 0;
    std::uint64_t last = 0;  // inclusive

    bool operator==(const ColumnRun& other) const {
        return first == other.first && last == other.last;
    }
};

// The occupied runs of each chunk row.
//
// Reading the bounding box would be correct and still wrong: the box of a diagonal cut is the whole
// image, while the cut touches one chunk per row. On a 4096^2 image that is 256 chunks decoded to
// use 16. The runs are what the walk reads instead, so the cost follows the regions rather than the
// rectangle that happens to contain them.
std::vector<std::vector<ColumnRun>> BuildColumnRuns(const ChunkBuckets& buckets) {
    std::vector<std::vector<ColumnRun>> rows(static_cast<std::size_t>(buckets.rows));
    for (std::uint64_t row = 0; row < buckets.rows; ++row) {
        auto& runs = rows.at(static_cast<std::size_t>(row));
        for (std::uint64_t column = 0; column < buckets.columns; ++column) {
            const auto cell = static_cast<std::size_t>((row * buckets.columns) + column);
            if (buckets.offsets.at(cell + 1) == buckets.offsets.at(cell)) {
                continue;
            }
            if (!runs.empty() && runs.back().last + 1 == column) {
                runs.back().last = column;
            } else {
                runs.push_back(ColumnRun{column, column});
            }
        }
    }
    return rows;
}

}  // namespace

Result<void> ReduceSpectral(const Store& store, const ImageDescriptor& descriptor,
                            const ChunkGeometry& geometry, const SpectralReduceRequest& request,
                            const SpectralSink& sink, const ReadOptions& options) {
    const auto& node = descriptor.id;
    if (!sink) {
        return MakeError(ErrorCode::invalid_argument, "A spectral reduction needs a sink", node);
    }
    auto axes = MapAxes(descriptor);
    if (!axes) {
        return axes.error();
    }
    const auto& map = axes.value();
    if (auto valid = ValidateRequest(descriptor, map, request); !valid) {
        return valid.error();
    }

    // The spectral range is checked here as well as inside each slab request, so that a bad range
    // is one error naming the axis rather than a partial reduction that fails on some later slab.
    const Range spectral = request.spectral;
    const auto channels = descriptor.axes.at(map.spectral).length;
    if (spectral.stride == 0 || spectral.count == 0 || spectral.start >= channels ||
        (spectral.count - 1) * spectral.stride > channels - 1 - spectral.start) {
        return MakeError(ErrorCode::invalid_argument, "The spectral range falls outside the image", node);
    }

    // The requested statistics, in the one order a block reports them.
    std::vector<Statistic> statistics;
    std::array<int, kStatisticOrder.size()> slot_of{};
    slot_of.fill(-1);
    for (std::size_t i = 0; i < kStatisticOrder.size(); ++i) {
        if (Contains(request.statistics, kStatisticOrder.at(i))) {
            slot_of.at(i) = static_cast<int>(statistics.size());
            statistics.push_back(kStatisticOrder.at(i));
        }
    }
    const std::size_t statistic_count = statistics.size();
    const int slot_num_pixels = slot_of.at(0);
    const int slot_nan_count = slot_of.at(1);
    const int slot_sum = slot_of.at(2);
    const int slot_sum_sq = slot_of.at(3);
    const int slot_min = slot_of.at(4);
    const int slot_max = slot_of.at(5);

    const auto chunk_width = std::max<std::uint64_t>(1, geometry.chunk_shape.at(map.x));
    const auto chunk_height = std::max<std::uint64_t>(1, geometry.chunk_shape.at(map.y));
    const auto chunk_depth = std::max<std::uint64_t>(1, geometry.chunk_shape.at(map.spectral));

    auto buckets_result =
        BuildChunkBuckets(request.regions, request.region_count, chunk_width, chunk_height, node);
    if (!buckets_result) {
        return buckets_result.error();
    }
    const auto& buckets = buckets_result.value();
    const auto runs_per_row = BuildColumnRuns(buckets);

    const bool apply_mask = options.apply_pixel_mask && descriptor.has_pixel_mask;
    // Budget the chunk data a slab decodes, not the pixels it keeps. A one-pixel region asks for
    // almost nothing and still decodes an entire chunk per chunk it touches, so sizing by the
    // region's own area would let a cursor-sized request pull an unbounded amount through.
    const std::uint64_t chunk_bytes = DecodedChunkBytes(descriptor, geometry) * (apply_mask ? 2 : 1);
    const std::size_t slab_budget_bytes = options.temporary_memory_limit_bytes != 0
                                              ? options.temporary_memory_limit_bytes
                                              : DefaultReadBytes(chunk_bytes);
    // The smallest slab that still decodes each spectral chunk once. Reading fewer channels than
    // this would decode a chunk and use part of it, then decode it again for the rest.
    const std::uint64_t least_channels = ((chunk_depth + spectral.stride - 1) / spectral.stride);

    // The chunks one spectral layer of the whole region set occupies.
    std::uint64_t layer_chunks = 0;
    for (const auto& runs : runs_per_row) {
        for (const auto& run : runs) {
            layer_chunks += run.last - run.first + 1;
        }
    }

    // How many channels one emitted block may hold. The hint is the caller's, the budgets are the
    // library's and the chunk alignment is the walk's; the smallest wins, and the block reports
    // what it used.
    //
    // Without a hint a block costs one budget of decoded bytes, the same invariant a piece of Read
    // carries, which is why it is free: the block spends whatever the spatial walk left over. A
    // small region leaves almost all of it, so the block spans many chunks along the spectrum. A
    // region covering the image spends the budget spatially, `spectral_chunks` below collapses to
    // one, and the block becomes the single chunk layer the walk is already reading -- the results
    // are handed over a layer at a time because that is when they are finished, not sooner.
    //
    // Emitting only at the end instead, which is what a zero used to mean, is silent for as long as
    // the whole reduction takes. One layer of a 7763x4742 image is 160 MiB and 70 ms; a thousand
    // channels of it is a minute of work with no partial answer and nowhere to cancel.
    const std::size_t bytes_per_channel = request.region_count * statistic_count * sizeof(double);
    const std::uint64_t budget_channels =
        std::max<std::uint64_t>(1, kSpectralEmitBudgetBytes / std::max<std::size_t>(1, bytes_per_channel));
    const std::uint64_t block_chunks = std::min(
        spectral.count,
        std::max<std::uint64_t>(1, slab_budget_bytes / std::max<std::uint64_t>(1, layer_chunks * chunk_bytes)));
    const std::uint64_t wanted_channels =
        std::min({request.emit_every_channels == 0 ? block_chunks * least_channels
                                                   : static_cast<std::uint64_t>(request.emit_every_channels),
                  budget_channels, spectral.count});

    const auto rank = descriptor.axes.size();
    std::vector<double> accumulator;
    std::vector<float> pixels;
    std::vector<std::uint8_t> mask;
    std::vector<ColumnRun> segments;

    for (std::uint64_t block_begin = 0; block_begin < spectral.count;) {
        const std::uint64_t block_end = AlignedBlockEnd(block_begin, wanted_channels, spectral.count,
                                                        spectral.start, spectral.stride, chunk_depth);
        const auto block_length = static_cast<std::size_t>(block_end - block_begin);
        const std::size_t statistic_stride = block_length;
        const std::size_t region_stride = statistic_count * statistic_stride;

        accumulator.assign(request.region_count * region_stride, 0.0);
        for (std::size_t r = 0; r < request.region_count; ++r) {
            if (slot_min >= 0) {
                auto* base = accumulator.data() + (r * region_stride) +
                             (static_cast<std::size_t>(slot_min) * statistic_stride);
                std::fill(base, base + block_length, kInfinity);
            }
            if (slot_max >= 0) {
                auto* base = accumulator.data() + (r * region_stride) +
                             (static_cast<std::size_t>(slot_max) * statistic_stride);
                std::fill(base, base + block_length, -kInfinity);
            }
        }

        const std::uint64_t block_spectral_chunks = (block_length + least_channels - 1) / least_channels;
        const std::uint64_t block_chunks_total =
            std::max<std::uint64_t>(1, std::max<std::uint64_t>(1, layer_chunks) * block_spectral_chunks);
        std::uint64_t block_chunks_done = 0;
        std::uint64_t reads_done = 0;

        // An extremum nothing contributed to is still its identity, which is the one value it must
        // not be reported as. Finding those needs no extra bookkeeping: a finite pixel can never
        // leave an infinity behind, so only the untouched entries are still infinite -- and by the
        // same argument the NaN that replaced one is the only NaN there, so the identity can be put
        // back when the walk is not finished with it.
        const auto settle_extrema = [&](bool report) {
            for (std::size_t r = 0; r < request.region_count; ++r) {
                for (const int slot : {slot_min, slot_max}) {
                    if (slot < 0) {
                        continue;
                    }
                    const double identity = slot == slot_min ? kInfinity : -kInfinity;
                    auto* base = accumulator.data() + (r * region_stride) +
                                 (static_cast<std::size_t>(slot) * statistic_stride);
                    for (std::size_t c = 0; c < block_length; ++c) {
                        if (report) {
                            if (std::isinf(base[c])) {
                                base[c] = std::numeric_limits<double>::quiet_NaN();
                            }
                        } else if (std::isnan(base[c])) {
                            base[c] = identity;
                        }
                    }
                }
            }
        };

        const auto hand_over = [&](bool complete) -> Result<void> {
            settle_extrema(true);
            SpectralBlock block;
            block.first_channel = block_begin;
            block.channel_count = block_length;
            block.values = accumulator.data();
            block.value_count = accumulator.size();
            block.region_stride = region_stride;
            block.statistic_stride = statistic_stride;
            block.statistics = statistics.data();
            block.statistic_count = statistic_count;
            block.complete = complete;
            block.completeness =
                complete ? 1.0
                         : static_cast<double>(block_chunks_done) / static_cast<double>(block_chunks_total);
            const bool keep_going = sink(block);
            if (!complete) {
                settle_extrema(false);
            }
            if (!keep_going) {
                return MakeError(ErrorCode::cancelled, "The spectral reduction was cancelled by its sink", node);
            }
            return {};
        };

        for (std::uint64_t row = 0; row < buckets.rows;) {
            const auto& runs = runs_per_row.at(static_cast<std::size_t>(row));
            if (runs.empty()) {
                ++row;
                continue;
            }

            // Rows below that repeat this row's runs exactly are read with it, so that a solid
            // rectangle becomes a few large requests while a diagonal stays one chunk per row.
            std::uint64_t widest = 0;
            for (const auto& run : runs) {
                widest = std::max(widest, run.last - run.first + 1);
            }
            const std::uint64_t row_bytes = std::max<std::uint64_t>(1, widest * chunk_bytes);
            const std::uint64_t band_limit = std::max<std::uint64_t>(1, slab_budget_bytes / row_bytes);
            std::uint64_t band_end = row + 1;
            while (band_end < buckets.rows && (band_end - row) < band_limit &&
                   runs_per_row.at(static_cast<std::size_t>(band_end)) == runs) {
                ++band_end;
            }

            const std::uint64_t chunk_y_begin = buckets.chunk_y0 + row;
            const std::uint64_t chunk_y_end = buckets.chunk_y0 + band_end;
            const std::uint64_t y_begin = std::max(buckets.y0, chunk_y_begin * chunk_height);
            const std::uint64_t y_end = std::min(buckets.y1, chunk_y_end * chunk_height);

            // A run wider than the budget is read in pieces, not in one request. Without this the
            // smallest request is a whole chunk row of the region: 157 chunks of a 80000-pixel-wide
            // region is 628 MiB, ten times the budget it was supposed to respect, and nothing to
            // report or cancel from until all of it lands.
            const std::uint64_t band_rows = std::max<std::uint64_t>(1, band_end - row);
            const std::uint64_t segment_limit =
                std::max<std::uint64_t>(1, slab_budget_bytes / (band_rows * chunk_bytes));
            segments.clear();
            for (const auto& whole : runs) {
                for (std::uint64_t first = whole.first; first <= whole.last;) {
                    const std::uint64_t width = std::min(segment_limit, whole.last - first + 1);
                    segments.push_back(ColumnRun{first, first + width - 1});
                    first += width;
                }
            }

            for (const auto& run : segments) {
                const std::uint64_t chunk_x_begin = buckets.chunk_x0 + run.first;
                const std::uint64_t chunk_x_end = buckets.chunk_x0 + run.last + 1;
                const std::uint64_t x_begin = std::max(buckets.x0, chunk_x_begin * chunk_width);
                const std::uint64_t x_end = std::min(buckets.x1, chunk_x_end * chunk_width);
                const std::uint64_t run_chunks =
                    std::max<std::uint64_t>(1, (run.last - run.first + 1) * (band_end - row));
                const std::uint64_t spectral_chunks =
                    std::max<std::uint64_t>(1, slab_budget_bytes / (run_chunks * chunk_bytes));
                const std::uint64_t slab_channels = std::max<std::uint64_t>(1, spectral_chunks * least_channels);

                for (std::uint64_t slab_begin = block_begin; slab_begin < block_end;) {
                    const std::uint64_t slab_end =
                        AlignedBlockEnd(slab_begin, std::min(slab_channels, block_end - slab_begin), block_end,
                                        spectral.start, spectral.stride, chunk_depth);
                    const std::uint64_t slab_length = slab_end - slab_begin;

                    // What is in hand before spending another budget on this block. A block that
                    // takes one read never gets here, so a small region still reports once.
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
                    read_request.axes.at(map.x) = Range{x_begin, x_end - x_begin, 1};
                    read_request.axes.at(map.y) = Range{y_begin, y_end - y_begin, 1};
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

                    // Destination strides of a densely packed logical read, axis 0 fastest. Derived
                    // rather than assumed, so the accumulation does not quietly depend on x, y and
                    // spectral being the first three logical axes.
                    std::uint64_t stride_x = 0;
                    std::uint64_t stride_y = 0;
                    std::uint64_t stride_z = 0;
                    std::uint64_t running = 1;
                    for (std::size_t i = 0; i < rank; ++i) {
                        if (i == map.x) stride_x = running;
                        if (i == map.y) stride_y = running;
                        if (i == map.spectral) stride_z = running;
                        running *= read_request.axes.at(i).count;
                    }
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
                        // Fold the flag into the pixels rather than carry it into the inner loop: a
                        // flagged pixel and a NaN pixel mean the same thing to every statistic here,
                        // and this is the rule Image::Read already applies.
                        for (std::size_t i = 0; i < elements; ++i) {
                            if (mask[i] == 0) {
                                pixels[i] = std::numeric_limits<float>::quiet_NaN();
                            }
                        }
                    }

                    for (std::uint64_t channel = 0; channel < slab_length; ++channel) {
                        const auto out_channel = static_cast<std::size_t>(slab_begin - block_begin + channel);
                        const float* plane = pixels.data() + (channel * stride_z);

                        for (std::uint64_t chunk_y = chunk_y_begin; chunk_y < chunk_y_end; ++chunk_y) {
                            const std::uint64_t cell_y0 = std::max(y_begin, chunk_y * chunk_height);
                            const std::uint64_t cell_y1 = std::min(y_end, (chunk_y + 1) * chunk_height);
                            if (cell_y0 >= cell_y1) {
                                continue;
                            }
                            for (std::uint64_t chunk_x = chunk_x_begin; chunk_x < chunk_x_end; ++chunk_x) {
                                const std::uint64_t cell_x0 = std::max(x_begin, chunk_x * chunk_width);
                                const std::uint64_t cell_x1 = std::min(x_end, (chunk_x + 1) * chunk_width);
                                if (cell_x0 >= cell_x1) {
                                    continue;
                                }

                                const auto cell = buckets.Cell(chunk_x, chunk_y);
                                const auto entry_end = buckets.offsets.at(cell + 1);
                                for (auto entry = buckets.offsets.at(cell); entry < entry_end; ++entry) {
                                    const std::size_t r = buckets.entries.at(static_cast<std::size_t>(entry));
                                    const auto& region = request.regions[r];

                                    const std::uint64_t rx0 = std::max(cell_x0, region.x_start);
                                    const std::uint64_t rx1 = std::min(cell_x1, region.x_start + region.width);
                                    const std::uint64_t ry0 = std::max(cell_y0, region.y_start);
                                    const std::uint64_t ry1 = std::min(cell_y1, region.y_start + region.height);
                                    if (rx0 >= rx1 || ry0 >= ry1) {
                                        continue;
                                    }

                                    double* out = accumulator.data() + (r * region_stride);
                                    for (std::uint64_t y = ry0; y < ry1; ++y) {
                                        const float* pixel_row =
                                            plane + ((y - y_begin) * stride_y) + ((rx0 - x_begin) * stride_x);
                                        const std::uint8_t* selected =
                                            region.mask == nullptr
                                                ? nullptr
                                                : region.mask + ((y - region.y_start) * region.width) +
                                                      (rx0 - region.x_start);
                                        RowTotals totals;
                                        const std::uint64_t run = rx1 - rx0;
                                        if (stride_x == 1) {
                                            if (selected == nullptr) {
                                                AccumulateRow<true, false>(pixel_row, 1, run, nullptr, totals);
                                            } else {
                                                AccumulateRow<true, true>(pixel_row, 1, run, selected, totals);
                                            }
                                        } else if (selected == nullptr) {
                                            AccumulateRow<false, false>(pixel_row, stride_x, run, nullptr, totals);
                                        } else {
                                            AccumulateRow<false, true>(pixel_row, stride_x, run, selected, totals);
                                        }

                                        if (slot_num_pixels >= 0) {
                                            out[(static_cast<std::size_t>(slot_num_pixels) * statistic_stride) +
                                                out_channel] += static_cast<double>(totals.good);
                                        }
                                        if (slot_nan_count >= 0) {
                                            out[(static_cast<std::size_t>(slot_nan_count) * statistic_stride) +
                                                out_channel] += static_cast<double>(totals.bad);
                                        }
                                        if (slot_sum >= 0) {
                                            out[(static_cast<std::size_t>(slot_sum) * statistic_stride) +
                                                out_channel] += totals.sum;
                                        }
                                        if (slot_sum_sq >= 0) {
                                            out[(static_cast<std::size_t>(slot_sum_sq) * statistic_stride) +
                                                out_channel] += totals.sum_sq;
                                        }
                                        if (slot_min >= 0) {
                                            double& current =
                                                out[(static_cast<std::size_t>(slot_min) * statistic_stride) +
                                                    out_channel];
                                            current = std::min(current, totals.smallest);
                                        }
                                        if (slot_max >= 0) {
                                            double& current =
                                                out[(static_cast<std::size_t>(slot_max) * statistic_stride) +
                                                    out_channel];
                                            current = std::max(current, totals.largest);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    block_chunks_done +=
                        run_chunks * ((slab_length + least_channels - 1) / least_channels);
                    slab_begin = slab_end;
                }
            }
            row = band_end;
        }

        if (auto handed = hand_over(true); !handed) {
            return handed.error();
        }

        block_begin = block_end;
    }
    return {};
}

}  // namespace carta::zarr::internal
