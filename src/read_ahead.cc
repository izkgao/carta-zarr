/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "read_ahead.h"

#include <utility>

namespace carta::zarr::internal {

Run RunOf(const ChunkGeometry&, const ReadRequest&) {
    return {};
}

std::uint64_t PlaneRunBytes(const ImageDescriptor&, const ChunkGeometry&, bool) {
    return 0;
}

Result<std::unique_ptr<ReadingAhead>> ReadingAhead::For(std::vector<std::shared_ptr<const RunSource>> sources) {
    return std::unique_ptr<ReadingAhead>(new ReadingAhead(std::move(sources)));
}

ReadingAhead::ReadingAhead(std::vector<std::shared_ptr<const RunSource>> sources) : _sources(std::move(sources)) {}

ReadingAhead::~ReadingAhead() {
    Cancel();
    Join();
}

void ReadingAhead::Served(Clock::time_point, bool, const std::vector<AnimatedPlane>&,
                          const std::vector<std::vector<AnimatedPlane>>&) {}

void ReadingAhead::Cancel() {
    _cancelled = true;
}

ReadAheadStats ReadingAhead::Stats() const {
    const std::scoped_lock lock(_mutex);
    return _stats;
}

bool ReadingAhead::UnderWay() const {
    const std::scoped_lock lock(_mutex);
    return _under_way;
}

void ReadingAhead::Join() {
    if (_worker.joinable()) {
        _worker.join();
    }
}

}  // namespace carta::zarr::internal
