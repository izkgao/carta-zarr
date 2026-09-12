/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CARTA_ZARR_SRC_WORK_POOL_H_
#define CARTA_ZARR_SRC_WORK_POOL_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace carta::zarr::internal {

/**
 * The library's own worker threads, for the per-pixel work a reduction does after a read returns.
 *
 * This is the only concurrency carta-zarr owns. Everything else is TensorStore's: its read pool
 * decompresses chunks, and until this existed the pixels those chunks produced were then visited on
 * one thread, which measured as 78% of a cube scan on a 28-core machine while the decode pool sat
 * idle (implementation-plan.md 6.2.3 and 6.2.5).
 *
 * Deliberately small, and deliberately not a general executor:
 *
 * - Threads are started once and parked, because the alternative -- starting them per slab -- costs
 *   more than the work when a slab is a few hundred thousand pixels.
 * - `Run` blocks until every worker has finished, so a caller's buffers stay alive for the whole
 *   span and there is no future to own. The walk is a loop over slabs, and it wants each slab done
 *   before it reuses the buffer for the next one.
 * - The body is handed a worker index as well as a task index, so a caller that needs private
 *   accumulation can address one slot per worker without a map or a lock.
 * - Exceptions are not propagated. Every body this pool runs is arithmetic over a buffer the caller
 *   already owns; if one of them throws, the process is already past saving.
 */
class WorkPool {
public:
    // `threads` of zero asks for one worker per hardware thread. One means everything runs inline
    // on the calling thread and no threads are started at all.
    explicit WorkPool(std::size_t threads);
    ~WorkPool();

    WorkPool(const WorkPool&) = delete;
    WorkPool& operator=(const WorkPool&) = delete;

    // How many bodies may run at once, counting the calling thread. Always at least one.
    std::size_t size() const noexcept {
        return _workers.empty() ? 1 : _workers.size() + 1;
    }

    /**
     * Run `body(task, worker)` for every task in [0, tasks), and return once all of them are done.
     *
     * `worker` is in [0, size()) and no two bodies running at the same time share one, so a private
     * accumulator indexed by it needs no lock. Tasks are claimed from a shared counter rather than
     * divided up front, so an uneven one does not leave workers waiting on the slowest slice.
     */
    void Run(std::size_t tasks, const std::function<void(std::size_t task, std::size_t worker)>& body);

private:
    void Worker(std::size_t index);

    std::vector<std::thread> _workers;
    std::mutex _mutex;
    std::condition_variable _wake;
    std::condition_variable _done;

    const std::function<void(std::size_t, std::size_t)>* _body = nullptr;
    std::size_t _tasks = 0;
    std::size_t _next = 0;
    std::size_t _running = 0;
    std::size_t _generation = 0;
    bool _stopping = false;
};

/**
 * How many contiguous pieces a plane of `rows` rows of `row_pixels` each should be split into.
 *
 * Separate from the pool, and pure, because it is the half of the split that has an answer worth
 * checking: a piece smaller than `least_pixels` costs more to hand out than to run, and a plane
 * cannot be split into more pieces than it has rows. Returning 1 means the caller should bin in
 * place and not go near the pool at all -- which is the right answer for a tiny image, and is why
 * a fixture of a few dozen pixels never reaches the parallel path.
 */
std::size_t PlanRowTasks(std::uint64_t row_pixels, std::uint64_t rows, std::size_t max_tasks,
                         std::uint64_t least_pixels);

}  // namespace carta::zarr::internal

#endif  // CARTA_ZARR_SRC_WORK_POOL_H_
