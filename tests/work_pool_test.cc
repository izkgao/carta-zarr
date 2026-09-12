/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// The library's own worker threads, and the rule that decides whether to wake them.
//
// This is the only concurrency carta-zarr owns, and the pixel fixtures are far too small to reach
// it -- PlanRowTasks answers 1 for a plane of twenty pixels, which is the correct answer and also
// means the reductions' own tests never run the parallel path. So it is tested here directly.

#include "work_pool.h"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <string>
#include <vector>

namespace {

using carta::zarr::internal::PlanRowTasks;
using carta::zarr::internal::WorkPool;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void EveryTaskRunsExactlyOnce() {
    for (const std::size_t threads : {std::size_t{1}, std::size_t{2}, std::size_t{8}}) {
        WorkPool pool(threads);
        for (const std::size_t tasks : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{1000}}) {
            std::vector<int> seen(tasks, 0);
            pool.Run(tasks, [&](std::size_t task, std::size_t) { ++seen[task]; });
            for (std::size_t task = 0; task < tasks; ++task) {
                Require(seen[task] == 1, "task " + std::to_string(task) + " ran " +
                                             std::to_string(seen[task]) + " times with " +
                                             std::to_string(threads) + " threads");
            }
        }
    }
}

// The contract a private accumulator relies on: no two bodies running at the same moment are given
// the same worker index, and no index is ever outside [0, size()).
void WorkerIndicesAreExclusiveWhileRunning() {
    WorkPool pool(8);
    const std::size_t size = pool.size();
    std::vector<std::atomic<int>> occupied(size);
    for (auto& slot : occupied) {
        slot.store(0);
    }
    std::atomic<bool> collided{false};
    std::atomic<bool> out_of_range{false};

    pool.Run(4096, [&](std::size_t, std::size_t worker) {
        if (worker >= size) {
            out_of_range.store(true);
            return;
        }
        if (occupied[worker].fetch_add(1) != 0) {
            collided.store(true);
        }
        // Long enough that two bodies sharing a slot would overlap here rather than pass in turn.
        for (volatile int spin = 0; spin < 2000; ++spin) {
        }
        occupied[worker].fetch_sub(1);
    });

    Require(!out_of_range.load(), "a body was given a worker index outside [0, size())");
    Require(!collided.load(), "two concurrent bodies shared a worker index");
}

// A pool of one starts no threads, so the body has to run on the caller's thread or not at all.
void SingleThreadedPoolRunsInline() {
    WorkPool pool(1);
    Require(pool.size() == 1, "a pool of one reports a size other than one");
    const auto caller = std::this_thread::get_id();
    bool elsewhere = false;
    pool.Run(16, [&](std::size_t, std::size_t worker) {
        if (std::this_thread::get_id() != caller || worker != 0) {
            elsewhere = true;
        }
    });
    Require(!elsewhere, "a pool of one ran a body off the calling thread");
}

// Runs are sequential and reusable: the second one must not inherit the first one's counters.
void RepeatedRunsDoNotLeak() {
    WorkPool pool(4);
    std::atomic<std::uint64_t> total{0};
    for (int round = 0; round < 200; ++round) {
        total.store(0);
        pool.Run(97, [&](std::size_t task, std::size_t) { total.fetch_add(task); });
        Require(total.load() == (96U * 97U) / 2, "round " + std::to_string(round) + " summed wrong");
    }
}

void TheSplitRuleIsConservative() {
    // Nothing to split.
    Require(PlanRowTasks(1000, 0, 8, 1U << 16U) == 0, "no rows should ask for no tasks");
    // A plane smaller than one task's worth stays in place. This is the pixel fixture's case.
    Require(PlanRowTasks(5, 4, 28, 1U << 16U) == 1, "a twenty-pixel plane should not be split");
    // A pool of one never splits, however large the plane.
    Require(PlanRowTasks(4742, 7763, 1, 1U << 16U) == 1, "a pool of one should not split");
    // Never more pieces than rows.
    Require(PlanRowTasks(1U << 20U, 3, 28, 1U << 16U) == 3, "a three-row plane should give three tasks");
    // Never more pieces than the pool.
    Require(PlanRowTasks(4742, 7763, 28, 1U << 16U) == 28, "a large plane should fill the pool");
    // The affordability cap bites between those two.
    Require(PlanRowTasks(64, 64, 28, 1U << 16U) == 1, "4096 pixels should not be split 28 ways");
    Require(PlanRowTasks(256, 512, 28, 1U << 16U) == 2, "131072 pixels should give two tasks");
}

// What the histogram does with the plan: contiguous pieces that cover every row exactly once.
void RowRangesCoverEveryRowOnce() {
    for (std::uint64_t rows : {std::uint64_t{1}, std::uint64_t{7}, std::uint64_t{28}, std::uint64_t{7763}}) {
        const std::size_t tasks = PlanRowTasks(1U << 20U, rows, 28, 1U << 16U);
        Require(tasks >= 1, "a plan of zero tasks for a non-empty plane");
        std::vector<int> seen(rows, 0);
        const std::uint64_t rows_per_task = (rows + tasks - 1) / tasks;
        for (std::size_t task = 0; task < tasks; ++task) {
            const std::uint64_t first = static_cast<std::uint64_t>(task) * rows_per_task;
            if (first >= rows) {
                continue;
            }
            for (std::uint64_t row = first; row < std::min(first + rows_per_task, rows); ++row) {
                ++seen[row];
            }
        }
        for (std::uint64_t row = 0; row < rows; ++row) {
            Require(seen[row] == 1, "row " + std::to_string(row) + " of " + std::to_string(rows) +
                                        " was covered " + std::to_string(seen[row]) + " times");
        }
    }
}

}  // namespace

int main() {
    try {
        EveryTaskRunsExactlyOnce();
        WorkerIndicesAreExclusiveWhileRunning();
        SingleThreadedPoolRunsInline();
        RepeatedRunsDoNotLeak();
        TheSplitRuleIsConservative();
        RowRangesCoverEveryRowOnce();
    } catch (const std::exception& error) {
        std::cerr << "work pool test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "work pool tests passed\n";
    return 0;
}
