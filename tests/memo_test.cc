/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// Memo and Lazy carry two decisions the Store depends on but never states: a value is computed
// exactly once, and a failed computation is remembered like any other. The second is what makes a
// Store a stable read-only view -- a node that was missing stays missing -- and until now it was
// only observable three layers up, through the metadata cache test.

#include "memo.h"

// Memo itself knows nothing of Result or Error; the failure test brings them.
#include "carta-zarr/error.h"
#include "carta-zarr/result.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using carta::zarr::internal::Lazy;
using carta::zarr::internal::Memo;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void TestComputesOncePerKey() {
    Memo<std::string, int> memo;
    int calls = 0;
    const auto compute = [&calls] {
        ++calls;
        return 42;
    };

    Require(memo.GetOrCompute("a", compute) == 42, "the computed value was not returned");
    Require(calls == 1, "the first read did not compute");
    Require(memo.GetOrCompute("a", compute) == 42, "the remembered value was not returned");
    Require(calls == 1, "a remembered key was computed again");
}

void TestKeysAreIndependent() {
    Memo<std::string, std::string> memo;
    Require(memo.GetOrCompute("a", [] { return std::string("first"); }) == "first", "unexpected value for a");
    Require(memo.GetOrCompute("b", [] { return std::string("second"); }) == "second", "unexpected value for b");
    Require(memo.GetOrCompute("a", [] { return std::string("third"); }) == "first", "key b displaced key a");
}

// A failure is a value. A Store is a read-only view, so a node that was missing when first asked for
// stays missing, and what a caller observes does not change under it.
void TestFailuresAreRemembered() {
    Memo<std::string, carta::zarr::Result<int>> memo;
    int calls = 0;
    const auto failing = [&calls]() -> carta::zarr::Result<int> {
        ++calls;
        return carta::zarr::Error{carta::zarr::ErrorCode::not_found, "absent", "node"};
    };

    const auto first = memo.GetOrCompute("node", failing);
    Require(!first && first.error().code == carta::zarr::ErrorCode::not_found, "the failure was not returned");

    // Even though the value could now be computed successfully, the remembered failure stands.
    const auto second = memo.GetOrCompute("node", []() -> carta::zarr::Result<int> { return 7; });
    Require(!second && second.error().code == carta::zarr::ErrorCode::not_found, "a remembered failure was recomputed");
    Require(calls == 1, "the failing computation ran more than once");
}

void TestLazyComputesOnce() {
    Lazy<int> lazy;
    int calls = 0;
    const auto compute = [&calls] {
        ++calls;
        return 9;
    };

    Require(lazy.GetOrCompute(compute) == 9, "the computed value was not returned");
    Require(lazy.GetOrCompute([] { return 11; }) == 9, "the remembered value was recomputed");
    Require(calls == 1, "Lazy computed more than once");
}

// The lock is held across the computation, so a key is computed exactly once however many threads
// arrive together. The computation is held open until every thread has reached the memo: without
// that, the first caller finishes before the others arrive and a memo that computed outside its
// lock would pass this test too.
void TestConcurrentReadersComputeOnce() {
    constexpr int kThreads = 8;
    Memo<std::string, int> memo;
    std::atomic<int> calls{0};
    std::atomic<int> ready{0};
    std::atomic<int> mismatches{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int index = 0; index < kThreads; ++index) {
        threads.emplace_back([&memo, &calls, &ready, &mismatches] {
            ready.fetch_add(1);
            while (ready.load() < kThreads) {
                std::this_thread::yield();
            }
            const int value = memo.GetOrCompute("shared", [&calls] {
                calls.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                return 5;
            });
            if (value != 5) {
                mismatches.fetch_add(1);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    Require(calls.load() == 1, "a contended key was computed more than once");
    Require(mismatches.load() == 0, "a reader observed a value it did not compute");
}

}  // namespace

int main() {
    try {
        TestComputesOncePerKey();
        TestKeysAreIndependent();
        TestFailuresAreRemembered();
        TestLazyComputesOnce();
        TestConcurrentReadersComputeOnce();
        std::cout << "carta-zarr memo tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carta-zarr memo tests failed: " << error.what() << '\n';
        return 1;
    }
}
