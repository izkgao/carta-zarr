/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// What a pool of its own is made of, asked of the TensorStore context underneath. A consumer sees a
// CachePool only by reading through one, and every read through one returns the same pixels whatever
// its size, so the three things that make it worth having -- that it is the size asked for, that it
// runs on the session's threads rather than its own, and that nothing keeps it once its holder lets
// go -- are visible only here. So is whether a cached chunk is used without asking storage about it
// first, which changes how long a read takes and not what it returns.

#include "zarr/store_context.h"

#include <tensorstore/internal/cache/cache_pool_resource.h>
#include <tensorstore/internal/data_copy_concurrency_resource.h>
#include <tensorstore/internal/file_io_concurrency_resource.h>

#include <tensorstore/tensorstore.h>

#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <unistd.h>

#include "support/check.h"

namespace {

using carta::zarr::testing::Require;
using carta::zarr::internal::StoreContextPtr;

std::size_t PoolLimit(const StoreContextPtr& store_context) {
    auto pool = store_context->context.GetResource<tensorstore::internal::CachePoolResource>();
    Require(pool.ok(), "the context has no cache pool resource");
    const auto& weak = **pool;
    Require(static_cast<bool>(weak), "the cache pool resource is empty");
    return weak->limits().total_bytes_limit;
}

template <typename Provider>
const void* ResourceOf(const StoreContextPtr& store_context) {
    auto resource = store_context->context.GetResource<Provider>();
    Require(resource.ok(), std::string("the context has no ") + Provider::id + " resource");
    return resource->get();
}

StoreContextPtr Session() {
    carta::zarr::ContextOptions options;
    options.cache_bytes = std::size_t{8} << 20;
    options.io_threads = 3;
    options.decode_threads = 5;
    auto session = carta::zarr::internal::MakeStoreContext(options);
    Require(static_cast<bool>(session), "the session context could not be made");
    return session.value();
}

// The size asked for, zero included: zero is the pool that holds nothing, which is what a scan that
// should leave the session's working set alone asks for.
void TestAPoolIsTheSizeAskedFor() {
    const auto session = Session();
    for (const std::size_t bytes : {std::size_t{0}, std::size_t{1}, std::size_t{3} << 30}) {
        auto own = session->WithCachePool(bytes);
        Require(static_cast<bool>(own), "a pool of " + std::to_string(bytes) + " bytes could not be made");
        Require(PoolLimit(own.value()) == bytes,
                "a pool of " + std::to_string(bytes) + " bytes holds " + std::to_string(PoolLimit(own.value())));
    }
    Require(PoolLimit(session) == (std::size_t{8} << 20), "making a pool of its own changed the session's");
}

// On the session's threads: a walk that ran on threads of its own would compete with the session's
// reads rather than take its turn among them.
void TestAPoolRunsOnTheSessionsThreads() {
    const auto session = Session();
    auto own = session->WithCachePool(std::size_t{1} << 20);
    Require(static_cast<bool>(own), "a pool of its own could not be made");
    Require(ResourceOf<tensorstore::internal::DataCopyConcurrencyResource>(own.value()) ==
                ResourceOf<tensorstore::internal::DataCopyConcurrencyResource>(session),
            "a pool of its own decodes on threads other than the session's");
    Require(ResourceOf<tensorstore::internal::FileIoConcurrencyResource>(own.value()) ==
                ResourceOf<tensorstore::internal::FileIoConcurrencyResource>(session),
            "a pool of its own reads files on threads other than the session's");
    Require(ResourceOf<tensorstore::internal::CachePoolResource>(own.value()) !=
                ResourceOf<tensorstore::internal::CachePoolResource>(session),
            "a pool of its own is the session's pool");
}

// Nothing keeps a pool once its holder lets go, and two asked for are two. The bypass this replaces
// was built once and kept for as long as the store, which cost nothing at zero bytes and would keep
// gigabytes of decoded chunks at the size a moment asks for.
void TestNothingKeepsAPoolItsHolderLetGo() {
    const auto session = Session();
    std::weak_ptr<const carta::zarr::internal::StoreContext> first;
    std::weak_ptr<const carta::zarr::internal::StoreContext> last;
    {
        auto own = session->WithCachePool(std::size_t{1} << 20);
        Require(static_cast<bool>(own), "a pool of its own could not be made");
        first = own.value();
        auto other = session->WithCachePool(std::size_t{1} << 20);
        Require(static_cast<bool>(other) && other.value() != own.value(), "two pools asked for are one");
        last = other.value();
    }
    Require(first.expired() && last.expired(), "a pool outlived the last holder of it");
}

// A chunk cached after its array was opened is used as it is: nothing asks storage whether it has
// changed. ADR 0015. Seen here by truncating every chunk of a copy once it has been read: through the
// handle that read them the array reads back unchanged, and opened afresh it no longer reads at all,
// so what answered the second read was the cache and not the files.
void TestACachedChunkIsNotCheckedAgain() {
    const auto copy = std::filesystem::temp_directory_path() /
                      ("carta-zarr-recheck-" + std::to_string(getpid()));
    std::filesystem::remove_all(copy);
    std::filesystem::copy(std::string(CARTA_ZARR_PIXEL_FIXTURE_WIDE) + "/SKY", copy,
                          std::filesystem::copy_options::recursive);

    carta::zarr::ContextOptions options;
    options.cache_bytes = std::size_t{64} << 20;
    const auto session = carta::zarr::internal::MakeStoreContext(options);
    Require(static_cast<bool>(session), "the session context could not be made");
    const auto read = [](const tensorstore::TensorStore<>& store) {
        return tensorstore::Read<tensorstore::zero_origin>(store).result();
    };
    const auto opened = session.value()->OpenArray(copy, "SKY");
    Require(static_cast<bool>(opened), "the copy did not open");
    const auto before = read(opened.value());
    Require(before.ok(), "the copy did not read");

    for (const auto& entry : std::filesystem::recursive_directory_iterator(copy / "c")) {
        if (entry.is_regular_file()) {
            std::filesystem::resize_file(entry.path(), 0);
        }
    }
    const auto after = read(opened.value());
    const auto afresh = carta::zarr::internal::MakeStoreContext(options).value()->OpenArray(copy, "SKY");
    const bool afresh_reads = afresh && read(afresh.value()).ok();
    std::filesystem::remove_all(copy);

    Require(after.ok(), "a cached chunk was checked against storage and found changed");
    const auto bytes = before->num_elements() * before->dtype().size();
    Require(after->num_elements() == before->num_elements() &&
                std::memcmp(after->data(), before->data(), bytes) == 0,
            "a cached chunk read back differently");
    Require(!afresh_reads, "truncated chunks still read, so this test shows nothing about the cache");
}

}  // namespace

int main() {
    try {
        TestAPoolIsTheSizeAskedFor();
        TestAPoolRunsOnTheSessionsThreads();
        TestNothingKeepsAPoolItsHolderLetGo();
        TestACachedChunkIsNotCheckedAgain();
    } catch (const std::exception& error) {
        std::cerr << "store context test failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
