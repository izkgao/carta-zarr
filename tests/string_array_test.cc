/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "zarr/string_array.h"

#include "zarr/array_metadata.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using carta::zarr::ErrorCode;
namespace zarr_metadata = carta::zarr::internal::zarr;

const std::filesystem::path kFixtureDir{CARTA_ZARR_STRING_FIXTURE_DIR};

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

carta::zarr::Result<std::vector<std::string>> ReadFixture(const std::string& name) {
    const std::filesystem::path array_dir = kFixtureDir / name;
    std::ifstream metadata_file(array_dir / "zarr.json");
    Require(metadata_file.is_open(), "Missing fixture " + array_dir.string());
    const nlohmann::json metadata = nlohmann::json::parse(metadata_file);

    auto array_metadata = zarr_metadata::ParseArrayMetadata(metadata, name);
    Require(static_cast<bool>(array_metadata), "Fixture " + name + " has unreadable array metadata");
    return zarr_metadata::ReadFixedLengthUtf32StringArray(array_dir, array_metadata.value(), metadata, name);
}

void ExpectValues(const std::string& name, const std::vector<std::string>& expected) {
    auto result = ReadFixture(name);
    Require(static_cast<bool>(result),
            "Fixture " + name + " failed to decode: " + (result ? std::string{} : result.error().message));
    Require(result.value() == expected, "Fixture " + name + " decoded unexpected values");
}

void ExpectError(const std::string& name, ErrorCode expected) {
    auto result = ReadFixture(name);
    Require(!result, "Fixture " + name + " decoded successfully but should have failed");
    Require(result.error().code == expected, "Fixture " + name + " reported " +
                                                 zarr_metadata::ErrorCodeName(result.error().code) + " instead of " +
                                                 zarr_metadata::ErrorCodeName(expected));
    Require(result.error().node_path == name, "Fixture " + name + " did not report its node path");
}

// Codec chains and chunk key encodings that XRADIO can emit must all decode to the same values.
void TestSupportedLayouts() {
    const std::vector<std::string> expected{"A", "BC"};
    ExpectValues("zstd_overhang", expected);
    ExpectValues("gzip", expected);
    ExpectValues("blosc", expected);
    ExpectValues("crc_before_after_zstd", expected);
    ExpectValues("v2_key", expected);
    ExpectValues("dot_key", expected);
}

// The bytes codec endianness must be honoured, including for multi-byte code points.
void TestBigEndian() {
    ExpectValues("big_endian", std::vector<std::string>{"\xCE\xA9", "\xF0\x9F\x99\x82"});
}

// A chunk that was never written falls back to the empty fill value.
void TestMissingChunk() {
    ExpectValues("missing_chunk", std::vector<std::string>{"", ""});
}

// Corrupted chunks must be reported, never decoded past the end of the buffer.
void TestCorruptChunks() {
    ExpectError("crc_mismatch", ErrorCode::decode_error);
    ExpectError("truncated", ErrorCode::decode_error);
    ExpectError("invalid_unicode", ErrorCode::decode_error);
}

}  // namespace

int main() {
    try {
        Require(std::filesystem::is_directory(kFixtureDir), "Missing string fixture directory " + kFixtureDir.string());
        TestSupportedLayouts();
        TestBigEndian();
        TestMissingChunk();
        TestCorruptChunks();
        std::cout << "carta-zarr string array tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carta-zarr string array tests failed: " << error.what() << '\n';
        return 1;
    }
}
