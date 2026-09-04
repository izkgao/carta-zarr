/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "string_array.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include <blosc.h>
#define ZLIB_CONST
#include <zlib.h>
#include <zstd.h>

namespace carta::zarr::internal::zarr {
namespace {

// Internal helpers report failures by throwing; ReadFixedLengthUtf32StringArray converts them into
// the library's structured errors at the boundary.
class DecodeFailure : public std::runtime_error {
public:
    DecodeFailure(ErrorCode code, const std::string& message) : std::runtime_error(message), code(code) {}

    ErrorCode code;
};

[[noreturn]] void Fail(ErrorCode code, const std::string& message) {
    throw DecodeFailure(code, message);
}

constexpr std::uint32_t kUnicodeMax = 0x10FFFF;
constexpr std::uint32_t kSurrogateMin = 0xD800;
constexpr std::uint32_t kSurrogateMax = 0xDFFF;

// Append a single Unicode code point to a UTF-8 string using the standard UTF-8 bit patterns.
void AppendUtf8(std::string& out, std::uint32_t code_point) {
    if (code_point > kUnicodeMax || (code_point >= kSurrogateMin && code_point <= kSurrogateMax)) {
        Fail(ErrorCode::decode_error, "Invalid Unicode code point");
    }
    if (code_point < 0x80) {
        // 1-byte sequence (ASCII): 0xxxxxxx
        out.push_back(static_cast<char>(code_point));
    } else if (code_point < 0x800) {
        // 2-byte sequence: 110xxxxx 10xxxxxx
        out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else if (code_point < 0x10000) {
        // 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
        out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else {
        // 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    }
}

// Read a 32-bit unsigned integer from a 4-byte buffer with the given endianness.
std::uint32_t ReadUint32(const std::uint8_t* bytes, bool little_endian) {
    if (little_endian) {
        return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
               (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
    }
    return static_cast<std::uint32_t>(bytes[3]) | (static_cast<std::uint32_t>(bytes[2]) << 8) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) | (static_cast<std::uint32_t>(bytes[0]) << 24);
}

// Decode a contiguous buffer of fixed-length UTF-32 strings into UTF-8 strings (null padding stripped).
std::vector<std::string> DecodeFixedLengthUtf32(const std::vector<std::uint8_t>& bytes, std::size_t num_elements,
                                                std::size_t length_bytes, bool little_endian) {
    const std::size_t chars_per_element = length_bytes / sizeof(std::uint32_t);
    std::vector<std::string> result;
    result.reserve(num_elements);

    for (std::size_t element = 0; element < num_elements; ++element) {
        std::string value;
        for (std::size_t code_index = 0; code_index < chars_per_element; ++code_index) {
            const std::size_t offset = (element * length_bytes) + (code_index * sizeof(std::uint32_t));
            if (offset + sizeof(std::uint32_t) > bytes.size()) {
                Fail(ErrorCode::decode_error, "String array chunk ended before the last element");
            }
            const std::uint32_t code_point = ReadUint32(bytes.data() + offset, little_endian);
            if (code_point == 0) {
                break;  // trailing null padding
            }
            AppendUtf8(value, code_point);
        }
        result.push_back(std::move(value));
    }

    return result;
}

enum class BytesCodec {
    blosc,
    gzip,
    zstd,
};

// Decompress a Zstandard frame into a byte buffer.
std::size_t ZstdDecompress(const std::vector<std::uint8_t>& compressed, std::vector<std::uint8_t>& decompressed) {
    const std::size_t decompressed_size =
        ZSTD_decompress(decompressed.data(), decompressed.size(), compressed.data(), compressed.size());
    if (ZSTD_isError(decompressed_size) != 0U) {
        Fail(ErrorCode::decode_error,
             std::string("Zstd decompression failed: ") + ZSTD_getErrorName(decompressed_size));
    }
    return decompressed_size;
}

std::size_t GzipDecompress(const std::vector<std::uint8_t>& compressed, std::vector<std::uint8_t>& decompressed) {
    if (compressed.size() > std::numeric_limits<uInt>::max()) {
        Fail(ErrorCode::decode_error, "Gzip buffer is too large to decompress");
    }
    if (decompressed.size() > std::numeric_limits<uInt>::max()) {
        Fail(ErrorCode::decode_error, "Gzip expected output is too large to decompress");
    }

    z_stream stream{};
    stream.next_in = reinterpret_cast<const Bytef*>(compressed.data());
    stream.avail_in = static_cast<uInt>(compressed.size());

    constexpr int gzip_window_bits = 16 + MAX_WBITS;
    int status = inflateInit2(&stream, gzip_window_bits);
    if (status != Z_OK) {
        Fail(ErrorCode::decode_error, "Failed to initialize gzip decompression");
    }

    stream.next_out = reinterpret_cast<Bytef*>(decompressed.data());
    stream.avail_out = static_cast<uInt>(decompressed.size());

    status = inflate(&stream, Z_FINISH);
    const std::size_t decompressed_size = stream.total_out;
    inflateEnd(&stream);

    if (status != Z_STREAM_END) {
        if (status == Z_OK && stream.avail_out == 0) {
            Fail(ErrorCode::decode_error, "Gzip decompressed size exceeds the expected array size");
        }
        Fail(ErrorCode::decode_error, std::string("Gzip decompression failed: ") + zError(status));
    }
    return decompressed_size;
}

std::size_t BloscDecompress(const std::vector<std::uint8_t>& compressed, std::vector<std::uint8_t>& decompressed) {
    static std::once_flag blosc_init_once;

    if (compressed.size() < BLOSC_MIN_HEADER_LENGTH) {
        Fail(ErrorCode::decode_error, "Blosc buffer is smaller than the header");
    }

    std::size_t decompressed_bytes = 0;
    std::size_t compressed_bytes = 0;
    std::size_t block_size = 0;
    blosc_cbuffer_sizes(compressed.data(), &decompressed_bytes, &compressed_bytes, &block_size);
    if (compressed_bytes > compressed.size()) {
        Fail(ErrorCode::decode_error, "Blosc buffer is truncated");
    }
    if (decompressed_bytes != decompressed.size()) {
        Fail(ErrorCode::decode_error, "Blosc decompressed size does not match the expected chunk size");
    }

    std::call_once(blosc_init_once, []() { blosc_init(); });
    const int decompressed_size = blosc_decompress(compressed.data(), decompressed.data(), decompressed.size());
    if (decompressed_size < 0) {
        Fail(ErrorCode::decode_error, "Blosc decompression failed");
    }
    if (static_cast<std::size_t>(decompressed_size) != decompressed.size()) {
        Fail(ErrorCode::decode_error, "Blosc produced an unexpected decompressed size");
    }
    return static_cast<std::size_t>(decompressed_size);
}

std::size_t DecompressBytesCodec(const std::vector<std::uint8_t>& compressed, BytesCodec codec,
                                 std::vector<std::uint8_t>& decompressed) {
    switch (codec) {
        case BytesCodec::blosc:
            return BloscDecompress(compressed, decompressed);
        case BytesCodec::gzip:
            return GzipDecompress(compressed, decompressed);
        case BytesCodec::zstd:
            return ZstdDecompress(compressed, decompressed);
    }
    Fail(ErrorCode::unsupported_codec, "Unsupported bytes codec");
}

// CRC32C (Castagnoli polynomial, RFC 3720), as required by the Zarr v3 crc32c codec.
std::uint32_t Crc32c(const std::uint8_t* data, std::size_t size) {
    constexpr std::uint32_t reversed_polynomial = 0x82F63B78U;
    static const std::array<std::uint32_t, 256> table = []() {
        std::array<std::uint32_t, 256> entries{};
        for (std::uint32_t index = 0; index < entries.size(); ++index) {
            std::uint32_t crc = index;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc & 1) ? ((crc >> 1) ^ reversed_polynomial) : (crc >> 1);
            }
            entries.at(index) = crc;
        }
        return entries;
    }();

    std::uint32_t crc = 0xFFFFFFFFU;
    for (std::size_t i = 0; i < size; ++i) {
        crc = table.at((crc ^ data[i]) & 0xFFU) ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFU;
}

// Verify and remove one 4-byte CRC32C suffix (stored little-endian regardless of the bytes codec
// endianness); returns the payload size without the checksum.
std::size_t StripCrc32c(const std::uint8_t* data, std::size_t size) {
    if (size < sizeof(std::uint32_t)) {
        Fail(ErrorCode::decode_error, "String array chunk is too small to contain a CRC32C checksum");
    }
    const std::size_t payload_size = size - sizeof(std::uint32_t);
    const std::uint32_t stored = ReadUint32(data + payload_size, true);
    const std::uint32_t computed = Crc32c(data, payload_size);
    if (stored != computed) {
        Fail(ErrorCode::decode_error, "String array CRC32C checksum mismatch");
    }
    return payload_size;
}

// Codec layout for a Zarr v3 string array: the optional bytes-to-bytes compressor, byte order, and
// the number of crc32c codecs on each side of the compressor (in codec-chain / encode order).
struct StringCodecInfo {
    std::optional<BytesCodec> bytes_codec;
    bool little_endian = true;
    std::size_t crc_before_compressor =
        0;  // encoded before the compressor: suffix ends up inside the compressed payload
    std::size_t crc_after_compressor = 0;  // encoded after the compressor: suffix wraps the stored chunk on disk
};

void SetBytesCodec(StringCodecInfo& info, BytesCodec codec) {
    if (info.bytes_codec) {
        Fail(ErrorCode::unsupported_codec,
             "String array uses multiple bytes compressors; only one of zstd, gzip, or blosc is supported");
    }
    info.bytes_codec = codec;
}

StringCodecInfo ParseStringCodecs(const nlohmann::json& metadata) {
    StringCodecInfo info;
    if (!metadata.contains("codecs") || !metadata.at("codecs").is_array()) {
        return info;
    }
    bool seen_bytes = false;
    for (const auto& codec : metadata.at("codecs")) {
        const std::string codec_name = codec.is_object() ? codec.value("name", "") : "";
        if (codec_name == "blosc") {
            SetBytesCodec(info, BytesCodec::blosc);
        } else if (codec_name == "gzip") {
            SetBytesCodec(info, BytesCodec::gzip);
        } else if (codec_name == "zstd") {
            SetBytesCodec(info, BytesCodec::zstd);
        } else if (codec_name == "bytes") {
            if (seen_bytes) {
                Fail(ErrorCode::unsupported_codec, "String array has multiple bytes codecs in its codec chain");
            }
            seen_bytes = true;
            if (codec.contains("configuration") && codec.at("configuration").is_object() &&
                codec.at("configuration").contains("endian") && codec.at("configuration").at("endian").is_string()) {
                info.little_endian = codec.at("configuration").at("endian").get<std::string>() != "big";
            }
        } else if (codec_name == "crc32c") {
            // The compressor (if any) is recorded once seen, so its presence distinguishes the two sides.
            ++(info.bytes_codec ? info.crc_after_compressor : info.crc_before_compressor);
        } else if (codec_name == "transpose") {
            // For a 1-D array the only valid permutation is the identity [0], which is a no-op.
            const bool identity = codec.contains("configuration") && codec.at("configuration").is_object() &&
                                  codec.at("configuration").contains("order") &&
                                  codec.at("configuration").at("order") == nlohmann::json::array({0});
            if (!identity) {
                Fail(ErrorCode::unsupported_codec,
                     "String array uses a non-identity transpose; unsupported for 1-D string decode");
            }
        } else {
            Fail(ErrorCode::unsupported_codec, "String array uses unsupported codec '" + codec_name +
                                                   "'; only bytes, transpose, zstd, gzip, blosc, and crc32c are "
                                                   "supported");
        }
    }
    return info;
}

struct StringArrayLayout {
    std::size_t num_elements;
    std::size_t chunk_elements;
    std::size_t length_bytes;
};

StringArrayLayout ParseStringArrayLayout(const ArrayMetadata& array_metadata) {
    if (!IsFixedLengthUtf32(array_metadata)) {
        Fail(ErrorCode::unsupported_data_type, "Array is not a valid fixed_length_utf32 string array");
    }
    if (array_metadata.shape.size() != 1 || array_metadata.chunk_shape.size() != 1) {
        Fail(ErrorCode::unsupported_data_type, "Only 1-D string arrays are supported");
    }
    if (array_metadata.shape.front() > array_metadata.chunk_shape.front()) {
        Fail(ErrorCode::unsupported_codec, "String array is multi-chunk; unsupported for string decode");
    }

    return StringArrayLayout{static_cast<std::size_t>(array_metadata.shape.front()),
                             static_cast<std::size_t>(array_metadata.chunk_shape.front()),
                             array_metadata.data_type_configuration.at("length_bytes").get<std::size_t>()};
}

std::filesystem::path GetStringChunkPath(const std::filesystem::path& array_dir, const nlohmann::json& metadata) {
    std::string key_encoding = "default";
    std::string separator = "/";
    if (metadata.contains("chunk_key_encoding") && metadata.at("chunk_key_encoding").is_object()) {
        const auto& encoding = metadata.at("chunk_key_encoding");
        if (encoding.contains("name") && encoding.at("name").is_string()) {
            key_encoding = encoding.at("name").get<std::string>();
        }
        if (encoding.contains("configuration") && encoding.at("configuration").is_object() &&
            encoding.at("configuration").contains("separator") &&
            encoding.at("configuration").at("separator").is_string()) {
            separator = encoding.at("configuration").at("separator").get<std::string>();
        }
    }
    if (key_encoding == "default") {
        if (separator != "/" && separator != ".") {
            Fail(ErrorCode::unsupported_codec, "String array has invalid chunk key separator '" + separator + "'");
        }
        return array_dir / ("c" + separator + "0");
    }
    if (key_encoding == "v2") {
        return array_dir / "0";
    }
    Fail(ErrorCode::unsupported_codec, "String array uses unsupported chunk key encoding '" + key_encoding + "'");
}

std::vector<std::uint8_t> ReadChunkFile(const std::filesystem::path& chunk_path) {
    std::ifstream chunk_file(chunk_path, std::ios::binary);
    if (!chunk_file) {
        Fail(ErrorCode::io_error, "Failed to open array data " + chunk_path.string());
    }
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(chunk_file)), std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> DecodeStringChunk(std::vector<std::uint8_t> bytes, const StringCodecInfo& codec_info,
                                            const StringArrayLayout& layout) {
    if (layout.length_bytes != 0 &&
        layout.chunk_elements > std::numeric_limits<std::size_t>::max() / layout.length_bytes) {
        Fail(ErrorCode::invalid_metadata, "String array expected byte count overflows size_t");
    }
    const std::size_t expected_chunk_bytes = layout.chunk_elements * layout.length_bytes;

    std::size_t bytes_size = bytes.size();
    // crc32c after the compressor wraps the stored chunk; verify and strip it before decompressing.
    for (std::size_t i = 0; i < codec_info.crc_after_compressor; ++i) {
        bytes_size = StripCrc32c(bytes.data(), bytes_size);
    }

    if (codec_info.bytes_codec) {
        // The decompressor input must be exactly the compressed stream (zstd rejects trailing bytes).
        bytes.resize(bytes_size);
        // crc32c before the compressor leaves its suffix inside the decompressed payload.
        if (codec_info.crc_before_compressor >
            (std::numeric_limits<std::size_t>::max() - expected_chunk_bytes) / sizeof(std::uint32_t)) {
            Fail(ErrorCode::invalid_metadata, "String array expected checksum byte count overflows size_t");
        }
        std::vector<std::uint8_t> decompressed(expected_chunk_bytes +
                                               (sizeof(std::uint32_t) * codec_info.crc_before_compressor));
        bytes_size = DecompressBytesCodec(bytes, *codec_info.bytes_codec, decompressed);
        bytes = std::move(decompressed);
    }

    // crc32c before the compressor wraps the (now decompressed) payload; strip it last.
    for (std::size_t i = 0; i < codec_info.crc_before_compressor; ++i) {
        bytes_size = StripCrc32c(bytes.data(), bytes_size);
    }

    if (bytes_size < expected_chunk_bytes) {
        Fail(ErrorCode::decode_error, "String array chunk is smaller than expected");
    }
    bytes.resize(bytes_size);
    return bytes;
}

}  // namespace

Result<std::vector<std::string>> ReadFixedLengthUtf32StringArray(const std::filesystem::path& array_dir,
                                                                 const ArrayMetadata& array_metadata,
                                                                 const nlohmann::json& metadata,
                                                                 std::string_view node) {
    try {
        const StringArrayLayout layout = ParseStringArrayLayout(array_metadata);
        const std::filesystem::path chunk_path = GetStringChunkPath(array_dir, metadata);

        // Missing chunk: all elements take the (empty) fill value.
        std::error_code error;
        if (!std::filesystem::exists(chunk_path, error)) {
            if (error) {
                return Error{ErrorCode::io_error, "Unable to inspect string array chunk: " + error.message(),
                             std::string(node)};
            }
            return std::vector<std::string>(layout.num_elements, std::string());
        }

        std::vector<std::uint8_t> bytes = ReadChunkFile(chunk_path);
        const StringCodecInfo codec_info = ParseStringCodecs(metadata);
        bytes = DecodeStringChunk(std::move(bytes), codec_info, layout);
        return DecodeFixedLengthUtf32(bytes, layout.num_elements, layout.length_bytes, codec_info.little_endian);
    } catch (const DecodeFailure& failure) {
        return Error{failure.code, failure.what(), std::string(node)};
    } catch (const std::exception& error) {
        return Error{ErrorCode::decode_error, error.what(), std::string(node)};
    }
}

}  // namespace carta::zarr::internal::zarr
