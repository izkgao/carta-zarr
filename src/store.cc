/*
 * This file is part of the CARTA Image Viewer: https://github.com/CARTAvis
 * Copyright 2026 Academia Sinica Institute of Astronomy and Astrophysics (ASIAA)
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "store.h"

#include "zarr/array_metadata.h"
#include "zarr/pixel_reader.h"
#include "zarr/string_array.h"
#include "zarr/transport.h"
#include "zarr/value_reader.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace carta::zarr::internal {
namespace {

namespace zarr_metadata = ::carta::zarr::internal::zarr;

// Parse metadata bytes handed up by a Transport. Parsing lives above the seam so that every
// Transport reports a malformed node the same way.
Result<nlohmann::json> ParseNodeMetadata(const std::string& bytes, std::string_view node_path) {
    try {
        return nlohmann::json::parse(bytes);
    } catch (const nlohmann::json::parse_error& error) {
        return Error{ErrorCode::invalid_metadata, "Invalid JSON in Zarr metadata: " + std::string(error.what()),
                     std::string(node_path)};
    } catch (const std::exception& error) {
        return Error{ErrorCode::io_error, "Unable to read Zarr metadata: " + std::string(error.what()),
                     std::string(node_path)};
    }
}

// A node name is a relative path carrying no ".." component. Validating it here rather than in a
// Transport holds every Transport to the same rule, and yields the key the caches are stored under.
//
// A "." component is dropped rather than refused, because "./SKY" names the node "SKY" and a store
// is entitled to spell it that way. This used to be the one place the two rules disagreed: a
// metadata read went through here and was served, and the pixel read that followed went through
// Transport::ArrayDirectory, which refused the same name -- so an image declaring `flag: "./MASK_0"`
// described perfectly, reported a pixel mask, and then failed every masked read.
Result<std::string> NormalizeNodeName(std::string_view node) {
    const std::filesystem::path relative(node);
    if (relative.empty() || relative.is_absolute() || relative.has_root_path()) {
        return Error{ErrorCode::invalid_argument, "Invalid Zarr node path", std::string(node)};
    }
    std::filesystem::path normalized;
    for (const auto& part : relative) {
        if (part == "..") {
            return Error{ErrorCode::invalid_argument, "Invalid Zarr node path", std::string(node)};
        }
        if (part == "." || part.empty()) {
            continue;
        }
        normalized /= part;
    }
    if (normalized.empty()) {
        return Error{ErrorCode::invalid_argument, "Invalid Zarr node path", std::string(node)};
    }
    return normalized.generic_string();
}

std::string NormalizeMetadataKey(std::string key) {
    while (!key.empty() && key.front() == '/') {
        key.erase(key.begin());
    }
    constexpr std::string_view suffix = "/zarr.json";
    if (key.size() >= suffix.size() && key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
        key.erase(key.size() - suffix.size());
    }
    if (key == "zarr.json") {
        key.clear();
    }
    return key;
}

// The documents are moved out of the root rather than copied out of it. The root is parsed here and
// nothing reads it again afterwards, and these are whole node metadata documents: copying them is
// the single largest cost of opening a consolidated store, and it buys a second copy of what the
// store is about to own anyway.
//
// Each key is filed under the node's canonical name, the one every read of that node is keyed by.
// A key is first stripped of what makes it name a document rather than a node -- a leading slash, a
// trailing zarr.json -- and then held to the same rule as any other node name. Filed under what the
// block happened to spell, a copy keyed "./SKY" was never found by a read of "SKY": the node was
// read again, cached twice, and listed under a name no caller asks for.
//
// Two ways the block can be wrong about itself are refusals rather than keys to pass over. For a
// consolidated store the block is the whole listing -- nothing enumerates the transport behind it --
// so a key dropped here is a node missing from the dataset, and a variable that disappears without
// a word is a worse answer than a store that is refused with one.
Result<void> CollectConsolidatedMetadata(nlohmann::json& metadata, const std::string& prefix,
                                         std::map<std::string, nlohmann::json>& output) {
    if (!metadata.is_object()) {
        return {};
    }
    for (auto& entry : metadata.items()) {
        const std::string spelled = prefix.empty() ? entry.key() : prefix + "/" + entry.key();
        std::string path = NormalizeMetadataKey(spelled);
        auto& value = entry.value();
        if (value.is_object() && value.contains("node_type")) {
            // The root's own document, which the store has already read as itself.
            if (path.empty()) {
                continue;
            }
            auto node = NormalizeNodeName(path);
            if (!node) {
                return Error{ErrorCode::invalid_metadata,
                             "Zarr consolidated_metadata lists '" + spelled + "', which is not a node path",
                             "zarr.json"};
            }
            if (!output.emplace(node.value(), std::move(value)).second) {
                return Error{ErrorCode::invalid_metadata,
                             "Zarr consolidated_metadata lists the node '" + node.value() + "' twice, once as '" +
                                 spelled + "'",
                             "zarr.json"};
            }
        } else if (value.is_object()) {
            if (auto nested = CollectConsolidatedMetadata(value, path, output); !nested) {
                return nested;
            }
        }
    }
    return {};
}

}  // namespace

Store::Store(TransportPtr transport, nlohmann::json root_attributes,
             std::map<std::string, nlohmann::json> consolidated_metadata, bool has_consolidated_metadata,
             StoreContextPtr context)
    : _transport(std::move(transport)),
      _root_attributes(std::move(root_attributes)),
      _has_consolidated_metadata(has_consolidated_metadata),
      _context(std::move(context)),
      _caches(std::make_shared<StoreCaches>()) {
    // The root's copy is node metadata this store has already read, so it goes where node metadata
    // is kept. Reading one of these nodes is then a lookup in the one table, rather than a second
    // table consulted first and a document copied out of it on every call.
    _consolidated_nodes.reserve(consolidated_metadata.size());
    for (auto& [node, metadata] : consolidated_metadata) {
        _consolidated_nodes.push_back(node);
        _caches->node_metadata.Insert(node, Result<nlohmann::json>{std::move(metadata)});
    }
}

Result<Store> OpenStore(std::string_view location, StoreContextPtr context) {
    auto transport = OpenFilesystemTransport(location);
    if (!transport) {
        return transport.error();
    }
    return OpenStore(std::move(transport.value()), std::move(context));
}

Result<Store> OpenStore(TransportPtr transport, StoreContextPtr context) {
    if (!transport) {
        return Error{ErrorCode::invalid_argument, "Zarr transport must not be null"};
    }

    auto bytes = transport->ReadNodeBytes({});
    if (!bytes) {
        // A transport with no root node is not a Zarr store at all, whatever else it holds.
        if (bytes.error().code == ErrorCode::not_found) {
            return Error{ErrorCode::not_zarr, "Zarr store is missing zarr.json", bytes.error().node_path};
        }
        return bytes.error();
    }

    auto metadata_result = ParseNodeMetadata(bytes.value(), "zarr.json");
    if (!metadata_result) {
        return metadata_result.error();
    }
    nlohmann::json& metadata = metadata_result.value();
    if (!metadata.is_object()) {
        return Error{ErrorCode::invalid_metadata, "Root Zarr metadata must be a JSON object", "zarr.json"};
    }
    if (!metadata.contains("zarr_format") ||
        !::carta::zarr::internal::zarr::IsNonNegativeInteger(metadata.at("zarr_format"))) {
        return Error{ErrorCode::invalid_metadata, "Root Zarr metadata has no valid zarr_format", "zarr.json"};
    }
    if (metadata.at("zarr_format").get<std::uint64_t>() != 3) {
        return Error{ErrorCode::unsupported_zarr_version, "Only Zarr format 3 is supported", "zarr.json"};
    }
    if (metadata.value("node_type", "") != "group") {
        return Error{ErrorCode::not_zarr, "The Zarr root must be a group", "zarr.json"};
    }

    // Taken before the children are moved out from under it, so that what the root says about itself
    // does not depend on the order of the two.
    auto root_attributes = metadata.value("attributes", nlohmann::json::object());

    std::map<std::string, nlohmann::json> consolidated;
    bool has_consolidated = false;
    // Null is no consolidation, as zarr-python reads it: the hierarchy is listed, as it is for a
    // root without the member at all.
    if (metadata.contains("consolidated_metadata") && !metadata.at("consolidated_metadata").is_null()) {
        auto& block = metadata.at("consolidated_metadata");
        if (!block.is_object() || !block.contains("metadata") || !block.at("metadata").is_object()) {
            return Error{ErrorCode::invalid_metadata,
                         "Zarr consolidated_metadata must contain an object metadata member", "zarr.json"};
        }
        has_consolidated = true;
        if (auto collected = CollectConsolidatedMetadata(block.at("metadata"), {}, consolidated); !collected) {
            return collected.error();
        }
    }

    return Store{std::move(transport), std::move(root_attributes), std::move(consolidated), has_consolidated,
                 std::move(context)};
}

const nlohmann::json& Store::RootAttributes() const noexcept {
    return _root_attributes;
}

const Result<nlohmann::json>& Store::ReadNodeMetadata(std::string_view node) const {
    auto node_name_result = NormalizeNodeName(node);
    if (!node_name_result) {
        // A rejected name is remembered like any other answer, so that everything handed back from
        // here is a reference to something the store owns. It is keyed by the name as asked for,
        // which cannot collide with a normalized one: this is the name normalization refused.
        return _caches->node_metadata.GetOrCompute(
            std::string(node), [&]() -> Result<nlohmann::json> { return node_name_result.error(); });
    }
    const std::string node_name = std::move(node_name_result.value());

    // The root's consolidated copy was put in this table when the store was built, so a node it
    // accounted for is found here and never read. Consolidated metadata is a copy that saves a
    // read, not a substitute for the document it copies: it carries `must_understand: false`, so a
    // reader that ignores it reads the same hierarchy, and the array data behind these names is
    // opened by TensorStore from each array's own metadata whatever this decides. A node the copy
    // does not mention is therefore read, not missing.
    return _caches->node_metadata.GetOrCompute(node_name, [&]() -> Result<nlohmann::json> {
        auto bytes = _transport->ReadNodeBytes(node_name);
        if (!bytes) {
            return bytes.error();
        }
        return ParseNodeMetadata(bytes.value(), node);
    });
}

const Result<zarr::ArrayMetadata>& Store::ReadArrayMetadata(std::string_view node) const {
    // Keyed by the normalized name, as the node metadata below it is: two spellings of one node are
    // one node, and keying by what the caller typed would read and parse it twice. A name that
    // cannot be normalized has no such key, so it is filed under what the caller typed -- the one
    // case where two spellings do not share an entry, and both of them are refusals anyway.
    auto key = NormalizeNodeName(node);
    const std::string cache_key = key ? key.value() : std::string(node);
    return _caches->array_metadata.GetOrCompute(cache_key, [&]() -> Result<zarr::ArrayMetadata> {
        if (!key) {
            return key.error();
        }
        const auto& metadata_result = ReadNodeMetadata(node);
        if (!metadata_result) {
            return metadata_result.error();
        }
        return zarr_metadata::ParseArrayMetadata(metadata_result.value(), node);
    });
}

const Result<std::vector<NodeEntry>>& Store::Inventory() const {
    using Listing = Result<std::vector<NodeEntry>>;
    return _caches->inventory.GetOrCompute([&]() -> Listing {
        std::vector<std::string> node_names;
        // The one place the copy is taken as the whole truth rather than as a first look. Listing
        // is what consolidated metadata exists to avoid -- on a store reached over a network,
        // enumerating a hierarchy is the expensive question -- so a store that consolidated its
        // metadata is taken at its word about which nodes it has. The cost is a node added after
        // the copy was written: it is invisible until the dataset is consolidated again, the same
        // staleness zarr-python's own consolidated open accepts. Reading a node it does not
        // mention still falls through to the node itself, above.
        if (_has_consolidated_metadata) {
            // Canonical already: they were filed that way when the store was built.
            node_names = _consolidated_nodes;
        } else {
            auto listed = _transport->ListNodes();
            if (!listed) {
                return listed.error();
            }
            // Held to the rule every other node name is held to, so that a listed name is the key
            // its reads are cached under. A name that rule refuses cannot be read at all, and a
            // transport that lists one is wrong about where its own nodes are -- which is not the
            // store's metadata being wrong, and is not something to diagnose around.
            node_names.reserve(listed.value().size());
            for (const auto& name : listed.value()) {
                auto canonical = NormalizeNodeName(name);
                if (!canonical) {
                    return Error{ErrorCode::io_error, "Zarr transport listed a node by an invalid path", name};
                }
                node_names.push_back(std::move(canonical.value()));
            }
        }

        std::sort(node_names.begin(), node_names.end());
        node_names.erase(std::unique(node_names.begin(), node_names.end()), node_names.end());

        // Every node is read here, so an inventory either accounts for the whole hierarchy or
        // reports why it cannot.
        std::vector<NodeEntry> entries;
        entries.reserve(node_names.size());
        for (auto& node : node_names) {
            const auto& metadata = ReadNodeMetadata(node);
            if (!metadata) {
                // A document that was read and would not parse says nothing about what the node is,
                // and that is an answer about the node rather than about the hierarchy: the other
                // nodes were read, and the images among them still open. CONTEXT.md says so of a
                // node whose metadata would not parse -- diagnosed rather than refused -- and this
                // used to refuse the whole dataset over one.
                //
                // Only that failure. A read that failed at all -- a transport error, a node listed
                // and then gone -- says the hierarchy could not be taken, and is reported as such.
                if (metadata.error().code != ErrorCode::invalid_metadata) {
                    return metadata.error();
                }
                entries.push_back(NodeEntry{std::move(node), NodeKind::unrecognised, nullptr, metadata.error()});
                continue;
            }
            NodeEntry entry{std::move(node), NodeKind::unrecognised, nullptr, std::nullopt};

            // Looked up rather than taken with a default, because a node_type that is present and
            // not a string is a document that does not say what it is -- and asking for it as a
            // string would throw rather than answer.
            const auto& document = metadata.value();
            const auto declared = document.find("node_type");
            const std::string node_type =
                declared != document.end() && declared->is_string() ? declared->get<std::string>() : std::string{};

            if (node_type == "group") {
                entry.kind = NodeKind::group;
            } else if (node_type == "array") {
                entry.kind = NodeKind::array;
                entry.array = &ReadArrayMetadata(entry.name);
            } else {
                entry.reason = Error{ErrorCode::invalid_metadata, "Zarr node is neither a group nor an array",
                                     entry.name};
            }
            entries.push_back(std::move(entry));
        }
        return entries;
    });
}

const NodeEntry* Store::FindNode(std::string_view name) const {
    const auto& inventory = Inventory();
    if (!inventory) {
        return nullptr;
    }
    const auto& entries = inventory.value();
    const auto found = std::lower_bound(entries.begin(), entries.end(), name,
                                        [](const NodeEntry& entry, std::string_view wanted) {
                                            return std::string_view(entry.name) < wanted;
                                        });
    return found != entries.end() && found->name == name ? &*found : nullptr;
}

Result<std::vector<double>> Store::ReadNumericArray(std::string_view node) const {
    auto key = NormalizeNodeName(node);
    if (!key) {
        return key.error();
    }
    return _caches->double_arrays.GetOrCompute(key.value(), [&] { return ReadNumericArrayUncached(node); });
}

Result<std::vector<double>> Store::ReadNumericArrayUncached(std::string_view node) const {
    auto array_path = ResolveArrayDirectory(node);
    if (!array_path) {
        return array_path.error();
    }
    try {
        return zarr_metadata::ReadNumericValues(array_path.value(), _context, node);
    } catch (const std::exception& e) {
        return Error{ErrorCode::io_error, e.what(), std::string(node)};
    }
}

// The one place the store asks where an array's bytes are. A transport answers with a location that
// does not depend on the caller's working directory, so nothing here or below normalizes: this ran
// weakly_canonical on every pixel read, and what it was protecting against belongs at the root,
// which is resolved once when the transport opens.
Result<std::filesystem::path> Store::ResolveArrayDirectory(std::string_view node) const {
    auto name = NormalizeNodeName(node);
    if (!name) {
        return name.error();
    }
    return _transport->ArrayDirectory(name.value());
}

template <typename T>
Result<void> Store::ReadPixelsInto(std::string_view node, const zarr::PixelSelection& selection,
                                   BufferView<T> destination, const ReadControl& control) const {
    try {
        const auto& metadata = ReadArrayMetadata(node);
        if (!metadata) {
            return metadata.error();
        }
        auto target_path = ResolveArrayDirectory(node);
        if (!target_path) {
            return target_path.error();
        }
        if constexpr (std::is_same_v<T, float>) {
            return zarr_metadata::ReadFloat32(target_path.value(), _context, node, metadata.value(),
                                              selection, destination, control);
        } else {
            return zarr_metadata::ReadMaskBytes(target_path.value(), _context, node, metadata.value(),
                                                selection, destination, control);
        }
    } catch (const std::exception& e) {
        return Error{ErrorCode::io_error, e.what(), std::string(node)};
    }
}

template Result<void> Store::ReadPixelsInto<float>(std::string_view, const zarr::PixelSelection&, BufferView<float>,
                                                   const ReadControl&) const;
template Result<void> Store::ReadPixelsInto<std::uint8_t>(std::string_view, const zarr::PixelSelection&,
                                                          BufferView<std::uint8_t>, const ReadControl&) const;

Result<std::vector<std::string>> Store::ReadStringArray1D(std::string_view node) const {
    auto key = NormalizeNodeName(node);
    if (!key) {
        return key.error();
    }
    return _caches->string_arrays.GetOrCompute(key.value(), [&] { return ReadStringArray1DUncached(node); });
}

Result<std::vector<std::string>> Store::ReadStringArray1DUncached(std::string_view node) const {
    const auto& array_meta_res = ReadArrayMetadata(node);
    if (!array_meta_res) {
        return array_meta_res.error();
    }
    auto array_path = ResolveArrayDirectory(node);
    if (!array_path) {
        return array_path.error();
    }
    try {
        return zarr_metadata::ReadFixedLengthUtf32StringArray(array_path.value(), array_meta_res.value(), node);
    } catch (const std::exception& e) {
        // io_error, as in every other read on this Store. What the decoder itself refuses comes back
        // as a Result with its own code; what escapes as an exception is a file or an allocation,
        // which is nothing the caller passed in.
        return Error{ErrorCode::io_error, e.what(), std::string(node)};
    }
}

Result<std::uint64_t> Store::StoredSizeBytes(std::chrono::steady_clock::time_point deadline) const {
    return _transport->StoredSizeBytes(deadline);
}

}  // namespace carta::zarr::internal
