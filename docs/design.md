# carta-zarr library design draft

Status: experimental discussion draft

Primary consumer: `carta-backend`

Initial scope: read-only `SKY` images following XRADIO v1.2, stored on a local filesystem

## 1. Motivation

Move Zarr-specific storage and XRADIO image interpretation out of `carta-backend` into an independently built and versioned C++ library.

The first concrete outcome is that TensorStore exists in and is built by `carta-zarr`, but does not appear in the `carta-backend` repository, public headers, or direct build dependencies. The backend consumes a separately built `carta-zarr` library and depends only on its public API. The library should eventually also be usable by unrelated C++ projects, including projects which install its development package through `apt`.

This is not merely a source-code move. The current implementation combines:

1. Zarr storage access and codec handling;
2. XRADIO-specific schema interpretation;
3. CARTA and casacore integration.

Those responsibilities need separate boundaries before the code becomes a reusable library.

## 2. Confirmed decisions

- `carta-zarr` owns and builds TensorStore. TensorStore is an implementation detail and must not leak into `carta-backend`.
- `carta-zarr` must not depend on casacore or CARTA protobuf. Translation to casacore, FITS headers, and CARTA protocol types remains in `carta-backend`.
- The project is experimental and may make frequent source and API changes while the XRADIO image schema is not finalized.
- APT packaging is deferred. The architecture should remain packageable later, but packaging and ABI stability are not current acceptance criteria.
- The current implementation baseline is XRADIO v1.2, including the Zarr version and encoding selected by that release line. Future work may advance with the official XRADIO schema, but each baseline update must record the exact XRADIO patch version or commit used to generate fixtures so tests remain reproducible.
- The initial transport is a local filesystem store only.
- The initial image data variable is `SKY` only.
- The public API provides schema detection before opening a dataset. It includes a general schema probe for future adapters and an XRADIO `SKY` convenience check.
- The library accepts a `time` axis of any supported length and exposes it to callers. Any CARTA limitation or selection policy is handled by the backend adapter.
- Data-type, codec, sharding, fill, mask, and NaN behavior initially preserve only what the migrated implementation actually supports. Unsupported and not-yet-implemented behavior must be reported explicitly.
- Licensing follows the other CARTAvis repositories: GPL version 3, with source files using the CARTA `SPDX-License-Identifier: GPL-3.0-or-later` convention.

## 3. Current implementation and constraints

The current branch in `../carta-backend` has the following access path:

```text
FileLoader::GetLoader()
        |
        +-- IsSupportedZarrImage()          fast format/schema detection
        |
        +-- ZarrLoader                      CARTA FileLoader integration
              |
              +-- CartaZarrImage            casacore::ImageInterface<float>
                    |
                    +-- ZarrImage            XRADIO interpretation
                          |
                          +-- ZarrStore       Zarr v3 and TensorStore access
```

The same general pattern is used for other CARTA formats: `FitsLoader` and `Hdf5Loader` adapt format-specific image implementations to `FileLoader`, while `CartaFitsImage` and `CartaHdf5Image` implement `casacore::ImageInterface<float>`. The Zarr integration should preserve this backend-facing pattern.

Important limitations in the current Zarr implementation are:

- only local filesystem stores are supported;
- only Zarr v3 is recognized;
- only the `SKY` image array is recognized;
- the image must have `time`, `frequency`, `polarization`, `l`, and `m` dimensions;
- `time` must currently have length at most one; this restriction belongs to the existing implementation and will not be carried into the library contract;
- metadata, coordinate labels, beam metadata, FITS-header synthesis, and storage information are implemented;
- pixel slice and mask reads through `CartaZarrImage::doGetSlice()` and `doGetMaskSlice()` are not implemented;
- numeric auxiliary arrays are read with TensorStore, while the `fixed_length_utf32` coordinate arrays use a custom decoder because TensorStore does not support that extension type;
- TensorStore types and `nlohmann::json` currently escape into internal class interfaces;
- process-wide TensorStore thread-pool and cache configuration currently lives in backend startup settings.

The phrase “supports XRADIO” must therefore include the tested upstream revision. The current baseline is XRADIO v1.2. As of this draft, the upstream XRADIO documentation still describes Sky and Aperture Images as design in progress, so the library must not infer long-term schema stability from the XRADIO package version. During the experimental phase, `carta-zarr` follows upstream changes rather than defining a separate stable CARTA schema. Each baseline update must update pinned generated fixtures and the compatibility notes in the same pull request.

Relevant upstream specifications:

- [XRADIO overview and schema versioning](https://xradio.readthedocs.io/en/latest/overview.html)
- [Zarr v3 core specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/)
- [Zarr v3 sharding codec](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/)

## 4. Goals

### Required for the experimental extraction

- Provide a read-only C++ library for the `SKY` image emitted according to the tested XRADIO v1.2 baseline and its selected Zarr version.
- Let `carta-backend` discover datasets, read `SKY` metadata, and ultimately read arbitrary image slices without including or linking directly to TensorStore APIs.
- Preserve the behavior expected by the existing CARTA file-list, file-info, HDU-selection, coordinate-system, beam, and image-loading flows.
- Keep all public types independent of CARTA protobuf, `FileLoader`, casacore, TensorStore, and `nlohmann::json`.
- Build the library independently from `carta-backend`, with installable public headers and CMake package metadata for a developer prefix.
- Define precise compatibility and error behavior rather than silently accepting partially understood datasets.
- Make concurrent reads from separate image handles safe. Document whether concurrent reads on one handle are supported.

### Design goals for later releases

- Add other Zarr-based astronomy schemas without changing the storage API or the backend adapter contract.
- Add more store transports, such as HTTP or S3, without changing image-level APIs.
- Add new Zarr codecs and data types without changing schema adapters.
- Permit consumers other than CARTA to use the normalized image metadata and slice API.
- Leave room for a stable C wrapper or language bindings, without requiring them during the experimental phase.
- Publish versioned runtime and development packages through APT when the API and schema expectations are mature enough.

## 5. Non-goals for the initial extraction

- Writing or modifying Zarr datasets.
- A general-purpose replacement for the complete Zarr or xarray APIs.
- Runtime loading of third-party schema plugins.
- A public casacore image class in the core package.
- Remote object stores, non-`SKY` image variables, mipmaps, precomputed statistics, or swizzled data.
- Stable ABI or Debian packages during the experimental phase.
- Supporting multiple historical XRADIO drafts simultaneously unless required by CARTA deployments.

## 6. Recommended architecture

```text
                              public, dependency-light API
                    +----------------------------------------+
other C++ projects ->| Dataset / Image / Descriptor / Read  |
                    +-------------------+--------------------+
                                        |
                    +-------------------v--------------------+
                    | built-in schema adapters               |
                    | XRADIO v1.2 SKY first; others later    |
                    +-------------------+--------------------+
                                        |
                    +-------------------v--------------------+
                    | schema-neutral Zarr store               |
                    | node paths, JSON, consolidated metadata |
                    | array metadata, storage layout, caches  |
                    +------+--------------------------+-------+
                           |                          |
              - - - - - - -+- - transport seam - - - -+- - - - - - -
                           |                          |
            +--------------v-------+   +--------------v-----------+
            | Transport            |   | coordinate value reads   |
            | filesystem | memory  |   | (outside the seam)       |
            | HTTP/S3 later        |   +--------------+-----------+
            +----------------------+                  |
                                                      |
                    | private implementation dependency|
                    +----------- TensorStore ----------+

carta-backend:
FileLoader -> ZarrLoader -> CartaZarrImage -> public carta-zarr API
                                      |
                                      +-> casacore coordinates/FITS adapter
```

### 6.1 Storage layer

The storage layer understands Zarr hierarchy, arrays, shapes, data types, chunk grids, chunk-key encoding, codec chains, consolidated metadata, and store access. It does not know that an array is a sky image, a beam, or a frequency coordinate.

The storage layer is split at the **transport** seam. A `Transport` supplies raw access only — one
node's `zarr.json` verbatim, the list of nodes, and a node's array path — while everything that
interprets those bytes stays above it, so every transport is interpreted identically. The filesystem
transport serves production and an in-memory one serves the schema profile tests; HTTP and S3 arrive
here without the schema layer changing. Coordinate value reads sit outside the seam and take a
filesystem path, which confines TensorStore to a single translation unit. See
[ADR 0004](adr/0004-store-seam-at-the-transport.md).

TensorStore lives here as a private implementation dependency and is built by the `carta-zarr` project. Until APT packages exist, developers and CI build/install `carta-zarr` into a separate prefix and then configure `carta-backend` with `CMAKE_PREFIX_PATH` or `CartaZarr_DIR`. The backend uses `find_package(CartaZarr CONFIG REQUIRED)`; it does not vendor `carta-zarr`, add its source tree as a subdirectory, or contain TensorStore configuration.

Public headers must not include TensorStore headers or return TensorStore objects. If TensorStore is linked statically into `libcarta-zarr.so`, it must be position-independent and its symbols should be hidden by default to avoid symbol collisions in consumer processes.

### 6.2 Schema layer

The schema layer determines whether a Zarr hierarchy matches a supported image schema and maps schema-specific names, axes, coordinates, and attributes into neutral public descriptors.

The first adapter implements the XRADIO v1.2 `SKY` image baseline and is expected to evolve with later official XRADIO revisions. Schema selection should be internal and registry-based so another built-in adapter can be added later. A stable public plugin ABI is intentionally deferred; committing to it now would unnecessarily freeze internal parsing concepts.

Schema detection must distinguish:

- not a Zarr hierarchy;
- valid Zarr but no recognized image schema;
- recognized schema but unsupported version or feature;
- recognized schema with invalid required metadata;
- supported dataset.

Detection must not rely only on the presence of an array named `SKY`. The XRADIO v1.2 detector validates the schema's identifying metadata, required image dimensions, and required coordinate relationships without reading image chunks. Its exact positive and negative criteria are fixture-tested and updated with each tracked XRADIO baseline.

The internal registry asks each built-in schema adapter to probe the already-opened schema-neutral store. Detection order must not silently select a schema when more than one adapter matches; an ambiguous match is returned explicitly. This registry is an internal extension point during the experimental phase, not a stable third-party plugin ABI.

### 6.3 CARTA adapter

`ZarrLoader` and `CartaZarrImage` remain in `carta-backend`. They translate neutral library descriptors into:

- `casacore::IPosition` and `casacore::DataType`;
- `casacore::CoordinateSystem`;
- `casacore::ImageBeamSet`;
- `casacore::Slicer` requests and `casacore::Array<float>` buffers.

Keeping this adapter in the backend prevents the library from acquiring casacore, CARTA protobuf, and backend logging dependencies. It also follows the existing FITS/HDF5 reader structure.

The current FITS-header synthesis in `ZarrImage` is removed rather than split. Descriptors carry enough to construct casacore coordinates directly, so the backend builds its `CoordinateSystem` from them and then reuses the generic `GetFITSHeader()` path it already applies to other image types to produce the header entries the file-info panel displays. See [ADR 0002](adr/0002-casacore-coordinates-without-fits.md). Generic user attributes should remain accessible without pretending every JSON value can be represented by a FITS card.

### 6.4 Source ownership after extraction

| Current backend area | Destination | Notes |
|---|---|---|
| `ZarrStore.*` | `carta-zarr` | Refactor behind PImpl; TensorStore and JSON stay private |
| schema-neutral parts of `ZarrUtil.*` | `carta-zarr` | Zarr metadata, safe hierarchy paths, and supported codec helpers |
| XRADIO parts of `ZarrImage.*` | `carta-zarr` | Replace casacore values with neutral descriptors; preserve full time axis |
| FITS-header/casacore parts of `ZarrImage.*` | `carta-backend` | Convert neutral descriptors into existing CARTA image metadata |
| `CartaZarrImage.*` | `carta-backend` | Remains the `casacore::ImageInterface<float>` adapter |
| `ZarrLoader.*` | `carta-backend` | Remains the `FileLoader` adapter and applies backend dimensional limits |
| Zarr program settings | `carta-backend` | Keep existing user-facing names; translate values into a library `Context` |
| TensorStore CMake setup | `carta-zarr` | Remove entirely from backend CMake and source tree |
| Zarr unit tests and fixtures | split | Storage/schema tests move; casacore and FileLoader integration tests remain |

## 7. Public API proposal

The following is a shape proposal, not final compilable syntax. Names are deliberately schema-neutral.

```cpp
namespace carta::zarr {

enum class ErrorCode {
    not_found,
    not_zarr,
    unsupported_zarr_version,
    unsupported_schema,
    unsupported_schema_version,
    ambiguous_schema,
    invalid_metadata,
    unsupported_data_type,
    unsupported_codec,
    invalid_slice,
    buffer_too_small,
    io_error,
    decode_error,
    cancelled,
};

struct Error {
    ErrorCode code;
    std::string message;
    std::string node_path;
};

template<class T>
class Result; // value-or-Error; exact implementation to be selected before ABI freeze

enum class ProbeKind {
    not_zarr,
    zarr_without_supported_schema,
    supported_dataset,
    invalid_dataset,
};

using SchemaId = std::string;

inline constexpr std::string_view kXradioImageSchema = "xradio.image";

enum class SchemaMatchKind {
    no_match,
    match,
    invalid,
};

struct SchemaProbeResult {
    SchemaMatchKind kind;
    SchemaId schema_id;
    std::string schema_version;
    std::vector<Diagnostic> diagnostics;
};

struct ProbeResult {
    ProbeKind kind;
    SchemaId schema_id;          // selected schema, e.g. "xradio.image"
    std::string schema_version;  // empty when not declared
    std::vector<std::string> image_ids;
    std::vector<Diagnostic> diagnostics;
};

struct OpenOptions {
    std::size_t cache_bytes = 0;
    unsigned int io_threads = 0;      // 0 means implementation default
    unsigned int decode_threads = 0;  // 0 means implementation default
};

class Context final {
public:
    static Result<Context> Create(const OpenOptions& options = {});

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

class Dataset final {
public:
    static Result<Dataset> Open(const Context& context,
                                std::string_view location);

    const DatasetDescriptor& descriptor() const noexcept;
    Result<Image> OpenImage(std::string_view image_id) const;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

enum class AxisRole { spatial_x, spatial_y, spectral, polarization, time, other };

struct AxisDescriptor {
    std::string name;       // original schema name
    AxisRole role;
    std::uint64_t length;
    std::string unit;
    std::size_t storage_index;
};

enum class DataType {
    boolean,
    int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    float16, float32, float64,
    complex64, complex128,
};

struct ImageDescriptor {
    std::string id;                          // the data variable name
    std::string image_role;                  // the variable's own type attribute, e.g. "sky"
    std::vector<std::string> data_groups;    // data groups referencing this image; descriptive only
    DataType stored_type;
    std::vector<AxisDescriptor> axes; // logical API order
    std::string unit;
    bool has_pixel_mask;
    std::optional<DirectionCoordinate> direction;
    std::optional<SpectralCoordinate> spectral;
    std::optional<PolarizationCoordinate> polarization;
    std::optional<TemporalCoordinate> temporal;
    std::optional<ObservationInfo> observation;
    std::optional<StorageLayout> storage;
    std::vector<Diagnostic> diagnostics;
};

struct Range {
    std::uint64_t start;
    std::uint64_t count;
    std::uint64_t stride = 1;
};

struct ReadRequest {
    std::vector<Range> axes; // same order and rank as ImageDescriptor::axes
    DataType output_type = DataType::float32;
};

struct MutableBufferView {
    void* data;
    std::size_t byte_size;
};

class Image final {
public:
    const ImageDescriptor& descriptor() const noexcept;
    Result<std::size_t> Read(const ReadRequest& request,
                             MutableBufferView destination) const;
    Result<std::vector<Beam>> ReadBeams() const;
};

ProbeResult Probe(std::string_view location,
                  const ProbeOptions& options = {});

Result<SchemaProbeResult> ProbeSchema(std::string_view location,
                                     std::string_view schema_id);

// Convenience wrapper around ProbeSchema(location, kXradioImageSchema).
// Returns an Error when the schema cannot be determined; false means a valid non-match.
Result<bool> IsXradioImage(std::string_view location);

} // namespace carta::zarr
```

### 7.1 API behavior

- `Probe()` is cheap, read-only, non-throwing, and suitable for CARTA file listing. It reads only the metadata needed for identification and returns diagnostics.
- `ProbeSchema()` checks one named schema without opening image data. `IsXradioImage()` is the convenience entry point requested by the first consumer, while the schema-ID form keeps the API extensible.
- A plain `bool IsXradio()` is intentionally avoided: XRADIO contains multiple dataset schemas, and a Boolean cannot distinguish a non-match from malformed metadata or an I/O failure.
- `Dataset::Open()` validates the selected schema profile sufficiently to make later descriptor access deterministic. It must not eagerly read the entire image.
- `image_ids` replaces the current use of Zarr array names as FITS HDUs, and lists every image in the dataset because CARTA presents them in a flat selector. An image ID is the data variable's name, which is unique within a dataset. `Probe()` reports the full list so that file listing does not have to open the dataset; consolidated metadata is used when the store provides it. See [ADR 0001](adr/0001-image-identity-and-discovery.md).
- An image is a data variable carrying every coordinate of its plane (`time`, `frequency`, `polarization`, `l`, `m`), holding a real data type, and not tagged `type: "flag"`. Optional coordinates that share only part of that set, such as the `right_ascension` and `declination` arrays XRADIO writes over `(l, m)`, are neither listed nor read. Aperture-plane and complex variables are reported with a diagnostic rather than hidden, but cannot be opened.
- Every store recognized as an image dataset must carry a coordinate array for every axis its image uses, the `time` axis included, because both XRADIO readers write all five unconditionally. A missing one is malformed metadata, reported as an invalid dataset rather than as a non-match.
- Coordinate descriptors are tabular first: the full channel frequency list and time coordinate values are always reported, and a linear reference-pixel/reference-value/increment description is optional, present only when the axis is uniformly spaced within tolerance. A reference pixel is located by searching for the reference world value rather than extrapolated from the first sample.
- `ImageDescriptor::axes` uses a logical order defined by the adapter. For XRADIO `SKY`, the proposed order is `x`, `y`, `spectral`, `polarization`, `time`; all stored time planes are preserved. `storage_index` records the mapping to stored order. A backend that only supports four-dimensional casacore images must explicitly reject or select a time plane without changing the library descriptor.
- `Read()` accepts arbitrary start/count/stride ranges and writes a densely packed result in logical axis order. Axis 0 is the fastest-varying destination dimension, matching casacore array storage. This maps directly to `casacore::Slicer` and avoids exposing TensorStore domains.
- The initial extraction exposes the stored type in metadata. When pixel reading is implemented, conversion to `float32` is required for the CARTA adapter; broader native-type reads can be added as the API evolves.
- The current implementation has no main-image pixel reader, so it does not yet define pixel fill, non-finite, or missing-chunk behavior. These semantics must be derived from the then-current XRADIO/Zarr behavior and locked down with fixtures when `Read()` is implemented.
- An image's pixel mask is the flag variable named by its `flag` attribute; when that attribute is absent, the dataset is scanned for a variable tagged `type: "flag"` whose shape matches, and an ambiguous match is reported rather than guessed. XRADIO stores flags with true meaning a good pixel.
- Descriptors are immutable after open. Handles are movable and cheap to copy only when sharing an immutable implementation is safe.
- Library code must not write directly to spdlog. Errors are returned as structured values; optional diagnostic callbacks can be added if operational logging is required.
- `location` is a string rather than a public filesystem-only type so URI transports can be added later. The initial implementation accepts local paths and may normalize `file://` URIs; unsupported schemes return a specific error.

### 7.2 Configuration ownership

The current process-wide “configure once before first store” behavior is fragile for a reusable library. Configuration should belong to an explicit `Context` or to `OpenOptions`, with shared resources owned by that context:

```cpp
auto context = carta::zarr::Context::Create(options);
auto dataset = carta::zarr::Dataset::Open(context, location);
```

The backend can own one context and reuse it for all datasets. Another consumer can own a context with different limits in the same process. If TensorStore prevents fully independent settings, the limitation must be documented and enforced instead of depending on static-initialization order.

The names should describe library behavior (`io_threads`, `decode_threads`, `cache_bytes`) rather than expose a TensorStore implementation detail. Backend command-line and preference names may remain compatible while being translated to these options.

## 8. Initial compatibility contract

During the experimental phase, compatibility is defined by the pinned XRADIO fixture generator and the migrated tests rather than a stable schema promise. The support matrix must be updated whenever the tracked upstream schema changes.

| Area | Proposed experimental behavior | Current branch |
|---|---|---|
| Access | Local directory; `file://` may be added as a normalization convenience | Local directory |
| Zarr format | Follow the version and encoding used by XRADIO v1.2 | v3-shaped metadata, version not separately declared |
| Schema | XRADIO v1.2 `SKY` baseline; record exact patch version or commit | Unversioned fixture-derived interpretation |
| Schema detection | General named-schema probe plus `IsXradioImage()` convenience function | `IsSupportedZarrImage()` checks for supported child arrays |
| Image variables | `SKY` only | `SKY` only |
| Stored axes | `time`, `frequency`, `polarization`, `l`, `m`, arbitrary stored order | Required, arbitrary stored order |
| Logical axes | `x`, `y`, `spectral`, `polarization`, `time` | Drops singleton time from the CARTA shape |
| Time | Preserve and accept all supported time planes | Rejects length greater than one; must be changed during extraction |
| Pixel data | Not part of the initial extraction; later behavior follows tracked XRADIO/Zarr and is fixture-defined | Not implemented |
| Coordinate strings | Required polarization labels; exact accepted dtype/codec combinations documented | Custom single-chunk `fixed_length_utf32` subset |
| Beams | Optional per-channel/per-polarization beam array | Metadata read implemented |
| Chunking | Regular chunk grid | TensorStore-dependent |
| Sharding | Declare supported codec combinations explicitly | Storage-info parsing exists; image read unimplemented |
| Compression | bytes plus selected zstd/gzip/blosc/checksum chains | Mixed TensorStore/custom behavior |
| Consolidated metadata | Inline consolidated metadata and per-node fallback | Supported |
| Missing chunks / fill | Unspecified until pixel reads are implemented | Not implemented |
| Masks / NaN | Unspecified until pixel reads are implemented | Not implemented |

Each implemented row needs positive fixtures, negative fixtures, and an expected error category. Fixtures generated by XRADIO and `zarr-python` must record exact versions or commits even though the project intentionally follows upstream rapidly.

## 9. Build, packaging, and consumption

### 9.1 Experimental developer workflow

`carta-zarr` and `carta-backend` remain separate builds:

```bash
cmake -S carta-zarr -B carta-zarr/build \
      -DCMAKE_INSTALL_PREFIX=/path/to/carta-zarr-prefix
cmake --build carta-zarr/build
cmake --install carta-zarr/build

cmake -S carta-backend -B carta-backend/build \
      -DCMAKE_PREFIX_PATH=/path/to/carta-zarr-prefix
```

This keeps TensorStore configuration and targets entirely inside the `carta-zarr` build. CI should exercise the same install-then-consume boundary. A convenience orchestration script may build both projects, but the backend CMake project must not use `FetchContent` or `add_subdirectory` for `carta-zarr`.

### 9.2 CMake package

Install a shared target and export it as:

```cmake
find_package(CartaZarr 0.1 CONFIG REQUIRED)
target_link_libraries(consumer PRIVATE CARTA::zarr)
```

The package should install:

- `libcarta-zarr.so.<SONAME>`;
- public headers under `include/carta-zarr/`;
- `CartaZarrConfig.cmake`, `CartaZarrConfigVersion.cmake`, and exported targets;
- optionally `carta-zarr.pc` for pkg-config;
- license, changelog, and compatibility documentation.

Use hidden symbol visibility by default and an explicit export macro for public symbols. Public headers should use PImpl for owning classes and avoid dependency types. During the experimental `0.x` period, API and ABI breaks are allowed and documented in the changelog; a stable SONAME policy begins before binary packages are published.

### 9.3 Future Debian packages

Recommended split:

- `libcarta-zarr1`: shared library and required runtime assets;
- `libcarta-zarr-dev`: headers, unversioned linker symlink, CMake config, and pkg-config file;
- optional `libcarta-zarr-tests` or a separately published fixture package only if downstream validation needs it.

When packaging is enabled, `carta-backend` build-depends on `libcarta-zarr-dev`; its runtime package depends automatically on the matching SONAME package. Packaging is intentionally deferred during rapid experimental development.

### 9.4 TensorStore ownership

`carta-zarr` fetches or locates, configures, builds, and links TensorStore. TensorStore remains a private implementation dependency: it is absent from installed public headers and from the exported target's public link interface. `carta-backend` only sees `CARTA::zarr`.

The exact static/shared packaging of TensorStore can remain an internal build choice during experimentation. Before APT packaging, symbol visibility, position-independent code, runtime dependency policy, and third-party license notices must be validated.

### 9.5 Licensing

The project follows the same licensing convention as the CARTAvis backend and frontend: GPL version 3, with source headers marked `SPDX-License-Identifier: GPL-3.0-or-later` and the repository carrying the canonical license text. Migrated files retain their existing copyright notices. TensorStore and all bundled dependencies must retain their required notices.

## 10. Error handling and diagnostics

Do not collapse all failures into `false`, an empty vector, or a log message. Callers need stable categories to decide whether to try another reader, reject a file, or report corruption.

Minimum distinctions:

- format mismatch: safe for `FileLoader` to try another reader;
- unsupported feature/version: valid input outside the library's compatibility contract;
- ambiguous schema: more than one built-in adapter claims the dataset;
- invalid schema metadata: claims to match but violates required structure;
- I/O failure: permissions, missing object, timeout, or transport failure;
- corrupt/invalid chunk: checksum, size, or decode failure;
- invalid caller request: rank, bounds, stride, type, or destination size;
- resource exhaustion/cancellation.

Errors should carry the hierarchy node path and operation but must avoid leaking credentials from future URIs.

## 11. Test and acceptance strategy

Before moving code, turn current behavior into an explicit compatibility baseline:

- preserve existing metadata, security, malformed-input, and integration fixtures from `carta-backend`;
- add schema-detection tests for valid XRADIO, valid non-XRADIO Zarr, malformed XRADIO-like metadata, missing paths, permission/I/O failures, and ambiguous adapter matches;
- when pixel reading is added, add end-to-end slice tests against arrays generated by pinned reference implementations;
- compare axis reordering and sliced values for multiple stored axis orders;
- test chunk boundaries, partial edge chunks, strides, missing chunks/fill values, endian conversion, compression, sharding, and checksums;
- test malformed metadata and integer-overflow paths;
- test concurrent reads and configured resource limits;
- test install-tree consumption with a tiny external CMake project;
- defer `.deb` installation, upgrade, and removal tests until packaging work begins;
- retain backend adapter tests that compare file info, coordinates, Stokes ordering, beams, and pixels with the prior behavior.

The backend migration is complete only when `rg tensorstore` over backend production sources and public build configuration returns no dependency, and a clean backend build does not download, configure, or compile TensorStore.

## 12. Incremental migration plan

### Phase 0: record the moving upstream baseline

- Record the exact XRADIO v1.2 patch version or commit and `zarr-python` version used by the fixture generator.
- Document which current behaviors are implemented and which are intentionally unsupported.
- Resolve the remaining open decisions in section 13.

### Phase 1: establish the standalone build boundary

- Create CMake targets, local install/export rules, symbol visibility, experimental versioning, and a package-consumer test.
- Introduce neutral error, descriptor, context, dataset, and image types.
- Introduce the schema registry, `ProbeSchema()`, and `IsXradioImage()` before moving backend format detection.
- Move schema-neutral Zarr access behind PImpl; keep TensorStore private.

### Phase 2: extract XRADIO interpretation

- Move validation and mapping into the built-in XRADIO adapter.
- Replace casacore types with neutral DTOs.
- Move existing fixtures and unit tests to this repository.
- Remove the existing `time <= 1` validation from the library layer and preserve the full time axis in descriptors.

### Phase 3: implement data reads

- Implement arbitrary slice reads, logical-axis reordering, type conversion, fill behavior, and agreed mask behavior.
- Add concurrency, cancellation, and resource-limit tests.

### Phase 4: integrate the backend

- Keep `ZarrLoader` and `CartaZarrImage` as thin adapters.
- Convert neutral metadata to casacore coordinates and beams in the backend, and drop the Zarr-specific FITS card generation in favour of the existing generic header path.
- Translate existing backend Zarr settings into a library `Context`.
- Replace TensorStore CMake targets with `CARTA::zarr` and verify the clean build acceptance criterion.

### Phase 5: stabilize and package when needed

- Decide when the public API is mature enough for an ABI policy.
- Build and publish runtime/development packages when there is an APT consumer requirement.
- Add package ABI checks and document supported platforms, compiler baseline, and deprecation policy.

## 13. Remaining decisions

The main architecture decisions are resolved. These implementation details still need answers before their corresponding phase:

1. Which exact XRADIO v1.2 patch version or source commit is the fixture and behavior baseline?
2. Besides generated fixtures, which real XRADIO v1.2 `SKY` datasets should be kept as integration/conformance samples?
3. Must one image handle support simultaneous reads from multiple backend worker threads? Is cancellation required for long reads?
4. When pixel reading begins, should `carta-zarr` expose native output types as well as `float32`, or implement only the conversion needed by CARTA first?
5. For `time > 1`, should the backend initially reject the dataset or select a time plane through a new CARTA-facing option? This does not change library acceptance but affects integration behavior.
6. Is support for additional schemas expected to remain built-in, or is a public third-party schema plugin mechanism likely to be needed later?

## 14. Current working defaults

- C++17 shared library, namespace `carta::zarr`, PImpl public handles, and project version `0.x` while the API is experimental.
- Read-only local stores using the Zarr version and encoding selected by XRADIO v1.2.
- One built-in adapter implementing the XRADIO v1.2 `SKY` baseline, designed to evolve with later official versions.
- Every sky-plane image in the dataset is openable, addressed by its data variable name; membership is decided by inspection, not by a name list or by `data_groups`.
- General named-schema probing plus an `IsXradioImage()` convenience wrapper; schema adapters remain built-in.
- The XRADIO profile is dataset-level: it discovers at least one valid sky-plane image by inspecting variables and applies the same required-coordinate validation to every store.
- Descriptors carry what casacore coordinate constructors need; the library produces no FITS headers.
- Preserve the full time axis, including `time > 1`; proposed logical order `x, y, spectral, polarization, time`.
- When slices are implemented, output is densely packed with `x` as the fastest-varying dimension; `float32` conversion supports the backend adapter.
- TensorStore is built and privately linked by `carta-zarr`; it is absent from backend source and direct build configuration.
- Explicit `Context` owned by the consumer; no configure-once global state.
- Backend retains the casacore/FITS adapter and existing preference names.
- Separate local install plus exported CMake config during experimentation; APT packaging later.
- GPL-3.0-or-later source convention matching CARTAvis.
- Built-in schema registry only during the experimental phase.
