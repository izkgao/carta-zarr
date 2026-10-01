# carta-zarr

A read-only C++17 library for XRADIO images stored as Zarr v3, extracted from
[carta-backend](https://github.com/CARTAvis/carta-backend) so that CARTA and other C++ consumers can
read these datasets without depending on TensorStore, casacore, or CARTA protobuf.

TensorStore is a private implementation detail: it is linked into this library and never appears in
its public headers, which need only the C++ standard library.

## What it reads

- **Image datasets** written by [XRADIO](https://github.com/casangi/xradio) 1.2.x: one Zarr v3 group
  holding sky-plane image variables and the coordinates they share. Schema id `xradio.image`,
  schema version `1.2`.
- **Images**: real-valued variables carrying all five of `time`, `frequency`, `polarization`, `l`
  and `m`. Complex and aperture-plane (`u`, `v`) variables are listed with a diagnostic saying why
  they cannot be opened.
- **Local filesystem stores.** Every node carries its own metadata document; a root
  `consolidated_metadata` copy, which is what zarr-python writes for a consolidated dataset, is used
  to answer for them without reading each one.

Writing is out of scope, and so is anything but the `xradio.image` profile.

## Requirements

CMake 3.24, a C++17 compiler, and `nlohmann_json` 3.11 or newer. TensorStore 0.1.84 is fetched and
built as part of the build, and brings its own zlib, zstd, blosc and protobuf.

## Build, test, install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix /your/prefix
```

The first configure builds TensorStore, which takes a while; later ones do not.

## Using it from CMake

```cmake
find_package(CartaZarr 0.1 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE CARTA::zarr)
```

`tests/consumer` is exactly this, built against the install tree by the `carta-zarr-package-consumer`
test, so the packaging stays consumable.

## Using the API

```c++
#include <carta-zarr/carta_zarr.h>

auto context = carta::zarr::Context::Create();          // one per process is enough
auto dataset = carta::zarr::Dataset::Open(context.value(), "/path/to/image.zarr");
auto image = dataset.value().OpenImage(*dataset.value().descriptor().default_image_id);

const auto& axes = image.value().descriptor().axes;
carta::zarr::ReadRequest request;                        // one Range per axis, each defaulting to
request.axes.resize(axes.size(), {0, 1, 1});             // the first plane of that axis
request.axes[*carta::zarr::AxisIndex(axes, carta::zarr::AxisRole::spatial_x)] = {0, 512, 1};
request.axes[*carta::zarr::AxisIndex(axes, carta::zarr::AxisRole::spatial_y)] = {0, 512, 1};
std::vector<float> pixels(512 * 512);
auto written = image.value().Read(request, {pixels.data(), pixels.size()});
```

`carta::zarr::ProbeSchema(path, carta::zarr::kXradioImageSchema)` answers "is this an image dataset
this library reads?" without opening an image: a `SchemaMatchKind` of match, no match, or a match
that is malformed. `Dataset::Open` lists the images in one, and `ImageEntry::openable` says which of
them will open; an openable entry also carries the axes it will open with, so a consumer can choose
among them without opening any.

### What the API promises

- **Everything is a `Result<T>`.** A failure carries an `ErrorCode`, a message and the node it came
  from. No public entry point throws: a malformed value in the metadata or a length no allocator can
  serve is reported as an error like any other. `Result` is `[[nodiscard]]` and reads like
  `std::expected`: `has_value`, `value`, `value_or`, `*`, `->` and `error`.
- **A handle always refers to something.** `Context`, `Dataset`, `Image` and `CachePool` come only
  from their factories, and moving one copies it, so the handle moved from still works. A consumer
  that fills one in later holds a `std::optional` of it.
- **Axes are found by role, not by position.** A descriptor lists an image's axes in a logical order
  that its schema profile chooses, and a request's ranges follow it; `AxisIndex(axes, role)` finds
  one. The XRADIO profile's order is `l`, `m`, `frequency`, `polarization`, `time`, whatever order
  the store holds them in, but that is the profile's choice rather than a promise. A read returns
  the axes densely packed with axis 0 varying fastest; `AxisDescriptor::storage_index` says where
  each one lives on disk.
- **Pixels come back as `float32`**, converted during the read rather than materialised in their
  stored type first. Reductions accumulate and report in `double`.
- **Pixel masks are applied by default.** An image with a flag variable reads a masked pixel as NaN,
  so a consumer that wants the mask reads it as finiteness; `ReadOptions::apply_pixel_mask` turns
  that off. A declared mask must be boolean and carry the image's own dimensions in order, or the
  image does not open.
- **Reads are cancellable.** `ReadOptions::control` carries a cancellation callback, a deadline and
  the cache pool the read keeps what it decodes in -- the context's shared one unless it brings its
  own from `Context::NewCachePool` -- which every read and reduction honours; `ReadOptions` adds a
  ceiling on the temporary memory a request may hold. Progress is an argument of `Image::Read` and of
  `Image::ComputeCubeHistogram`, and a reduction's arrives with its blocks; a request is only ever
  data.
- **Threading**: `Context`, `Dataset` and `Image` handles may be shared and read from several
  threads at once. Reductions running concurrently share one worker pool, which runs one read's
  arithmetic at a time: two reductions interleave read by read rather than running side by side.

## Documentation

- [CONTEXT.md](CONTEXT.md) — the vocabulary this codebase uses, and the words it avoids.
- [docs/design.md](docs/design.md) — why the library is split from the backend the way it is.
- [docs/read-api.md](docs/read-api.md) — the read and reduction API, and the reasoning behind its
  output types and streaming shape.
- [docs/storage-tuning.md](docs/storage-tuning.md) — choosing a Zarr layout and carta-backend's reader
  settings for Lustre or BeeGFS, what measurements found, and how to measure your own.
- [docs/adr](docs/adr) — decisions and their alternatives.
- [AGENTS.md](AGENTS.md) — building and testing without drowning in TensorStore's output.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
