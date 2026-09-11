# carta-zarr

A standalone C++ reader library for XRADIO Zarr images, extracted from `carta-backend` so that
CARTA and other C++ consumers can read these datasets without depending on TensorStore, casacore,
or CARTA protobuf.

## Language

### The dataset

**Image dataset**:
One Zarr v3 group holding a complete XRADIO image, marked by the root attribute `type:
"image_dataset"`.
It is the unit a consumer opens by path.
_Avoid_: file, store, image (on its own), group

**Data variable**:
One named array inside an image dataset. Some are images, others are flags, beam tables, or
normalization factors.
_Avoid_: array, dataset variable, HDU

**Coordinate**:
A named axis of the image dataset (`time`, `frequency`, `polarization`, `l`, `m`, `u`, `v`), stored
once at dataset level and referenced by every data variable that uses it.
_Avoid_: dimension, axis array

**Data group**:
A named set of related data variables produced by one imaging run, mapping a role (`sky`, `flag`,
`primary_beam`, …) to the variable that fills it. Several data groups can reference the same
variable.
_Avoid_: image group, group, dataset group

### What can be opened

**Image**:
A data variable a consumer can open and read pixels from: real-valued, carrying every coordinate of
its plane, and not a flag. Identified by its variable name, which is unique within the dataset.
Carrying only some of the plane's coordinates makes something an optional coordinate, not an image.
_Avoid_: image array, HDU, plane, layer

**Image role**:
The part an image plays in its data group — sky, model, residual, point spread function, primary
beam, deconvolution mask. Carried on the data variable's own `type` attribute, the same attribute
that spells `flag` for pixel masks. Descriptive only; it never identifies an image.
_Avoid_: image type, kind, category

**Sky plane**:
The tangent plane to the celestial sphere, carrying the `l` and `m` direction-cosine coordinates.
_Avoid_: image plane, lm plane

**Aperture plane**:
The Fourier-conjugate plane, carrying the `u` and `v` coordinates. Variables on it are outside this
library's supported set.
_Avoid_: uv plane, visibility plane, Fourier plane

**Flag**:
A boolean data variable marking which pixels of its image are valid, distinguished by the attribute
`type: "flag"` rather than by its name. True means the pixel is good. It becomes the image's pixel
mask; it is never itself an image.
_Avoid_: mask, internal mask, blanking

**Deconvolution mask**:
An image marking the region a deconvolution was allowed to work in. It is an ordinary image, opened
and displayed like any other.
_Avoid_: mask, flag

**Beam fit parameters**:
A data variable holding fitted major axis, minor axis, and position angle per plane, named by the
owning image's `beam_fit_params` attribute. Different images may point at different ones.
_Avoid_: beam table, beam array, restoring beam

### Reading

**Schema profile**:
A named, versioned description of how an image dataset is laid out, which the library matches a
store against. The only built-in profile is the XRADIO image profile.
_Avoid_: schema, format, flavor

**Probe**:
A cheap, read-only inspection that decides whether a path is a supported image dataset and lists the
images in it, without opening any of them.
_Avoid_: detect, sniff, validate

**Transport**:
Where an image dataset's bytes live, and the only thing the library swaps out to read from
somewhere else. It hands up one node's metadata verbatim and lists the nodes; it never parses and
never decides what a node means, so every transport is interpreted identically.
_Avoid_: driver, backend, kvstore, store

**Descriptor**:
The immutable metadata the library reports for a dataset or an image. It never contains pixels.
_Avoid_: info, header, metadata (on its own)

**Linear description**:
The reference pixel, reference value, and increment that let a consumer treat a coordinate as an
evenly spaced axis. A coordinate carries one only when its samples support it; without one the
consumer builds a tabular axis from the values instead. A direction axis is linear by construction
and always reports one.
_Avoid_: linear triple, FITS keywords, WCS

**Stored order**:
The order the coordinates appear in a data variable's own dimensions, as written on disk.
_Avoid_: native order, disk order, physical order

**Logical order**:
The coordinate order the library reports and reads in, chosen by the schema profile rather than by
the file.
_Avoid_: canonical order, CARTA order, display order
