#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.1",
#   "zarr==3.2.1",
# ]
# ///

"""Generate Zarr v3 fixtures using zarr-python as the reference implementation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BytesCodec, Crc32cCodec, GzipCodec, ZstdCodec


OUTPUT_DIR = Path(__file__).parent / "images" / "zarr"
STRING_VALUES = np.asarray(["A", "BC"], dtype="U2")


def create_string_array(
    path: Path,
    *,
    serializer: object,
    compressors: list[object] | None = None,
    chunk_key_encoding: dict[str, object] | None = None,
    shape: tuple[int, ...] = (2,),
    chunks: tuple[int, ...] = (2,),
    write_values: bool = True,
    values: np.ndarray = STRING_VALUES,
) -> zarr.Array:
    array = zarr.create_array(
        store=path,
        shape=shape,
        chunks=chunks,
        dtype=values.dtype,
        zarr_format=3,
        chunk_key_encoding=chunk_key_encoding
        or {"name": "default", "configuration": {"separator": "/"}},
        serializer=serializer,
        compressors=compressors or [],
        fill_value="",
    )
    if write_values:
        array[:] = values
    return array


def chunk_path(array_path: Path) -> Path:
    metadata = zarr.open_array(array_path, mode="r").metadata
    if metadata.chunk_key_encoding.name == "v2":
        return array_path / "0"
    separator = metadata.chunk_key_encoding.separator
    return array_path / ("c.0" if separator == "." else "c/0")


def generate_string_fixtures() -> None:
    output_dir = OUTPUT_DIR / "string"
    output_dir.mkdir(parents=True)

    create_string_array(
        output_dir / "zstd_overhang",
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=1)],
        chunks=(4,),
    )
    create_string_array(
        output_dir / "v2_key",
        serializer=BytesCodec(endian="little"),
        chunk_key_encoding={"name": "v2", "configuration": {"separator": "."}},
    )

    gzip_path = output_dir / "gzip"
    create_string_array(
        gzip_path,
        serializer=BytesCodec(endian="little"),
        compressors=[GzipCodec(level=1)],
    )
    gzip_chunk = bytearray(chunk_path(gzip_path).read_bytes())
    if gzip_chunk[:2] != bytes((0x1F, 0x8B)):
        raise RuntimeError("zarr-python did not produce a gzip stream")
    gzip_chunk[4:8] = bytes(
        4
    )  # Normalize the gzip MTIME header for reproducible fixtures.
    chunk_path(gzip_path).write_bytes(gzip_chunk)

    create_string_array(
        output_dir / "blosc",
        serializer=BytesCodec(endian="little"),
        compressors=[BloscCodec(cname="zstd", clevel=1, shuffle="noshuffle")],
    )
    create_string_array(
        output_dir / "big_endian",
        serializer=BytesCodec(endian="big"),
        values=np.asarray(["Ω", "🙂"], dtype="U2"),
    )
    create_string_array(
        output_dir / "dot_key",
        serializer=BytesCodec(endian="little"),
        chunk_key_encoding={"name": "default", "configuration": {"separator": "."}},
    )
    create_string_array(
        output_dir / "missing_chunk",
        serializer=BytesCodec(endian="little"),
        write_values=False,
    )
    create_string_array(
        output_dir / "crc_before_after_zstd",
        serializer=BytesCodec(endian="little"),
        compressors=[Crc32cCodec(), ZstdCodec(level=1), Crc32cCodec()],
    )

    crc_mismatch = output_dir / "crc_mismatch"
    create_string_array(
        crc_mismatch,
        serializer=BytesCodec(endian="little"),
        compressors=[Crc32cCodec()],
    )
    crc_bytes = bytearray(chunk_path(crc_mismatch).read_bytes())
    crc_bytes[-1] ^= 0xFF
    chunk_path(crc_mismatch).write_bytes(crc_bytes)

    truncated = output_dir / "truncated"
    create_string_array(truncated, serializer=BytesCodec(endian="little"))
    truncated_bytes = chunk_path(truncated).read_bytes()
    chunk_path(truncated).write_bytes(truncated_bytes[:-4])

    invalid_unicode = output_dir / "invalid_unicode"
    create_string_array(invalid_unicode, serializer=BytesCodec(endian="little"))
    invalid_bytes = bytearray(chunk_path(invalid_unicode).read_bytes())
    invalid_bytes[:4] = bytes((0x00, 0xD8, 0x00, 0x00))  # UTF-32LE surrogate U+D800.
    chunk_path(invalid_unicode).write_bytes(invalid_bytes)


def create_numeric_array(
    path: Path,
    values: np.ndarray,
    *,
    dimension_names: tuple[str, ...],
    chunks: tuple[int, ...] | None = None,
    attributes: dict[str, Any] | None = None,
) -> zarr.Array:
    array = zarr.create_array(
        store=path,
        shape=values.shape,
        chunks=chunks or values.shape,
        dtype=values.dtype,
        zarr_format=3,
        dimension_names=dimension_names,
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=1)],
        fill_value=0,
        attributes=attributes,
    )
    array[:] = values
    return array


def add_consolidated_metadata(path: Path) -> None:
    root_metadata_path = path / "zarr.json"
    root_metadata = json.loads(root_metadata_path.read_text())
    metadata = {
        node.parent.relative_to(path).as_posix(): json.loads(node.read_text())
        for node in sorted(path.rglob("zarr.json"))
        if node != root_metadata_path
    }
    root_metadata["consolidated_metadata"] = {"kind": "inline", "metadata": metadata}
    root_metadata_path.write_text(json.dumps(root_metadata, indent=2) + "\n")


def generate_xradio_fixture(
    path: Path, *, typed: bool, consolidated: bool, uniform_beams: bool = False
) -> None:
    root = zarr.open_group(store=path, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "coordinate_system_info": {
                "projection": "SIN",
                "reference_direction": {
                    "data": [1.0, 0.5],
                    "attrs": {"frame": "fk5", "equinox": "J2000"},
                },
                "native_pole_direction": {"data": [0.0, 1.5707963267948966]},
                "projection_parameters": [0.25, -0.5],
                "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]],
            }
        }
    )
    if typed:
        root.attrs.update(
            {
                "type": "image_dataset",
                "data_groups": {
                    "base": {"sky": "SKY", "model": "MODEL", "residual": "RESIDUAL"},
                    "deconvolution": {"sky": "SKY", "mask_deconvolve": "MASK_DECONVOLVE"},
                },
            }
        )

    sky_shape = (1, 3, 2, 4, 5)
    create_numeric_array(
        path / "SKY",
        np.zeros(sky_shape, dtype=np.float32),
        dimension_names=("time", "frequency", "polarization", "l", "m"),
        chunks=(1, 1, 1, 2, 5),
        attributes={
            "units": "Jy/beam",
            "type": "sky",
            "flag": "MASK_0",
            "object_name": "Zarr test source",
            "observer": "CARTA",
            "obsdate": {"data": 59000.0, "attrs": {"format": "MJD", "scale": "UTC"}},
            "telescope": {
                "name": "Test scope",
                "direction": {"data": [0.0, 0.0]},
                "distance": {"data": [6371000.0]},
            },
            "user": {"origin": "zarr-python", "exposure": 12.5},
            "beam_fit_params": "BEAM",
        },
    )
    create_numeric_array(
        path / "l",
        np.asarray([-0.001, 0.0, 0.001, 0.002], dtype=np.float64),
        dimension_names=("l",),
    )
    create_numeric_array(
        path / "m",
        np.asarray([-0.002, -0.001, 0.0, 0.001, 0.002], dtype=np.float64),
        dimension_names=("m",),
    )
    create_numeric_array(
        path / "frequency",
        np.asarray([1.4e9, 1.401e9, 1.403e9], dtype=np.float64),
        dimension_names=("frequency",),
        attributes={
            "reference_frequency": {"data": 1.401e9, "attrs": {"units": "Hz", "observer": "lsrk"}},
            "rest_frequency": {"data": 1.420405751e9},
        },
    )
    create_numeric_array(
        path / "time",
        np.asarray([1.6e9], dtype=np.float64),
        dimension_names=("time",),
        attributes={"units": "s", "scale": "utc", "format": "unix"},
    )
    polarization = create_string_array(
        path / "polarization",
        serializer=BytesCodec(endian="little"),
        values=np.asarray(["I", "Q"], dtype="U1"),
    )
    polarization.attrs.update({"dimension_names": ["polarization"]})

    parameter_labels = create_string_array(
        path / "beam_params_label",
        serializer=BytesCodec(endian="little"),
        shape=(3,),
        chunks=(3,),
        values=np.asarray(["minor", "pa", "major"], dtype="U5"),
    )
    parameter_labels.attrs.update({"dimension_names": ["beam_params_label"]})

    # A table with one row per plane holding the same beam on every one of them is what a
    # converter writes for an image that was restored with a single beam. It has to stay
    # distinguishable from a table whose planes really differ, because a consumer that sees
    # "multiple beams" reconciles them -- ImageMoments convolves the whole cube to a common
    # beam first -- and doing that to planes that already agree costs a full copy of the cube
    # for no change.
    beam_values = np.zeros((1, 3, 2, 3), dtype=np.float64)
    for channel in range(3):
        for stokes in range(2):
            if uniform_beams:
                major = 2.0e-5
                beam_values[0, channel, stokes] = (major / 2.0, 0.1, major)
            else:
                major = 2.0e-5 + (channel * 1.0e-6) + (stokes * 1.0e-7)
                beam_values[0, channel, stokes] = (major / 2.0, 0.1 + channel * 0.01, major)
    create_numeric_array(
        path / "BEAM",
        beam_values,
        dimension_names=("time", "frequency", "polarization", "beam_params_label"),
        chunks=(1, 1, 1, 3),
        attributes={"units": "rad"},
    )

    create_numeric_array(
        path / "MODEL",
        np.ones(sky_shape, dtype=np.float32),
        dimension_names=("time", "frequency", "polarization", "l", "m"),
        chunks=(1, 1, 1, 2, 5),
        attributes={"units": "Jy/beam", "type": "model"},
    )
    create_numeric_array(
        path / "RESIDUAL",
        np.full(sky_shape, 2.0, dtype=np.float32),
        dimension_names=("time", "frequency", "polarization", "l", "m"),
        chunks=(1, 1, 1, 2, 5),
        attributes={"units": "Jy/beam", "type": "residual"},
    )
    create_numeric_array(
        path / "MASK_DECONVOLVE",
        np.ones(sky_shape, dtype=np.float32),
        dimension_names=("time", "frequency", "polarization", "l", "m"),
        chunks=(1, 1, 1, 2, 5),
        attributes={"units": "", "type": "mask_deconvolve"},
    )
    create_numeric_array(
        path / "MASK_0",
        np.ones(sky_shape, dtype=bool),
        dimension_names=("time", "frequency", "polarization", "l", "m"),
        chunks=(1, 1, 1, 2, 5),
        attributes={"type": "flag"},
    )
    create_numeric_array(
        path / "APERTURE",
        np.zeros(sky_shape, dtype=np.float32),
        dimension_names=("time", "frequency", "polarization", "u", "v"),
        chunks=(1, 1, 1, 2, 5),
        attributes={"type": "aperture"},
    )
    create_numeric_array(
        path / "COMPLEX",
        np.zeros(sky_shape, dtype=np.complex64),
        dimension_names=("time", "frequency", "polarization", "l", "m"),
        chunks=(1, 1, 1, 2, 5),
        attributes={"type": "sky"},
    )

    # XRADIO writes these optional coordinates by default (do_sky_coords=True). They share SKY's
    # spatial axes, are real-valued, and carry no type attribute, so they are the exact shape that a
    # discovery rule keyed only on "has l and m" would mistake for openable images.
    lm_shape = (sky_shape[3], sky_shape[4])
    create_numeric_array(
        path / "right_ascension",
        np.zeros(lm_shape, dtype=np.float64),
        dimension_names=("l", "m"),
    )
    create_numeric_array(
        path / "declination",
        np.zeros(lm_shape, dtype=np.float64),
        dimension_names=("l", "m"),
    )
    create_numeric_array(
        path / "velocity",
        np.asarray([100.0, 200.0, 300.0], dtype=np.float64),
        dimension_names=("frequency",),
        attributes={"type": "doppler", "units": "m/s"},
    )

    if consolidated:
        add_consolidated_metadata(path)


def generate_pixel_fixture(path: Path, *, l_fastest: bool = False) -> None:
    """An XRADIO image whose pixels are readable and self-describing.

    The other XRADIO fixtures carry only fill values, so they pin metadata and say nothing about a
    pixel read. Every axis here has a different length, so a read that permutes axes wrongly cannot
    still produce the right shape, and each value spells its own logical coordinates. One chunk is
    deleted after writing and the flag marks a known pattern false, which is how the fill-value and
    pixel-mask paths get a definition instead of an assumption.
    """
    time_size, frequency_size, polarization_size, l_size, m_size = 1, 2, 3, 4, 5
    # XRADIO writes m last, so m is the axis a plane is contiguous along. A store that writes l last
    # is the same image with the spatial pair swapped, and a reader that decides anything from the
    # position of an axis rather than from its name gets a different answer for one of the two.
    if l_fastest:
        names = ("time", "frequency", "polarization", "m", "l")
        shape = (time_size, frequency_size, polarization_size, m_size, l_size)
        chunks = (1, 1, 1, 5, 2)
        missing_chunk_key = ("0", "1", "2", "0", "1")
    else:
        names = ("time", "frequency", "polarization", "l", "m")
        shape = (time_size, frequency_size, polarization_size, l_size, m_size)
        chunks = (1, 1, 1, 2, 5)
        missing_chunk_key = ("0", "1", "2", "1", "0")

    root = zarr.open_group(store=path, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "type": "image_dataset",
            "data_groups": {"base": {"sky": "SKY"}},
            "coordinate_system_info": {
                "projection": "SIN",
                "reference_direction": {
                    "data": [1.0, 0.5],
                    "attrs": {"frame": "fk5", "equinox": "J2000"},
                },
                "native_pole_direction": {"data": [0.0, 1.5707963267948966]},
                "projection_parameters": [0.0, 0.0],
                "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]],
            },
        }
    )

    # value = t*10000 + f*1000 + p*100 + l*10 + m, so a misplaced element names where it came from.
    indices = np.indices(shape)
    l_index = indices[4] if l_fastest else indices[3]
    m_index = indices[3] if l_fastest else indices[4]
    values = (
        indices[0] * 10000 + indices[1] * 1000 + indices[2] * 100 + l_index * 10 + m_index
    ).astype(np.float32)

    sky = zarr.create_array(
        store=path / "SKY",
        shape=shape,
        chunks=chunks,
        dtype=np.float32,
        zarr_format=3,
        dimension_names=names,
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=1)],
        fill_value=float("nan"),
        attributes={
            "units": "Jy/beam",
            "type": "sky",
            "flag": "FLAG",
            "object_name": "Zarr pixel source",
            "observer": "CARTA",
            "obsdate": {"data": 59000.0, "attrs": {"format": "MJD", "scale": "UTC"}},
            # direction is (longitude, latitude) in radians and distance is a radius in metres.
            # Both are needed before an observatory position exists at all, and the values are
            # deliberately off-axis so that a consumer mixing up x, y and z cannot still agree.
            "telescope": {
                "name": "Test scope",
                "direction": {"data": [2.0, -0.5], "attrs": {"units": "rad"}},
                "distance": {"data": [6371000.0], "attrs": {"units": "m"}},
            },
        },
    )
    sky[:] = values

    # True means a good pixel. The pattern crosses chunk boundaries so a mask read that ignores the
    # transpose cannot accidentally agree.
    flags = ((l_index + m_index) % 3 != 0)
    flag = zarr.create_array(
        store=path / "FLAG",
        shape=shape,
        chunks=chunks,
        dtype=np.bool_,
        zarr_format=3,
        dimension_names=names,
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=1)],
        fill_value=False,
        attributes={"type": "flag"},
    )
    flag[:] = flags

    # Drop one written chunk so that a read crossing it has to fall back to the fill value.
    missing_chunk = path / "SKY" / "c" / Path(*missing_chunk_key)
    if not missing_chunk.exists():
        raise RuntimeError(f"expected chunk {missing_chunk} to exist before deleting it")
    missing_chunk.unlink()

    create_numeric_array(
        path / "l",
        np.asarray([-0.002, -0.001, 0.0, 0.001], dtype=np.float64),
        dimension_names=("l",),
    )
    create_numeric_array(
        path / "m",
        np.asarray([-0.002, -0.001, 0.0, 0.001, 0.002], dtype=np.float64),
        dimension_names=("m",),
    )
    create_numeric_array(
        path / "frequency",
        np.asarray([1.4e9, 1.401e9], dtype=np.float64),
        dimension_names=("frequency",),
        attributes={"rest_frequency": {"data": 1.420405751e9}},
    )
    create_numeric_array(
        path / "time",
        np.asarray([1.6e9], dtype=np.float64),
        dimension_names=("time",),
        attributes={"units": "s", "scale": "utc", "format": "unix"},
    )
    polarization = create_string_array(
        path / "polarization",
        serializer=BytesCodec(endian="little"),
        shape=(polarization_size,),
        chunks=(polarization_size,),
        values=np.asarray(["I", "Q", "U"], dtype="U1"),
    )
    polarization.attrs.update({"dimension_names": ["polarization"]})


def generate_wide_pixel_fixture(path: Path) -> None:
    """A pixel fixture large enough for the reductions to split the work.

    The other pixel fixtures are a few dozen pixels, which is the right size for pinning what a read
    returns but is below every threshold the parallel paths have: a plane must hold more than 65,536
    pixels before a histogram divides it between workers, and a chunk must before a spectral
    reduction does. Everything those paths do -- the private accumulators, the merge, the split of a
    read across threads -- went untested on the small fixtures and was checked against real cubes by
    hand instead.

    So this one is wide rather than interesting: no absent chunk and no flag array, because the fill
    and mask paths are defined on the small fixtures and adding them here would only double what the
    walk reads. What it does keep is the rule that no two axes are the same length, so a read that
    permutes them cannot still produce the right shape.
    """
    time_size, frequency_size, polarization_size, l_size, m_size = 1, 4, 2, 512, 520
    names = ("time", "frequency", "polarization", "l", "m")
    shape = (time_size, frequency_size, polarization_size, l_size, m_size)
    # Four chunks to a plane and two channels deep, so a read covers several of them and the walk
    # has bands and slabs to divide rather than one chunk that is the whole image.
    chunks = (1, 2, 1, 256, 260)

    root = zarr.open_group(store=path, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "type": "image_dataset",
            "data_groups": {"base": {"sky": "SKY"}},
            "coordinate_system_info": {
                "projection": "SIN",
                "reference_direction": {
                    "data": [1.0, 0.5],
                    "attrs": {"frame": "fk5", "equinox": "J2000"},
                },
                "native_pole_direction": {"data": [0.0, 1.5707963267948966]},
                "projection_parameters": [0.0, 0.0],
                "pixel_coordinate_transformation_matrix": [[1.0, 0.0], [0.0, 1.0]],
            },
        }
    )

    # value = (f * P + p) * 1e6 + (l // 8) * 1000 + (m // 8), which still says where an element came
    # from -- the quotient names the plane, the rest names the block of the plane -- but holds the
    # same value across each eight-by-eight block. That is what makes the fixture compressible: a
    # per-pixel ramp of float32 varies eighteen mantissa bits every element and zstd can do nothing
    # with it, which cost six megabytes against under one for this.
    #
    # The largest value is 7,063,064, well inside the 16,777,216 a float32 counts to exactly, so a
    # test recomputes it rather than approximating it.
    indices = np.indices(shape)
    plane_index = indices[1] * polarization_size + indices[2]
    values = (plane_index * 1_000_000 + (indices[3] // 8) * 1000 + (indices[4] // 8)).astype(np.float32)

    sky = zarr.create_array(
        store=path / "SKY",
        shape=shape,
        chunks=chunks,
        dtype=np.float32,
        zarr_format=3,
        dimension_names=names,
        serializer=BytesCodec(endian="little"),
        compressors=[ZstdCodec(level=9)],
        fill_value=float("nan"),
        attributes={
            "units": "Jy/beam",
            "type": "sky",
            "object_name": "Zarr wide pixel source",
            "observer": "CARTA",
            "obsdate": {"data": 59000.0, "attrs": {"format": "MJD", "scale": "UTC"}},
            "telescope": {
                "name": "Test scope",
                "direction": {"data": [2.0, -0.5], "attrs": {"units": "rad"}},
                "distance": {"data": [6371000.0], "attrs": {"units": "m"}},
            },
        },
    )
    sky[:] = values

    create_numeric_array(
        path / "l",
        (np.arange(l_size, dtype=np.float64) - (l_size // 2)) * 1e-5,
        dimension_names=("l",),
    )
    create_numeric_array(
        path / "m",
        (np.arange(m_size, dtype=np.float64) - (m_size // 2)) * 1e-5,
        dimension_names=("m",),
    )
    create_numeric_array(
        path / "frequency",
        1.4e9 + np.arange(frequency_size, dtype=np.float64) * 1e6,
        dimension_names=("frequency",),
        attributes={"rest_frequency": {"data": 1.420405751e9}},
    )
    create_numeric_array(
        path / "time",
        np.asarray([1.6e9], dtype=np.float64),
        dimension_names=("time",),
        attributes={"units": "s", "scale": "utc", "format": "unix"},
    )
    polarization = create_string_array(
        path / "polarization",
        serializer=BytesCodec(endian="little"),
        shape=(polarization_size,),
        chunks=(polarization_size,),
        values=np.asarray(["I", "Q"], dtype="U1"),
    )
    polarization.attrs.update({"dimension_names": ["polarization"]})


def main() -> None:
    # Remove only what this script owns. xradio/conformance lives under the same directory but is
    # written by generate_conformance_fixtures.py against a pinned XRADIO, and wiping the whole tree
    # here would delete a fixture this script cannot rebuild.
    for owned in (
        "string",
        "xradio/minimal",
        "xradio/uniform_beam",
        "xradio/legacy",
        "xradio/pixels",
        "xradio/pixels_l_fastest",
        "xradio/pixels_wide",
    ):
        shutil.rmtree(OUTPUT_DIR / owned, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_string_fixtures()
    generate_xradio_fixture(OUTPUT_DIR / "xradio" / "minimal", typed=True, consolidated=True)
    generate_xradio_fixture(OUTPUT_DIR / "xradio" / "legacy", typed=False, consolidated=False)
    generate_xradio_fixture(
        OUTPUT_DIR / "xradio" / "uniform_beam", typed=True, consolidated=True, uniform_beams=True
    )
    generate_pixel_fixture(OUTPUT_DIR / "xradio" / "pixels")
    generate_pixel_fixture(OUTPUT_DIR / "xradio" / "pixels_l_fastest", l_fastest=True)
    generate_wide_pixel_fixture(OUTPUT_DIR / "xradio" / "pixels_wide")


if __name__ == "__main__":
    main()
