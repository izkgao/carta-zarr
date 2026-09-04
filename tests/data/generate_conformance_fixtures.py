#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "xradio[zarr]==1.2.3",
#   "casatools==6.7.6.14",
#   "casaconfig",
#   "astropy==7.1.0",
#   "numpy==2.3.1",
# ]
# ///

"""Generate the conformance fixture by converting a FITS image with XRADIO itself.

`generate_zarr_fixtures.py` builds stores with zarr-python directly, which pins our *belief* about
what XRADIO writes. This script pins XRADIO instead: it makes a FITS image with astropy and hands it
to `xradio.image.open_image()` / `write_image()`, so the result is whatever the released XRADIO
actually produces. Regenerate it when the pinned XRADIO version is bumped, and let the conformance
test report which of our assumptions the new version broke.

XRADIO's FITS reader is strict. Every axis needs CTYPE and CUNIT (including the unitless Stokes
axis), and the header must carry LONPOLE, LATPOLE, the PC matrix, and DATE-OBS.

Note that `open_image()` imports the casacore layer even for FITS input, and XRADIO excludes
python-casacore on macOS, so casatools is required on every platform here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

OUTPUT_DIR = Path(__file__).parent / "images" / "zarr" / "xradio"
FIXTURE_NAME = "conformance"

AXES = [
    # (CTYPE, CRVAL, CDELT, CUNIT)
    ("RA---SIN", 45.0, -0.001, "deg"),
    ("DEC--SIN", 30.0, 0.001, "deg"),
    ("STOKES", 1.0, 1.0, ""),
    ("FREQ", 1.4e9, 1.0e6, "Hz"),
]


def write_fits(path: Path) -> None:
    # FITS axis order is the reverse of the numpy shape: NAXIS1 is the fastest-varying axis.
    values = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    hdu = fits.PrimaryHDU(values)
    header = hdu.header
    for index, (ctype, crval, cdelt, cunit) in enumerate(AXES, start=1):
        header[f"CTYPE{index}"] = ctype
        header[f"CRVAL{index}"] = crval
        header[f"CDELT{index}"] = cdelt
        header[f"CRPIX{index}"] = 1.0
        header[f"CUNIT{index}"] = cunit

    header["BUNIT"] = "Jy/beam"
    header["BTYPE"] = "Intensity"
    header["OBJECT"] = "Zarr conformance source"
    header["OBSERVER"] = "CARTA"
    header["TELESCOP"] = "Test scope"
    header["RADESYS"] = "FK5"
    header["EQUINOX"] = 2000.0
    header["SPECSYS"] = "LSRK"
    header["RESTFRQ"] = 1.420405751e9
    header["DATE-OBS"] = "2020-05-31T12:00:00.000"
    header["TIMESYS"] = "UTC"
    header["LONPOLE"] = 180.0
    header["LATPOLE"] = 30.0
    header["PC1_1"], header["PC1_2"] = 1.0, 0.0
    header["PC2_1"], header["PC2_2"] = 0.0, 1.0
    header["BMAJ"], header["BMIN"], header["BPA"] = 2.0e-5, 1.0e-5, 0.1

    hdu.writeto(path, overwrite=True)


def main() -> None:
    from xradio.image import open_image, write_image

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fixture = OUTPUT_DIR / FIXTURE_NAME
    shutil.rmtree(fixture, ignore_errors=True)

    fits_path = OUTPUT_DIR / f"{FIXTURE_NAME}.fits"
    write_fits(fits_path)
    try:
        xds = open_image(str(fits_path))
        write_image(xds, str(fixture), out_format="zarr", overwrite=True)
    finally:
        fits_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
