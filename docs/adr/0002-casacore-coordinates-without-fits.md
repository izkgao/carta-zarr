# Descriptors feed casacore coordinate construction directly, bypassing FITS headers

`carta-backend` currently converts XRADIO metadata into FITS header strings and hands them to
`casacore::ImageFITSConverter::getCoordinateSystem()`. This library will instead expose descriptors
rich enough for the backend to construct `DirectionCoordinate`, `SpectralCoordinate`,
`StokesCoordinate`, `LinearCoordinate`, and `TabularCoordinate` directly, and will not produce FITS
headers at all.

The FITS intermediate is lossy in ways that have already caused work: a FITS header describes an
axis linearly, through `CRVALn`, `CDELTn`, and `CRPIXn`, which cannot express a non-uniform
polarization ordering or a non-uniform frequency axis. The backend already detects the first case
and overwrites the converter's polarization coordinate with a `StokesCoordinate` built from the
per-plane list. XRADIO explicitly warns that neighbouring frequency channels need not be evenly
spaced, so the same defect applies to the spectral axis, where the correct target is a
`TabularCoordinate` built from the full channel list.

## Consequences

Coordinate descriptors are tabular first. `SpectralCoordinate::channel_frequencies` is always
populated and is the authoritative value; the linear triple of reference pixel, reference value, and
increment is optional and is filled only when the channels are uniformly spaced within tolerance. A
consumer that receives no linear description cannot silently assume one — it has to build a
`TabularCoordinate`, which is the correct outcome. The same reasoning is why time coordinate values
are reported even though the backend's casacore images are four-dimensional today.

`DirectionCoordinate` must carry everything casacore's constructor takes, including the projection
parameters and the native pole direction that become `longPole` and `latPole`. XRADIO writes both
and the previous implementation ignored them; omitting them shifts coordinates silently rather than
raising an error.

The header entries CARTA displays in its file information panel are no longer this library's
concern. Once the backend holds a correct `CoordinateSystem`, it can use the generic
`GetFITSHeader()` path it already applies to other image types, and delete its Zarr-specific FITS
card generation.
