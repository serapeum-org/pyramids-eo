"""MTG-FCI L1C (FDHSI) reader.

`read_fci` stitches a channel across the ~40 NetCDF chunks of one FCI repeat
cycle, calibrates the raw radiance to reflectance / brightness temperature via
the sensor registry, and returns a geolocated pyramids `Dataset` ready for
`to_crs` / `warped_view`.

.. warning::
    The default per-chunk extraction assumes each chunk exposes the requested
    channel's radiance as a NetCDF **variable named like the channel**. Real FCI
    L1C FDHSI stores radiance in nested groups
    (``data/<channel>/measured/effective_radiance``) with the calibration
    coefficients in group attributes — so for real granules pass a custom
    `open_chunk` (or the coefficients from metadata). The stitch + calibrate +
    geolocate logic is unit-tested on synthetic chunks; the exact FCI variable
    layout still needs validation against a real granule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids_eo.errors import ReaderError
from pyramids_eo.readers._common import calibrate_channel


def _default_open_chunk(path: Any, channel: str) -> Any:
    """Open one FCI chunk and return the channel's radiance as a `Dataset`.

    Reads `path` with `pyramids.netcdf.NetCDF` and extracts the variable named
    `channel`. See the module warning: real FCI L1C uses nested groups, so a
    custom `open_chunk` is expected for real granules.

    Args:
        path: Path to a chunk NetCDF file.
        channel: Channel identifier / variable name.

    Returns:
        A pyramids `Dataset` of the channel's raw radiance.
    """
    from pyramids.netcdf import NetCDF

    return NetCDF.read_file(str(path)).get_variable(channel)


def read_fci(
    chunks: Any,
    channel: str,
    *,
    sensor: str = "fci",
    calibrate: bool = True,
    sun_earth_distance: float = 1.0,
    cos_sza: Any = None,
    coeffs: dict[str, Any] | None = None,
    open_chunk: Any = None,
) -> Any:
    """Read one FCI channel across its chunk set into a calibrated `Dataset`.

    Orders the chunks north -> south by their top-left latitude (so the caller
    may pass them in any order), stitches the channel's radiance row-wise,
    calibrates it to reflectance (solar) or brightness temperature (thermal),
    and returns a geolocated pyramids `Dataset` carrying the northernmost chunk's
    CRS + geotransform. The chunks must share a CRS, cell size and column count
    and be vertically contiguous (validated).

    Args:
        chunks: An ordered iterable of chunks, each either a pyramids `Dataset`
            already holding the channel radiance, or a value accepted by
            `open_chunk` (by default a NetCDF path).
        channel: Channel identifier (e.g. `"ir_105"`, `"vis_06"`).
        sensor: Registry sensor name (default `"fci"`).
        calibrate: When `True` (default), calibrate to a physical quantity; when
            `False`, return the stitched raw radiance.
        sun_earth_distance: Sun-earth distance (AU) for solar-channel reflectance.
        cos_sza: Cosine of the solar zenith angle for the reflectance sun-angle
            correction, or `None`.
        coeffs: Per-granule calibration coefficients preferred over the registry
            fallback (see `calibrate_channel`), or `None` to use the registry.
        open_chunk: Callable `(chunk, channel) -> Dataset` used for chunks that
            are not already Datasets. Defaults to a NetCDF reader (see the module
            warning about the FCI layout).

    Returns:
        A pyramids `Dataset` of the calibrated (or raw) channel on the stitched
        grid.

    Raises:
        ReaderError: When `chunks` is empty.
        CalibrationError: When a channel lacks the constants its kind needs.
        UnknownSensorError: When the sensor / channel is not in the registry.
    """
    chunk_list = list(chunks)
    if not chunk_list:
        raise ReaderError("read_fci: no chunks given")

    opener = open_chunk or _default_open_chunk
    datasets = [
        chunk if hasattr(chunk, "read_array") else opener(chunk, channel)
        for chunk in chunk_list
    ]
    _validate_chunk_grid(datasets)

    # Order the chunks north -> south by their top-left latitude/y so the stitch
    # and geolocation are correct regardless of the order chunks were passed in
    # (FCI files are commonly numbered south -> north).
    ordered = sorted(datasets, key=lambda ds: ds.geotransform[3], reverse=True)
    radiance = np.concatenate(
        [np.asarray(ds.read_array(), dtype=float) for ds in ordered], axis=0
    )
    data = (
        calibrate_channel(
            radiance, channel, sensor, sun_earth_distance, cos_sza, coeffs=coeffs
        )
        if calibrate
        else radiance
    )

    from pyramids.dataset import Dataset

    north = ordered[0]
    return Dataset.create_from_array(data, geo=north.geotransform, epsg=north.epsg)


def _validate_chunk_grid(datasets: list) -> None:
    """Check the chunks share a CRS / cell size / width and are contiguous.

    Args:
        datasets: The per-chunk pyramids `Dataset`s (in any vertical order).

    Raises:
        ReaderError: When the chunks have a mixed CRS, cell size or column count,
            or are not vertically contiguous once ordered north -> south.
    """
    first = datasets[0]
    for ds in datasets[1:]:
        if ds.epsg != first.epsg:
            raise ReaderError("read_fci: chunks have mixed CRS")
        if ds.geotransform[1] != first.geotransform[1] or (
            ds.geotransform[5] != first.geotransform[5]
        ):
            raise ReaderError("read_fci: chunks have mixed cell size")
        if ds.columns != first.columns:
            raise ReaderError("read_fci: chunks have mixed column count")

    ordered = sorted(datasets, key=lambda ds: ds.geotransform[3], reverse=True)
    for upper, lower in zip(ordered, ordered[1:]):
        upper_bottom = upper.geotransform[3] + upper.rows * upper.geotransform[5]
        if not np.isclose(upper_bottom, lower.geotransform[3]):
            raise ReaderError("read_fci: chunks are not vertically contiguous")
