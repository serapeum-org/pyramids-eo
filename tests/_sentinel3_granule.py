"""Synthetic Sentinel-3 `.SEN3` granule builders for the OLCI / SLSTR reader tests.

These write **real** NetCDF files through pyramids' bundled GDAL netCDF driver, so the
reader tests exercise the actual GDAL subdataset path (`NETCDF:"<file>":<variable>`), the
real CF attribute names (`scale_factor` / `add_offset` / `_FillValue`) and the real
packing arithmetic -- rather than a mock whose shape could drift from GDAL's.

Two deliberate limitations of the fixtures, neither of which the readers depend on:

* GDAL's netCDF *writer* emits one variable per file, so a fixture `geo_coordinates.nc`
  holds a single `longitude` array where a real granule holds `longitude`, `latitude` and
  `altitude`. The readers only require the geolocation file to *exist* -- they reference
  its variables by name in the `GEOLOCATION` domain and never open it themselves (GDAL
  does, when a caller warps) -- so the tests assert the emitted domain strings instead of
  performing a warp.
* GDAL's writer does not round-trip `valid_range`, so the out-of-range masking branch is
  covered by a monkeypatched metadata case in the reader tests and by a direct unit test
  of `_valid_bounds`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Default packing for a fixture measurement variable.
SCALE = 0.5
OFFSET = -1.0
FILL = 65535


def write_variable(
    path: Path,
    variable: str,
    array: np.ndarray,
    *,
    scale: float | None = SCALE,
    offset: float | None = OFFSET,
    fill: int | None = FILL,
    units: str | None = None,
) -> Path:
    """Write `array` as a single named variable in a NetCDF file at `path`.

    Args:
        path: The NetCDF file to create (parents must exist).
        variable: The NetCDF variable name (what the reader addresses).
        array: The raw, still-packed values.
        scale: `scale_factor` to record, or `None` to omit.
        offset: `add_offset` to record, or `None` to omit.
        fill: `_FillValue` to record, or `None` to omit.
        units: A `units` attribute to record, or `None` to omit.

    Returns:
        The created file's path.
    """
    from osgeo import gdal

    gdal_type = gdal.GDT_UInt16 if array.dtype.kind in "ui" else gdal.GDT_Float32
    source = gdal.GetDriverByName("MEM").Create(
        "", array.shape[1], array.shape[0], 1, gdal_type
    )
    band = source.GetRasterBand(1)
    band.WriteArray(array)
    # NETCDF_VARNAME is what the netCDF driver uses to name the variable it writes, which
    # is how a fixture gets the product's real variable name (e.g. "Oa08_radiance").
    band.SetMetadataItem("NETCDF_VARNAME", variable)
    if scale is not None:
        band.SetScale(scale)
    if offset is not None:
        band.SetOffset(offset)
    if fill is not None:
        band.SetNoDataValue(fill)
    if units is not None:
        band.SetMetadataItem("units", units)
    gdal.GetDriverByName("netCDF").CreateCopy(str(path), source)
    del band, source
    return path


def packed_counts() -> np.ndarray:
    """Return a small raw count array containing one `_FillValue` pixel.

    Returns:
        A `(2, 3)` uint16 array whose `[0, 2]` element is the fill value.
    """
    return np.array([[10, 20, FILL], [30, 40, 50]], dtype="uint16")


def expected_radiance(counts: np.ndarray) -> np.ndarray:
    """Unpack `counts` the way the readers do, with the fill value as NaN.

    Args:
        counts: The raw packed counts.

    Returns:
        A float array of `counts * SCALE + OFFSET`, NaN where `counts == FILL`.
    """
    values = counts.astype(float) * SCALE + OFFSET
    values[counts == FILL] = np.nan
    return values


def olci_granule(
    root: Path,
    *,
    bands: dict[str, np.ndarray] | None = None,
    geolocation: bool = True,
    name: str = "S3A_OL_1_EFR.SEN3",
) -> Path:
    """Build a synthetic OLCI L1B `.SEN3` granule.

    Args:
        root: Directory to create the granule inside (e.g. `tmp_path`).
        bands: `{channel: raw counts}` to write, defaulting to one `Oa08` band.
        geolocation: When `True`, also write a `geo_coordinates.nc`.
        name: The granule directory name.

    Returns:
        The granule directory's path.
    """
    granule = root / name
    granule.mkdir(parents=True, exist_ok=True)
    # `is None` not `or`: an explicitly empty dict means "write no bands", which a
    # falsy-default would silently turn back into the default band.
    requested = {"Oa08": packed_counts()} if bands is None else bands
    for channel, counts in requested.items():
        write_variable(
            granule / f"{channel}_radiance.nc",
            f"{channel}_radiance",
            counts,
            units="mW.m-2.sr-1.nm-1",
        )
    if geolocation:
        write_variable(
            granule / "geo_coordinates.nc",
            "longitude",
            np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype="float32"),
            scale=None,
            offset=None,
            fill=None,
        )
    return granule


def slstr_granule(
    root: Path,
    *,
    channels: dict[str, np.ndarray] | None = None,
    view: str = "n",
    geodetic: bool = True,
    name: str = "S3A_SL_1_RBT.SEN3",
) -> Path:
    """Build a synthetic SLSTR L1 RBT `.SEN3` granule.

    Writes each channel on its conventional grid -- `S1`-`S6` radiance on the `a` grid,
    `S7`-`S9` brightness temperature on `i`, `F1`-`F2` on `f` -- plus the matching
    `geodetic_<grid><view>.nc` annotation files.

    Args:
        root: Directory to create the granule inside (e.g. `tmp_path`).
        channels: `{channel: raw counts}` to write, defaulting to one `S5` channel.
        view: One-letter view suffix (`"n"` nadir, `"o"` oblique).
        geodetic: When `True`, also write the geodetic file for each grid written.
        name: The granule directory name.

    Returns:
        The granule directory's path.
    """
    from pyramids_eo.sensors.readers.slstr import _CHANNEL_GRID

    granule = root / name
    granule.mkdir(parents=True, exist_ok=True)
    grids = set()
    # `is None` not `or`, for the same reason as `olci_granule`'s `bands`.
    requested = {"S5": packed_counts()} if channels is None else channels
    for channel, counts in requested.items():
        grid, quantity = _CHANNEL_GRID[channel]
        grids.add(grid)
        stem = f"{channel}_{quantity}_{grid}{view}"
        write_variable(
            granule / f"{stem}.nc",
            stem,
            counts,
            units="mW.m-2.sr-1.nm-1" if quantity == "radiance" else "K",
        )
    if geodetic:
        for grid in grids:
            write_variable(
                granule / f"geodetic_{grid}{view}.nc",
                f"longitude_{grid}{view}",
                np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype="float32"),
                scale=None,
                offset=None,
                fill=None,
            )
    return granule
