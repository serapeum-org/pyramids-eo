"""Synthetic Sentinel-3 `.SEN3` granule builders for the OLCI / SLSTR reader tests.

These build granules from the **product specification** rather than from invented
conventions: the real filenames, the real variable names, and the real CF
attributes, including the packing the geolocation actually ships with (`int32`
micro-degrees, `scale_factor = 1e-06`, `_FillValue = -2147483648`, and
`valid_min` / `valid_max` in *packed* units). That matters because a fixture with
invented contents makes a wrong attribute name, a wrong filename or a
packed-versus-unpacked confusion equally invisible, however high the coverage.

The files are written by `tests/_netcdf3.write_netcdf`, a minimal NetCDF-3
writer, because GDAL's netCDF *writer* cannot put several cleanly-named
variables in one file -- a multi-band write suffixes them (`longitude1`), the
multidimensional API produces files this GDAL build cannot reopen classically,
and `APPEND_SUBDATASET=YES` overwrites instead of appending. A real
`geo_coordinates.nc` carries `longitude` *and* `latitude`, so expressing that is
a requirement, not a nicety.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests._netcdf3 import write_netcdf

#: Default packing for a fixture measurement variable.
SCALE = 0.5
OFFSET = -1.0
FILL = 65535

#: The geolocation packing the Sentinel-3 products use.
GEO_SCALE = 1e-06
GEO_FILL = -2147483648

#: The grid and measured quantity each SLSTR channel really ships on, per the
#: SLSTR L1 RBT product-composition table. Note the fire channels are
#: **asymmetric**: F1 has its own `f` grid while F2 stays on the 1 km `i` grid.
#: S4-S6 additionally appear on the alternative 500 m `b` grid.
PDFS_LAYOUT: dict[str, tuple[tuple[str, ...], str]] = {
    "S1": (("a",), "radiance"),
    "S2": (("a",), "radiance"),
    "S3": (("a",), "radiance"),
    "S4": (("a", "b"), "radiance"),
    "S5": (("a", "b"), "radiance"),
    "S6": (("a", "b"), "radiance"),
    "S7": (("i",), "BT"),
    "S8": (("i",), "BT"),
    "S9": (("i",), "BT"),
    "F1": (("f",), "BT"),
    "F2": (("i",), "BT"),
}


def packed_counts() -> np.ndarray:
    """Return a small raw count array containing one `_FillValue` pixel.

    Returns:
        A `(2, 3)` uint16-valued int32 array whose `[0, 2]` element is the fill
        value (int32 because the NetCDF-3 writer has no unsigned types; the
        values stay in uint16 range).
    """
    return np.array([[10, 20, FILL], [30, 40, 50]], dtype="int32")


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


def write_measurement(
    path: Path,
    variable: str,
    counts: np.ndarray,
    *,
    units: str = "mW.m-2.sr-1.nm-1",
    valid_min: int | None = None,
    valid_max: int | None = None,
) -> Path:
    """Write one packed measurement variable with its CF attributes.

    Args:
        path: The NetCDF file to create.
        variable: The variable name (matching the file stem, as the products do).
        counts: The raw, still-packed counts.
        units: The `units` attribute.
        valid_min: An optional `valid_min`, in *packed* units.
        valid_max: An optional `valid_max`, in *packed* units.

    Returns:
        The created file's path.
    """
    attributes: dict[str, object] = {
        "scale_factor": SCALE,
        "add_offset": OFFSET,
        "_FillValue": FILL,
        "units": units,
    }
    if valid_min is not None:
        attributes["valid_min"] = valid_min
    if valid_max is not None:
        attributes["valid_max"] = valid_max
    return write_netcdf(path, {variable: (counts, attributes)})


def write_geolocation(
    path: Path,
    *,
    longitude_variable: str = "longitude",
    latitude_variable: str = "latitude",
    shape: tuple[int, int] = (2, 3),
    packed: bool = True,
) -> Path:
    """Write a geolocation annotation file carrying longitude **and** latitude.

    Args:
        path: The NetCDF file to create.
        longitude_variable: Longitude variable name (`longitude` for OLCI,
            `longitude_an` etc. for SLSTR).
        latitude_variable: Latitude variable name.
        shape: The `(rows, columns)` grid shape.
        packed: When `True` (the product's form), write `int32` micro-degrees
            with `scale_factor = 1e-06`; when `False`, write plain `float64`
            degrees, which is what the tie-point grids use.

    Returns:
        The created file's path.
    """
    rows, columns = shape
    degrees_lon = np.tile(np.linspace(4.0, 5.0, columns), (rows, 1))
    degrees_lat = np.repeat(np.linspace(52.0, 51.0, rows)[:, None], columns, axis=1)
    if packed:
        attributes = {"scale_factor": GEO_SCALE, "_FillValue": GEO_FILL}
        longitude = np.round(degrees_lon / GEO_SCALE).astype("int32")
        latitude = np.round(degrees_lat / GEO_SCALE).astype("int32")
    else:
        attributes = {}
        longitude = degrees_lon.astype("float64")
        latitude = degrees_lat.astype("float64")
    return write_netcdf(
        path,
        {
            longitude_variable: (longitude, {**attributes, "units": "degrees_east"}),
            latitude_variable: (latitude, {**attributes, "units": "degrees_north"}),
        },
    )


def olci_granule(
    root: Path,
    *,
    bands: dict[str, np.ndarray] | None = None,
    geolocation: bool = True,
    packed_geolocation: bool = True,
    product: str = "EFR",
    name: str | None = None,
) -> Path:
    """Build a synthetic OLCI L1B `.SEN3` granule.

    Args:
        root: Directory to create the granule inside (e.g. `tmp_path`).
        bands: `{channel: raw counts}` to write, defaulting to one `Oa08` band.
            An explicitly empty dict writes no bands.
        geolocation: When `True`, also write `geo_coordinates.nc`.
        packed_geolocation: Passed to `write_geolocation` as `packed`.
        product: `"EFR"` (full resolution) or `"ERR"` (reduced), which the
            granule name encodes exactly as the real products do.
        name: An explicit granule directory name, overriding `product`.

    Returns:
        The granule directory's path.
    """
    granule = root / (name or f"S3A_OL_1_{product}____20260101T101500.SEN3")
    granule.mkdir(parents=True, exist_ok=True)
    # `is None` not `or`: an explicitly empty dict means "write no bands", which a
    # falsy-default would silently turn back into the default band.
    requested = {"Oa08": packed_counts()} if bands is None else bands
    for channel, counts in requested.items():
        write_measurement(
            granule / f"{channel}_radiance.nc", f"{channel}_radiance", counts
        )
    if geolocation:
        write_geolocation(granule / "geo_coordinates.nc", packed=packed_geolocation)
    return granule


def slstr_granule(
    root: Path,
    *,
    channels: dict[str, np.ndarray] | None = None,
    view: str = "n",
    geodetic: bool = True,
    name: str = "S3A_SL_1_RBT____20260101T101500.SEN3",
) -> Path:
    """Build a synthetic SLSTR L1 RBT granule on each channel's conventional grid.

    Args:
        root: Directory to create the granule inside (e.g. `tmp_path`).
        channels: `{channel: raw counts}` to write, defaulting to one `S5`
            channel. Each is written on the first grid `PDFS_LAYOUT` gives it.
        view: One-letter view suffix (`"n"` nadir, `"o"` oblique).
        geodetic: When `True`, also write the geodetic file for each grid written.
        name: The granule directory name.

    Returns:
        The granule directory's path.
    """
    granule = root / name
    granule.mkdir(parents=True, exist_ok=True)
    requested = {"S5": packed_counts()} if channels is None else channels
    grids = set()
    for channel, counts in requested.items():
        grid = PDFS_LAYOUT[channel][0][0]
        quantity = PDFS_LAYOUT[channel][1]
        grids.add(grid)
        stem = f"{channel}_{quantity}_{grid}{view}"
        write_measurement(
            granule / f"{stem}.nc",
            stem,
            counts,
            units="mW.m-2.sr-1.nm-1" if quantity == "radiance" else "K",
        )
    if geodetic:
        for grid in grids:
            _write_geodetic(granule, grid, view)
    return granule


def _write_geodetic(granule: Path, grid: str, view: str) -> Path:
    """Write the `geodetic_<grid><view>.nc` annotation file for one SLSTR grid."""
    suffix = f"{grid}{view}"
    return write_geolocation(
        granule / f"geodetic_{suffix}.nc",
        longitude_variable=f"longitude_{suffix}",
        latitude_variable=f"latitude_{suffix}",
    )


def slstr_pdfs_granule(
    root: Path,
    *,
    views: tuple[str, ...] = ("n", "o"),
    name: str = "S3A_SL_1_RBT____20260101T101500.SEN3",
) -> Path:
    """Build an SLSTR granule reproducing the PDFS filename set verbatim.

    Writes every channel on **every** grid the product composition gives it, in
    both views by default, with the `*_quality_*` annotation sibling that sits
    beside each measurement file in a real granule -- the files a discovery glob
    can wrongly match.

    Args:
        root: Directory to create the granule inside (e.g. `tmp_path`).
        views: The views to populate (`"n"` nadir, `"o"` oblique).
        name: The granule directory name.

    Returns:
        The granule directory's path.
    """
    granule = root / name
    granule.mkdir(parents=True, exist_ok=True)
    grids = set()
    for channel, (channel_grids, quantity) in PDFS_LAYOUT.items():
        for grid in channel_grids:
            for view in views:
                grids.add((grid, view))
                stem = f"{channel}_{quantity}_{grid}{view}"
                write_measurement(
                    granule / f"{stem}.nc",
                    stem,
                    packed_counts(),
                    units="mW.m-2.sr-1.nm-1" if quantity == "radiance" else "K",
                )
                # The quality sibling: same naming shape, different content, not
                # imagery. Its variable deliberately does not match its stem,
                # mirroring the real per-scan uncertainty terms.
                quality = f"{channel}_quality_{grid}{view}"
                write_measurement(
                    granule / f"{quality}.nc",
                    f"{channel}_dL_BB_{grid}{view}",
                    packed_counts(),
                    units="mW.m-2.sr-1.nm-1",
                )
    for grid, view in grids:
        _write_geodetic(granule, grid, view)
    return granule
