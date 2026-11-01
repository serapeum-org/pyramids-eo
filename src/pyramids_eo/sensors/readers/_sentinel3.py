"""Shared decoding helpers for the Sentinel-3 `.SEN3` NetCDF readers.

OLCI and SLSTR ship the same way: a `.SEN3` *directory* holding one NetCDF file
per measurement variable plus annotation files (geolocation, geometry, quality)
and an `xfdumanifest.xml`. Both are read here through pyramids' bundled GDAL
NetCDF driver -- no `netCDF4` dependency, and nothing from the satpy stack.

Unlike the geostationary SEVIRI / FCI readers, these are **swath** instruments: a
pushbroom sensor sweeps a strip of ground, so the pixels land on a curved,
irregular footprint that no affine geotransform can describe. The product
therefore carries a latitude and a longitude for *every* pixel, in its
geolocation annotation file. The readers express that the way GDAL does -- a
`GEOLOCATION` metadata domain pointing at those arrays (see
`geolocation_domain`) -- so a caller can warp on demand while these readers
resample nothing (resampling is deliberately out of scope).

Consequently a swath `Dataset` from here carries a **placeholder** pixel/line
geotransform, not a geographic one; `attach_geolocation` documents that contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._common import finite_float, invalid_mask

#: The identity pixel/line geotransform a swath `Dataset` carries in place of a
#: geographic one. A swath has no affine mapping to the ground, so this is an
#: explicit placeholder: real locations come from the `GEOLOCATION` domain.
PLACEHOLDER_GEOTRANSFORM: tuple[float, float, float, float, float, float] = (
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    -1.0,
)

#: Directory suffix of a Sentinel-3 granule (SAFE-style product directory).
SEN3_SUFFIX = ".SEN3"


def resolve_granule(source: Any, reader: str) -> Path:
    """Resolve a reader `source` to the `.SEN3` granule directory holding its files.

    Args:
        source: A `.SEN3` directory, or any file inside one (e.g. a band NetCDF).
        reader: The calling reader's name, for error messages.

    Returns:
        The granule directory as a `Path`.

    Raises:
        ReaderError: When `source` is neither a directory nor a file inside one.
    """
    path = Path(str(source))
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    raise ReaderError(
        f"{reader}: {str(source)!r} is not a Sentinel-3 granule directory (or a "
        f"file inside one); pass the {SEN3_SUFFIX} directory"
    )


def granule_file(granule: Path, filename: str, reader: str) -> Path:
    """Return `filename` inside `granule`, checking it exists.

    Args:
        granule: The `.SEN3` granule directory.
        filename: The file to locate (e.g. `"geo_coordinates.nc"`).
        reader: The calling reader's name, for error messages.

    Returns:
        The file's `Path`.

    Raises:
        ReaderError: When the file is not in the granule.
    """
    path = granule / filename
    if not path.is_file():
        raise ReaderError(
            f"{reader}: granule {granule.name!r} has no {filename!r}; the product "
            f"looks incomplete or is not the expected Sentinel-3 level/type"
        )
    return path


def subdataset(path: Any, variable: str) -> str:
    """Build the GDAL NetCDF subdataset string for `variable` inside `path`.

    Args:
        path: Path to the NetCDF file.
        variable: The variable name inside it.

    Returns:
        A `NETCDF:"<path>":<variable>` connection string.

    Examples:
        - A variable in a band file becomes a GDAL subdataset string:
            ```python
            >>> from pyramids_eo.sensors.readers._sentinel3 import subdataset
            >>> subdataset("granule.SEN3/Oa08_radiance.nc", "Oa08_radiance")
            'NETCDF:"granule.SEN3/Oa08_radiance.nc":Oa08_radiance'

            ```
    """
    return f'NETCDF:"{path}":{variable}'


def _open(path: Any, variable: str, reader: str) -> Any:
    """Open one NetCDF variable as a GDAL raster, or raise `ReaderError`."""
    from osgeo import gdal

    source = subdataset(path, variable)
    try:
        raster = gdal.Open(source)
    except RuntimeError as exc:  # GDAL exceptions enabled
        raise ReaderError(f"{reader}: cannot open {source} ({exc!r})") from exc
    if raster is None:  # GDAL exceptions disabled
        raise ReaderError(f"{reader}: cannot open {source}")
    return raster


def read_packed_variable(
    path: Any, variable: str, *, reader: str
) -> tuple[np.ndarray, dict[str, str]]:
    """Read one CF-packed NetCDF variable, unpacked and masked to NaN.

    Applies the CF unpacking the Sentinel-3 products use -- `raw * scale_factor +
    add_offset` -- and masks the `_FillValue` and anything outside the variable's
    validity limits to NaN, via the shared CF decoder in `_common` (so the
    attribute spellings are resolved the same way for every reader). A variable
    with no `scale_factor` / `add_offset` (the geolocation and already-geophysical
    variables) is read as-is, so this serves both the packed measurement
    variables and the plain annotation ones.

    Args:
        path: Path to the NetCDF file.
        variable: The variable to read.
        reader: The calling reader's name, for error messages.

    Returns:
        A `(array, metadata)` pair. `array` is float64 with invalid pixels NaN;
        `metadata` is the variable's GDAL attribute dict.

    Raises:
        ReaderError: When the variable cannot be opened, or carries a
            `scale_factor` / `add_offset` that is not a finite number.
    """
    raster = _open(path, variable, reader)
    band = raster.GetRasterBand(1)
    raw = np.asarray(band.ReadAsArray())
    meta = dict(band.GetMetadata())
    # Release both handles before any raise below: a live band reference keeps the
    # owning dataset open, so drop them as soon as the data is in memory.
    del raster, band

    values = raw.astype(float)
    raw_scale, raw_offset = meta.get("scale_factor"), meta.get("add_offset")
    if raw_scale is not None or raw_offset is not None:
        scale = 1.0 if raw_scale is None else finite_float(raw_scale)
        offset = 0.0 if raw_offset is None else finite_float(raw_offset)
        if scale is None or offset is None:
            raise ReaderError(
                f"{reader}: {variable!r} has a scale_factor / add_offset that is "
                f"not a finite number ({raw_scale!r} / {raw_offset!r})"
            )
        values = values * scale + offset

    values[invalid_mask(raw, meta)] = np.nan
    return values, meta


def geolocation_domain(
    geo_path: Any,
    *,
    longitude_variable: str = "longitude",
    latitude_variable: str = "latitude",
    pixel_step: int = 1,
    line_step: int = 1,
) -> dict[str, str]:
    """Build the GDAL `GEOLOCATION` metadata domain for a swath grid.

    This is how a swath product states where its pixels are without an affine
    geotransform: the domain names the NetCDF subdatasets holding a longitude and
    a latitude for every pixel, and GDAL reads them when a caller warps. It
    describes the geolocation rather than applying it, so no resampling happens
    here.

    Args:
        geo_path: Path to the geolocation NetCDF file (e.g.
            `geo_coordinates.nc` for OLCI, `geodetic_an.nc` for SLSTR).
        longitude_variable: Longitude variable name inside it.
        latitude_variable: Latitude variable name inside it.
        pixel_step: Column stride between geolocation samples (`1` for a
            full-resolution per-pixel grid).
        line_step: Row stride between geolocation samples.

    Returns:
        The `GEOLOCATION` domain as a dict of GDAL metadata items: the longitude /
        latitude subdataset strings and bands, the offsets and steps, plus the
        `SRS` (lon/lat on WGS 84, which the arrays are in) and a
        `GEOREFERENCING_CONVENTION` of `PIXEL_CENTER`. GDAL needs the whole set,
        so a domain missing any of it is silently unusable by a warp.

    Examples:
        - A full-resolution OLCI geolocation grid:
            ```python
            >>> from pyramids_eo.sensors.readers._sentinel3 import geolocation_domain
            >>> domain = geolocation_domain("g.SEN3/geo_coordinates.nc")
            >>> domain["X_DATASET"]
            'NETCDF:"g.SEN3/geo_coordinates.nc":longitude'
            >>> domain["PIXEL_STEP"], domain["LINE_STEP"]
            ('1', '1')

            ```
        - A subsampled tie-point grid records its stride:
            ```python
            >>> domain = geolocation_domain(
            ...     "g.SEN3/tie_geo_coordinates.nc", pixel_step=64
            ... )
            >>> domain["PIXEL_STEP"]
            '64'

            ```
    """
    return {
        "X_DATASET": subdataset(geo_path, longitude_variable),
        "X_BAND": "1",
        "Y_DATASET": subdataset(geo_path, latitude_variable),
        "Y_BAND": "1",
        "PIXEL_OFFSET": "0",
        "LINE_OFFSET": "0",
        "PIXEL_STEP": str(pixel_step),
        "LINE_STEP": str(line_step),
        # The geolocation arrays are plain lon/lat degrees on WGS 84.
        "SRS": "EPSG:4326",
        "GEOREFERENCING_CONVENTION": "PIXEL_CENTER",
    }


def wrap_swath(data: np.ndarray, domain: dict[str, str]) -> Any:
    """Wrap a swath array in a `Dataset` carrying its `GEOLOCATION` domain.

    The returned `Dataset` has NaN nodata and a **placeholder** pixel/line
    geotransform (`PLACEHOLDER_GEOTRANSFORM`) with no CRS, because a swath has no
    affine mapping to the ground. Its real geolocation is the attached
    `GEOLOCATION` domain, which `gdal.Warp(..., geoloc=True)` consumes.

    Args:
        data: The `(row, col)` swath array, NaN where invalid.
        domain: The `GEOLOCATION` domain from `geolocation_domain`, or an empty
            dict to attach none -- which is what a reader's `geolocation=False`
            passes, leaving a result with no geolocation at all.

    Returns:
        A pyramids `Dataset` of `data`, with the domain set on its GDAL handle
        when `domain` is non-empty. The domain is copied, so the caller's dict is
        not adopted.
    """
    from pyramids.dataset import Dataset, GeoReference

    dataset = Dataset.from_array(
        np.asarray(data, dtype=float),
        geo_ref=GeoReference(geo=PLACEHOLDER_GEOTRANSFORM, epsg=None),
        no_data_value=np.nan,
    )
    dataset.raster.SetMetadata(dict(domain), "GEOLOCATION")
    return dataset
