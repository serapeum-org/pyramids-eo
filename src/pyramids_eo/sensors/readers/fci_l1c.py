"""Real MTG-FCI L1C FDHSI granule reader.

`read_fci_l1c` decodes and stitches one channel across the real FCI L1C FDHSI
chunk files of a repeat cycle. Unlike the generic `read_fci` (which stitches
already-opened radiance `Dataset`s), this reads the actual granule layout:

* the packed `uint16` radiance from the nested group
  `data/<channel>/measured/effective_radiance`, unpacked to physical radiance
  via the variable's `scale_factor` / `add_offset` and masked by its
  `valid_range` / `_FillValue`;
* the **per-granule** calibration coefficients carried as sibling *variables* in
  the same group (`radiance_to_bt_conversion_coefficient_a` / `_b` /
  `_wavenumber` for thermal channels, `channel_effective_solar_irradiance` for
  solar ones), preferred over the nominal registry table;
* the chunks are ordered and stitched by their geostationary geotransform Y
  origin (FCI's `start_position_row` runs opposite to the geospatial Y, so it is
  read as metadata but is not the stitch key); a chunk that carries no radiance
  for the channel — e.g. the `CHK-TRAIL` trailer — is skipped.

The granule stores the grid in geostationary *angular* (radian) coordinates with
a metre geostationary CRS; the metre geotransform is reconstructed as
`angular_geotransform * satellite_height` (from the CRS `+h`). This addresses the
axis-unit mismatch behind pyramids #706 by keeping the CRS explicit rather than
letting it be misread as lon/lat.

Validated against real MTI1/Meteosat-12 FDHSI chunks (see issue #40): `ir_105`
stitches to brightness temperature in the expected range on the geostationary
grid. The reader uses GDAL (already vendored by pyramids) to read the nested
groups and scalar variables — no extra dependency.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._common import calibrate_channel

#: Group holding a channel's measured radiance and per-granule coefficients.
_MEASURED_GROUP = "/data/{channel}/measured"
#: netCDF default fill for a float coefficient (solar irradiance on thermal bands).
_COEFF_FILL = 1e30


def _measured_group(path: Any, channel: str) -> tuple[Any, Any]:
    """Open a chunk and return its `(dataset, /data/<channel>/measured group)`.

    The owning multidimensional `Dataset` is returned alongside the group so the
    caller can keep it alive while reading the group's arrays (a GDAL child
    `Group` does not keep its parent dataset alive on its own), then release it.

    Args:
        path: Path to an FCI L1C FDHSI chunk NetCDF file.
        channel: Channel identifier (e.g. `"ir_105"`).

    Returns:
        A `(dataset, group)` pair, or `(None, None)` when the file has no such
        group (e.g. the `CHK-TRAIL` trailer).
    """
    from osgeo import gdal

    try:
        dataset = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER)
    except RuntimeError as exc:  # GDAL exceptions enabled: unopenable/corrupt file
        raise ReaderError(f"read_fci_l1c: cannot open {path} ({exc!r})") from exc
    if dataset is None:  # GDAL exceptions disabled
        raise ReaderError(f"read_fci_l1c: cannot open {path}")
    try:
        group = dataset.GetRootGroup().OpenGroupFromFullname(
            _MEASURED_GROUP.format(channel=channel)
        )
    except RuntimeError:
        # No such group -> this file does not carry the channel (e.g. CHK-TRAIL).
        return None, None
    return dataset, group


def _scalar(group: Any, name: str) -> float | None:
    """Read a 0-D coefficient/position variable from `group` as a float.

    Args:
        group: A GDAL multidim group.
        name: Variable name.

    Returns:
        The scalar value, or `None` when the variable is absent.
    """
    try:
        array = group.OpenMDArray(name)
    except RuntimeError:
        return None
    if array is None:
        return None
    return float(np.asarray(array.ReadAsArray()).ravel()[0])


def _granule_coeffs(group: Any) -> dict[str, Any]:
    """Build the per-granule calibration coefficients from group variables.

    A solar channel carries a finite `channel_effective_solar_irradiance`; a
    thermal one carries the inverse-Planck `a` / `b` / `wavenumber` triple.

    Args:
        group: The channel's `measured` multidim group.

    Returns:
        A `coeffs` mapping for `calibrate_channel` (keys `kind` plus the
        channel-kind-specific constants).
    """
    solar_irradiance = _scalar(group, "channel_effective_solar_irradiance")
    if solar_irradiance is not None and solar_irradiance < _COEFF_FILL:
        return {"kind": "solar", "solar_irradiance": solar_irradiance}
    return {
        "kind": "thermal",
        "central_wavenumber_cm1": _scalar(
            group, "radiance_to_bt_conversion_coefficient_wavenumber"
        ),
        "alpha": _scalar(group, "radiance_to_bt_conversion_coefficient_a"),
        "beta": _scalar(group, "radiance_to_bt_conversion_coefficient_b"),
    }


#: Matches a signed integer/decimal number, optionally in scientific notation.
_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _valid_bounds(text: Any) -> tuple[float | None, float | None]:
    """Parse `(min, max)` from a `valid_range` metadata string.

    Args:
        text: The GDAL `valid_range` metadata value (e.g. `"{0, 4095}"`), or
            `None`/an unparseable string.

    Returns:
        The `(valid_min, valid_max)` pair, or `(None, None)` when the string is
        missing or does not contain two numbers.
    """
    if not text:
        return None, None
    numbers = re.findall(_NUMBER, str(text))
    if len(numbers) < 2:
        return None, None
    return float(numbers[0]), float(numbers[-1])


def _unpack_radiance(
    path: Any, channel: str
) -> tuple[np.ndarray, tuple[float, ...], str]:
    """Read + unpack a chunk's radiance, with its angular geotransform + CRS.

    Reads the packed `uint16` `effective_radiance` via a GDAL NetCDF subdataset,
    unpacks it (`raw * scale_factor + add_offset`), and masks counts outside the
    variable's `valid_range` (and the `_FillValue`) to NaN.

    Args:
        path: Path to the chunk NetCDF file.
        channel: Channel identifier.

    Returns:
        A `(radiance, geotransform, crs_wkt)` tuple, where `geotransform` is the
        granule's native angular (radian) geotransform.
    """
    from osgeo import gdal

    subdataset = f'NETCDF:"{path}":/data/{channel}/measured/effective_radiance'
    try:
        raster = gdal.Open(subdataset)
    except RuntimeError as exc:  # GDAL exceptions enabled
        raise ReaderError(f"read_fci_l1c: cannot open {subdataset} ({exc!r})") from exc
    if raster is None:  # GDAL exceptions disabled
        raise ReaderError(f"read_fci_l1c: cannot open {subdataset}")
    band = raster.GetRasterBand(1)
    raw = np.asarray(band.ReadAsArray())
    meta = band.GetMetadata()
    try:
        scale = float(meta["scale_factor"])
        offset = float(meta["add_offset"])
    except (KeyError, ValueError) as exc:
        raise ReaderError(
            f"read_fci_l1c: effective_radiance for {channel!r} lacks a numeric "
            f"scale_factor / add_offset ({exc!r})"
        ) from exc
    radiance = raw.astype(float) * scale + offset

    invalid = np.zeros(raw.shape, dtype=bool)
    valid_min, valid_max = _valid_bounds(meta.get("valid_range"))
    if valid_min is not None:
        invalid |= raw < valid_min
    if valid_max is not None:
        invalid |= raw > valid_max
    if "_FillValue" in meta:
        invalid |= raw == float(meta["_FillValue"])
    radiance[invalid] = np.nan
    geotransform, crs = raster.GetGeoTransform(), raster.GetProjection()
    # Drop both the band and the dataset: a live band reference keeps the owning
    # dataset open, so release both to free the handle at function return.
    del raster, band
    return radiance, geotransform, crs


def read_fci_l1c_chunk(path: Any, channel: str) -> dict[str, Any] | None:
    """Decode one real FCI L1C FDHSI chunk for `channel`.

    Args:
        path: Path to a single FCI L1C FDHSI chunk NetCDF file.
        channel: Channel identifier (e.g. `"ir_105"`).

    Returns:
        A record with `radiance` (unpacked, NaN-masked), the `start_row` /
        `end_row` position, the per-granule `coeffs`, the native angular
        `geotransform`, and the geostationary `crs` WKT — or `None` when the file
        does not carry the channel's radiance (e.g. the `CHK-TRAIL` trailer).
    """
    dataset, group = _measured_group(path, channel)
    if group is None or "effective_radiance" not in group.GetMDArrayNames():
        return None
    start_row = _scalar(group, "start_position_row")
    end_row = _scalar(group, "end_position_row")
    coeffs = _granule_coeffs(group)
    # Release the multidimensional handle before opening the radiance raster, so
    # only one GDAL file handle is held at a time and the group's reads all
    # happen while its owning dataset is still alive.
    del dataset, group
    radiance, geotransform, crs = _unpack_radiance(path, channel)
    return {
        "radiance": radiance,
        "start_row": start_row,
        "end_row": end_row,
        "coeffs": coeffs,
        "geotransform": geotransform,
        "crs": crs,
    }


def _satellite_height(crs_wkt: str) -> float:
    """Extract the geostationary satellite height (metres) from a CRS WKT.

    Args:
        crs_wkt: The geostationary CRS as WKT (its PROJ form carries `+h=`).

    Returns:
        The perspective-point height in metres.

    Raises:
        ReaderError: When the WKT carries no satellite height.
    """
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)
    match = re.search(rf"\+h=({_NUMBER})", srs.ExportToProj4())
    if not match:
        raise ReaderError("FCI CRS carries no satellite height (+h)")
    height = float(match.group(1))
    if not 1.0e6 < height < 1.0e9:
        raise ReaderError(f"FCI CRS satellite height {height} m is implausible")
    return height


def _validate_chunks(chunks: list) -> None:
    """Check the ordered chunks form one consistent geostationary mosaic.

    The chunks must be pre-sorted north -> south by their geotransform Y origin.
    Validates that they share a CRS, cell size and column count, and that each
    chunk's bottom edge meets the next chunk's top edge (no vertical gap or
    overlap) — so concatenating their arrays yields a correctly geolocated grid.

    Args:
        chunks: The chunk records, sorted by `geotransform[3]` descending.

    Raises:
        ReaderError: On mixed CRS / cell size / column count, or a vertical gap /
            overlap between chunks (which would silently mis-stitch the scene).
    """
    first = chunks[0]
    columns = first["radiance"].shape[1]
    for chunk in chunks[1:]:
        if chunk["crs"] != first["crs"]:
            raise ReaderError("read_fci_l1c: chunks have mixed CRS")
        if not np.isclose(
            chunk["geotransform"][1], first["geotransform"][1]
        ) or not np.isclose(chunk["geotransform"][5], first["geotransform"][5]):
            raise ReaderError("read_fci_l1c: chunks have mixed cell size")
        if chunk["radiance"].shape[1] != columns:
            raise ReaderError("read_fci_l1c: chunks have mixed column count")

    # Tolerance scaled to the (angular) row pixel, so it tracks the grid rather
    # than depending on the coordinate magnitude.
    atol = abs(first["geotransform"][5]) * 1e-3
    for upper, lower in zip(chunks, chunks[1:]):
        upper_bottom = (
            upper["geotransform"][3]
            + upper["radiance"].shape[0] * upper["geotransform"][5]
        )
        if not np.isclose(upper_bottom, lower["geotransform"][3], rtol=0.0, atol=atol):
            raise ReaderError(
                "read_fci_l1c: chunks are not vertically contiguous "
                f"({upper_bottom} -> {lower['geotransform'][3]})"
            )


def read_fci_l1c(
    paths: Any,
    channel: str,
    *,
    calibrate: bool = True,
    sun_earth_distance: float = 1.0,
    cos_sza: Any = None,
) -> Any:
    """Read one channel across a set of real FCI L1C FDHSI chunk files.

    Decodes each chunk (`read_fci_l1c_chunk`), drops those without the channel's
    radiance (e.g. `CHK-TRAIL`), orders the rest north -> south by their
    geotransform Y origin, checks they are vertically contiguous (and share a
    grid), stitches the radiance, and calibrates it with
    the **per-granule** coefficients (reflectance for a solar channel, brightness
    temperature for a thermal one). The result is a geolocated pyramids `Dataset`
    on the granule's geostationary grid (metre geotransform reconstructed from the
    angular grid and the CRS satellite height).

    Args:
        paths: Iterable of FCI L1C FDHSI chunk file paths (any order; trailer /
            non-imagery chunks are skipped).
        channel: Channel identifier (e.g. `"ir_105"`, `"vis_06"`).
        calibrate: When `True` (default), calibrate to reflectance / brightness
            temperature; when `False`, return the stitched raw radiance.
        sun_earth_distance: Sun-earth distance (AU) for solar-channel reflectance.
        cos_sza: Cosine of the solar zenith angle for the reflectance sun-angle
            correction, or `None`.

    Returns:
        A pyramids `Dataset` of the calibrated (or raw) channel on the stitched
        geostationary grid, with NaN nodata and the granule's geostationary CRS.

    Raises:
        ReaderError: When no chunk carries the channel, or the chunks are
            inconsistent (mixed CRS / cell size / column count, or not vertically
            contiguous on the geostationary grid — see `_validate_chunks`).
        CalibrationError: When a channel lacks the constants its kind needs.
        UnknownSensorError: When the channel is not in the registry.
    """
    chunks = [
        chunk
        for chunk in (read_fci_l1c_chunk(path, channel) for path in paths)
        if chunk is not None
    ]
    if not chunks:
        raise ReaderError(f"read_fci_l1c: no chunk carries channel {channel!r}")

    # Order north -> south by the geotransform Y origin — the geostationary grid,
    # NOT the row index, is the geolocation source. (FCI's start_position_row runs
    # the opposite way to the geospatial Y, so ordering by it would flip the scene;
    # start/end_position_row are kept on the record as metadata only.)
    chunks.sort(key=lambda chunk: chunk["geotransform"][3], reverse=True)
    _validate_chunks(chunks)

    radiance = np.concatenate([chunk["radiance"] for chunk in chunks], axis=0)
    data = (
        calibrate_channel(
            radiance,
            channel,
            "fci",
            sun_earth_distance,
            cos_sza,
            coeffs=chunks[0]["coeffs"],
        )
        if calibrate
        else radiance
    )

    from pyramids.dataset import Dataset

    top = chunks[0]
    height = _satellite_height(top["crs"])
    # The granule's geotransform is in geostationary radians; scale it by the
    # satellite height to get the metre grid the geostationary CRS expects.
    geo = tuple(term * height for term in top["geotransform"])
    dataset = Dataset.create_from_array(data, geo=geo, epsg=None, no_data_value=np.nan)
    dataset.crs = top["crs"]
    return dataset
