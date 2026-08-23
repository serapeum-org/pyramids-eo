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
* the vertical placement from `start_position_row` / `end_position_row`, used to
  order and stitch the chunks (a chunk that carries no radiance for the channel —
  e.g. the `CHK-TRAIL` trailer — is skipped).

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


def _measured_group(path: Any, channel: str) -> Any:
    """Return the `/data/<channel>/measured` multidim group, or `None`.

    Args:
        path: Path to an FCI L1C FDHSI chunk NetCDF file.
        channel: Channel identifier (e.g. `"ir_105"`).

    Returns:
        The GDAL multidim group, or `None` when the file has no such group
        (e.g. the `CHK-TRAIL` trailer).
    """
    from osgeo import gdal

    dataset = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER)
    root = dataset.GetRootGroup()
    try:
        return root.OpenGroupFromFullname(_MEASURED_GROUP.format(channel=channel))
    except RuntimeError:
        return None


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


def _unpack_radiance(path: Any, channel: str) -> tuple[np.ndarray, tuple, str]:
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
    raster = gdal.Open(subdataset)
    band = raster.GetRasterBand(1)
    raw = np.asarray(band.ReadAsArray())
    meta = band.GetMetadata()
    scale = float(meta["scale_factor"])
    offset = float(meta["add_offset"])
    radiance = raw.astype(float) * scale + offset

    valid_max = None
    if "valid_range" in meta:
        valid_max = float(re.findall(r"[-\d.eE+]+", meta["valid_range"])[-1])
    fill = float(meta["_FillValue"]) if "_FillValue" in meta else None
    invalid = np.zeros(raw.shape, dtype=bool)
    if valid_max is not None:
        invalid |= raw > valid_max
    if fill is not None:
        invalid |= raw == fill
    radiance[invalid] = np.nan
    return radiance, raster.GetGeoTransform(), raster.GetProjection()


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
    group = _measured_group(path, channel)
    if group is None or "effective_radiance" not in set(group.GetMDArrayNames()):
        return None
    radiance, geotransform, crs = _unpack_radiance(path, channel)
    return {
        "radiance": radiance,
        "start_row": _scalar(group, "start_position_row"),
        "end_row": _scalar(group, "end_position_row"),
        "coeffs": _granule_coeffs(group),
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
    match = re.search(r"\+h=([\d.]+)", srs.ExportToProj4())
    if not match:
        raise ReaderError("FCI CRS carries no satellite height (+h)")
    return float(match.group(1))


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
    radiance (e.g. `CHK-TRAIL`), orders the rest by `start_position_row`, checks
    they are exactly row-contiguous, stitches the radiance, and calibrates it with
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
        ReaderError: When no chunk carries the channel, or the chunks are not
            row-contiguous.
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

    chunks.sort(key=lambda chunk: chunk["start_row"])
    for upper, lower in zip(chunks, chunks[1:]):
        if upper["end_row"] + 1 != lower["start_row"]:
            raise ReaderError(
                "read_fci_l1c: chunks are not row-contiguous "
                f"({upper['end_row']} -> {lower['start_row']})"
            )

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
