"""Real MTG-FCI L1C FDHSI granule reader.

`read_fci_l1c` decodes and stitches one channel — or several in a single pass
(`channels=[...]` returns a `dict`, reading each chunk's structure and
coefficients once for the whole set) — across the real FCI L1C FDHSI chunk files
of a repeat cycle, and `available_channels` lists which channels a chunk carries.
Unlike the generic `read_fci` (which stitches already-opened radiance
`Dataset`s), this reads the actual granule layout:

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

import os
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._common import calibrate_channel, resolve_channels

#: Group holding a channel's measured radiance and per-granule coefficients.
_MEASURED_GROUP = "/data/{channel}/measured"


def _open_root(path: Any) -> tuple[Any, Any]:
    """Open a chunk as a multidimensional dataset and return `(dataset, root_group)`.

    The owning `Dataset` is returned alongside its root `Group` so the caller can
    keep it alive while reading (a GDAL child `Group` does not keep its parent
    dataset alive), then release both.

    Args:
        path: Path to an FCI L1C FDHSI chunk NetCDF file.

    Returns:
        A `(dataset, root_group)` pair.

    Raises:
        ReaderError: When the file cannot be opened.
    """
    from osgeo import gdal

    try:
        dataset = gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER)
    except RuntimeError as exc:  # GDAL exceptions enabled: unopenable/corrupt file
        raise ReaderError(f"read_fci_l1c: cannot open {path} ({exc!r})") from exc
    if dataset is None:  # GDAL exceptions disabled
        raise ReaderError(f"read_fci_l1c: cannot open {path}")
    return dataset, dataset.GetRootGroup()


#: Upper-bound threshold separating a real band solar irradiance (tens-hundreds)
#: from the fill sentinel a thermal band carries for it (the netCDF float fill is
#: ~9.97e36, well above this).
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
    dataset, root = _open_root(path)
    try:
        group = root.OpenGroupFromFullname(_MEASURED_GROUP.format(channel=channel))
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
        channel-kind-specific constants). A constant whose variable is absent is
        omitted, so `calibrate_channel` falls back to the registry value rather
        than being handed a `None`. When the granule carries neither a solar
        irradiance nor any thermal coefficient, no `kind` is set so the registry
        decides — a solar channel merely missing its irradiance is not mistyped
        as thermal (which would fail with a misleading "no central_wavenumber").
    """
    solar_irradiance = _scalar(group, "channel_effective_solar_irradiance")
    if solar_irradiance is not None and solar_irradiance < _COEFF_FILL:
        return {"kind": "solar", "solar_irradiance": solar_irradiance}
    coeffs: dict[str, Any] = {}
    for key, variable in (
        ("central_wavenumber_cm1", "radiance_to_bt_conversion_coefficient_wavenumber"),
        ("alpha", "radiance_to_bt_conversion_coefficient_a"),
        ("beta", "radiance_to_bt_conversion_coefficient_b"),
    ):
        value = _scalar(group, variable)
        if value is not None:
            coeffs[key] = value
    # Only claim the thermal kind when the granule actually carries a thermal
    # coefficient; otherwise leave it unset for the registry to resolve.
    if coeffs:
        coeffs["kind"] = "thermal"
    return coeffs


#: Matches a signed integer/decimal number, optionally in scientific notation.
#: Written without overlapping quantifiers so it has linear (no-backtracking) time.
_NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


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


def read_fci_l1c_chunks(
    path: Any, channels: Sequence[str]
) -> dict[str, dict[str, Any] | None]:
    """Decode several channels from one chunk, opening its structure once.

    Opens the chunk's multidimensional dataset a **single** time to read every
    requested channel's group, per-granule coefficients and row positions (rather
    than re-opening it once per channel), then reads each present channel's
    radiance raster. A channel the file does not carry (e.g. on the `CHK-TRAIL`
    trailer) maps to `None`.

    Args:
        path: Path to a single FCI L1C FDHSI chunk NetCDF file.
        channels: The channel identifiers to decode.

    Returns:
        A `{channel: record | None}` mapping, each record shaped like
        `read_fci_l1c_chunk`'s.
    """
    dataset, root = _open_root(path)
    structure: dict[str, dict[str, Any] | None] = {}
    for channel in channels:
        try:
            group = root.OpenGroupFromFullname(_MEASURED_GROUP.format(channel=channel))
        except RuntimeError:
            group = None
        if group is None or "effective_radiance" not in group.GetMDArrayNames():
            structure[channel] = None
            continue
        structure[channel] = {
            "start_row": _scalar(group, "start_position_row"),
            "end_row": _scalar(group, "end_position_row"),
            "coeffs": _granule_coeffs(group),
        }
    # Release the multidim handle before the per-channel radiance opens, so the
    # coefficient reads all happen while the owning dataset is alive and only one
    # handle is held at a time.
    del root, dataset

    records: dict[str, dict[str, Any] | None] = {}
    for channel, meta in structure.items():
        if meta is None:
            records[channel] = None
            continue
        # GDAL's netCDF driver reads the per-channel radiance grid (it applies the
        # channel's own scale/offset and geolocation, incl. dual-calibration
        # subtleties), so the radiance raster is read per channel.
        radiance, geotransform, crs = _unpack_radiance(path, channel)
        records[channel] = {
            "radiance": radiance,
            "start_row": meta["start_row"],
            "end_row": meta["end_row"],
            "coeffs": meta["coeffs"],
            "geotransform": geotransform,
            "crs": crs,
        }
    return records


def available_channels(chunks: Any) -> list[str]:
    """List the VIS/IR channels present across FCI L1C FDHSI chunk file(s).

    Opens each chunk's multidimensional dataset and reports the `data/<channel>`
    groups that carry an `effective_radiance` array, so a caller can discover the
    channel names instead of hard-coding them. The union across chunks is returned
    (a trailer chunk contributes none), sorted.

    Args:
        chunks: A single chunk path, or an iterable of chunk paths. (For
            convenience this discovery helper also accepts a lone path, whereas
            `read_fci_l1c` / `read_fci` take an iterable of chunk paths.)

    Returns:
        The sorted channel identifiers available in the chunk(s).
    """
    paths = [chunks] if isinstance(chunks, (str, bytes, os.PathLike)) else list(chunks)
    found: set[str] = set()
    for path in paths:
        dataset, root = _open_root(path)
        try:
            try:
                data = root.OpenGroup("data")
            except RuntimeError:
                data = None
            if data is None:  # no /data group (e.g. a trailer / malformed chunk)
                continue
            for name in data.GetGroupNames():
                # Handle both GDAL modes: OpenGroup raises (exceptions on) or
                # returns None (exceptions off) when a group is absent.
                try:
                    channel_group = data.OpenGroup(name)
                    measured = (
                        channel_group.OpenGroup("measured") if channel_group else None
                    )
                except RuntimeError:
                    measured = None
                if (
                    measured is not None
                    and "effective_radiance" in measured.GetMDArrayNames()
                ):
                    found.add(name)
        finally:
            # Always release the multidim handle, even on an unexpected error.
            del root, dataset
    return sorted(found)


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
        ReaderError: On a non-north-up grid, mixed CRS / cell size / column count /
            calibration coefficients, or a vertical gap / overlap between chunks
            (which would silently mis-stitch the scene).
    """
    first = chunks[0]
    # The sort-descending + top-down concatenation assumes a north-up grid; make
    # that a first-class check rather than letting a south-up grid fall through to
    # a misleading "not vertically contiguous" error.
    if first["geotransform"][5] >= 0:
        raise ReaderError(
            "read_fci_l1c: expected a north-up grid (geotransform[5] < 0)"
        )
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
        if chunk["coeffs"] != first["coeffs"]:
            raise ReaderError(
                "read_fci_l1c: chunks have mixed calibration coefficients"
            )

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


def _assemble_channel(
    records: list,
    channel: str,
    *,
    calibrate: bool,
    sun_earth_distance: float,
    cos_sza: Any,
) -> Any:
    """Order, validate, stitch and calibrate one channel's chunk records.

    Args:
        records: The channel's decoded chunk records (from `read_fci_l1c_chunks`),
            in any order; must be non-empty.
        channel: Channel identifier (for calibration + errors).
        calibrate: When `True`, calibrate to reflectance / brightness temperature;
            when `False`, return the stitched raw radiance.
        sun_earth_distance: Sun-earth distance (AU) for solar-channel reflectance.
        cos_sza: Cosine of the solar zenith angle, or `None`.

    Returns:
        A geolocated pyramids `Dataset` on the stitched geostationary grid.

    Raises:
        ReaderError: When `records` is empty, or the chunks are inconsistent.
    """
    if not records:
        raise ReaderError(f"read_fci_l1c: no chunk carries channel {channel!r}")

    # Order north -> south by the geotransform Y origin — the geostationary grid,
    # NOT the row index, is the geolocation source. (FCI's start_position_row runs
    # the opposite way to the geospatial Y, so ordering by it would flip the scene;
    # start/end_position_row are kept on the record as metadata only.) Sort into a
    # new list rather than mutating the caller's.
    ordered = sorted(records, key=lambda chunk: chunk["geotransform"][3], reverse=True)
    _validate_chunks(ordered)

    radiance = np.concatenate([chunk["radiance"] for chunk in ordered], axis=0)
    data = (
        calibrate_channel(
            radiance,
            channel,
            "fci",
            sun_earth_distance,
            cos_sza,
            coeffs=ordered[0]["coeffs"],
        )
        if calibrate
        else radiance
    )

    from pyramids.dataset import Dataset

    top = ordered[0]
    height = _satellite_height(top["crs"])
    # The granule's geotransform is in geostationary radians; scale it by the
    # satellite height to get the metre grid the geostationary CRS expects.
    geo = tuple(term * height for term in top["geotransform"])
    dataset = Dataset.create_from_array(data, geo=geo, epsg=None, no_data_value=np.nan)
    dataset.crs = top["crs"]
    return dataset


def read_fci_l1c(
    paths: Any,
    channel: str | None = None,
    *,
    channels: Sequence[str] | None = None,
    calibrate: bool = True,
    sun_earth_distance: float = 1.0,
    cos_sza: Any = None,
) -> Any:
    """Read one or several channels across a set of real FCI L1C FDHSI chunks.

    Decodes each chunk once for the whole requested channel set
    (`read_fci_l1c_chunks` opens the chunk's structure a single time), drops chunks
    without a channel's radiance (e.g. `CHK-TRAIL`), then for each channel orders
    the chunks north -> south by their geotransform Y origin, checks they are
    vertically contiguous (and share a grid), stitches the radiance, and calibrates
    it with the **per-granule** coefficients (reflectance for a solar channel,
    brightness temperature for a thermal one). Each result is a geolocated pyramids
    `Dataset` on the granule's geostationary grid (metre geotransform reconstructed
    from the angular grid and the CRS satellite height).

    Pass exactly one of `channel` (returns a `Dataset`) or `channels` (returns a
    `dict[str, Dataset]`); `read_fci_l1c(paths, "ir_105")` is unchanged.

    Args:
        paths: Iterable of FCI L1C FDHSI chunk file paths (any order; trailer /
            non-imagery chunks are skipped).
        channel: A single channel identifier (e.g. `"ir_105"`, `"vis_06"`) for a
            `Dataset` result; mutually exclusive with `channels`.
        channels: A sequence of channel identifiers for a `dict[str, Dataset]`
            result (each chunk opened once for the whole set); mutually exclusive
            with `channel`.
        calibrate: When `True` (default), calibrate to reflectance / brightness
            temperature; when `False`, return the stitched raw radiance.
        sun_earth_distance: Sun-earth distance (AU) for solar-channel reflectance.
            The default `1.0` leaves reflectance up to ~3.4% off (`d` ranges
            ~0.983-1.017 AU over the year); pass the granule's `d` for absolute
            accuracy.
        cos_sza: Cosine of the solar zenith angle for the reflectance sun-angle
            correction, or `None`.

    Returns:
        A pyramids `Dataset` (for `channel`) or a `dict[str, Dataset]` keyed by
        channel (for `channels`) of the calibrated (or raw) channel(s) on the
        stitched geostationary grid, with NaN nodata and the geostationary CRS.

    Raises:
        ReaderError: When neither / both of `channel` / `channels` are given, when
            no chunk carries a requested channel, or the chunks are inconsistent
            (mixed CRS / cell size / column count, or not vertically contiguous —
            see `_validate_chunks`).
        CalibrationError: When a channel lacks the constants its kind needs.
        UnknownSensorError: When a channel is not in the registry.
    """
    requested, single = resolve_channels(channel, channels, "read_fci_l1c")

    per_channel: dict[str, list] = {name: [] for name in requested}
    for path in paths:
        for name, record in read_fci_l1c_chunks(path, requested).items():
            if record is not None:
                per_channel[name].append(record)

    results = {
        name: _assemble_channel(
            per_channel[name],
            name,
            calibrate=calibrate,
            sun_earth_distance=sun_earth_distance,
            cos_sza=cos_sza,
        )
        for name in requested
    }
    return results[requested[0]] if single else results
