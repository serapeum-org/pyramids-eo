"""MTG-FCI L1C (FDHSI) reader.

`read_fci` stitches a channel across the ~40 NetCDF chunks of one FCI repeat
cycle, calibrates the raw radiance to reflectance / brightness temperature via
the sensor registry, and returns a geolocated pyramids `Dataset` ready for
`to_crs` / `warped_view`.

Warning:
    The zero-config default (`_default_open_chunk`) assumes each chunk exposes
    the requested channel's radiance as a NetCDF **variable named like the
    channel**. Real FCI L1C FDHSI stores radiance in nested groups
    (`data/<channel>/measured/effective_radiance`) with the calibration
    coefficients in group attributes — for those granules pass
    `open_fci_l1c_chunk` (which reads that nested layout) as `open_chunk`, plus
    the coefficients via `coeffs`. The stitch + calibrate + geolocate logic is
    unit-tested on synthetic chunks, and `open_fci_l1c_chunk` is unit-tested for
    its group-path resolution, but the exact FCI variable layout and the
    per-granule coefficient attributes still need validation against a real
    granule (see issue #40).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._common import calibrate_channel, resolve_channels


def _default_open_chunk(path: Any, channel: str) -> Any:
    """Open one FCI chunk and return the channel's radiance as a NetCDF view.

    Reads `path` with `pyramids.netcdf.NetCDF` and extracts the variable named
    `channel`. See the module warning: real FCI L1C uses nested groups, so a
    custom `open_chunk` is expected for real granules.

    Args:
        path: Path to a chunk NetCDF file.
        channel: Channel identifier / variable name.

    Returns:
        A pyramids `NetCDF` variable view of the channel's raw radiance (a
        `Dataset`-like object exposing `read_array` / `geotransform` / `epsg` /
        `columns` / `rows`, as `read_fci` consumes it).
    """
    from pyramids.netcdf import NetCDF

    return NetCDF.read_file(str(path)).get_variable(channel)


#: Group-qualified variable-name template for FCI L1C FDHSI channel radiance.
_FCI_RADIANCE_GROUP = "data/{channel}/measured/effective_radiance"


def open_fci_l1c_chunk(
    path: Any,
    channel: str,
    *,
    radiance_group: str = _FCI_RADIANCE_GROUP,
) -> Any:
    """Open one real FCI L1C FDHSI chunk from its nested group layout.

    Unlike `_default_open_chunk` (which assumes a flat variable named like the
    channel), this reads the radiance from the nested group path a real FCI L1C
    FDHSI chunk uses — by default `data/<channel>/measured/effective_radiance` —
    through pyramids' group-qualified `NetCDF.get_variable`, which navigates the
    `/`-separated group path. Pass it to `read_fci` as `open_chunk` for real
    granules:

    ```python
    from pyramids_eo.sensors.readers import open_fci_l1c_chunk, read_fci

    scene = read_fci(chunk_paths, "ir_105", open_chunk=open_fci_l1c_chunk)
    ```

    Warning:
        The nested group path matches the documented FDHSI layout, but has not
        been validated byte-for-byte against a real EUMETSAT granule, and this
        opener does not (yet) extract the per-granule calibration coefficients
        from the chunk's group attributes — supply them via
        `read_fci(..., coeffs=...)` or rely on the registry fallback. See issue
        #40.

    Args:
        path: Path to a single FCI L1C FDHSI chunk NetCDF file.
        channel: Channel identifier (e.g. `"ir_105"`), substituted into
            `radiance_group`.
        radiance_group: Group-qualified variable-name template with a single
            `{channel}` placeholder. Defaults to
            `data/{channel}/measured/effective_radiance`. `read_fci` calls the
            opener positionally, so to thread a non-default template through
            `read_fci` wrap it with
            `functools.partial(open_fci_l1c_chunk, radiance_group=...)`.

    Returns:
        A pyramids `NetCDF` variable view of the channel's raw effective radiance
        (a `Dataset`-like object exposing `read_array` / `geotransform` / `epsg` /
        `columns` / `rows`, as `read_fci` consumes it).
    """
    from pyramids.netcdf import NetCDF

    try:
        variable = radiance_group.format(channel=channel)
    except (KeyError, IndexError, ValueError, AttributeError, TypeError) as exc:
        raise ReaderError(
            f"invalid radiance_group template {radiance_group!r}: it must be a "
            f"format string with a single '{{channel}}' field ({exc!r})"
        ) from exc
    # Group navigation via get_variable needs the file opened as multidimensional;
    # pass it explicitly rather than relying on read_file's default.
    return NetCDF.read_file(str(path), open_as_multi_dimensional=True).get_variable(
        variable
    )


def read_fci(
    chunks: Any,
    channel: str | None = None,
    *,
    channels: Sequence[str] | None = None,
    sensor: str = "fci",
    calibrate: bool = True,
    sun_earth_distance: float = 1.0,
    cos_sza: Any = None,
    coeffs: dict[str, Any] | None = None,
    open_chunk: Any = None,
) -> Any:
    """Read one or several FCI channels across a chunk set into `Dataset`s.

    For each requested channel, orders the chunks north -> south by their top-left
    latitude (so the caller may pass them in any order), stitches the radiance
    row-wise, calibrates it to reflectance (solar) or brightness temperature
    (thermal), and returns a geolocated pyramids `Dataset` carrying the
    northernmost chunk's CRS + geotransform. The chunks must share a CRS, cell
    size and column count and be vertically contiguous (validated per channel,
    since channels differ in resolution).

    Pass exactly one of `channel` (returns a `Dataset`) or `channels` (returns a
    `dict[str, Dataset]`); `read_fci(chunks, "ir_105")` is unchanged.

    Args:
        chunks: An ordered iterable of chunks, each either a pyramids `Dataset`
            already holding the channel radiance, or a value accepted by
            `open_chunk` (by default a NetCDF path).
        channel: A single channel identifier (e.g. `"ir_105"`, `"vis_06"`) for a
            `Dataset` result; mutually exclusive with `channels`.
        channels: A sequence of channel identifiers for a `dict[str, Dataset]`
            result; mutually exclusive with `channel`. (Each channel is opened via
            `open_chunk`, whose contract is per-channel; the grid is validated and
            ordered per channel because channels differ in resolution.)
        sensor: Registry sensor name (default `"fci"`).
        calibrate: When `True` (default), calibrate to a physical quantity; when
            `False`, return the stitched raw radiance.
        sun_earth_distance: Sun-earth distance (AU) for solar-channel reflectance.
        cos_sza: Cosine of the solar zenith angle for the reflectance sun-angle
            correction, or `None`.
        coeffs: Per-granule calibration coefficients preferred over the registry
            fallback (see `calibrate_channel`), or `None` to use the registry. They
            override a **single** channel's calibration, so they are only accepted
            with `channel=` — passing `coeffs` together with `channels=[...]` is an
            error (each channel would need its own).
        open_chunk: Callable `(chunk, channel) -> Dataset` used for chunks that
            are not already Datasets. Defaults to a NetCDF reader (see the module
            warning about the FCI layout).

    Returns:
        A pyramids `Dataset` (for `channel`) or a `dict[str, Dataset]` keyed by
        channel (for `channels`) of the calibrated (or raw) channel(s) on the
        stitched grid.

    Raises:
        ReaderError: When neither / both of `channel` / `channels` are given, or
            when `chunks` is empty.
        CalibrationError: When a channel lacks the constants its kind needs.
        UnknownSensorError: When the sensor / channel is not in the registry.
    """
    requested, single = resolve_channels(channel, channels, "read_fci")
    chunk_list = list(chunks)
    if not chunk_list:
        raise ReaderError("read_fci: no chunks given")
    # A pre-opened Dataset holds exactly one channel's radiance, so it cannot serve
    # a multi-channel request — every channel would alias the same array. Reject it
    # rather than return a physically meaningless dict.
    if not single and any(hasattr(chunk, "read_array") for chunk in chunk_list):
        raise ReaderError(
            "read_fci: channels=[...] needs path-like chunks read per channel via "
            "open_chunk; a pre-opened Dataset already holds a single channel"
        )
    # `coeffs` overrides one channel's calibration; sharing it across a channel set
    # would mis-calibrate the others. Require the single-channel form for an override.
    if not single and coeffs is not None:
        raise ReaderError(
            "read_fci: `coeffs` overrides a single channel's calibration; pass it "
            "with `channel=`, not `channels=[...]` (each channel calibrates on its own)"
        )
    opener = open_chunk or _default_open_chunk

    from pyramids.dataset import Dataset, GeoReference

    results = {}
    for name in requested:
        datasets = [
            chunk if hasattr(chunk, "read_array") else opener(chunk, name)
            for chunk in chunk_list
        ]
        _validate_chunk_grid(datasets)
        # Order the chunks north -> south by their top-left latitude/y so the
        # stitch and geolocation are correct regardless of the order the chunks
        # were passed in (FCI files are commonly numbered south -> north).
        ordered = sorted(datasets, key=lambda ds: ds.geotransform[3], reverse=True)
        radiance = np.concatenate(
            [np.asarray(ds.read_array(), dtype=float) for ds in ordered], axis=0
        )
        data = (
            calibrate_channel(
                radiance, name, sensor, sun_earth_distance, cos_sza, coeffs=coeffs
            )
            if calibrate
            else radiance
        )
        north = ordered[0]
        # Calibration can produce NaN (terminator reflectance / non-positive
        # radiance), so declare NaN as nodata rather than the default -9999.
        results[name] = Dataset.from_array(
            data,
            geo_ref=GeoReference(geo=north.geotransform, epsg=north.epsg),
            no_data_value=np.nan,
        )
    return results[requested[0]] if single else results


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
        if not np.isclose(ds.geotransform[1], first.geotransform[1]) or (
            not np.isclose(ds.geotransform[5], first.geotransform[5])
        ):
            raise ReaderError("read_fci: chunks have mixed cell size")
        if ds.columns != first.columns:
            raise ReaderError("read_fci: chunks have mixed column count")

    ordered = sorted(datasets, key=lambda ds: ds.geotransform[3], reverse=True)
    # Tolerance is a small fraction of the pixel height, so it scales with the
    # grid rather than depending on the coordinate magnitude (geostationary
    # metres are ~5.4e6, where np.isclose's absolute default would be far too
    # tight).
    atol = abs(first.geotransform[5]) * 1e-3
    for upper, lower in zip(ordered, ordered[1:]):
        upper_bottom = upper.geotransform[3] + upper.rows * upper.geotransform[5]
        if not np.isclose(upper_bottom, lower.geotransform[3], rtol=0.0, atol=atol):
            raise ReaderError("read_fci: chunks are not vertically contiguous")
