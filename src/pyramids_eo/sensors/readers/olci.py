"""Sentinel-3 OLCI Level-1B (EFR / ERR) reader -- native, satpy-free.

`read_olci(granule, channel="Oa08")` decodes one OLCI radiance band (or several
via `channels=[...]`) out of a `.SEN3` granule into a pyramids `Dataset`, through
pyramids' bundled GDAL NetCDF driver. No `netCDF4`, and nothing from the satpy
stack.

OLCI is a **pushbroom swath** instrument, so the result carries its geolocation
as a GDAL `GEOLOCATION` domain pointing at the granule's per-pixel
`geo_coordinates.nc` longitude / latitude arrays -- not an affine geotransform,
which cannot describe a swath. The reader resamples nothing; a caller who wants a
map grid warps the result themselves (`gdal.Warp(..., geoloc=True)`).

Per issue #136 this decodes to arrays plus standard CF metadata only:
radiometric calibration choices (radiance -> reflectance) stay with the callers
and the `composites` tier, and the granule's own `instrument_data.nc` carries the
`solar_flux` needed for that conversion.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._common import resolve_channels
from pyramids_eo.sensors.readers._sentinel3 import (
    geolocation_domain,
    granule_file,
    read_packed_variable,
    resolve_granule,
    wrap_swath,
)

#: Sensor key for the bundled channel table (`registry/data/olci.yaml`).
SENSOR = "olci"

#: OLCI carries 21 solar bands, `Oa01`..`Oa21`.
_BAND_COUNT = 21

#: Geolocation annotation file: a longitude / latitude for every pixel.
_GEO_FILE = "geo_coordinates.nc"

#: Accepts `Oa08`, `oa8`, or `Oa08_radiance` and canonicalises to `Oa08`.
_CHANNEL_PATTERN = re.compile(r"^oa(\d{1,2})(?:_radiance)?$", re.IGNORECASE)


def _canonical_channel(channel: Any, reader: str) -> str:
    """Normalise an OLCI channel spelling to its canonical `OaNN` form.

    Args:
        channel: A channel name (`"Oa08"`, `"oa8"`, `"Oa08_radiance"`).
        reader: The calling reader's name, for error messages.

    Returns:
        The canonical zero-padded name, e.g. `"Oa08"`.

    Raises:
        ReaderError: When the name is not an OLCI band, or is out of the
            `Oa01`..`Oa21` range.

    Examples:
        - Several spellings canonicalise to the same band:
            ```python
            >>> from pyramids_eo.sensors.readers.olci import _canonical_channel
            >>> _canonical_channel("oa8", "read_olci")
            'Oa08'
            >>> _canonical_channel("Oa08_radiance", "read_olci")
            'Oa08'
            >>> _canonical_channel("Oa21", "read_olci")
            'Oa21'

            ```
        - An out-of-range band is rejected rather than silently read:
            ```python
            >>> _canonical_channel("Oa22", "read_olci")
            Traceback (most recent call last):
                ...
            pyramids_eo.errors.ReaderError: read_olci: 'Oa22' is not an OLCI band; expected Oa01..Oa21

            ```
    """
    match = _CHANNEL_PATTERN.match(str(channel).strip())
    number = int(match.group(1)) if match else 0
    if not match or not 1 <= number <= _BAND_COUNT:
        raise ReaderError(
            f"{reader}: {str(channel)!r} is not an OLCI band; expected "
            f"Oa01..Oa{_BAND_COUNT:02d}"
        )
    return f"Oa{number:02d}"


def _band_filename(channel: str) -> str:
    """Return the per-band NetCDF filename for a canonical channel."""
    return f"{channel}_radiance.nc"


def available_olci_channels(source: Any) -> list[str]:
    """List the OLCI radiance bands a granule actually carries.

    Args:
        source: The `.SEN3` granule directory, or a file inside it.

    Returns:
        The canonical channel names present, sorted (`["Oa01", "Oa02", ...]`).

    Raises:
        ReaderError: When `source` is not a granule directory or a file in one.
    """
    granule = resolve_granule(source, "available_olci_channels")
    found = []
    for number in range(1, _BAND_COUNT + 1):
        channel = f"Oa{number:02d}"
        if (granule / _band_filename(channel)).is_file():
            found.append(channel)
    return found


def _channel_metadata(
    channel: str, granule: Path, variable_meta: dict[str, str]
) -> dict[str, str]:
    """Build the CF-style metadata carried onto a decoded band.

    Keeps the variable's own CF attributes (`units`, `long_name`,
    `standard_name`) when the granule declares them, and adds the channel's
    registry `wavelength_um` / `resolution_m` / `kind` plus provenance (sensor,
    channel, granule name and a note that the geolocation is a swath domain).
    Registry lookup failure is tolerated and simply omits those three fields:
    the decode must not depend on the bundled table.

    Args:
        channel: Canonical channel name.
        granule: The granule directory (for provenance).
        variable_meta: The NetCDF variable's GDAL attribute dict.

    Returns:
        A flat `str -> str` metadata dict for the output `Dataset`.
    """
    metadata = {
        "sensor": SENSOR,
        "channel": channel,
        "granule": granule.name,
        "geolocation": "swath (see the GEOLOCATION metadata domain)",
    }
    for key in ("units", "long_name", "standard_name"):
        if key in variable_meta:
            metadata[key] = str(variable_meta[key])
    try:
        from pyramids_eo.sensors.registry import get_sensor

        entry = get_sensor(SENSOR).get_channel(channel)
    except Exception:  # noqa: BLE001 - registry is descriptive, never required
        return metadata
    metadata["wavelength_um"] = str(entry.wavelength_um)
    metadata["resolution_m"] = str(entry.resolution_m)
    metadata["kind"] = entry.kind
    return metadata


def read_olci_band(source: Any, channel: Any, *, geolocation: bool = True) -> Any:
    """Decode a single OLCI radiance band from a `.SEN3` granule.

    Args:
        source: The `.SEN3` granule directory, or a file inside it.
        channel: The band to read (`"Oa08"`, `"oa8"`, `"Oa08_radiance"`).
        geolocation: When `True` (default), attach the granule's per-pixel
            `GEOLOCATION` domain. `False` returns the array on its placeholder
            pixel grid only -- useful when the geolocation file is absent or the
            caller only wants the counts.

    Returns:
        A pyramids `Dataset` of the unpacked radiance, NaN where the granule
        flags a fill / out-of-range pixel, with CF metadata and (by default) the
        swath `GEOLOCATION` domain.

    Raises:
        ReaderError: When the channel name is not an OLCI band, the granule has
            no such band file (or no geolocation file when `geolocation=True`),
            or the variable cannot be decoded.
    """
    reader = "read_olci"
    name = _canonical_channel(channel, reader)
    granule = resolve_granule(source, reader)
    band_path = granule_file(granule, _band_filename(name), reader)

    variable = f"{name}_radiance"
    data, variable_meta = read_packed_variable(band_path, variable, reader=reader)

    domain: dict[str, str] = {}
    if geolocation:
        geo_path = granule_file(granule, _GEO_FILE, reader)
        domain = geolocation_domain(geo_path, reader=reader)
    dataset = wrap_swath(data, domain)
    dataset.band_names = [name]
    dataset.raster.SetMetadata(_channel_metadata(name, granule, variable_meta))
    return dataset


def read_olci(
    source: Any,
    channel: str | None = None,
    *,
    channels: Sequence[str] | None = None,
    geolocation: bool = True,
) -> Any:
    """Read one or several OLCI L1B radiance bands from a `.SEN3` granule.

    Decodes each requested band's packed `OaNN_radiance` variable (`raw *
    scale_factor + add_offset`, with the `_FillValue` and anything outside
    `valid_range` masked to NaN) and returns it as a swath `Dataset` carrying the
    granule's per-pixel geolocation as a GDAL `GEOLOCATION` domain.

    Pass exactly one of `channel` (returns a `Dataset`) or `channels` (returns a
    `dict[str, Dataset]` keyed by canonical channel name).

    This reader **decodes only** -- it does not convert radiance to reflectance
    (issue #136 scopes radiometric calibration choices out). The granule's
    `instrument_data.nc` carries the per-detector `solar_flux` a caller needs for
    that conversion.

    Args:
        source: The `.SEN3` granule directory, or any file inside it.
        channel: A single band (`"Oa08"`); mutually exclusive with `channels`.
        channels: A sequence of bands for a `dict` result; mutually exclusive
            with `channel`.
        geolocation: When `True` (default), attach each result's per-pixel
            `GEOLOCATION` domain from `geo_coordinates.nc`.

    Returns:
        A pyramids `Dataset` (for `channel`) or a `dict[str, Dataset]` (for
        `channels`), each on the granule's native swath grid with NaN nodata.

    Raises:
        ReaderError: When neither / both of `channel` / `channels` are given,
            `channels` is empty or has duplicates, a band name is not an OLCI
            band, or the granule lacks a requested band / its geolocation file.
    """
    requested, single = resolve_channels(channel, channels, "read_olci")
    # Canonicalise before the duplicate check: `resolve_channels` compares the raw
    # strings, so aliases of one band ("Oa08" and "oa8") slip through it and would
    # otherwise collapse into a single dict entry, silently dropping a requested band.
    canonical = [_canonical_channel(name, "read_olci") for name in requested]
    if len(set(canonical)) != len(canonical):
        duplicates = sorted({c for c in canonical if canonical.count(c) > 1})
        raise ReaderError(
            f"read_olci: `channels` names the same band more than once (after "
            f"normalising spellings): {duplicates}"
        )
    results = {
        name: read_olci_band(source, name, geolocation=geolocation)
        for name in canonical
    }
    return results[canonical[0]] if single else results
