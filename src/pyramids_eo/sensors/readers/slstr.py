"""Sentinel-3 SLSTR Level-1 RBT reader -- native, satpy-free.

`read_slstr(granule, channel="S5")` decodes one SLSTR channel (or several via
`channels=[...]`) out of a `.SEN3` granule into a pyramids `Dataset`, through
pyramids' bundled GDAL NetCDF driver. No `netCDF4`, and nothing from the satpy
stack.

SLSTR is **dual-view and multi-grid**, which drives this reader's two extra
arguments:

* `view` -- the instrument images each scene twice, `"nadir"` (straight down)
  and `"oblique"` (slanted, ~55 deg forward). Each view is a separate grid.
* `grid` -- channels live on different resolutions: the solar channels `S1`-`S6`
  on the 500 m `a` grid, the thermal `S7`-`S9` on the 1 km `i` grid, and the fire
  channels `F1`-`F2` on the 1 km `f` grid. `read_slstr` picks the channel's usual
  grid and falls back to discovering the file when a product spells it
  differently; `grid=` forces a specific one (e.g. the alternative 500 m `b`).

Because the grids differ in shape, a multi-channel read returns a `dict` rather
than a stack -- `channels=["S5", "S8"]` yields arrays of different sizes.

Like OLCI, SLSTR is a swath instrument, so a result carries its geolocation as a
GDAL `GEOLOCATION` domain over the matching `geodetic_*` annotation file rather
than an affine geotransform, and this reader resamples nothing.

Per issue #136 this decodes to arrays plus standard CF metadata only. Note the
RBT product is already geophysical: `S1`-`S6` are radiances and `S7`-`S9` /
`F1`-`F2` are **brightness temperatures in kelvin** as shipped, so no Planck
conversion is applied (or needed) here.
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

#: Sensor key for the bundled channel table (`registry/data/slstr.yaml`).
SENSOR = "slstr"

#: The usual grid and measured quantity per channel. `a` = 500 m solar,
#: `i` = 1 km thermal, `f` = 1 km fire. A product that spells a channel's grid
#: differently is still found by `_discover_band_file`.
_CHANNEL_GRID: dict[str, tuple[str, str]] = {
    "S1": ("a", "radiance"),
    "S2": ("a", "radiance"),
    "S3": ("a", "radiance"),
    "S4": ("a", "radiance"),
    "S5": ("a", "radiance"),
    "S6": ("a", "radiance"),
    "S7": ("i", "BT"),
    "S8": ("i", "BT"),
    "S9": ("i", "BT"),
    "F1": ("f", "BT"),
    "F2": ("f", "BT"),
}

#: `view=` spellings mapped to the one-letter product suffix.
_VIEWS: dict[str, str] = {
    "nadir": "n",
    "n": "n",
    "oblique": "o",
    "o": "o",
}

_CHANNEL_PATTERN = re.compile(r"^([sf])(\d)$", re.IGNORECASE)


def _canonical_channel(channel: Any, reader: str) -> str:
    """Normalise an SLSTR channel spelling to its canonical form.

    Args:
        channel: A channel name (`"S5"`, `"s5"`, `"f1"`).
        reader: The calling reader's name, for error messages.

    Returns:
        The canonical name, e.g. `"S5"` or `"F1"`.

    Raises:
        ReaderError: When the name is not an SLSTR channel.

    Examples:
        - Case is normalised:
            ```python
            >>> from pyramids_eo.sensors.readers.slstr import _canonical_channel
            >>> _canonical_channel("s5", "read_slstr")
            'S5'
            >>> _canonical_channel("f2", "read_slstr")
            'F2'

            ```
        - A channel the instrument does not have is rejected:
            ```python
            >>> _canonical_channel("S0", "read_slstr")
            Traceback (most recent call last):
                ...
            pyramids_eo.errors.ReaderError: read_slstr: 'S0' is not an SLSTR channel; expected one of F1, F2, S1, S2, S3, S4, S5, S6, S7, S8, S9

            ```
    """
    match = _CHANNEL_PATTERN.match(str(channel).strip())
    name = f"{match.group(1).upper()}{match.group(2)}" if match else ""
    if name not in _CHANNEL_GRID:
        raise ReaderError(
            f"{reader}: {str(channel)!r} is not an SLSTR channel; expected one of "
            f"{', '.join(sorted(_CHANNEL_GRID))}"
        )
    return name


def _resolve_view(view: Any, reader: str) -> str:
    """Map a `view` argument to its one-letter product suffix.

    Args:
        view: `"nadir"` / `"n"` or `"oblique"` / `"o"` (case-insensitive).
        reader: The calling reader's name, for error messages.

    Returns:
        `"n"` for the nadir view, `"o"` for the oblique one.

    Raises:
        ReaderError: When `view` is not one of the accepted spellings.

    Examples:
        - Both the long and short spellings work:
            ```python
            >>> from pyramids_eo.sensors.readers.slstr import _resolve_view
            >>> _resolve_view("nadir", "read_slstr"), _resolve_view("o", "read_slstr")
            ('n', 'o')

            ```
    """
    key = str(view).strip().lower()
    if key not in _VIEWS:
        raise ReaderError(
            f"{reader}: view={view!r} is not an SLSTR view; expected 'nadir' or 'oblique'"
        )
    return _VIEWS[key]


def _discover_band_file(
    granule: Path, channel: str, view: str, reader: str, grid: str | None
) -> Path:
    """Locate a channel's measurement file, by convention then by discovery.

    Tries the channel's usual `<channel>_<quantity>_<grid><view>.nc` name (or the
    caller's `grid`), then falls back to globbing `<channel>_*<view>.nc` so a
    product that places a channel on a different grid is still read rather than
    rejected.

    Args:
        granule: The `.SEN3` granule directory.
        channel: Canonical channel name.
        view: One-letter view suffix (`"n"` / `"o"`).
        reader: The calling reader's name, for error messages.
        grid: An explicit grid letter overriding the channel's usual one, or
            `None`.

    Returns:
        The measurement file's `Path`.

    Raises:
        ReaderError: When the granule holds no file for the channel and view; when
            `grid` was given but matches none of the files that do exist; or when
            several grids carry the channel and no `grid` was given to choose one.
    """
    default_grid, quantity = _CHANNEL_GRID[channel]
    candidate = granule / f"{channel}_{quantity}_{grid or default_grid}{view}.nc"
    if candidate.is_file():
        return candidate
    matches = sorted(
        path for path in granule.glob(f"{channel}_*{view}.nc") if path.is_file()
    )
    if len(matches) == 1:
        return matches[0]
    if matches:
        if grid is not None:
            raise ReaderError(
                f"{reader}: granule {granule.name!r} has no {candidate.name!r}; "
                f"grid={grid!r} does not match any of "
                f"{[path.name for path in matches]}"
            )
        raise ReaderError(
            f"{reader}: {channel!r} ({view!r} view) is ambiguous in granule "
            f"{granule.name!r} -- {[path.name for path in matches]} all match; "
            f"pass grid= to choose one"
        )
    raise ReaderError(
        f"{reader}: granule {granule.name!r} has no file for channel {channel!r} "
        f"in the {'nadir' if view == 'n' else 'oblique'} view (expected "
        f"{candidate.name!r})"
    )


def _geodetic_domain(granule: Path, band_path: Path, reader: str) -> dict[str, str]:
    """Build the `GEOLOCATION` domain for the grid a band file sits on.

    The grid+view suffix is taken from the band file's own name (its last
    underscore-separated token, e.g. `an` in `S1_radiance_an.nc`), so the
    geolocation always matches the grid actually read -- including when
    `_discover_band_file` fell back to a non-default grid.

    Args:
        granule: The `.SEN3` granule directory.
        band_path: The measurement file being read.
        reader: The calling reader's name, for error messages.

    Returns:
        The `GEOLOCATION` domain for the matching `geodetic_<grid><view>.nc`.

    Raises:
        ReaderError: When the granule has no geodetic file for that grid.
    """
    suffix = band_path.stem.rsplit("_", 1)[-1]
    geo_path = granule_file(granule, f"geodetic_{suffix}.nc", reader)
    return geolocation_domain(
        geo_path,
        longitude_variable=f"longitude_{suffix}",
        latitude_variable=f"latitude_{suffix}",
    )


def _channel_metadata(
    channel: str, granule: Path, band_path: Path, variable_meta: dict[str, str]
) -> dict[str, str]:
    """Build the CF-style metadata carried onto a decoded channel.

    Keeps the variable's own CF attributes (`units`, `long_name`,
    `standard_name`) when the granule declares them, and adds the channel's
    registry `wavelength_um` / `resolution_m` / `kind` plus provenance (sensor,
    channel, granule name, and the grid and view actually read). Registry lookup
    failure is tolerated and simply omits those three fields.

    Args:
        channel: Canonical channel name.
        granule: The granule directory (for provenance).
        band_path: The measurement file read (names the grid and view).
        variable_meta: The NetCDF variable's GDAL attribute dict.

    Returns:
        A flat `str -> str` metadata dict for the output `Dataset`.
    """
    suffix = band_path.stem.rsplit("_", 1)[-1]
    metadata = {
        "sensor": SENSOR,
        "channel": channel,
        "granule": granule.name,
        "grid": suffix,
        "view": "nadir" if suffix.endswith("n") else "oblique",
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


def read_slstr_band(
    source: Any,
    channel: Any,
    *,
    view: str = "nadir",
    grid: str | None = None,
    geolocation: bool = True,
) -> Any:
    """Decode a single SLSTR channel from a `.SEN3` granule.

    Args:
        source: The `.SEN3` granule directory, or a file inside it.
        channel: The channel to read (`"S5"`, `"s5"`, `"F1"`).
        view: `"nadir"` (default) or `"oblique"`.
        grid: An explicit grid letter (`"a"`, `"b"`, `"i"`, `"f"`) overriding the
            channel's usual grid.
        geolocation: When `True` (default), attach the `GEOLOCATION` domain from
            the `geodetic_*` file matching the grid read.

    Returns:
        A pyramids `Dataset` of the channel -- radiance for `S1`-`S6`, brightness
        temperature in kelvin for `S7`-`S9` / `F1`-`F2` -- NaN where the granule
        flags a fill / out-of-range pixel, with CF metadata and (by default) the
        swath `GEOLOCATION` domain.

    Raises:
        ReaderError: When the channel or view is not an SLSTR one, the granule
            has no matching band file (or its geodetic file), the channel is
            ambiguous across grids, or the variable cannot be decoded.
    """
    reader = "read_slstr"
    name = _canonical_channel(channel, reader)
    suffix = _resolve_view(view, reader)
    granule = resolve_granule(source, reader)
    band_path = _discover_band_file(granule, name, suffix, reader, grid)

    data, variable_meta = read_packed_variable(band_path, band_path.stem, reader=reader)
    domain = _geodetic_domain(granule, band_path, reader) if geolocation else {}
    dataset = wrap_swath(data, domain)
    dataset.band_names = [name]
    dataset.raster.SetMetadata(
        _channel_metadata(name, granule, band_path, variable_meta)
    )
    return dataset


def read_slstr(
    source: Any,
    channel: str | None = None,
    *,
    channels: Sequence[str] | None = None,
    view: str = "nadir",
    grid: str | None = None,
    geolocation: bool = True,
) -> Any:
    """Read one or several SLSTR L1 RBT channels from a `.SEN3` granule.

    Decodes each requested channel's packed variable (`raw * scale_factor +
    add_offset`, with the `_FillValue` and anything outside `valid_range` masked
    to NaN) and returns it as a swath `Dataset` carrying the matching grid's
    per-pixel geolocation as a GDAL `GEOLOCATION` domain.

    Pass exactly one of `channel` (returns a `Dataset`) or `channels` (returns a
    `dict[str, Dataset]`). Channels on different grids have **different array
    shapes**, which is why a channel set returns a dict rather than a stack.

    This reader **decodes only** -- issue #136 scopes radiometric calibration
    choices out. The RBT product is already geophysical, so `S7`-`S9` / `F1`-`F2`
    arrive as brightness temperature in kelvin with no conversion applied.

    Args:
        source: The `.SEN3` granule directory, or any file inside it.
        channel: A single channel (`"S5"`); mutually exclusive with `channels`.
        channels: A sequence of channels for a `dict` result; mutually exclusive
            with `channel`.
        view: `"nadir"` (default) or `"oblique"` -- SLSTR images each scene twice.
        grid: An explicit grid letter overriding each channel's usual grid.
        geolocation: When `True` (default), attach each result's `GEOLOCATION`
            domain.

    Returns:
        A pyramids `Dataset` (for `channel`) or a `dict[str, Dataset]` (for
        `channels`), each on its native swath grid with NaN nodata.

    Raises:
        ReaderError: When neither / both of `channel` / `channels` are given,
            `channels` is empty or names a channel twice, a channel or the view
            is not an SLSTR one, or the granule lacks a requested channel.
    """
    requested, single = resolve_channels(channel, channels, "read_slstr")
    # Canonicalise before the duplicate check: `resolve_channels` compares the raw
    # strings, so aliases of one channel ("S5" and "s5") slip through it and would
    # otherwise collapse into a single dict entry, silently dropping a request.
    canonical = [_canonical_channel(name, "read_slstr") for name in requested]
    if len(set(canonical)) != len(canonical):
        duplicates = sorted({c for c in canonical if canonical.count(c) > 1})
        raise ReaderError(
            f"read_slstr: `channels` names the same channel more than once (after "
            f"normalising spellings): {duplicates}"
        )
    results = {
        name: read_slstr_band(
            source, name, view=view, grid=grid, geolocation=geolocation
        )
        for name in canonical
    }
    return results[canonical[0]] if single else results
