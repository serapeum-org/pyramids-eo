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
  on the 500 m `a` grid (with `S4`-`S6` duplicated onto the alternative 500 m
  `b`), the thermal `S7`-`S9` on the 1 km `i` grid, and of the fire channels
  **`F1` on the 1 km `f` grid but `F2` on `i`** -- they are asymmetric, and there
  is no `F2_BT_f*.nc`. `read_slstr` tries each channel's documented grids in
  order (`F1` falls back to `i` for products predating the `f` baseline), and
  `grid=` restricts the read to one grid. `available_slstr_channels` reports what
  a particular granule actually carries.

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

#: The grid(s) and measured quantity each channel ships on, per the SLSTR L1 RBT
#: product-composition table, in preference order. `a`/`b` are the 500 m solar
#: grids, `i` the 1 km thermal grid and `f` the 1 km fire grid.
#:
#: Two asymmetries are easy to get wrong and are the reason this is an explicit
#: table rather than a rule: the fire channels differ from each other -- F1 has
#: its own `f` grid while **F2 stays on `i`** (there is no `F2_BT_fn.nc`) -- and
#: F1 is additionally found on `i` in products predating the `f`-grid baseline.
#: S4-S6 are duplicated onto the alternative 500 m `b` grid.
_CHANNEL_GRIDS: dict[str, tuple[tuple[str, ...], str]] = {
    "S1": (("a",), "radiance"),
    "S2": (("a",), "radiance"),
    "S3": (("a",), "radiance"),
    "S4": (("a", "b"), "radiance"),
    "S5": (("a", "b"), "radiance"),
    "S6": (("a", "b"), "radiance"),
    "S7": (("i",), "BT"),
    "S8": (("i",), "BT"),
    "S9": (("i",), "BT"),
    "F1": (("f", "i"), "BT"),
    "F2": (("i",), "BT"),
}

#: The grid letters `grid=` accepts, and their ground sampling distance in metres.
_GRID_RESOLUTION: dict[str, int] = {"a": 500, "b": 500, "i": 1000, "f": 1000}

#: `view=` spellings mapped to the one-letter product suffix.
_VIEWS: dict[str, str] = {
    "nadir": "n",
    "n": "n",
    "oblique": "o",
    "o": "o",
}

#: Accepts the bare channel (`S5`, `f1`) and, symmetrically with the OLCI
#: reader's `Oa08_radiance`, the product's own variable spelling with its
#: quantity and grid+view suffix (`S5_radiance_an`, `S8_BT_in`, `F1_BT_fo`).
#: The suffix is **honoured**, not merely tolerated: a spelled grid and view
#: select the file named, and a spelled quantity is checked against the channel.
_CHANNEL_PATTERN = re.compile(
    r"^([sf])(\d)(?:_(radiance|bt)_([abif])([no]))?$", re.IGNORECASE
)


def _canonical_channel(channel: Any, reader: str) -> str:
    """Normalise an SLSTR channel spelling to its canonical form.

    This keeps only the channel name: a spelled grid and view are validated but
    then **discarded**, so this is for the places that key by channel (the
    duplicate check and the result dict), not for choosing a file. Use
    `_parse_channel` where the spelled grid and view have to be honoured.

    Args:
        channel: A channel name (`"S5"`, `"s5"`, `"f1"`), or the product's own
            variable spelling (`"S5_radiance_an"`, `"S8_BT_in"`).
        reader: The calling reader's name, for error messages.

    Returns:
        The canonical name, e.g. `"S5"` or `"F1"`.

    Raises:
        ReaderError: When the name is not an SLSTR channel, or a spelled quantity
            is not the one that channel ships (both come from `_parse_channel`).

    Examples:
        - Case is normalised:
            ```python
            >>> from pyramids_eo.sensors.readers.slstr import _canonical_channel
            >>> _canonical_channel("s5", "read_slstr")
            'S5'
            >>> _canonical_channel("f2", "read_slstr")
            'F2'

            ```
        - The product's variable spelling is accepted too, as OLCI accepts
          `Oa08_radiance` -- but the grid and view it carries are dropped here,
          which is why the readers resolve the file from `_parse_channel` instead:
            ```python
            >>> _canonical_channel("S5_radiance_an", "read_slstr")
            'S5'
            >>> _canonical_channel("S5_radiance_bo", "read_slstr")
            'S5'

            ```
        - A channel the instrument does not have is rejected:
            ```python
            >>> _canonical_channel("S0", "read_slstr")
            Traceback (most recent call last):
                ...
            pyramids_eo.errors.ReaderError: read_slstr: 'S0' is not an SLSTR channel; expected one of F1, F2, S1, S2, S3, S4, S5, S6, S7, S8, S9

            ```

    See Also:
        _parse_channel: Returns the spelled grid and view instead of dropping them.
    """
    return _parse_channel(channel, reader)[0]


def _parse_channel(channel: Any, reader: str) -> tuple[str, str | None, str | None]:
    """Split a channel spelling into its name and any grid / view it encodes.

    Accepts the bare channel (`"S5"`) and the product's own variable spelling
    (`"S5_radiance_an"`). A spelled suffix is **honoured** -- it names a specific
    file, so its grid and view are returned for the caller to use, rather than
    being parsed and thrown away, which would read a different file than the one
    named. A spelled quantity is checked against the channel's radiometric kind,
    so a combination no product has (`"S9_radiance_an"` -- S9 is thermal) is
    rejected instead of silently resolving to the BT file.

    Args:
        channel: A channel name or product variable spelling.
        reader: The calling reader's name, for error messages.

    Returns:
        A `(name, grid, view)` triple. `grid` and `view` are `None` for a bare
        channel, and the one-letter forms when the spelling carried them.

    Raises:
        ReaderError: When the name is not an SLSTR channel, or the spelled
            quantity is not the one that channel ships.

    Examples:
        - A bare channel carries no grid or view:
            ```python
            >>> from pyramids_eo.sensors.readers.slstr import _parse_channel
            >>> _parse_channel("s5", "read_slstr")
            ('S5', None, None)

            ```
        - The product spelling carries both, and they are returned to be used:
            ```python
            >>> _parse_channel("S5_radiance_bn", "read_slstr")
            ('S5', 'b', 'n')
            >>> _parse_channel("S8_BT_io", "read_slstr")
            ('S8', 'i', 'o')

            ```
        - A quantity the channel does not ship is rejected, not quietly swapped:
            ```python
            >>> _parse_channel("S9_radiance_an", "read_slstr")
            Traceback (most recent call last):
                ...
            pyramids_eo.errors.ReaderError: read_slstr: 'S9_radiance_an' names quantity 'radiance', but S9 ships 'BT'

            ```

    See Also:
        _canonical_channel: The same validation, keeping only the channel name.
        _resolve_channel_target: Reconciles a spelled suffix with `grid=` / `view=`.
    """
    match = _CHANNEL_PATTERN.match(str(channel).strip())
    name = f"{match.group(1).upper()}{match.group(2)}" if match else ""
    if match is None or name not in _CHANNEL_GRIDS:
        raise ReaderError(
            f"{reader}: {str(channel)!r} is not an SLSTR channel; expected one of "
            f"{', '.join(sorted(_CHANNEL_GRIDS))}"
        )
    quantity = match.group(3)
    if quantity is None:
        return name, None, None
    expected = _CHANNEL_GRIDS[name][1]
    if quantity.lower() != expected.lower():
        raise ReaderError(
            f"{reader}: {str(channel)!r} names quantity {quantity.lower()!r}, but "
            f"{name} ships {expected!r}"
        )
    return name, match.group(4).lower(), match.group(5).lower()


def _resolve_grid(grid: Any, reader: str) -> str | None:
    """Validate an explicit `grid` argument, or pass `None` through.

    Args:
        grid: A grid letter (`"a"`, `"b"`, `"i"`, `"f"`), or `None` to use the
            channel's documented grids.
        reader: The calling reader's name, for error messages.

    Returns:
        The lower-cased grid letter, or `None`.

    Raises:
        ReaderError: When `grid` is not an SLSTR grid letter. Validating it
            matters because the letter is interpolated into a filename, so an
            unchecked value silently built a nonsense path (`""` meant "the
            default grid", and `".."` or `"*"` built separator-bearing names).

    Examples:
        - A real grid letter is accepted, case-insensitively, and `None` passes through:
            ```python
            >>> from pyramids_eo.sensors.readers.slstr import _resolve_grid
            >>> _resolve_grid("i", "read_slstr"), _resolve_grid("A", "read_slstr")
            ('i', 'a')
            >>> _resolve_grid(None, "read_slstr") is None
            True

            ```
        - Anything else is rejected rather than interpolated into a path:
            ```python
            >>> _resolve_grid("", "read_slstr")
            Traceback (most recent call last):
                ...
            pyramids_eo.errors.ReaderError: read_slstr: grid='' is not an SLSTR grid; expected one of a, b, i, f

            ```
    """
    if grid is None:
        return None
    key = str(grid).strip().lower()
    if key not in _GRID_RESOLUTION:
        raise ReaderError(
            f"{reader}: grid={grid!r} is not an SLSTR grid; expected one of "
            f"{', '.join(_GRID_RESOLUTION)}"
        )
    return key


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
) -> tuple[Path, str]:
    """Locate a channel's measurement file from its documented grid candidates.

    Each candidate is an exact `<channel>_<quantity>_<grid><view>.nc` name built
    from `_CHANNEL_GRIDS`, tried in order; `grid` restricts the search to that one
    grid. Nothing is globbed, which matters twice: a `*_quality_*` annotation
    sibling can never be mistaken for the measurement, and an explicit `grid=` can
    never silently fall back to a different resolution.

    Args:
        granule: The `.SEN3` granule directory.
        channel: Canonical channel name.
        view: One-letter view suffix (`"n"` / `"o"`).
        reader: The calling reader's name, for error messages.
        grid: A validated grid letter restricting the search, or `None` to try
            the channel's documented grids in order.

    Returns:
        The measurement file's `Path`, and the grid letter it was found on.

    Raises:
        ReaderError: When no candidate exists. The message names the grids tried
            and the measurement files the granule does hold for that channel, so
            a channel on an unexpected grid is immediately apparent.
    """
    candidates, quantity = _CHANNEL_GRIDS[channel]
    wanted = (grid,) if grid is not None else candidates
    for letter in wanted:
        path = granule / f"{channel}_{quantity}_{letter}{view}.nc"
        if path.is_file():
            return path, letter
    # Only for the error message, and restricted to the measured quantity so a
    # quality sibling is never offered as a choice.
    present = sorted(
        path.name for path in granule.glob(f"{channel}_{quantity}_*{view}.nc")
    )
    tried = f"grid {grid!r}" if grid is not None else f"grid(s) {', '.join(wanted)}"
    holds = present if present else "no measurement files for this channel"
    raise ReaderError(
        f"{reader}: granule {granule.name!r} has no {channel!r} measurement for the "
        f"{'nadir' if view == 'n' else 'oblique'} view on {tried}; it holds {holds}"
    )


def _geodetic_domain(
    granule: Path, grid: str, view: str, reader: str, shape: tuple[int, int]
) -> dict[str, str]:
    """Build the `GEOLOCATION` domain for the grid a channel was read from.

    Args:
        granule: The `.SEN3` granule directory.
        grid: The grid letter the measurement was found on.
        view: One-letter view suffix (`"n"` / `"o"`).
        reader: The calling reader's name, for error messages.
        shape: The channel's `(rows, columns)`, which the geodetic grid must match
            -- each SLSTR grid has its own size, so a mismatch means the wrong
            annotation file for the stripe that was read.

    Returns:
        The `GEOLOCATION` domain for the matching `geodetic_<grid><view>.nc`.

    Raises:
        ReaderError: When the granule has no geodetic file for that grid, it lacks
            the longitude / latitude variables, or its grid does not match `shape`.
    """
    suffix = f"{grid}{view}"
    geo_path = granule_file(granule, f"geodetic_{suffix}.nc", reader)
    return geolocation_domain(
        geo_path,
        reader=reader,
        longitude_variable=f"longitude_{suffix}",
        latitude_variable=f"latitude_{suffix}",
        shape=shape,
    )


def _channel_metadata(
    channel: str,
    granule: Path,
    grid: str,
    view: str,
    variable_meta: dict[str, str],
    *,
    geolocated: bool,
) -> dict[str, str]:
    """Build the CF-style metadata carried onto a decoded channel.

    Keeps the variable's own CF attributes (`units`, `long_name`,
    `standard_name`) when the granule declares them, and adds the channel's
    registry `wavelength_um` / `kind` plus provenance (sensor, channel, granule
    name, and the grid and view actually read). Registry lookup failure is
    tolerated and simply omits those fields.

    `resolution_m` comes from the **grid actually read**, not from the channel's
    nominal registry entry, so a channel read off a non-default grid is labelled
    with that grid's ground sampling distance rather than its usual one.

    Args:
        channel: Canonical channel name.
        granule: The granule directory (for provenance).
        grid: The grid letter the measurement was read from.
        view: One-letter view suffix (`"n"` / `"o"`).
        variable_meta: The NetCDF variable's GDAL attribute dict.
        geolocated: Whether a `GEOLOCATION` domain was attached, so the
            `geolocation` key describes what the result actually carries.

    Returns:
        A flat `str -> str` metadata dict for the output `Dataset`.
    """
    metadata = {
        "sensor": SENSOR,
        "channel": channel,
        "granule": granule.name,
        "grid": f"{grid}{view}",
        "view": "nadir" if view == "n" else "oblique",
        "resolution_m": str(_GRID_RESOLUTION[grid]),
        "geolocation": (
            "swath (see the GEOLOCATION metadata domain)"
            if geolocated
            else "none (placeholder pixel grid)"
        ),
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
    metadata["kind"] = entry.kind
    return metadata


def available_slstr_channels(
    source: Any, *, view: str = "nadir"
) -> dict[str, list[str]]:
    """Report which channels a granule carries in one view, and on which grids.

    This is how a caller discovers a product's real layout rather than assuming
    it -- including that `F2` lives on the 1 km `i` grid while `F1` is on `f`, and
    which of `S4`-`S6` are duplicated onto the alternative 500 m `b` grid.

    A grid is listed only when the granule holds **both** the measurement file and
    the `geodetic_<grid><view>.nc` the measurement needs, so what this reports is
    what `read_slstr` can actually return -- a measurement whose annotation file is
    missing would raise there, and listing it would promise a read that fails. A
    granule stripped of its geodetic files therefore reports `{}` even though the
    measurement files are all present.

    Args:
        source: The `.SEN3` granule directory, or a file inside it.
        view: `"nadir"` (default) or `"oblique"`.

    Returns:
        A `{channel: [grid letters]}` mapping listing only the channels the granule
        holds a measurement **and** a matching geodetic file for, in the product's
        own channel order -- `S1`..`S9` then `F1`, `F2` -- which is not the
        alphabetical order the error messages use. Each channel's grid letters are
        likewise in the order `read_slstr` would try them.

    Raises:
        ReaderError: When `source` is not a granule, or `view` is not an SLSTR
            view.
    """
    reader = "available_slstr_channels"
    suffix = _resolve_view(view, reader)
    granule = resolve_granule(source, reader)
    found: dict[str, list[str]] = {}
    for channel, (grids, quantity) in _CHANNEL_GRIDS.items():
        present = [
            letter
            for letter in grids
            if (granule / f"{channel}_{quantity}_{letter}{suffix}.nc").is_file()
            and (granule / f"geodetic_{letter}{suffix}.nc").is_file()
        ]
        if present:
            found[channel] = present
    return found


def _resolve_channel_target(
    channel: Any, view: Any, grid: Any, reader: str
) -> tuple[str, str | None, str]:
    """Reconcile a channel spelling with explicit `view` / `grid` arguments.

    A spelled channel such as `"S5_radiance_bn"` names one specific file, so its
    grid and view are used. An explicit `view=` / `grid=` is used when the
    spelling carries none. When both are given and **disagree**, that is an error:
    silently preferring either one would read a different file than the caller
    named, which is the whole class of defect this reader guards against.

    Args:
        channel: A channel name or product variable spelling.
        view: The caller's `view`, or `None`.
        grid: The caller's `grid`, or `None`.
        reader: The calling reader's name, for error messages.

    Returns:
        A `(name, grid, view_suffix)` triple, where `grid` may be `None` to mean
        "try the channel's documented grids" and `view_suffix` is `"n"` or `"o"`.
        With neither a spelled nor an explicit view, `view_suffix` falls back to
        `"n"`, which is what makes nadir the effective default of `view=None`.

    Raises:
        ReaderError: When the channel, view or grid is invalid, or a spelled grid
            or view contradicts the explicit argument.

    Examples:
        - A bare channel leaves the grid open and falls back to the nadir view:
            ```python
            >>> from pyramids_eo.sensors.readers.slstr import _resolve_channel_target
            >>> _resolve_channel_target("S5", None, None, "read_slstr")
            ('S5', None, 'n')
            >>> _resolve_channel_target("S5", "oblique", "b", "read_slstr")
            ('S5', 'b', 'o')

            ```
        - A spelled suffix supplies both, and an agreeing explicit argument is
          accepted rather than treated as a conflict:
            ```python
            >>> _resolve_channel_target("S5_radiance_bo", None, None, "read_slstr")
            ('S5', 'b', 'o')
            >>> _resolve_channel_target("S5_radiance_bo", "o", "b", "read_slstr")
            ('S5', 'b', 'o')

            ```
        - A disagreeing pair is an error, not a silent preference either way:
            ```python
            >>> _resolve_channel_target("S5_radiance_bn", None, "a", "read_slstr")
            Traceback (most recent call last):
                ...
            pyramids_eo.errors.ReaderError: read_slstr: 'S5_radiance_bn' names grid 'b' but grid='a' was given; pass one or the other

            ```

    See Also:
        _parse_channel: Splits the spelling the reconciliation starts from.
    """
    name, spelled_grid, spelled_view = _parse_channel(channel, reader)
    wanted_grid = _resolve_grid(grid, reader)
    wanted_view = None if view is None else _resolve_view(view, reader)

    if (
        spelled_grid is not None
        and wanted_grid is not None
        and spelled_grid != wanted_grid
    ):
        raise ReaderError(
            f"{reader}: {str(channel)!r} names grid {spelled_grid!r} but grid="
            f"{grid!r} was given; pass one or the other"
        )
    if (
        spelled_view is not None
        and wanted_view is not None
        and spelled_view != wanted_view
    ):
        raise ReaderError(
            f"{reader}: {str(channel)!r} names the "
            f"{'nadir' if spelled_view == 'n' else 'oblique'} view but view={view!r} "
            f"was given; pass one or the other"
        )
    grid_letter = spelled_grid if spelled_grid is not None else wanted_grid
    view_suffix = spelled_view or wanted_view or "n"
    return name, grid_letter, view_suffix


def read_slstr_band(
    source: Any,
    channel: Any,
    *,
    view: str | None = None,
    grid: str | None = None,
    geolocation: bool = True,
) -> Any:
    """Decode a single SLSTR channel from a `.SEN3` granule.

    Args:
        source: The `.SEN3` granule directory, or a file inside it.
        channel: The channel to read (`"S5"`, `"s5"`, `"F1"`), or the product's
            own variable spelling (`"S5_radiance_bn"`), whose grid and view then
            select the file -- passing a conflicting `grid=` / `view=` is an error
            rather than one of the two being silently ignored.
        view: `"nadir"` or `"oblique"`. `None` (default) takes the view from a
            spelled channel, falling back to nadir.
        grid: An explicit grid letter (`"a"`, `"b"`, `"i"`, `"f"`) restricting the
            read to that one grid. `None` (default) tries the channel's documented
            grids in order. The restriction is honoured even when only one
            candidate file exists, so a read never silently returns a different
            resolution than the one asked for.
        geolocation: When `True` (default), attach the `GEOLOCATION` domain from
            the `geodetic_*` file matching the grid read.

    Returns:
        A pyramids `Dataset` of the channel -- radiance for `S1`-`S6`, brightness
        temperature in kelvin for `S7`-`S9` / `F1`-`F2` -- NaN where the granule
        flags a fill / out-of-range pixel, with CF metadata and (by default) the
        swath `GEOLOCATION` domain.

    Raises:
        ReaderError: When the channel, view or `grid` is not an SLSTR one, when a
            spelled channel names a quantity that channel does not ship
            (`"S9_radiance_an"`), when a spelled channel's grid / view contradicts
            an explicit `grid=` / `view=`, or when the granule has no measurement
            for that channel, view and grid. With `geolocation=True` it also raises
            when the matching geodetic file or its longitude / latitude variables
            are missing, when that annotation grid's shape does not match the band
            (an oversized one would otherwise place every pixel wrongly, silently),
            or when its `scale_factor` / `add_offset` is non-finite or the scale is
            zero.

    Note:
        A `_FillValue` that is not a finite number cannot mask anything, so the
        decode emits a `UserWarning` and the result's nodata is incomplete rather
        than the read failing (see `invalid_mask`).
    """
    reader = "read_slstr"
    name, letter, suffix = _resolve_channel_target(channel, view, grid, reader)
    granule = resolve_granule(source, reader)
    band_path, found_grid = _discover_band_file(granule, name, suffix, reader, letter)

    data, variable_meta = read_packed_variable(band_path, band_path.stem, reader=reader)
    domain = (
        _geodetic_domain(granule, found_grid, suffix, reader, data.shape)
        if geolocation
        else {}
    )
    dataset = wrap_swath(data, domain)
    dataset.band_names = [name]
    dataset.raster.SetMetadata(
        _channel_metadata(
            name,
            granule,
            found_grid,
            suffix,
            variable_meta,
            geolocated=bool(domain),
        )
    )
    return dataset


def read_slstr(
    source: Any,
    channel: str | None = None,
    *,
    channels: Sequence[str] | None = None,
    view: str | None = None,
    grid: str | None = None,
    geolocation: bool = True,
) -> Any:
    """Read one or several SLSTR L1 RBT channels from a `.SEN3` granule.

    Decodes each requested channel's packed variable (`raw * scale_factor +
    add_offset`, with the `_FillValue` and anything outside the variable's
    validity limits masked to NaN) and returns it as a swath `Dataset` carrying
    the matching grid's per-pixel geolocation as a GDAL `GEOLOCATION` domain. The
    limits come from the `valid_min` / `valid_max` scalars the SLSTR L1
    specification carries -- not from a combined `valid_range`, which appears
    nowhere in it -- and are compared against the raw, still-packed values.

    Pass exactly one of `channel` (returns a `Dataset`) or `channels` (returns a
    `dict[str, Dataset]`). Channels on different grids have **different array
    shapes**, which is why a channel set returns a dict rather than a stack.

    Either may also be given in the product's own variable spelling
    (`"S5_radiance_bn"`), and the grid and view it carries then **select the
    file**: they are honoured, not merely tolerated. So a channel set may mix views
    and grids per channel -- `channels=["S5_radiance_an", "S8_BT_io"]` reads one
    nadir and one oblique -- while `view=` / `grid=` supply whatever a bare
    spelling leaves open. A spelling that contradicts an explicit `view=` / `grid=`
    is an error rather than one of the two being silently ignored.

    This reader **decodes only** -- issue #136 scopes radiometric calibration
    choices out. The RBT product is already geophysical, so `S7`-`S9` / `F1`-`F2`
    arrive as brightness temperature in kelvin with no conversion applied.

    Args:
        source: The `.SEN3` granule directory, or any file inside it.
        channel: A single channel (`"S5"`, or `"S5_radiance_bn"`); mutually
            exclusive with `channels`.
        channels: A sequence of channels for a `dict` result; mutually exclusive
            with `channel`. The dict is keyed by **canonical** channel name, so two
            spellings of one channel -- even on different grids, as
            `["S5_radiance_an", "S5_radiance_bn"]` -- are rejected as a duplicate
            rather than collapsing into one entry.
        view: `"nadir"` or `"oblique"` -- SLSTR images each scene twice. `None`
            (default) takes each channel's view from its own spelling, falling back
            to nadir for a bare one.
        grid: An explicit grid letter (`"a"`, `"b"`, `"i"`, `"f"`) restricting
            every channel to that one grid. `None` (default) uses a spelled grid
            where one is given and otherwise tries the channel's documented grids
            in order. A restricted read never falls back to another resolution: a
            channel the granule does not carry on that grid raises instead.
        geolocation: When `True` (default), attach each result's `GEOLOCATION`
            domain.

    Returns:
        A pyramids `Dataset` (for `channel`) or a `dict[str, Dataset]` (for
        `channels`), each on its native swath grid with NaN nodata.

    Raises:
        ReaderError: When neither / both of `channel` / `channels` are given,
            `channels` is empty or names a channel twice, a channel or the view
            is not an SLSTR one, a spelled channel names a quantity that channel
            does not ship or a grid / view contradicting an explicit `grid=` /
            `view=`, `grid` is not one of the four SLSTR grid letters, or the
            granule lacks a requested channel on the grid(s) tried. With
            `geolocation=True`, also when a channel's geodetic annotation is
            missing, is on a grid whose shape differs from the band, or carries an
            unusable `scale_factor` / `add_offset`.

    Note:
        A `_FillValue` that is not a finite number cannot mask anything, so the
        decode emits a `UserWarning` per affected channel and the result's nodata
        is incomplete rather than the read failing (see `invalid_mask`).
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
    # Pass the caller's ORIGINAL spelling through, not the canonical name: a
    # spelled grid and view select a specific file, and canonicalising first would
    # strip them and read a different one.
    # Resolve the granule once for the whole set rather than per channel, matching
    # read_fci_l1c's and read_olci's read-once pattern.
    granule = resolve_granule(source, "read_slstr")
    results = {
        name: read_slstr_band(
            granule, spelling, view=view, grid=grid, geolocation=geolocation
        )
        for name, spelling in zip(canonical, requested)
    }
    return results[canonical[0]] if single else results
