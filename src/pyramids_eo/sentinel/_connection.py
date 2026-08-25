"""GDAL connection-string grammar for the Sentinel drivers.

The `SENTINEL2` and `SAFE` drivers address the rasters inside a product through
structured connection strings rather than plain paths. pyramids' `subdatasets`
lists them, but selecting one by a short human token (``"60m"``, ``"IW_VV"``)
means parsing those strings — and `Dataset.open_subdataset` matches only the
*full* string, so this module owns both directions:

* :func:`parse_s2` / :func:`parse_s1` — split a driver connection string into
  its typed parts.
* :func:`select` — pick the one subdataset in a list whose parsed parts match a
  short token (resolution / EPSG for S2; swath / polarisation / calibration for
  S1).

Keeping the grammar here is the deliberate answer to serapeum-org/pyramids#1030
leaving `open_subdataset` full-string-only: the per-driver knowledge lives in
one place instead of being re-spelled at every call site.

Syntax reference (verified against GDAL 3.13.1):

* S2  ``SENTINEL2_<LEVEL>:<mtd.xml>:<RES>:EPSG_<code>`` where ``LEVEL`` is
  ``L1B`` / ``L1C`` / ``L2A`` and ``RES`` is ``10m`` / ``20m`` / ``60m`` /
  ``PREVIEW`` / ``TCI``.
* S1  ``SENTINEL1_CALIB:<CALIB>:<manifest.safe>:<SWATH>[_<POL>]:<UNIT>`` where
  ``CALIB`` is ``SIGMA0`` / ``BETA0`` / ``GAMMA`` / ``UNCALIB`` and ``UNIT`` is
  ``AMPLITUDE`` / ``COMPLEX`` / ``INTENSITY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyramids_eo.errors import ProductError

#: GDAL driver connection prefixes.
_S2_PREFIX = "SENTINEL2_"
_S1_PREFIX = "SENTINEL1_CALIB:"


@dataclass(frozen=True)
class S2Connection:
    """Parsed parts of a ``SENTINEL2_<LEVEL>:…`` connection string.

    Attributes:
        level: Processing level token — ``"L1B"`` / ``"L1C"`` / ``"L2A"``.
        source: The product metadata XML path the string points at.
        resolution: Resolution token — ``"10m"`` / ``"20m"`` / ``"60m"`` /
            ``"PREVIEW"`` / ``"TCI"``.
        epsg: UTM EPSG code parsed from the ``EPSG_<code>`` tail, or ``None``
            when the string carries no CRS token (e.g. some previews).
    """

    level: str
    source: str
    resolution: str
    epsg: int | None


@dataclass(frozen=True)
class S1Connection:
    """Parsed parts of a ``SENTINEL1_CALIB:…`` connection string.

    Attributes:
        calibration: ``"UNCALIB"`` / ``"SIGMA0"`` / ``"BETA0"`` / ``"GAMMA"``.
        source: The ``manifest.safe`` (or ``.SAFE``) path the string points at.
        swath: Swath token (e.g. ``"IW"``), optionally suffixed ``_<POL>``.
        polarisation: Polarisation (``"VV"`` / ``"VH"`` / …) when the swath
            token carries one, else ``None`` (all polarisations as bands).
        unit: ``"AMPLITUDE"`` / ``"COMPLEX"`` / ``"INTENSITY"``.
    """

    calibration: str
    source: str
    swath: str
    polarisation: str | None
    unit: str


def _epsg_from_token(token: str) -> int | None:
    """Return the integer code from an ``EPSG_<code>`` token, else ``None``."""
    if token.upper().startswith("EPSG_"):
        try:
            return int(token.split("_", 1)[1])
        except (ValueError, IndexError):
            return None
    return None


def parse_s2(connection: str) -> S2Connection:
    """Parse a ``SENTINEL2_<LEVEL>:<xml>:<RES>[:EPSG_<code>]`` string.

    The path itself can contain a Windows drive colon (``C:``), so the split
    is anchored on the known head (level) and tail (resolution / EPSG) tokens
    rather than a naive ``split(":")``.

    Args:
        connection: A ``SENTINEL2_*`` connection string, e.g. from a pyramids
            ``SubDataset.name``.

    Returns:
        The parsed :class:`S2Connection`.

    Raises:
        ProductError: The string is not a ``SENTINEL2_*`` connection string,
            or its tail cannot be read.
    """
    if not connection.startswith(_S2_PREFIX):
        raise ProductError(f"not a SENTINEL2 connection string: {connection!r}")
    head, _, rest = connection.partition(":")
    level = head[len(_S2_PREFIX) :]
    # The tail is the last one or two colon-separated tokens: <RES>[:EPSG_code].
    # Everything before them is the (possibly colon-bearing) source path.
    parts = rest.rsplit(":", 2)
    epsg: int | None = None
    if len(parts) == 3 and _epsg_from_token(parts[2]) is not None:
        source, resolution, epsg_token = parts
        epsg = _epsg_from_token(epsg_token)
    else:
        # No trailing EPSG token: the resolution is the final token.
        source, _, resolution = rest.rpartition(":")
    if not source or not resolution:
        raise ProductError(f"malformed SENTINEL2 connection string: {connection!r}")
    return S2Connection(level=level, source=source, resolution=resolution, epsg=epsg)


def parse_s1(connection: str) -> S1Connection:
    """Parse a ``SENTINEL1_CALIB:<CALIB>:<manifest>:<SWATH[_POL]>:<UNIT>`` string.

    Args:
        connection: A ``SENTINEL1_CALIB:*`` connection string.

    Returns:
        The parsed :class:`S1Connection`.

    Raises:
        ProductError: The string is not a ``SENTINEL1_CALIB:*`` string or is
            malformed.
    """
    if not connection.startswith(_S1_PREFIX):
        raise ProductError(f"not a SENTINEL1_CALIB connection string: {connection!r}")
    body = connection[len(_S1_PREFIX) :]
    calib, _, rest = body.partition(":")
    # tail = <SWATH[_POL]>:<UNIT>; the source path (with a drive colon) is the
    # middle, so anchor on the two trailing tokens.
    middle, _, unit = rest.rpartition(":")
    source, _, swath_pol = middle.rpartition(":")
    if not (calib and source and swath_pol and unit):
        raise ProductError(f"malformed SENTINEL1 connection string: {connection!r}")
    swath, _, pol = swath_pol.partition("_")
    return S1Connection(
        calibration=calib,
        source=source,
        swath=swath,
        polarisation=pol or None,
        unit=unit,
    )


def select(subdatasets: list[Any], **tokens: str | int | None) -> Any:
    """Return the single subdataset whose parsed connection matches ``tokens``.

    ``subdatasets`` is a list of pyramids ``SubDataset`` value objects (from
    ``Dataset.subdatasets``). Each is parsed with :func:`parse_s2` or
    :func:`parse_s1` (chosen by the connection prefix) and compared field-wise
    against the non-``None`` ``tokens``. Comparison is case-insensitive for
    strings; the ``epsg`` token compares as an integer.

    Args:
        subdatasets: The candidate subdatasets.
        **tokens: Field constraints, e.g. ``resolution="60m", epsg=32632`` for
            S2, or ``swath="IW", polarisation="VV", calibration="UNCALIB"`` for
            S1. Keys must be attributes of the parsed connection; ``None``
            values are ignored (match anything).

    Returns:
        The matching ``SubDataset``.

    Raises:
        ProductError: No subdataset matches, or more than one does.
    """
    wanted = {k: v for k, v in tokens.items() if v is not None}
    matches = []
    for sd in subdatasets:
        parsed = _parse_any(sd.name)
        if parsed is None:
            continue
        if all(_field_eq(getattr(parsed, k, None), v) for k, v in wanted.items()):
            matches.append(sd)
    if not matches:
        raise ProductError(
            f"no subdataset matches {wanted!r}; "
            f"available: {[sd.name.rsplit(':', 2)[-2:] for sd in subdatasets]}"
        )
    if len(matches) > 1:
        raise ProductError(
            f"{len(matches)} subdatasets match {wanted!r}; tighten the query"
        )
    return matches[0]


def _parse_any(connection: str) -> S2Connection | S1Connection | None:
    """Parse a connection string of either driver, or ``None`` if neither."""
    if connection.startswith(_S2_PREFIX):
        return parse_s2(connection)
    if connection.startswith(_S1_PREFIX):
        return parse_s1(connection)
    return None


def _field_eq(actual: Any, wanted: str | int) -> bool:
    """Case-insensitive equality for tokens; integer equality for EPSG codes."""
    if actual is None:
        return False
    if isinstance(wanted, int):
        return int(actual) == wanted
    return str(actual).upper() == str(wanted).upper()
