"""The MTG-FCI L1C (FDHSI) driver for the ``pyramids.raster_drivers`` plugin hook.

A real FCI L1C FDHSI granule is a *set* of NetCDF chunk files, so it is named to ``read_file`` by
a **directory** holding the chunks, a **glob** matching them, or an ``fci://`` path. Any of
``read_file(granule_dir)`` (bare — GDAL cannot open a directory, so it falls through to this
driver), ``read_file("fci://" + granule_dir)``, or ``read_file(granule, driver="fci")`` routes
into :func:`pyramids_eo.sensors.readers.read_fci_l1c`. The driver expands the single source string
into the chunk list the reader stitches; the reader orders the chunks north→south itself, so the
order of discovery does not matter.

``read_fci_l1c`` reads **one channel** (or a channel set via ``channels=``), so the driver
requires the channel to be named — via ``bands=["ir_105"]`` or ``open_options={"channel":
"ir_105"}`` — and passes the remaining options (``calibrate``, ``sun_earth_distance``, …) through.
"""

from __future__ import annotations

from collections.abc import Mapping
from glob import glob
from pathlib import Path
from typing import Any

from pyramids.dataset import Dataset

from pyramids_eo.drivers._base import (
    RASTER_DRIVER_API_VERSION,
    expand_sources,
    options_as_dict,
)

#: Filename glob a real EUMETSAT FCI L1C FDHSI chunk matches (…FCI-1C-RRAD-FDHSI…CHK-BODY….nc).
_FDHSI_GLOB = "*FDHSI*.nc"

#: Optional URL-style scheme naming an FCI granule explicitly (``fci://<dir-or-glob>``).
_SCHEME = "fci://"


def _strip_scheme(source: str) -> str:
    """Return ``source`` without a leading ``fci://`` scheme (case-insensitive)."""
    return source[len(_SCHEME) :] if source.lower().startswith(_SCHEME) else source


def _bands_to_channels(bands: Any) -> tuple[Any, Any]:
    """Map a ``bands`` argument to ``(channel, channels)`` for :func:`read_fci_l1c`.

    Args:
        bands: A list of channel names (one for a single channel, several for a set).

    Returns:
        ``(name, None)`` for a single band, ``(None, [names])`` for several, and ``(None, None)``
        for an empty list (which names no channel).

    Raises:
        ValueError: ``bands`` is not a list of channel names — a string, bytes, mapping, or other
            non-sequence.
    """
    if isinstance(bands, (str, bytes, Mapping)):
        raise ValueError(
            f"bands must be a list of channel names, not a "
            f"{type(bands).__name__} ({bands!r}); pass bands=['ir_105']."
        )
    try:
        names = list(bands)
    except TypeError as error:
        raise ValueError(
            f"bands must be a list of channel names, not a "
            f"{type(bands).__name__} ({bands!r}); pass bands=['ir_105']."
        ) from error
    if len(names) == 1:
        return names[0], None
    if len(names) > 1:
        return None, names
    return None, None


class FciDriver:
    """Read one or more MTG-FCI L1C FDHSI channels from a granule's chunk set.

    Attributes:
        name: The driver's registry name and the value accepted by ``read_file(driver=…)``.
        long_name: Human-readable source name for ``describe_drivers``.
        api_version: The driver-contract version (see
            :data:`~pyramids_eo.drivers._base.RASTER_DRIVER_API_VERSION`).
        suffixes: Empty — a granule is a directory or glob of chunks, not a single-file suffix.
        schemes: ``("fci",)`` — an ``fci://`` path names a granule explicitly (and is the routable
            declaration the contract requires, since a granule has no file suffix). ``can_open``
            also claims a bare granule directory.
        capabilities: ``read`` only (one granule is one timestep).
        extra: ``None`` — the reader is pure pyramids-eo over pyramids' bundled GDAL, with no
            optional backend to gate on.
    """

    name = "fci"
    long_name = "MTG-FCI L1C FDHSI granule"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes: tuple[str, ...] = ()
    schemes: tuple[str, ...] = ("fci",)
    capabilities = frozenset({"read"})
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return whether ``source`` names a real FCI L1C FDHSI chunk set.

        Args:
            source: The source handed by ``Dataset.read_file`` (stringified) — an ``fci://`` path,
                a directory, or a glob string.

        Returns:
            True for an ``fci://`` source, a directory containing at least one ``*FDHSI*.nc``
            chunk, or a glob whose matches include one; False for an empty source. It does a
            bounded, non-recursive directory listing / glob (no file open, no network) — more than
            the string-only ``can_open`` of the other drivers, but it runs only when GDAL has
            already failed to open the source, a very broad user glob (e.g. ``**``) is the caller's
            cost, and a filesystem error (e.g. an unreadable directory) is swallowed to False so
            the method never raises. ``driver="fci"`` bypasses this, so a non-standard chunk set can
            still be read explicitly.

        Examples:
            - An ``fci://`` source is always claimed (a cheap string check):
                ```python
                >>> from pyramids_eo.drivers.fci import FciDriver
                >>> FciDriver.can_open("fci://MTG_FCI_granule")
                True

                ```
            - A non-string source is never claimed:
                ```python
                >>> FciDriver.can_open(42)
                False

                ```
        """
        if not isinstance(source, str) or not source:
            return False
        if source.lower().startswith(_SCHEME):
            return True
        # Resolve an existing path literally first, so a real granule directory whose own name
        # contains a glob metacharacter (e.g. ``granule[1]``) is stat-checked, not treated as a
        # pattern. Only a non-existent source is interpreted as a glob. The filesystem probes are
        # wrapped so can_open honours its contract of never raising (an unreadable dir -> False).
        try:
            path = Path(source)
            if path.is_dir():
                return any(path.glob(_FDHSI_GLOB))
            if path.exists():
                return False
            if any(character in source for character in "*?["):
                return any(
                    "fdhsi" in Path(match).name.lower() for match in glob(source)
                )
        except (OSError, ValueError):
            return False
        return False

    def read(
        self,
        source: Any,
        *,
        bands: Any = None,
        options: Any = None,
        **_ignored: Any,
    ) -> Dataset:
        """Read one or more FCI L1C channels into a :class:`~pyramids.dataset.Dataset`.

        Args:
            source: A directory of FDHSI ``.nc`` chunks, a glob matching them, a single chunk, or
                any of those prefixed ``fci://``.
            bands: The channel(s) to read — one name for a single-channel ``Dataset``, several for
                a ``dict[str, Dataset]``. A direct-call convenience only: ``Dataset.read_file``
                forwards just ``options`` to a driver, so through ``read_file`` the channel must
                come from ``options={"channel": …}`` / ``options={"channels": [...]}``, and a
                ``channel``/``channels`` in ``options`` takes precedence over ``bands``.
            options: Extra :func:`~pyramids_eo.sensors.readers.read_fci_l1c` keywords — ``channel``
                / ``channels``, ``calibrate``, ``sun_earth_distance``, ``cos_sza`` — as a dict or
                ``KEY=VALUE`` list.
            **_ignored: Any other generic driver-read parameter (``bbox`` / ``crs`` / ``auth``),
                accepted and ignored — a granule is read whole, carries its own geostationary CRS,
                and needs no credentials.

        Returns:
            A single channel's ``Dataset``, or a ``dict[str, Dataset]`` when several channels were
            requested.

        Raises:
            ValueError: No channel was given (via ``bands`` or ``options``, counting an empty
                ``channels`` list as none); both a single ``channel`` and a ``channels`` list were
                given; ``bands`` is not a list of channel names (a string, bytes, mapping, or other
                non-sequence); or the source resolved to no chunk files.
        """
        from pyramids_eo.sensors.readers import read_fci_l1c

        settings = options_as_dict(options)
        channel = settings.pop("channel", None)
        channels = settings.pop("channels", None)
        if channel is not None and channels is not None:
            raise ValueError(
                "pass either a single channel or a channels list, not both."
            )
        if channel is None and channels is None and bands is not None:
            channel, channels = _bands_to_channels(bands)
        if isinstance(channels, (list, tuple)) and len(channels) == 0:
            channels = None  # an empty channels list names no channel
        if channel is None and channels is None:
            raise ValueError(
                "an FCI read needs a channel: pass bands=['ir_105'], "
                "open_options={'channel': 'ir_105'}, or open_options={'channels': [...]}."
            )
        paths = expand_sources(_strip_scheme(str(source)), pattern="*.nc")
        if not paths:
            raise ValueError(
                f"no chunk files found for FCI source {source!r}; expected a directory, glob, or "
                "fci:// path resolving to one or more .nc chunks."
            )
        return read_fci_l1c(paths, channel=channel, channels=channels, **settings)
