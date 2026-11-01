"""The Sentinel-3 SLSTR L1 RBT driver for the ``pyramids.raster_drivers`` plugin hook.

An SLSTR granule is a ``.SEN3`` *directory* holding one NetCDF file per channel, view and grid,
plus the annotation files, so it is named to ``read_file`` by that directory (GDAL ships no SLSTR
driver, so a bare ``.SEN3`` path falls through to this one) or by an explicit ``slstr://`` path.
Either routes into :func:`pyramids_eo.sensors.readers.read_slstr`.

``read_slstr`` reads **one channel** (or a set via ``channels=``), so the driver requires the
channel to be named — via ``bands=["S5"]`` or ``open_options={"channel": "S5"}`` — and passes the
remaining options (``view``, ``grid``, ``geolocation``) through.
"""

from __future__ import annotations

from fnmatch import fnmatch
from glob import glob
from pathlib import Path
from typing import Any

from pyramids.dataset import Dataset

from pyramids_eo.drivers._base import (
    RASTER_DRIVER_API_VERSION,
    bands_to_channels,
    coerce_booleans,
    options_as_dict,
)

#: Filename globs an SLSTR L1 RBT measurement file matches — radiance for the solar channels
#: (``S1_radiance_an.nc``) and brightness temperature for the thermal / fire ones (``S8_BT_in.nc``).
_MEASUREMENT_GLOBS = ("S*_radiance_*.nc", "S*_BT_*.nc", "F*_BT_*.nc")

#: Optional URL-style scheme naming an SLSTR granule explicitly (``slstr://<dir>``).
_SCHEME = "slstr://"


def _strip_scheme(source: str) -> str:
    """Return ``source`` without a leading ``slstr://`` scheme (case-insensitive)."""
    return source[len(_SCHEME) :] if source.lower().startswith(_SCHEME) else source


def _holds_measurement(path: Path) -> bool:
    """Return whether ``path`` is a directory holding any SLSTR measurement file.

    One pass over the directory, short-circuiting on the first match, rather than a
    separate glob per pattern (which enumerated the whole directory up to three times
    when nothing matched).
    """
    if not path.is_dir():
        return False
    return any(
        fnmatch(entry.name, pattern)
        for entry in path.iterdir()
        for pattern in _MEASUREMENT_GLOBS
    )


class SlstrDriver:
    """Read one or more Sentinel-3 SLSTR L1 RBT channels from a ``.SEN3`` granule.

    Attributes:
        name: The driver's registry name and the value accepted by ``read_file(driver=…)``.
        long_name: Human-readable source name for ``describe_drivers``.
        api_version: The driver-contract version (see
            :data:`~pyramids_eo.drivers._base.RASTER_DRIVER_API_VERSION`).
        suffixes: ``(".sen3",)`` — a granule directory is named ``*.SEN3``.
        schemes: ``("slstr",)`` — a ``slstr://`` path names a granule explicitly, and
            distinguishes SLSTR from OLCI when both share the ``.SEN3`` suffix.
        capabilities: ``read`` only (one granule is one acquisition).
        extra: ``None`` — the reader is pure pyramids-eo over pyramids' bundled GDAL.
    """

    name = "slstr"
    long_name = "Sentinel-3 SLSTR L1 RBT (.SEN3)"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes: tuple[str, ...] = (".sen3",)
    schemes: tuple[str, ...] = ("slstr",)
    capabilities = frozenset({"read"})
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return whether ``source`` names a real SLSTR L1 RBT granule.

        A ``.SEN3`` directory alone is not enough to claim, because OLCI granules share that
        suffix; the directory must actually hold an SLSTR measurement file.

        Args:
            source: The source handed by ``Dataset.read_file`` (stringified) — a ``slstr://``
                path, a ``.SEN3`` directory, or a glob string.

        Returns:
            True for a ``slstr://`` source, a directory containing at least one measurement
            file, or a glob whose matches include such a directory; False otherwise. The
            directory listing short-circuits on the first match but enumerates the whole
            directory when nothing matches; it runs only once GDAL has failed to open the
            source, and any filesystem error is swallowed to False so the method never raises.
            ``driver="slstr"`` bypasses this entirely.

        Examples:
            - A ``slstr://`` source is always claimed (a cheap string check):
                ```python
                >>> from pyramids_eo.drivers.slstr import SlstrDriver
                >>> SlstrDriver.can_open("slstr://S3A_SL_1_RBT.SEN3")
                True

                ```
            - A non-string source is never claimed:
                ```python
                >>> SlstrDriver.can_open(None)
                False

                ```
        """
        if not isinstance(source, str) or not source:
            return False
        if source.lower().startswith(_SCHEME):
            return True
        # Resolve an existing path literally first, so a real granule whose own name contains a
        # glob metacharacter is stat-checked rather than treated as a pattern. The probes are
        # wrapped so can_open honours its contract of never raising.
        try:
            path = Path(source)
            if path.is_dir():
                return _holds_measurement(path)
            if path.exists():
                return False
            if any(character in source for character in "*?["):
                return any(_holds_measurement(Path(match)) for match in glob(source))
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
    ) -> Dataset | dict[str, Dataset]:
        """Read one or more SLSTR L1 RBT channels into a :class:`~pyramids.dataset.Dataset`.

        Args:
            source: A ``.SEN3`` granule directory, or that path prefixed ``slstr://``.
            bands: The channel(s) to read — one name for a single-channel ``Dataset``, several for
                a ``dict[str, Dataset]``. A direct-call convenience only: ``Dataset.read_file``
                forwards just ``options``, so through ``read_file`` the channel must come from
                ``options={"channel": …}`` / ``options={"channels": [...]}``, and a
                ``channel``/``channels`` in ``options`` takes precedence over ``bands``.
            options: Extra :func:`~pyramids_eo.sensors.readers.read_slstr` keywords — ``channel``
                / ``channels``, ``view``, ``grid`` and ``geolocation`` — as a dict or
                ``KEY=VALUE`` list.
            **_ignored: Any other generic driver-read parameter (``bbox`` / ``crs`` / ``auth``),
                accepted and ignored — a granule is read whole, carries its own swath geolocation,
                and needs no credentials.


        A channel **set** is a direct-call feature only: ``Dataset.read_file`` requires a driver
        to return one ``Dataset``, so an ``options={'channels': [...]}`` read raises ``TypeError``
        from pyramids' plugin layer. Use the reader function (``read_slstr(..., channels=[...])``)
        for several channels at once. This is shared with the other channel-at-a-time drivers.

        Returns:
            A single channel's ``Dataset``, or a ``dict[str, Dataset]`` when several were
            requested.

        Raises:
            ValueError: No channel was given (via ``bands`` or ``options``, counting an empty
                ``channels`` list as none); both a single ``channel`` and a ``channels`` list were
                given; or ``bands`` is not a list of channel names.
        """
        from pyramids_eo.sensors.readers import read_slstr

        settings = options_as_dict(options)
        coerce_booleans(settings, ("geolocation",), driver='slstr')
        channel = settings.pop("channel", None)
        channels = settings.pop("channels", None)
        if channel is not None and channels is not None:
            raise ValueError(
                "pass either a single channel or a channels list, not both."
            )
        if channel is None and channels is None and bands is not None:
            channel, channels = bands_to_channels(bands, example="S5")
        if isinstance(channels, (list, tuple)) and len(channels) == 0:
            channels = None  # an empty channels list names no channel
        if channel is None and channels is None:
            raise ValueError(
                "an SLSTR read needs a channel: pass bands=['S5'], "
                "open_options={'channel': 'S5'}."
            )
        return read_slstr(
            _strip_scheme(str(source)), channel=channel, channels=channels, **settings
        )
