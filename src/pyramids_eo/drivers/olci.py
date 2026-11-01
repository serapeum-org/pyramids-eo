"""The Sentinel-3 OLCI L1B driver for the ``pyramids.raster_drivers`` plugin hook.

An OLCI granule is a ``.SEN3`` *directory* holding one NetCDF file per radiance band plus the
annotation files, so it is named to ``read_file`` by that directory (GDAL ships no OLCI driver, so
a bare ``.SEN3`` path falls through to this one) or by an explicit ``olci://`` path. Either routes
into :func:`pyramids_eo.sensors.readers.read_olci`.

``read_olci`` reads **one band** (or a set via ``channels=``), so the driver requires the band to be
named — via ``bands=["Oa08"]`` or ``open_options={"channel": "Oa08"}`` — and passes the remaining
options (``geolocation``) through.
"""

from __future__ import annotations

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

#: Filename glob an OLCI L1B radiance band file matches (``Oa01_radiance.nc`` … ``Oa21_radiance.nc``).
_RADIANCE_GLOB = "Oa*_radiance.nc"

#: Optional URL-style scheme naming an OLCI granule explicitly (``olci://<dir>``).
_SCHEME = "olci://"


def _strip_scheme(source: str) -> str:
    """Return ``source`` without a leading ``olci://`` scheme (case-insensitive)."""
    return source[len(_SCHEME) :] if source.lower().startswith(_SCHEME) else source


class OlciDriver:
    """Read one or more Sentinel-3 OLCI L1B radiance bands from a ``.SEN3`` granule.

    Attributes:
        name: The driver's registry name and the value accepted by ``read_file(driver=…)``.
        long_name: Human-readable source name for ``describe_drivers``.
        api_version: The driver-contract version (see
            :data:`~pyramids_eo.drivers._base.RASTER_DRIVER_API_VERSION`).
        suffixes: ``(".sen3",)`` — a granule directory is named ``*.SEN3``.
        schemes: ``("olci",)`` — an ``olci://`` path names a granule explicitly, and distinguishes
            OLCI from SLSTR when both share the ``.SEN3`` suffix.
        capabilities: ``read`` only (one granule is one acquisition).
        extra: ``None`` — the reader is pure pyramids-eo over pyramids' bundled GDAL.
    """

    name = "olci"
    long_name = "Sentinel-3 OLCI L1B (.SEN3)"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes: tuple[str, ...] = (".sen3",)
    schemes: tuple[str, ...] = ("olci",)
    capabilities = frozenset({"read"})
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return whether ``source`` names a real OLCI L1B granule.

        A ``.SEN3`` directory alone is not enough to claim, because SLSTR granules share that
        suffix; the directory must actually hold an ``Oa*_radiance.nc`` band file.

        Args:
            source: The source handed by ``Dataset.read_file`` (stringified) — an ``olci://``
                path, a ``.SEN3`` directory, or a glob string.

        Returns:
            True for an ``olci://`` source, a directory containing at least one
            ``Oa*_radiance.nc``, or a glob whose matches include one; False otherwise. The
            directory listing short-circuits on the first match but enumerates the whole
            directory when nothing matches; it runs only once GDAL has failed to open the
            source, and any filesystem error is swallowed to False so the method never raises.
            ``driver="olci"`` bypasses this entirely.

        Examples:
            - An ``olci://`` source is always claimed (a cheap string check):
                ```python
                >>> from pyramids_eo.drivers.olci import OlciDriver
                >>> OlciDriver.can_open("olci://S3A_OL_1_EFR.SEN3")
                True

                ```
            - A non-string source is never claimed:
                ```python
                >>> OlciDriver.can_open(42)
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
                return any(path.glob(_RADIANCE_GLOB))
            if path.exists():
                return False
            if any(character in source for character in "*?["):
                return any(
                    Path(match).is_dir() and any(Path(match).glob(_RADIANCE_GLOB))
                    for match in glob(source)
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
    ) -> Dataset | dict[str, Dataset]:
        """Read one or more OLCI L1B bands into a :class:`~pyramids.dataset.Dataset`.

        Args:
            source: A ``.SEN3`` granule directory, or that path prefixed ``olci://``.
            bands: The band(s) to read — one name for a single-band ``Dataset``, several for a
                ``dict[str, Dataset]``. A direct-call convenience only: ``Dataset.read_file``
                forwards just ``options``, so through ``read_file`` the band must come from
                ``options={"channel": …}`` / ``options={"channels": [...]}``, and a
                ``channel``/``channels`` in ``options`` takes precedence over ``bands``.
            options: Extra :func:`~pyramids_eo.sensors.readers.read_olci` keywords — ``channel``
                / ``channels`` and ``geolocation`` — as a dict or ``KEY=VALUE`` list.
            **_ignored: Any other generic driver-read parameter (``bbox`` / ``crs`` / ``auth``),
                accepted and ignored — a granule is read whole, carries its own swath geolocation,
                and needs no credentials.

        Returns:
            A single band's ``Dataset``, or a ``dict[str, Dataset]`` when several were requested.

        Raises:
            ValueError: No band was given (via ``bands`` or ``options``, counting an empty
                ``channels`` list as none); both a single ``channel`` and a ``channels`` list were
                given; ``bands`` is not a list of band names; ``options`` is a bare string or
                carries an item that is not ``"KEY=VALUE"``; or ``geolocation`` is a string that
                is not a recognised boolean spelling (see
                :func:`~pyramids_eo.drivers._base.coerce_booleans`).
            ReaderError: Propagated from :func:`~pyramids_eo.sensors.readers.read_olci` — the
                band is not an OLCI one, or the granule lacks it or its geolocation file.

        Note:
            A channel **set** is a direct-call feature only: ``Dataset.read_file`` requires a
            driver to return one ``Dataset``, so an ``options={'channels': [...]}`` read raises
            ``TypeError`` from pyramids' plugin layer. Use the reader function
            (``read_olci(..., channels=[...])``) for several channels at once. This is shared
            with the other channel-at-a-time drivers.
        """
        from pyramids_eo.sensors.readers import read_olci

        settings = options_as_dict(options)
        coerce_booleans(settings, ("geolocation",), driver='olci')
        channel = settings.pop("channel", None)
        channels = settings.pop("channels", None)
        if channel is not None and channels is not None:
            raise ValueError(
                "pass either a single channel or a channels list, not both."
            )
        if channel is None and channels is None and bands is not None:
            channel, channels = bands_to_channels(bands, example="Oa08")
        if isinstance(channels, (list, tuple)) and len(channels) == 0:
            channels = None  # an empty channels list names no band
        if channel is None and channels is None:
            raise ValueError(
                "an OLCI read needs a band: pass bands=['Oa08'], "
                "open_options={'channel': 'Oa08'}."
            )
        return read_olci(
            _strip_scheme(str(source)), channel=channel, channels=channels, **settings
        )
