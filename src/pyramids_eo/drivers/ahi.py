"""The Himawari AHI driver for the ``pyramids.raster_drivers`` plugin hook.

Registers Himawari Standard Data (HSD) granules so
``Dataset.read_file("HS_H08_….DAT", driver="ahi")`` — or a bare
``read_file("HS_H08_….DAT")``, which GDAL cannot open and so falls through to this driver —
routes into :func:`pyramids_eo.sensors.readers.read_ahi`.

Unlike the other drivers here, this one does **not** claim its extension. ``.dat`` is the most
generic suffix in use and says nothing about the content, so claiming every ``.dat`` would hijack
unrelated files that GDAL (or another plugin) should get. Instead ``can_open`` matches the HSD
file-name convention of Guide Table 4 — ``HS_aaa_yyyymmdd_hhnn_Bbb_cccc_Rjj_Skkll.DAT`` — which
is distinctive enough to identify a granule without opening it. A granule that has been renamed
is still readable by passing ``driver="ahi"`` explicitly, or by calling ``read_ahi`` directly.

An HSD granule names its own band in calibration block #5, so unlike the SEVIRI driver this one
does not require the band to be given. Passing it (via ``bands=["B13"]`` or
``open_options={"band": "B13"}``) asserts the expectation instead, and a granule carrying a
different band is an error rather than a silent surprise. The read covers the whole granule, so
``bbox`` / ``crs`` do not apply and are ignored.
"""

from __future__ import annotations

import re
from typing import Any

from pyramids.dataset import Dataset

from pyramids_eo.drivers._base import (
    RASTER_DRIVER_API_VERSION,
    coerce_booleans,
    coerce_floats,
    options_as_dict,
)

#: The HSD file-name convention (Guide Table 4), optionally ``.bz2``-compressed. Matched on the
#: base name only, so a granule in any directory is recognised.
_HSD_NAME_RE = re.compile(
    r"^HS_H\d{2}_\d{8}_\d{4}_B\d{2}_[A-Za-z0-9]{4}_R\d{2}_S\d{4}\.DAT(\.bz2)?$",
    re.IGNORECASE,
)


class AhiDriver:
    """Read one Himawari-8/9 AHI band from a Himawari Standard Data granule.

    Attributes:
        name: The driver's registry name and the value accepted by ``read_file(driver=…)``.
        long_name: Human-readable source name for ``describe_drivers``.
        api_version: The driver-contract version (see
            :data:`~pyramids_eo.drivers._base.RASTER_DRIVER_API_VERSION`).
        suffixes: ``(".dat",)`` — declared so the driver is discoverable for HSD granules, while
            ``can_open`` narrows the claim to the HSD name convention (see the module docstring).
        schemes: Empty — an HSD granule is a local file.
        capabilities: ``read`` only (one granule is one band at one timestep).
        extra: ``None`` — HSD is decoded in-process from the standard library, so there is no
            optional backend to gate on.
    """

    name = "ahi"
    long_name = "Himawari-8/9 AHI Himawari Standard Data (.DAT)"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes: tuple[str, ...] = (".dat",)
    schemes: tuple[str, ...] = ()
    capabilities = frozenset({"read"})
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return whether ``source`` names a Himawari Standard Data granule.

        Args:
            source: The source handed by ``Dataset.read_file`` (stringified).

        Returns:
            True when the base name follows the HSD convention
            ``HS_aaa_yyyymmdd_hhnn_Bbb_cccc_Rjj_Skkll.DAT`` (optionally ``.bz2``), else False.
            Cheap: no import, no network, no file access, and safe for a path that does not
            exist — matching the name rather than the generic ``.dat`` suffix keeps the driver
            from claiming unrelated files.

        Examples:
            - An HSD granule is claimed; a bare ``.dat`` or another suffix is not:
                ```python
                >>> from pyramids_eo.drivers.ahi import AhiDriver
                >>> AhiDriver.can_open("HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT")
                True
                >>> AhiDriver.can_open("/data/HS_H09_20240101_0000_B03_FLDK_R05_S0110.DAT.bz2")
                True
                >>> AhiDriver.can_open("scene.dat")
                False
                >>> AhiDriver.can_open("scene.tif")
                False

                ```
        """
        if not isinstance(source, str):
            return False
        base = source.replace("\\", "/").rsplit("/", 1)[-1]
        return _HSD_NAME_RE.match(base) is not None

    def read(
        self,
        source: Any,
        *,
        bands: Any = None,
        options: Any = None,
        **_ignored: Any,
    ) -> Dataset:
        """Read one AHI band into a :class:`~pyramids.dataset.Dataset`.

        Args:
            source: Path to an HSD ``.DAT`` / ``.DAT.bz2`` granule.
            bands: Optionally the band to read, as a one-element list (e.g. ``["B13"]``); an
                alternative to ``options={"band": …}``. A granule names its own band, so this
                only asserts the expectation. An AHI read is a single band, so more than one is
                rejected rather than silently truncated.
            options: Extra :func:`~pyramids_eo.sensors.readers.read_ahi` keywords — ``band``,
                ``calibrate``, ``coeffs``, ``sun_earth_distance``, … — as a dict or
                ``KEY=VALUE`` list. ``read_file`` normalises options to the ``KEY=VALUE`` form,
                so ``calibrate`` and the numeric keyword are coerced from their string
                spellings (see :func:`~pyramids_eo.drivers._base.coerce_booleans`); without
                that, the truthy string ``"false"`` would switch calibration *on*.
            **_ignored: The generic driver-read parameters an AHI read does not use — ``bbox``,
                ``crs``, ``auth`` and any future one — are accepted and ignored: a granule is
                read whole, carries its own geostationary CRS, and needs no credentials.

        Returns:
            The calibrated, geolocated band.

        Raises:
            ValueError: More than one band was given, or a boolean / numeric option could not
                be read from its ``KEY=VALUE`` string spelling (``calibrate``,
                ``sun_earth_distance``).
        """
        from pyramids_eo.sensors.readers import read_ahi

        settings = options_as_dict(options)
        coerce_booleans(settings, ("calibrate",), driver="ahi")
        coerce_floats(settings, ("sun_earth_distance",), driver="ahi")
        band = settings.pop("band", None)
        if band is None and bands:
            if isinstance(bands, (str, bytes)):
                raise ValueError(
                    f"bands must be a list of band names, not a bare "
                    f"{type(bands).__name__} ({bands!r}); pass bands=['B13']."
                )
            bands = list(bands)  # materialise a generator/iterator so len() is safe
            if len(bands) != 1:
                raise ValueError(
                    f"an AHI read is one band, but {len(bands)} bands were given "
                    f"({bands}); pass a single band or options={{'band': 'B13'}}."
                )
            band = bands[0]
        return read_ahi(source, band, **settings)
