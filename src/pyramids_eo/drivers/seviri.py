"""The MSG SEVIRI driver for the ``pyramids.raster_drivers`` plugin hook.

Registers SEVIRI native ``.nat`` granules so ``Dataset.read_file("scene.nat", driver="seviri")``
(or a bare ``read_file("scene.nat")``, which GDAL cannot open and so falls through to this
driver) routes into :func:`pyramids_eo.sensors.readers.read_seviri`.

``read_seviri`` reads **one channel** and takes an injectable ``.nat`` decoder (``parse=``), so
the driver requires the channel to be named — via ``bands=["IR_108"]`` or
``open_options={"channel": "IR_108"}`` — and passes any remaining options (``parse``,
``calibrate``, ``coeffs``, …) straight through. The read covers the whole granule, so ``bbox`` /
``crs`` do not apply and are ignored.

``.nat`` is the EUMETSAT Native format, shared across instruments (MSG SEVIRI, MFG MVIRI, …), so
``can_open`` cannot tell a SEVIRI granule from another native file by extension alone — and it
must not try, since the contract requires it to be cheap, open nothing, and answer for a path
that does not exist (the ``read_file`` fallback). This driver is pyramids-eo's only ``.nat``
reader and is deliberately scoped to SEVIRI: a non-SEVIRI ``.nat`` routed here (by an explicit
``driver="seviri"`` or the bare-``read_file`` fallback) is read by ``read_seviri`` and fails
there with a clear decode error, never silently mis-interpreted.
"""

from __future__ import annotations

from typing import Any

from pyramids.dataset import Dataset

from pyramids_eo.drivers._base import RASTER_DRIVER_API_VERSION, options_as_dict


class SeviriDriver:
    """Read one MSG SEVIRI channel from a native ``.nat`` granule.

    Attributes:
        name: The driver's registry name and the value accepted by ``read_file(driver=…)``.
        long_name: Human-readable source name for ``describe_drivers``.
        api_version: The driver-contract version (see
            :data:`~pyramids_eo.drivers._base.RASTER_DRIVER_API_VERSION`).
        suffixes: ``(".nat",)`` — the driver claims SEVIRI native granules by extension.
        schemes: Empty — a ``.nat`` granule is a local file.
        capabilities: ``read`` only (a single granule is one timestep).
        extra: ``None`` — the ``.nat`` decoder is injected via ``parse=``, so there is no hard
            backend to gate on.
    """

    name = "seviri"
    long_name = "MSG SEVIRI native (.nat)"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes: tuple[str, ...] = (".nat",)
    schemes: tuple[str, ...] = ()
    capabilities = frozenset({"read"})
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return whether ``source`` is a SEVIRI native ``.nat`` granule.

        Args:
            source: The source handed by ``Dataset.read_file`` (stringified).

        Returns:
            True for a string ending ``.nat`` (case-insensitive), else False. Cheap: no import,
            no network, no file access — ``.nat`` is the shared EUMETSAT Native extension, so a
            non-SEVIRI native file also answers True and fails later in ``read_seviri`` (see the
            module docstring), which is preferred over opening the file here.

        Examples:
            - A ``.nat`` granule is claimed; other suffixes are not:
                ```python
                >>> from pyramids_eo.drivers.seviri import SeviriDriver
                >>> SeviriDriver.can_open("MSG_SEVIRI.nat")
                True
                >>> SeviriDriver.can_open("scene.tif")
                False

                ```
        """
        return isinstance(source, str) and source.lower().endswith(".nat")

    def read(
        self,
        source: Any,
        *,
        bbox: Any = None,
        bands: Any = None,
        crs: Any = None,
        options: Any = None,
        auth: Any = None,
    ) -> Dataset:
        """Read one SEVIRI channel into a :class:`~pyramids.dataset.Dataset`.

        Args:
            source: Path to a SEVIRI native ``.nat`` granule.
            bbox: Ignored — ``read_seviri`` reads the whole granule.
            bands: The channel to read, as a one-element list (e.g. ``["IR_108"]``); an
                alternative to ``options={"channel": …}``. A SEVIRI read is a single channel, so
                more than one band is rejected rather than silently truncated.
            crs: Ignored — the granule carries its own geostationary CRS.
            options: Extra :func:`~pyramids_eo.sensors.readers.read_seviri` keywords —
                ``channel``, ``parse`` (the ``.nat`` decoder), ``calibrate``, ``coeffs``, … — as
                a dict or ``KEY=VALUE`` list.
            auth: Ignored — a local granule needs no credentials.

        Returns:
            The calibrated, geolocated channel.

        Raises:
            ValueError: No channel was given (via ``bands`` or ``options``), or more than one
                band was given.
        """
        from pyramids_eo.sensors.readers import read_seviri

        settings = options_as_dict(options)
        channel = settings.pop("channel", None)
        if channel is None and bands:
            if len(bands) != 1:
                raise ValueError(
                    f"a SEVIRI read is one channel, but {len(bands)} bands were given "
                    f"({list(bands)}); pass a single band or options={{'channel': 'IR_108'}}."
                )
            channel = bands[0]
        if channel is None:
            raise ValueError(
                "a SEVIRI read needs a channel: pass bands=['IR_108'] or "
                "open_options={'channel': 'IR_108'}."
            )
        return read_seviri(source, channel, **settings)
