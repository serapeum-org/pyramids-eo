"""The Earth Engine raster driver for the ``pyramids.raster_drivers`` plugin hook.

Registers Earth Engine assets so ``Dataset.read_file("ee://<asset-id>", driver="earthengine")``
(or a bare ``read_file("ee://…")``, which GDAL cannot open and so falls through to this driver)
routes into :func:`pyramids_eo.from_earthengine`. An asset id has no file and no suffix, so the
driver claims the ``ee://`` URL scheme rather than an extension; a bare asset id with no scheme
is ambiguous with a GDAL path and stays on the typed :func:`~pyramids_eo.from_earthengine`
facade.

Earth Engine is reached through GDAL's EEDAI driver, which is bundled with pyramids' GDAL — not
through the ``earthengine-api`` (``ee``) package, which pyramids-eo does not depend on. There is
therefore no optional backend to gate on, so :attr:`extra` is ``None`` and the driver is always
discovered; an EE read still needs credentials, which is what the ``needs_auth`` capability
declares and :func:`pyramids_eo.from_earthengine` enforces at read time. The reader is imported
inside :meth:`read` — never at module load — so discovery stays cheap.
"""

from __future__ import annotations

from typing import Any

from pyramids.dataset import Dataset, DatasetCollection

from pyramids_eo.drivers._base import RASTER_DRIVER_API_VERSION, options_as_dict

_SCHEME = "ee://"


def _strip_scheme(source: str) -> str:
    """Return the asset id behind an ``ee://`` source."""
    return source[len(_SCHEME) :]


class EarthEngineDriver:
    """Read an Earth Engine ``Image`` or ``ImageCollection`` via an ``ee://`` source.

    Attributes:
        name: The driver's registry name and the value accepted by ``read_file(driver=…)``.
        long_name: Human-readable source name for ``describe_drivers``.
        api_version: The driver-contract version (see
            :data:`~pyramids_eo.drivers._base.RASTER_DRIVER_API_VERSION`).
        suffixes: Empty — an Earth Engine asset has no file.
        schemes: ``("ee",)`` — the driver claims ``ee://…`` sources.
        capabilities: ``read`` (single image), ``read_collection`` (temporal stack), and
            ``needs_auth`` (an Earth Engine read needs credentials).
        extra: ``None`` — EE is reached through pyramids' bundled GDAL EEDAI driver, so there is
            no separate optional backend to gate discovery on.
    """

    name = "earthengine"
    long_name = "Google Earth Engine"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes: tuple[str, ...] = ()
    schemes: tuple[str, ...] = ("ee",)
    capabilities = frozenset({"read", "read_collection", "needs_auth"})
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return whether ``source`` is an ``ee://`` Earth Engine asset id.

        Args:
            source: The source handed by ``Dataset.read_file`` (stringified).

        Returns:
            True for a string beginning ``ee://`` (case-insensitive), else False. Cheap: no
            import, no network.

        Examples:
            - An ``ee://`` source is claimed; a plain path or a bare asset id is not:
                ```python
                >>> from pyramids_eo.drivers.earthengine import EarthEngineDriver
                >>> EarthEngineDriver.can_open("ee://COPERNICUS/S2_SR_HARMONIZED")
                True
                >>> EarthEngineDriver.can_open("scene.tif")
                False
                >>> EarthEngineDriver.can_open("COPERNICUS/S2_SR_HARMONIZED")
                False

                ```
        """
        return isinstance(source, str) and source.lower().startswith(_SCHEME)

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
        """Read a single Earth Engine image into a :class:`~pyramids.dataset.Dataset`.

        Args:
            source: An ``ee://<asset-id>`` string.
            bbox: Optional AOI ``(min_x, min_y, max_x, max_y)`` for the output window.
            bands: Optional band subset.
            crs: Optional target CRS for the output window.
            options: Extra :func:`~pyramids_eo.from_earthengine` keywords (e.g. ``reducer``,
                ``start``, ``end``, ``tile_size``, ``path``) as a dict or ``KEY=VALUE`` list.
                Because ``Dataset.read_file`` forwards only ``options`` to a driver, the window,
                band, and credential inputs may also be given here as ``bbox``/``crs``/``window``,
                ``bands``, and ``credentials`` (or ``auth``) keys; an explicit keyword below wins
                over the same key in ``options``, and each is consumed so it cannot collide.
            auth: Credentials for the Earth Engine read, passed as ``credentials=``.

        Returns:
            The windowed image.
        """
        from pyramids_eo import Window, from_earthengine

        settings = options_as_dict(options)
        # Pop every reserved key unconditionally so a leftover can never collide with the matching
        # from_earthengine keyword below; then let an explicit read keyword win over the option.
        opt_bbox = settings.pop("bbox", None)
        opt_crs = settings.pop("crs", None)
        opt_bands = settings.pop("bands", None)
        opt_window = settings.pop("window", None)
        opt_cred = settings.pop("credentials", None)
        opt_auth = settings.pop("auth", None)
        bbox = bbox if bbox is not None else opt_bbox
        crs = crs if crs is not None else opt_crs
        bands = bands if bands is not None else opt_bands
        if auth is None:
            auth = opt_cred if opt_cred is not None else opt_auth
        window = opt_window
        if window is None:
            window = (
                Window(bbox=bbox, crs=crs)
                if bbox is not None or crs is not None
                else Window()
            )
        return from_earthengine(
            _strip_scheme(str(source)),
            window=window,
            bands=bands,
            credentials=auth,
            **settings,
        )

    def read_collection(
        self, source: Any, *, time: Any = None, **kwargs: Any
    ) -> DatasetCollection:
        """Read an Earth Engine ``ImageCollection`` into a temporal ``DatasetCollection``.

        pyramids 0.70's :meth:`Dataset.read_file` dispatches only :meth:`read` (single image) —
        it has no collection path — so this method is reached by a direct call, and is here for
        that use and for forward-compatibility if pyramids adds collection dispatch.

        Args:
            source: An ``ee://<asset-id>`` string.
            time: Optional ``(start, end)`` date pair; an alternative to passing ``start`` and
                ``end`` in ``kwargs``. Must be a two-element sequence.
            **kwargs: :func:`~pyramids_eo.collection_from_earthengine` keywords — ``start``,
                ``end`` (both required), ``window``, ``bands``, and ``credentials`` (or ``auth``).
                An explicit ``start``/``end`` keyword takes precedence over the matching element
                of ``time``, and an explicit ``credentials`` takes precedence over ``auth``.

        Returns:
            One aligned timestep per scene.

        Raises:
            ValueError: ``time`` is not a two-element pair, or the resolved ``start``/``end`` are
                not both given and non-``None``.
        """
        from pyramids_eo import collection_from_earthengine

        settings = dict(kwargs)
        if time is not None:
            if not isinstance(time, (tuple, list)) or len(time) != 2:
                raise ValueError(
                    f"time must be a (start, end) pair, got {time!r}; pass a two-element "
                    "sequence or start=… and end=… instead."
                )
            start, end = time
            settings.setdefault(
                "start", start
            )  # an explicit start= keyword wins over time
            settings.setdefault("end", end)
        auth = settings.pop("auth", None)
        if auth is not None:
            settings.setdefault(
                "credentials", auth
            )  # an explicit credentials= wins over auth
        if settings.get("start") is None or settings.get("end") is None:
            raise ValueError(
                "an Earth Engine collection read needs a date range: pass time=(start, end) "
                "or start=… and end=… (both non-None)."
            )
        return collection_from_earthengine(_strip_scheme(str(source)), **settings)
