"""The Earth Engine raster driver for the ``pyramids.raster_drivers`` plugin hook.

Registers Earth Engine assets so ``Dataset.read_file("ee://<asset-id>", driver="earthengine")``
(or a bare ``read_file("ee://…")``, which GDAL cannot open and so falls through to this driver)
routes into :func:`pyramids_eo.from_earthengine`. An asset id has no file and no suffix, so the
driver claims the ``ee://`` URL scheme rather than an extension; a bare asset id with no scheme
is ambiguous with a GDAL path and stays on the typed :func:`~pyramids_eo.from_earthengine`
facade.

The heavy ``earthengine-api`` backend (imported as ``ee``) is named in :attr:`extra`, so
discovery skips this driver with a warning when it is not installed, and the backend is imported
inside :meth:`read` — never at module load — so a bare pyramids-eo install discovers the driver
without importing it.
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
        extra: ``"ee"`` — the import name of the optional ``earthengine-api`` backend.
    """

    name = "earthengine"
    long_name = "Google Earth Engine"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes: tuple[str, ...] = ()
    schemes: tuple[str, ...] = ("ee",)
    capabilities = frozenset({"read", "read_collection", "needs_auth"})
    extra: str | None = "ee"

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
            auth: Credentials for the Earth Engine read, passed as ``credentials=``.

        Returns:
            The windowed image.
        """
        from pyramids_eo import Window, from_earthengine

        window_kwargs: dict[str, Any] = {}
        if bbox is not None:
            window_kwargs["bbox"] = bbox
        if crs is not None:
            window_kwargs["crs"] = crs
        window = Window(**window_kwargs)
        return from_earthengine(
            _strip_scheme(str(source)),
            window=window,
            bands=bands,
            credentials=auth,
            **options_as_dict(options),
        )

    def read_collection(
        self, source: Any, *, time: Any = None, **kwargs: Any
    ) -> DatasetCollection:
        """Read an Earth Engine ``ImageCollection`` into a temporal ``DatasetCollection``.

        Args:
            source: An ``ee://<asset-id>`` string.
            time: Optional ``(start, end)`` date pair; an alternative to passing ``start`` and
                ``end`` in ``kwargs``.
            **kwargs: :func:`~pyramids_eo.collection_from_earthengine` keywords — ``start``,
                ``end`` (both required), ``window``, ``bands``, and ``credentials`` (or ``auth``).

        Returns:
            One aligned timestep per scene.

        Raises:
            ValueError: Neither ``time`` nor both ``start`` and ``end`` were given.
        """
        from pyramids_eo import collection_from_earthengine

        settings = dict(kwargs)
        if time is not None:
            settings.setdefault("start", time[0])
            settings.setdefault("end", time[1])
        if "auth" in settings:
            settings.setdefault("credentials", settings.pop("auth"))
        if "start" not in settings or "end" not in settings:
            raise ValueError(
                "an Earth Engine collection read needs a date range: pass time=(start, end) "
                "or start=… and end=…."
            )
        return collection_from_earthengine(_strip_scheme(str(source)), **settings)
