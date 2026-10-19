"""The Sentinel-2 driver for the ``pyramids.raster_drivers`` plugin hook.

Registers Sentinel-2 ``.SAFE`` products so ``Dataset.read_file(path, driver="s2")`` routes into
:func:`pyramids_eo.from_sentinel2` (reflectance, SCL masking, band selection) instead of GDAL's
own container view.

GDAL ships a ``SENTINEL2`` driver and *opens* a ``.SAFE`` itself, so on the plain
``read_file(path)`` path GDAL wins and returns a bare container — this driver is reached only via
an explicit ``driver="s2"``. A bare-path auto-route awaits pyramids letting a driver pre-empt GDAL
for a suffix it claims (serapeum-org/pyramids#1344); ``can_open`` already claims ``.SAFE`` so it
will work the moment that lands, and it still catches the rare case where GDAL fails on a
``.SAFE``. :func:`~pyramids_eo.from_sentinel2` reads a single ``.SAFE``, so no source expansion is
needed.
"""

from __future__ import annotations

from typing import Any

from pyramids.dataset import Dataset

from pyramids_eo.drivers._base import RASTER_DRIVER_API_VERSION, options_as_dict


class Sentinel2Driver:
    """Read a Sentinel-2 ``.SAFE`` product into a :class:`~pyramids.dataset.Dataset`.

    Attributes:
        name: The driver's registry name and the value accepted by ``read_file(driver=…)``.
        long_name: Human-readable source name for ``describe_drivers``.
        api_version: The driver-contract version (see
            :data:`~pyramids_eo.drivers._base.RASTER_DRIVER_API_VERSION`).
        suffixes: ``(".safe", ".safe.zip")`` — the driver claims a Sentinel-2 product directory or
            a zipped product.
        schemes: Empty — a ``.SAFE`` is a local product.
        capabilities: ``read`` only (one product is one timestep; a time series goes through
            :func:`pyramids_eo.collection_from_sentinel2`).
        extra: ``None`` — the reader is pure pyramids-eo over pyramids' bundled GDAL, with no
            optional backend to gate on.
    """

    name = "s2"
    long_name = "Sentinel-2 (.SAFE)"
    api_version = RASTER_DRIVER_API_VERSION
    suffixes: tuple[str, ...] = (".safe", ".safe.zip")
    schemes: tuple[str, ...] = ()
    capabilities = frozenset({"read"})
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return whether ``source`` is a Sentinel-2 ``.SAFE`` product.

        Args:
            source: The source handed by ``Dataset.read_file`` (stringified).

        Returns:
            True for a string ending ``.SAFE`` or ``.SAFE.zip`` (case-insensitive, trailing slash
            tolerated), else False. Cheap: no import, no file access, no network.

        Examples:
            - A ``.SAFE`` product (zipped or not) is claimed; a plain raster or bare zip is not:
                ```python
                >>> from pyramids_eo.drivers.sentinel2 import Sentinel2Driver
                >>> Sentinel2Driver.can_open("S2A_MSIL2A.SAFE")
                True
                >>> Sentinel2Driver.can_open("S2A_MSIL2A.SAFE.zip")
                True
                >>> Sentinel2Driver.can_open("scene.tif")
                False

                ```
        """
        if not isinstance(source, str):
            return False
        lowered = source.rstrip("/\\").lower()
        return lowered.endswith((".safe", ".safe.zip"))

    def read(
        self,
        source: Any,
        *,
        bbox: Any = None,
        bands: Any = None,
        crs: Any = None,
        options: Any = None,
        auth: Any = None,
        **_ignored: Any,
    ) -> Dataset:
        """Read a Sentinel-2 product into a :class:`~pyramids.dataset.Dataset`.

        Args:
            source: Path to a ``.SAFE`` product directory.
            bbox: Optional AOI ``(min_x, min_y, max_x, max_y)``.
            bands: Optional band subset (e.g. ``["B04", "B03", "B02"]``).
            crs: Optional target CRS.
            options: Extra :func:`~pyramids_eo.from_sentinel2` keywords (e.g. ``resolution``,
                ``epsg``, ``reflectance``, ``mask_scl``, ``resample``) as a dict or ``KEY=VALUE``
                list. Because ``Dataset.read_file`` forwards only ``options`` to a driver,
                ``bbox`` / ``bands`` / ``crs`` may also be given here; a non-``None`` explicit
                keyword wins over the same key in ``options``, and each is consumed so it cannot
                collide. A ``KEY=VALUE`` list keeps every value a **string**, so an option that the
                reader needs typed — ``resolution`` / ``epsg`` as ``int`` — must be passed in the
                dict form (``{"resolution": 10}``), not as ``["resolution=10"]``.
            auth: Ignored — a local ``.SAFE`` needs no credentials.
            **_ignored: Any other generic driver-read parameter, accepted and ignored.

        Returns:
            The read Sentinel-2 dataset.
        """
        from pyramids_eo import from_sentinel2

        settings = options_as_dict(options)
        opt_bbox = settings.pop("bbox", None)
        opt_bands = settings.pop("bands", None)
        opt_crs = settings.pop("crs", None)
        settings.pop("auth", None)
        bbox = bbox if bbox is not None else opt_bbox
        bands = bands if bands is not None else opt_bands
        crs = crs if crs is not None else opt_crs
        return from_sentinel2(str(source), bands=bands, bbox=bbox, crs=crs, **settings)
