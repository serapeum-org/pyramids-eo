"""The Sentinel-2 driver for the ``pyramids.raster_drivers`` plugin hook.

Registers Sentinel-2 ``.SAFE`` products so ``Dataset.read_file`` routes into
:func:`pyramids_eo.from_sentinel2` (reflectance, SCL masking, band selection) instead of GDAL's
own container view.

How each form reaches this driver differs, because GDAL ships its own ``SENTINEL2`` / ``SAFE``
drivers that compete for the same paths:

* a product **directory** arrives through the post-GDAL ``can_open`` fallback — GDAL cannot open
  one at all, so a bare ``read_file("<product>.SAFE")`` already reads the product.
* a **zipped** product is the one form GDAL *does* open, handing back an empty container. This
  driver therefore declares ``preempts = (".zip",)`` — the opt-in pyramids-core 0.71 added
  (serapeum-org/pyramids-core#1344) — so a ``.SAFE.zip`` is routed here *before* GDAL.
* an explicit ``driver="s2"`` short-circuits ahead of both.

ESA shares ``.SAFE`` (and ``.SAFE.zip``) across missions, so the suffix alone cannot say whether a
product is Sentinel-2 — a Sentinel-1 product carries the same one, and GDAL reads it correctly.
``can_open`` therefore identifies the mission from the product's **contents**, which is what keeps
a Sentinel-1 product with GDAL instead of failing in this reader.

:func:`~pyramids_eo.from_sentinel2` reads a single ``.SAFE``, so no source expansion is needed.
"""

from __future__ import annotations

import fnmatch
import os
import zipfile
from typing import Any

from pyramids.dataset import Dataset

from pyramids_eo.drivers._base import RASTER_DRIVER_API_VERSION, options_as_dict

#: Top-level names that identify a Sentinel-2 product. ESA shares the ``.SAFE`` suffix across
#: missions, so the suffix alone cannot tell a Sentinel-2 product from a Sentinel-1 one — these
#: metadata files can. A Sentinel-1 product carries none of them (it has ``annotation/`` and
#: ``measurement/`` beside a ``manifest.safe``). Mirrors the globs
#: :mod:`pyramids_eo.sentinel.product` already uses to find a product's metadata.
_S2_MARKERS: tuple[str, ...] = ("MTD_MSIL*.xml", "*_MTD_SAF*.xml")


def _holds_s2_metadata(names: Any) -> bool:
    """Return whether any of ``names`` is a Sentinel-2 metadata file.

    Args:
        names: Iterable of file names, each compared against :data:`_S2_MARKERS`
            case-insensitively and by base name only.

    Returns:
        bool: True when at least one name identifies a Sentinel-2 product.
    """
    for name in names:
        base = name.rstrip("/").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if any(fnmatch.fnmatch(base.upper(), glob.upper()) for glob in _S2_MARKERS):
            return True
    return False


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
        preempts: ``(".zip",)`` — a zipped product is tried *before* GDAL, which would
            otherwise open the archive and return an empty container. pyramids tokenises a
            source's **final** suffix, so a ``.SAFE.zip`` presents as ``.zip``; ``".safe.zip"``
            could never match. Declaring the bare ``".zip"`` is safe because ``can_open`` gates
            every pre-empted claim, so an ordinary archive is left to GDAL. ``".safe"`` is
            **not** declared: ESA shares that suffix with Sentinel-1, which GDAL reads itself, and
            pre-empting it would route those products into this reader.
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
    #: Tried before GDAL for a zipped product; ``can_open`` keeps ordinary archives out, and
    #: ``.safe`` is excluded because Sentinel-1 shares it. See the class docstring.
    preempts: tuple[str, ...] = (".zip",)
    capabilities = frozenset({"read"})
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return whether ``source`` is a Sentinel-2 ``.SAFE`` product.

        Args:
            source: The source handed by ``Dataset.read_file`` (stringified).

        Returns:
            True only for a Sentinel-2 product. The ``.SAFE`` suffix is **not** sufficient: ESA
            shares it with Sentinel-1, which GDAL reads itself, so claiming by suffix alone would
            take those products and fail on them. When the source exists this therefore looks
            inside it for Sentinel-2 metadata (:data:`_S2_MARKERS`) — a directory listing, or a
            zip's index. A ``manifest.safe`` is never claimed: it is the metadata file *inside* a
            product and Sentinel-1's entry point. A zipped product is claimed only when its
            extension is a lowercase ``.zip``, which is what the archive reader downstream
            accepts. A ``.SAFE`` path that does not exist is claimed, so the reader can raise its
            own clear error rather than GDAL's.

        Note:
            This is no longer free of I/O: identifying the mission requires reading a directory
            listing or a zip's central directory. The cost is bounded — pyramids only calls
            ``can_open`` once the suffix or scheme already matched — and it buys the mission
            discrimination that a filename cannot give.

        Examples:
            - A plain raster, a bare archive and a manifest are all rejected without I/O:
                ```python
                >>> from pyramids_eo.drivers.sentinel2 import Sentinel2Driver
                >>> Sentinel2Driver.can_open("scene.tif")
                False
                >>> Sentinel2Driver.can_open("archive.zip")
                False
                >>> Sentinel2Driver.can_open("S2A_MSIL2A.SAFE/manifest.safe")
                False

                ```
            - A `.SAFE` path that is not on disk is claimed, so the reader reports the problem:
                ```python
                >>> from pyramids_eo.drivers.sentinel2 import Sentinel2Driver
                >>> Sentinel2Driver.can_open("no_such_product.SAFE")
                True

                ```
        """
        if not isinstance(source, str):
            return False
        trimmed = source.rstrip("/\\")
        lowered = trimmed.lower()
        base = lowered.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if base == "manifest.safe":
            return False
        if lowered.endswith(".safe.zip"):
            # Only a lowercase `.zip` is claimed: the archive reader infers the kind from the
            # extension and rejects `.ZIP`, so claiming one would route a read it cannot serve.
            if not trimmed.endswith(".zip"):
                return False
            if not os.path.exists(trimmed):
                # Nothing to inspect: claim it so the reader reports the problem, which
                # matches how a missing `.SAFE` directory is handled below.
                return True
            try:
                with zipfile.ZipFile(trimmed) as archive:
                    return _holds_s2_metadata(archive.namelist())
            except (OSError, zipfile.BadZipFile, ValueError):
                return False
        if not lowered.endswith(".safe"):
            return False
        try:
            if not os.path.isdir(trimmed):
                # Not on disk (or not a directory): let the reader raise the clear error.
                return not os.path.exists(trimmed)
            return _holds_s2_metadata(os.listdir(trimmed))
        except OSError:
            return False

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
