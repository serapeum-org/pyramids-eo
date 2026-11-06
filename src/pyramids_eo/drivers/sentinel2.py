"""The Sentinel-2 driver for the ``pyramids.raster_drivers`` plugin hook.

Registers Sentinel-2 ``.SAFE`` products so ``Dataset.read_file`` routes into
:func:`pyramids_eo.from_sentinel2` (reflectance, SCL masking, band selection) instead of GDAL's
own container view.

GDAL ships its own ``SENTINEL2`` / ``SAFE`` drivers, so which reader serves a path depends on
whether GDAL can open it:

* a product **directory** reaches this driver through the post-GDAL ``can_open`` fallback — GDAL
  cannot open one, so a bare ``read_file("<product>.SAFE")`` already reads the product.
* a **zipped** product is opened by GDAL, which would return a bare container. This driver
  therefore declares ``preempts = (".zip",)`` — the opt-in pyramids-core 0.71 added
  (serapeum-org/pyramids-core#1344) — so a ``.SAFE.zip`` is routed here *before* GDAL and a bare
  ``read_file("<product>.SAFE.zip")`` reads the product too.

  Only ``.zip`` is declared, not ``.safe.zip``: pyramids matches a source's **final** suffix, so a
  zipped product presents as ``.zip`` and ``".safe.zip"`` could never match. The bare ``.zip`` is
  safe to declare because ``can_open`` gates every pre-empted claim, and ``.safe`` is not declared
  because a directory needs no help.

  Reading a whole product is expensive — 13 bands harmonised onto a 20984x20980 10 m grid, about
  10 GiB — so pass the ``bands`` and ``resolution`` you want. Without them the reader is asked for
  everything at the finest resolution, which takes minutes: on a fake L1C fixture the zipped form
  returned after 228 s and the directory form raised ``MemoryError`` after 202 s. GDAL's container
  returned in about 0.1 s, with nothing useful in it.

ESA shares ``.SAFE`` (and ``.SAFE.zip``) across missions, so the suffix alone cannot say whether a
product is Sentinel-2 — a Sentinel-1 product carries the same one, and GDAL reads it correctly.
``can_open`` therefore identifies the mission from the product's **contents**, so a source this
driver cannot actually read is never claimed.

:func:`~pyramids_eo.from_sentinel2` reads a single ``.SAFE``, so no source expansion is needed.
"""

from __future__ import annotations

import os
import zipfile
from collections.abc import Iterable
from fnmatch import fnmatchcase
from typing import Any

from pyramids.dataset import Dataset

from pyramids_eo.drivers._base import RASTER_DRIVER_API_VERSION, options_as_dict

#: Top-level names that identify a Sentinel-2 product. ESA shares the ``.SAFE`` suffix across
#: missions, so the suffix alone cannot tell a Sentinel-2 product from a Sentinel-1 one — these
#: metadata files can. A Sentinel-1 product carries none of them (it has ``annotation/`` and
#: ``measurement/`` beside a ``manifest.safe``). These are the first two of the three globs
#: :mod:`pyramids_eo.sentinel.product` uses to locate a product's metadata; its third,
#: ``manifest.safe``, is deliberately left out here because every mission ships one, so it
#: cannot discriminate — which is also why ``can_open`` rejects a bare manifest outright.
_S2_MARKERS: tuple[str, ...] = ("MTD_MSIL*.xml", "*_MTD_SAF*.xml")


#: :data:`_S2_MARKERS` pre-upper-cased, so the comparison does not re-fold them per name.
_S2_MARKERS_UPPER: tuple[str, ...] = tuple(glob.upper() for glob in _S2_MARKERS)


def _holds_s2_metadata(names: Iterable[str]) -> bool:
    """Return whether ``names`` holds a Sentinel-2 metadata file at the product root.

    Args:
        names: Entry names to test — a directory listing, or a zip's index. A
            listing is already flat; an archive normally wraps the product in one
            folder, so an entry is considered only at depth 0 or 1. Anything
            deeper is *inside* the product and cannot identify it: a zipped
            Sentinel-1 product carrying a stray ``notes/backup/MTD_MSIL1C.xml``
            is not a Sentinel-2 product, and matching at any depth would claim
            it away from GDAL. Comparison is by base name, case-insensitively,
            and a folder entry (trailing separator) is skipped.

    Returns:
        bool: True when at least one entry identifies a Sentinel-2 product.
    """
    for name in names:
        normalised = name.replace("\\", "/")
        if normalised.endswith("/"):
            continue
        parts = [part for part in normalised.split("/") if part not in ("", ".")]
        if len(parts) > 2:
            continue
        base = parts[-1] if parts else ""
        if any(fnmatchcase(base.upper(), glob) for glob in _S2_MARKERS_UPPER):
            return True
    return False


def _is_plain_local_path(source: str) -> bool:
    """Return whether `source` is a plain local path this driver can inspect.

    A URL, a GDAL `/vsi*` virtual path or a UNC share cannot be sniffed with
    `os`/`zipfile`, and GDAL may well read it when pyramids-eo cannot. Those are
    left to GDAL rather than claimed on the strength of a suffix.

    Args:
        source: The already-trimmed source string.

    Returns:
        bool: True for a plain local path, False for a URL, `/vsi*` path or UNC
        share.
    """
    lowered = source.lower()
    if "://" in lowered or lowered.startswith("/vsi"):
        return False
    return not source.startswith(("//", "\\\\"))


class Sentinel2Driver:
    """Read a Sentinel-2 ``.SAFE`` product into a :class:`~pyramids.dataset.Dataset`.

    Attributes:
        name: The driver's registry name and the value accepted by ``read_file(driver=…)``.
        long_name: Human-readable source name for ``describe_drivers``.
        api_version: The driver-contract version (see
            :data:`~pyramids_eo.drivers._base.RASTER_DRIVER_API_VERSION`).
        suffixes: ``(".safe", ".safe.zip")`` — the two spellings a Sentinel-2 product comes in.
            Declaration only, and the second entry is descriptive rather than functional:
            pyramids matches a source's **final** suffix, so a zipped product presents as
            ``.zip`` and never as ``.safe.zip``. Routing is decided by :meth:`can_open`, and the
            pre-emption by ``preempts``.
        schemes: Empty — a ``.SAFE`` is a local product.
        preempts: ``(".zip",)`` — a zipped product is tried *before* GDAL, which would otherwise
            open the archive and hand back an empty container. ``".safe.zip"`` is not usable here
            (pyramids matches the final suffix, so a zipped product presents as ``.zip``), and the
            bare ``".zip"`` is safe because ``can_open`` gates every claim — an ordinary archive,
            and a zipped Sentinel-1 product, are both left to GDAL. ``".safe"`` is not declared:
            GDAL cannot open a product directory, so the fallback already serves it.
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
    #: Tried before GDAL for a zipped product; see the class docstring for why the bare
    #: ``.zip`` is safe and why ``.safe`` is absent.
    preempts: tuple[str, ...] = (".zip",)
    capabilities = frozenset({"read"})
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return whether ``source`` is a Sentinel-2 ``.SAFE`` product.

        Args:
            source: The source handed by ``Dataset.read_file`` (stringified) — a product
                directory or a lowercase ``*.SAFE.zip`` archive, both local.

        Returns:
            True for a source this driver should read, which differs by form because the two
            forms reach it by different routes.

            A **zipped** product is pre-empted *before* GDAL, so the claim has to be exact: the
            extension must be a lowercase ``.zip`` (what the archive reader accepts), and the
            archive must hold Sentinel-2 metadata (:data:`_S2_MARKERS`) at the product root.
            ESA shares ``.SAFE.zip`` with Sentinel-1, which GDAL reads itself, so a name alone
            must never win the claim.

            A **directory** is only reached through the post-GDAL fallback, where GDAL has
            already declined the source, so any ``.SAFE`` directory is claimed — no marker check,
            which would otherwise turn away a product whose root metadata is a bare
            ``manifest.safe``.

            In both forms a ``manifest.safe`` *as the source* is refused: it is the file inside a
            product and Sentinel-1's entry point. A **plain local** path that does not exist is
            claimed so the reader can report it, but a URL, a GDAL ``/vsi*`` path or a UNC share
            never is — this driver cannot read one and GDAL often can.

        Note:
            This is no longer free of I/O: identifying the mission requires reading a directory
            listing or a zip's central directory. Pre-emption consults only drivers whose
            ``preempts`` matches the source's suffix, but the post-GDAL fallback
            (``select_driver``) asks **every** registered driver, with no suffix filter — so this
            runs for any source GDAL failed to open. The cheap string tests come first and return
            before any I/O, so the filesystem is touched only for a source actually spelled
            ``*.SAFE`` or ``*.SAFE.zip``.

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
                # Nothing to inspect. Only shortcut a plain local path, where the reader
                # gives a better "no such product" message than GDAL. A URL, `/vsi*` or
                # UNC source is left to GDAL: this driver cannot read it, and claiming it
                # would pre-empt GDAL out of a read it can serve.
                return _is_plain_local_path(trimmed)
            try:
                with zipfile.ZipFile(trimmed) as archive:
                    return _holds_s2_metadata(archive.namelist())
            except (OSError, zipfile.BadZipFile, ValueError):
                return False
        if not lowered.endswith(".safe"):
            return False
        # A directory claim is NOT gated on the metadata markers. This branch is only
        # ever reached through the post-GDAL fallback — ``.safe`` is not in ``preempts``
        # — and GDAL reads a Sentinel-1 ``.SAFE`` directory itself, so by the time this
        # runs GDAL has already declined the source. Requiring a marker here only turned
        # away products whose root metadata is a bare ``manifest.safe``, which
        # ``from_sentinel2`` reads perfectly well, while buying no mission
        # discrimination that GDAL has not already provided.
        try:
            if not os.path.isdir(trimmed):
                # Not a directory on disk. A plain local path that is simply absent is
                # claimed so the reader reports it; anything remote or virtual is left to
                # GDAL, and an existing non-directory is not a product.
                return not os.path.exists(trimmed) and _is_plain_local_path(trimmed)
            return True
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
