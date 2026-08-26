"""Product model: a Sentinel product is a *catalog of rasters*, not a raster.

Opening a Sentinel ``.SAFE`` (or its metadata XML / ``.zip``) with GDAL yields a
**0-band** dataset whose payload is a set of subdatasets — one per resolution ×
UTM zone for Sentinel-2, one per swath × polarisation × calibration for
Sentinel-1. `Dataset` models a raster; it cannot model that catalog. So the
product is its own small class family:

* :class:`SentinelProduct` — the abstract container (path, driver, mission,
  product type, raw metadata, the subdataset list).
* :func:`open_product` — sniffs the GDAL driver behind a path and returns the
  right concrete product (currently :class:`~pyramids_eo.sentinel.s2.S2Product`).

Nothing here subclasses `Dataset`: the readers *return* plain pyramids `Dataset`
objects, and the product only *describes* what can be read (see the
pyramids-eo/earthengine precedent — no engine objects leak into public types).
"""

from __future__ import annotations

# isort: off
import pyramids as _pyramids_bootstrap  # noqa: F401  (activates the bundled osgeo)

# isort: on

import abc
from pathlib import Path
from typing import Any

from pyramids.dataset import Dataset

from pyramids_eo.errors import ProductError, UnsupportedProductError

#: GDAL driver short names this package can model, mapped to a product family.
_DRIVER_FAMILY: dict[str, str] = {
    "SENTINEL2": "s2",
    "SAFE": "s1",
}


class SentinelProduct(abc.ABC):
    """A Sentinel product on disk — a catalog of openable rasters.

    Concrete subclasses (:class:`~pyramids_eo.sentinel.s2.S2Product`, and a
    future ``S1Product``) add the mission-specific model. This base holds only
    what every Sentinel product shares.

    Attributes:
        path: The path (or connection root) the product was opened from.
        driver: The GDAL driver short name (``"SENTINEL2"`` / ``"SAFE"``).
        metadata: The container's default-domain metadata (``Dataset.meta_data``).
        subdatasets: The pyramids ``SubDataset`` value objects the container
            exposes, each openable with ``.open()``.
    """

    def __init__(self, path: str, container: Dataset) -> None:
        self.path = path
        self.driver = container.raster.GetDriver().ShortName
        self.metadata: dict[str, str] = dict(container.meta_data or {})
        self.subdatasets: list[Any] = list(container.subdatasets)

    @property
    @abc.abstractmethod
    def mission(self) -> str:
        """Mission identifier, e.g. ``"sentinel-2"``."""

    @property
    @abc.abstractmethod
    def product_type(self) -> str:
        """Product type token, e.g. ``"S2MSI2A"`` / ``"GRD"``."""

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"{type(self).__name__}(driver={self.driver!r}, "
            f"product_type={self.product_type!r}, subdatasets={len(self.subdatasets)})"
        )


#: Product-level metadata filenames, most specific first. Matches the S2
#: compact (``MTD_MSIL2A.xml``) and legacy (``S2A_…_MTD_SAFL2A.xml``) namings and
#: the S1 ``manifest.safe`` — but not the deeper granule/datastrip MTDs.
_METADATA_GLOBS = ("MTD_MSIL*.xml", "*_MTD_SAF*.xml", "manifest.safe")


def _resolve_container_path(path: str | Path) -> str:
    """Return a GDAL-openable container path.

    GDAL opens a Sentinel product through its **metadata file**, not the
    ``.SAFE`` directory (opening the directory fails). So a directory or a
    ``.zip`` is resolved to the product-level metadata inside it; a metadata
    file / ``manifest.safe`` / driver connection string is returned unchanged.
    A ``.zip`` is read in place via ``/vsizip/`` — no extraction to disk.

    Args:
        path: The product path, ``.SAFE`` directory, or ``.zip`` holding one.

    Returns:
        A path (possibly ``/vsizip/…``) GDAL can open as the container.

    Raises:
        ProductError: No product metadata is found inside a directory / archive.
    """
    text = str(path)
    if text.lower().endswith(".zip"):
        return _resolve_in_zip(text)
    candidate = Path(text)
    if candidate.is_dir():
        return _find_metadata_in_dir(candidate)
    return text


def open_connection(connection: str) -> Dataset:
    """Open a GDAL driver connection string as a pyramids ``Dataset``.

    Prefer ``Dataset.read_file``, but a connection string whose source is a
    ``/vsizip/`` (or other ``/vsi``) path — e.g. a ``SubDataset.name`` from a
    zipped product, ``SENTINEL2_L2A:/vsizip/…:60m:EPSG_32632`` — trips pyramids'
    archive auto-detection: it sees the embedded ``.zip`` and re-wraps the whole
    string in a second ``/vsizip/``, which then fails to open. Those are opened
    through GDAL directly and wrapped, bypassing the re-sniff.

    Args:
        connection: A GDAL connection string / path.

    Returns:
        The opened :class:`~pyramids.dataset.Dataset`.

    Raises:
        ProductError: GDAL cannot open the connection.
    """
    if "/vsi" in connection:
        from osgeo import gdal

        try:
            handle = gdal.Open(connection)
        except RuntimeError as exc:  # UseExceptions turns a bad open into a raise
            raise ProductError(f"GDAL could not open {connection!r}: {exc}") from exc
        if handle is None:
            raise ProductError(f"GDAL could not open {connection!r}")
        return Dataset(handle)
    return Dataset.read_file(connection)


def _find_metadata_in_dir(directory: Path) -> str:
    """Return the product-level metadata file inside a ``.SAFE`` directory."""
    for pattern in _METADATA_GLOBS:
        matches = sorted(directory.glob(pattern))
        if matches:
            return str(matches[0])
    raise ProductError(
        f"no product metadata ({', '.join(_METADATA_GLOBS)}) found in {directory}"
    )


def _resolve_in_zip(zip_path: str) -> str:
    """Return the ``/vsizip/`` path of the product metadata inside ``zip_path``.

    ``zipfile.namelist`` gives the *full recursive* member list (pyramids'
    ``archive_members`` is top-level only), so the nested product MTD is found;
    ``_io.archive_dir_vsi`` supplies the platform-correct ``/vsizip/`` prefix.
    """
    import zipfile

    from pyramids import _io  # local import: archive helpers are private-by-name

    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
    vsi_dir = _io.archive_dir_vsi(zip_path, "auto")
    # Prefer a shallow (product-level) metadata member over a deep granule one.
    for pattern in _METADATA_GLOBS:
        hits = sorted(
            (m for m in members if _fnmatch_base(m, pattern)),
            key=lambda m: m.count("/"),
        )
        if hits:
            return f"{vsi_dir}/{hits[0]}"
    raise ProductError(
        f"no Sentinel product metadata found inside archive: {zip_path!r}"
    )


def _fnmatch_base(member: str, pattern: str) -> bool:
    """Case-insensitive fnmatch of an archive member's basename against ``pattern``."""
    import fnmatch

    return fnmatch.fnmatch(member.rsplit("/", 1)[-1].lower(), pattern.lower())


def open_product(path: str | Path) -> SentinelProduct:
    """Open a Sentinel product and return its typed model.

    Sniffs the GDAL driver behind ``path`` and dispatches to the concrete
    product class. Accepts a ``.SAFE`` directory, a product metadata XML, a
    ``manifest.safe``, a ``.zip`` holding a ``.SAFE`` (opened in place), or a
    driver connection string.

    Args:
        path: The product to open.

    Returns:
        A :class:`SentinelProduct` — currently an
        :class:`~pyramids_eo.sentinel.s2.S2Product` for Sentinel-2.

    Raises:
        UnsupportedProductError: The path opens, but its driver is not a
            Sentinel product family this package models.
        ProductError: The path cannot be opened as a product.

    Examples:
        - Open a Level-2A product and inspect its subdatasets:
            ```python
            >>> from pyramids_eo.sentinel import open_product  # doctest: +SKIP
            >>> product = open_product("S2A_..._MSIL2A.SAFE")   # doctest: +SKIP
            >>> product.product_type                            # doctest: +SKIP
            'S2MSI2A'

            ```
    """
    container_path = _resolve_container_path(path)
    try:
        container = Dataset.read_file(container_path)
    except Exception as exc:  # noqa: BLE001 - re-raise as the package's error
        raise ProductError(f"cannot open product {path!r}: {exc}") from exc

    driver = container.raster.GetDriver().ShortName
    family = _DRIVER_FAMILY.get(driver)
    if family == "s2":
        from pyramids_eo.sentinel.s2.product import S2Product

        return S2Product(str(path), container)
    if family == "s1":
        raise UnsupportedProductError(
            f"Sentinel-1 ({driver}) products are not yet modelled "
            "(planned as Phase 2); Sentinel-2 is supported today."
        )
    raise UnsupportedProductError(
        f"{driver!r} is not a Sentinel product driver this package models "
        f"(known: {sorted(_DRIVER_FAMILY)})."
    )
