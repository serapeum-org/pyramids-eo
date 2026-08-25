"""Turnkey Sentinel-2 reader → a pyramids ``Dataset``.

:func:`from_sentinel2` mirrors the shape of
:func:`pyramids_eo.from_earthengine`: one flat-keyword call that opens a
Sentinel-2 product, picks the requested bands at a resolution, optionally
converts DN → reflectance (as lazy scale/offset tags), masks cloud/shadow via
the Scene-Classification band, and crops / reprojects — returning a plain
pyramids ``Dataset`` with no Sentinel objects leaking out.

Built entirely on pyramids-gis 0.55+ APIs: ``Dataset.subdatasets`` /
``open_subdataset`` for the catalog, ``Dataset.bands.select`` for band subsetting
(1-based, or by name), ``Dataset.read_array(scaled=True)`` for the lazy
reflectance read the scale/offset tags drive, and ``crop`` / ``to_crs`` for the
spatial ops.
"""

from __future__ import annotations

# isort: off
import pyramids as _pyramids_bootstrap  # noqa: F401  (activates the bundled osgeo)

# isort: on

from pathlib import Path
from typing import Any

from pyramids_eo.errors import ProductError
from pyramids_eo.sentinel.product import open_product
from pyramids_eo.sentinel.s2 import masks as _masks
from pyramids_eo.sentinel.s2 import scaling as _scaling
from pyramids_eo.sentinel.s2.product import (
    S2Product,
    S2Subdataset,
    _has_band,
    _normalise_band,
    is_spectral_band,
)

BBox = tuple[float, float, float, float]


def from_sentinel2(  # NOSONAR(S107) - flat keyword reader API mirroring from_earthengine
    path: str | Path | S2Product,
    *,
    bands: list[str] | None = None,
    resolution: int | None = None,
    epsg: int | None = None,
    bbox: BBox | None = None,
    crs: str | int | None = None,
    reflectance: bool = True,
    mask_scl: list[Any] | None = None,
    path_out: str | Path | None = None,
) -> Any:
    """Read a Sentinel-2 product into a pyramids ``Dataset``.

    Args:
        path: A product path (``.SAFE`` / ``MTD_*.xml`` / ``.zip``) or an
            already-open :class:`S2Product`.
        bands: Band names to read (e.g. ``["B04", "B08"]``; ``B4`` == ``B04``).
            ``None`` reads every spectral band at the chosen resolution.
        resolution: Native resolution in metres (10 / 20 / 60). ``None`` picks
            the finest resolution whose subdataset carries all requested bands.
        epsg: UTM EPSG code, required only for a multi-zone product.
        bbox: Optional crop window ``(minx, miny, maxx, maxy)`` in the output CRS.
        crs: Optional target CRS (EPSG int or WKT/……) to reproject to.
        reflectance: When ``True`` (default), tag the spectral bands so
            ``read_array(scaled=True)`` yields reflectance (DN / quantification
            + baseline offset). The returned array stays DN until read scaled.
        mask_scl: L2A only — classes to mask out (see
            :class:`~pyramids_eo.sentinel.s2.masks.SclClass`); masked pixels
            become no-data.
        path_out: When given, also write the result there (COG by extension).

    Returns:
        A pyramids ``Dataset`` of the requested bands.

    Raises:
        ProductError: The product is missing a requested band / resolution, or
            a multi-zone product is read without ``epsg``.

    Examples:
        - Read the red and NIR bands as reflectance:
            ```python
            >>> from pyramids_eo.sentinel import from_sentinel2  # doctest: +SKIP
            >>> ds = from_sentinel2("S2A_..._MSIL2A.SAFE",       # doctest: +SKIP
            ...                     bands=["B04", "B08"])        # doctest: +SKIP
            >>> reflectance = ds.read_array(scaled=True)         # doctest: +SKIP

            ```
    """
    product = path if isinstance(path, S2Product) else open_product(path)
    if not isinstance(
        product, S2Product
    ):  # pragma: no cover - open_product guarantees it
        raise ProductError(f"{path!r} is not a Sentinel-2 product")

    epsg = _resolve_epsg(product, epsg)
    wanted = _resolve_bands(product, bands, resolution, epsg)
    target_res = (
        resolution
        if resolution is not None
        else _finest_resolution_for(product, wanted, epsg)
    )
    subdataset = product.subdataset_for(target_res, epsg)
    _check_bands_present(subdataset, wanted)

    dataset = _select_bands(subdataset, wanted)
    _set_nodata(dataset, product)

    if reflectance:
        dataset = _scaling.tag_reflectance(dataset, product)
    if mask_scl:
        dataset = _apply_scl_mask(dataset, product, target_res, epsg, mask_scl)
    if bbox is not None:
        dataset = dataset.crop(list(bbox))
    if crs is not None:
        dataset = dataset.to_crs(crs)
    if path_out is not None:
        dataset.to_file(str(path_out))
    return dataset


# -- resolution / band planning -------------------------------------------


def _resolve_epsg(product: S2Product, epsg: int | None) -> int | None:
    """Return the EPSG to read, defaulting to the sole zone of the product."""
    zones = product.epsg_codes
    if epsg is not None:
        return epsg
    if len(zones) == 1:
        return zones[0]
    if len(zones) == 0:
        return None
    raise ProductError(f"product spans UTM zones {zones}; pass epsg= to choose one.")


def _resolve_bands(
    product: S2Product,
    bands: list[str] | None,
    resolution: int | None,
    epsg: int | None,
) -> list[str]:
    """Return the requested band names, defaulting to all spectral bands."""
    if bands:
        return list(bands)
    if resolution is not None:
        sd = product.subdataset_for(resolution, epsg)
        return [b for b in sd.bands if is_spectral_band(b)]
    return sorted(
        {
            b
            for sd in product.image_subdatasets
            for b in sd.bands
            if is_spectral_band(b)
        },
        key=_normalise_band,
    )


def _finest_resolution_for(
    product: S2Product, wanted: list[str], epsg: int | None
) -> int:
    """Finest resolution whose single subdataset carries every wanted band.

    Raises:
        ProductError: No single resolution carries all the requested bands
            (cross-resolution harmonisation is a planned follow-up).
    """
    for res in product.resolutions:  # ascending
        try:
            sd = product.subdataset_for(res, epsg)
        except ProductError:
            continue
        if all(_has_band(sd.bands, b) for b in wanted):
            return res
    spread = {b: product.resolution_of(b) for b in wanted}
    raise ProductError(
        "requested bands span resolutions "
        f"{spread}; pass an explicit resolution= that contains them all "
        "(cross-resolution harmonise is a planned follow-up)."
    )


def _check_bands_present(subdataset: S2Subdataset, wanted: list[str]) -> None:
    """Raise if any wanted band is absent from ``subdataset``."""
    missing = [b for b in wanted if not _has_band(subdataset.bands, b)]
    if missing:
        raise ProductError(
            f"bands {missing} not at {subdataset.resolution_m}m; "
            f"available there: {subdataset.bands}"
        )


# -- reads -----------------------------------------------------------------


def _select_bands(subdataset: S2Subdataset, wanted: list[str]) -> Any:
    """Open ``subdataset`` and select ``wanted`` bands (1-based, arg order)."""
    ds = subdataset.open()
    indices = [_band_index(subdataset.bands, b) for b in wanted]
    return ds.bands.select(indices)


def _band_index(band_list: list[str], wanted: str) -> int:
    """1-based position of ``wanted`` in ``band_list`` (zero-pad tolerant)."""
    target = _normalise_band(wanted)
    for i, name in enumerate(band_list):
        if _normalise_band(name) == target:
            return i + 1
    raise ProductError(f"band {wanted!r} not found in {band_list}")


def _set_nodata(dataset: Any, product: S2Product) -> None:
    """Set the S2 DN no-data value (``SPECIAL_VALUE_NODATA`` or 0) on ``dataset``."""
    raw = product.metadata.get("SPECIAL_VALUE_NODATA")
    try:
        nodata = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        nodata = 0.0
    try:
        dataset.no_data_value = [nodata] * dataset.band_count
    except Exception:  # noqa: BLE001 - nodata is advisory; don't fail the read
        pass


def _apply_scl_mask(
    dataset: Any,
    product: S2Product,
    target_res: int,
    epsg: int | None,
    classes: list[Any],
) -> Any:
    """Read SCL (aligning it to ``dataset``'s grid) and mask ``dataset``."""
    scl_sd = _scl_subdataset(product, epsg)
    if scl_sd is None:
        raise ProductError("no SCL band in this product (SCL is Level-2A only)")
    scl_ds = _select_bands(scl_sd, ["SCL"])
    if scl_sd.resolution_m != target_res:
        # Align the coarser/finer SCL grid onto the data grid (nearest — SCL is
        # categorical). harmonise returns the aligned band.
        from pyramids_eo.readers.harmonise import harmonise

        scl_ds = harmonise([scl_ds], dataset, method="nearest")[0]
    return _masks.scl_mask(dataset, classes, scl=scl_ds)


def _scl_subdataset(product: S2Product, epsg: int | None) -> S2Subdataset | None:
    """Finest image subdataset that carries an ``SCL`` band, or ``None``."""
    candidates = [
        sd
        for sd in product.image_subdatasets
        if (epsg is None or sd.epsg == epsg) and _has_band(sd.bands, "SCL")
    ]
    return min(candidates, key=lambda sd: sd.resolution_m) if candidates else None
