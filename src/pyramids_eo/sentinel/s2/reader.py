"""Turnkey Sentinel-2 reader → a pyramids ``Dataset``.

:func:`from_sentinel2` mirrors the shape of
:func:`pyramids_eo.from_earthengine`: one flat-keyword call that opens a
Sentinel-2 product, picks the requested bands at a resolution, optionally
converts DN → reflectance (as lazy scale/offset tags), masks cloud/shadow via
the Scene-Classification band, and crops / reprojects — returning a plain
pyramids ``Dataset`` with no Sentinel objects leaking out.

Built entirely on pyramids-gis 0.56+ APIs: ``Dataset.subdatasets`` /
``open_subdataset`` for the catalog, ``Dataset.bands.select`` for band subsetting
(1-based, or by name), ``Dataset.read_array(scaled=True)`` for the lazy
reflectance read the scale/offset tags drive, and ``crop`` / ``to_crs`` for the
spatial ops.
"""

from __future__ import annotations

# isort: off
import pyramids as _pyramids_bootstrap  # noqa: F401  (activates the bundled osgeo)

# isort: on

from collections.abc import Sequence
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
    resample: str = "nearest",
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
        bbox: Optional crop window ``(minx, miny, maxx, maxy)`` in the product's
            native CRS (cropping happens before any ``crs`` reprojection).
        crs: Optional target CRS (an EPSG integer or a WKT string) to reproject to.
        reflectance: When ``True`` (default), tag the spectral bands so
            ``read_array(scaled=True)`` yields reflectance
            ``(DN + offset) / quantification`` (the baseline-≥04.00 offset is 0
            on older products). The returned array stays DN until read scaled.
        mask_scl: L2A only — classes to mask out (see
            :class:`~pyramids_eo.sentinel.s2.masks.SclClass`); masked pixels
            become no-data.
        path_out: When given, also write the result there (COG by extension).

    Returns:
        A pyramids ``Dataset`` of the requested bands.

    Note:
        No-data is declared in the **DN domain** (the GDAL convention). Because
        reflectance is applied as lazy ``scale`` / ``offset`` tags,
        ``read_array(scaled=True)`` returns ``DN * scale + offset`` for *every*
        pixel — so a DN no-data (and any SCL-masked pixel) reads back as the
        *scaled* sentinel, not the declared ``no_data_value``. On a
        baseline-≥04.00 product (offset ≈ −1000) a DN-0 no-data reads as
        ≈ −0.1, not 0. Test for no-data against the raw value (or an unscaled
        read), not against a scaled reflectance.

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
    dataset, target_res, offsets = _read(product, wanted, resolution, epsg, resample)
    _set_nodata(dataset, product)

    # Do every DN-domain transform first (mask, crop, reproject), then tag
    # reflectance LAST. The mask's rebuild and pyramids-gis `crop` both reset
    # per-band scale/offset (only `to_crs` / `to_file` preserve them), so tags
    # applied earlier would be silently dropped — tagging last guarantees the
    # returned dataset carries the calibration. `offsets` is captured from the
    # original read and passed through, so tagging after a crop stays correct.
    if mask_scl:
        dataset = _apply_scl_mask(dataset, product, target_res, epsg, mask_scl)
    if bbox is not None:
        dataset = _crop_to_bbox(dataset, bbox)
    if crs is not None:
        dataset = dataset.to_crs(crs)
    if reflectance:
        dataset = _scaling.tag_reflectance(
            dataset,
            product,
            offsets=offsets,
            spectral=[is_spectral_band(b) for b in wanted],
        )
    if path_out is not None:
        dataset.to_file(str(path_out))
    return dataset


def collection_from_sentinel2(
    paths: Sequence[str | Path],
    *,
    root_dir: str | Path,
    **kwargs: Any,
) -> Any:
    """Read a time series of Sentinel-2 products into a ``DatasetCollection``.

    Each product in ``paths`` is read with :func:`from_sentinel2` (passing
    ``**kwargs`` through) and written as a GeoTIFF under ``root_dir``; the
    written files are then assembled into a ``DatasetCollection``.

    The collection is **file-backed on purpose**: ``DatasetCollection``'s dask
    path (time-axis reductions, ``to_zarr``, out-of-core scale) works only for a
    file-backed collection, so the per-scene rasters are materialised to disk
    rather than held in memory (mirroring ``collection_from_earthengine``).

    Args:
        paths: The Sentinel-2 products (any form :func:`from_sentinel2` accepts).
        root_dir: Directory the per-scene GeoTIFFs are written to (created if
            absent).
        **kwargs: Forwarded to :func:`from_sentinel2` (``bands`` / ``resolution``
            / ``epsg`` / ``reflectance`` / ``mask_scl`` / …), applied to every
            scene.

    Returns:
        A ``pyramids.dataset.DatasetCollection`` over the written scenes.

    Raises:
        ProductError: ``paths`` is empty.
    """
    from pyramids.dataset import DatasetCollection

    products = list(paths)
    if not products:
        raise ProductError("collection_from_sentinel2: no product paths given")
    if "path_out" in kwargs:
        raise ProductError(
            "collection_from_sentinel2 manages per-scene output itself; pass "
            "root_dir=, not path_out= (it is set per scene)."
        )

    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, product_path in enumerate(products):
        out = root / f"{_scene_stem(product_path, index)}.tif"
        from_sentinel2(product_path, path_out=out, **kwargs)
        written.append(out)
    return DatasetCollection.from_files(written)


def _scene_stem(path: str | Path | S2Product, index: int) -> str:
    """A unique output stem for a scene (product basename + index)."""
    if isinstance(path, S2Product):
        base = Path(path.path).stem or "scene"
    else:
        base = Path(str(path)).stem or "scene"
    return f"{index:03d}_{base}"


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
    if bands is not None:
        if not bands:
            raise ProductError(
                "bands=[] is empty; omit bands= (or pass None) to read all spectral bands"
            )
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


def _single_resolution_for(
    product: S2Product, wanted: list[str], epsg: int | None
) -> int | None:
    """Finest resolution whose one subdataset carries every wanted band.

    Returns ``None`` when no single resolution holds them all — the caller then
    reads each band from its native resolution and harmonises onto a common
    grid.
    """
    for res in product.resolutions:  # ascending
        try:
            sd = product.subdataset_for(res, epsg)
        except ProductError:
            continue
        if all(_has_band(sd.bands, b) for b in wanted):
            return res
    return None


def _read(
    product: S2Product,
    wanted: list[str],
    resolution: int | None,
    epsg: int | None,
    resample: str,
) -> tuple[Any, int, list[float]]:
    """Read ``wanted`` bands into one ``Dataset``; return it, its resolution, and
    the per-band radiometric offsets.

    Single-subdataset when one resolution carries every band (or ``resolution``
    is pinned); otherwise each band is read at its native resolution and
    harmonised onto the finest requested grid.
    """
    if resolution is not None:
        subdataset = product.subdataset_for(resolution, epsg)
        _check_bands_present(subdataset, wanted)
        dataset = _select_bands(subdataset, wanted)
        return dataset, resolution, _offsets_of(dataset)

    single_res = _single_resolution_for(product, wanted, epsg)
    if single_res is not None:
        subdataset = product.subdataset_for(single_res, epsg)
        dataset = _select_bands(subdataset, wanted)
        return dataset, single_res, _offsets_of(dataset)

    return _read_harmonised(product, wanted, epsg, resample)


def _read_harmonised(
    product: S2Product, wanted: list[str], epsg: int | None, resample: str
) -> tuple[Any, int, list[float]]:
    """Read each band at its native resolution and stack onto the finest grid.

    The bands span resolutions (e.g. B04 at 10 m and B11 at 20 m). Each band is
    read from its native subdataset, coarser bands are resampled onto the finest
    band's grid via :func:`pyramids_eo.sensors.readers.harmonise`, and the aligned
    single-band results are stacked into one multi-band ``Dataset`` (band order
    follows ``wanted``).
    """
    import numpy as np
    from pyramids.dataset import Dataset

    from pyramids_eo.sensors.readers.harmonise import harmonise

    native = [(b, _native_subdataset(product, b, epsg)) for b in wanted]
    target_res = min(sd.resolution_m for _, sd in native)

    per_band = []  # (band, resolution_m, single_band_dataset, offset)
    for band, sd in native:
        band_ds = _select_bands(sd, [band])
        per_band.append((band, sd.resolution_m, band_ds, _offsets_of(band_ds)[0]))

    reference = next(ds for _, res, ds, _ in per_band if res == target_res)
    arrays = []
    for _band, res, band_ds, _off in per_band:
        aligned = (
            band_ds
            if res == target_res
            else harmonise([band_ds], reference, method=resample)[0]
        )
        arrays.append(np.asarray(aligned.read_array(band=0)))

    combined = Dataset.create_from_array(
        arr=np.stack(arrays, axis=0),
        geo=reference.raster.GetGeoTransform(),
        epsg=reference.epsg,
    )
    combined.band_names = list(wanted)
    offsets = [off for _, _, _, off in per_band]
    return combined, target_res, offsets


def _native_subdataset(product: S2Product, band: str, epsg: int | None) -> S2Subdataset:
    """Finest image subdataset (at ``epsg``) that carries ``band``."""
    candidates = [
        sd
        for sd in product.image_subdatasets
        if (epsg is None or sd.epsg == epsg) and _has_band(sd.bands, band)
    ]
    if not candidates:
        raise ProductError(
            f"band {band!r} not in product; available: {sorted(product.available_bands)}"
        )
    return min(candidates, key=lambda sd: sd.resolution_m)


def _offsets_of(dataset: Any) -> list[float]:
    """Per-band radiometric offsets (BOA/RADIO_ADD_OFFSET), ``0.0`` if absent."""
    return [_scaling._band_offset(meta) for meta in dataset.band_meta_data]


def _check_bands_present(subdataset: S2Subdataset, wanted: list[str]) -> None:
    """Raise if any wanted band is absent from ``subdataset``."""
    missing = [b for b in wanted if not _has_band(subdataset.bands, b)]
    if missing:
        raise ProductError(
            f"bands {missing} not at {subdataset.resolution_m}m; "
            f"available there: {subdataset.bands}"
        )


# -- crop ------------------------------------------------------------------


def _crop_to_bbox(dataset: Any, bbox: BBox) -> Any:
    """Window ``dataset`` to ``bbox`` (in its own CRS), keeping the full window.

    ``Dataset.crop(bbox=)`` reads the bbox window and then trims all-no-data
    border rows/columns — and for a multi-band array a row is trimmed only when
    *every* band is no-data there, whereas a single band trims on that one band.
    So the output grid shrinks by an amount that depends on how many bands are
    read, and a single-band or masked read of the same ``bbox`` comes back on a
    different, smaller grid than a multi-band read (see #81).

    This reads exactly the bbox pixel window and rebuilds it **without** the
    trim, so the returned grid (rows × cols and geotransform) is a deterministic
    function of the ``bbox`` and the resolution — identical regardless of band
    selection, masking, or whether the window straddles the granule's no-data
    fill. The snapping matches pyramids' own windowed-crop path (floor the near
    edges, ceil the far edges).

    Contract: this runs only inside the Sentinel-2 read path, on a **north-up**
    grid in its own (UTM) CRS, before any ``to_crs`` reprojection. It enforces
    north-up (raising otherwise), carries the per-band no-data values (defaulting
    a genuinely-unset value to 0.0, the S2 DN fill), and carries the source EPSG.

    Args:
        dataset: The dataset to window (its bands are read over the bbox).
        bbox: ``(minx, miny, maxx, maxy)`` in ``dataset``'s CRS.

    Returns:
        A new pyramids ``Dataset`` covering the bbox window. A bbox that extends
        beyond the raster is clipped to the dataset extent (no no-data fill is
        added outside it).

    Raises:
        ProductError: The grid is not north-up, or the bbox does not overlap the
            dataset's extent.
    """
    import math

    import numpy as np
    from pyramids.dataset import Dataset

    minx, miny, maxx, maxy = bbox
    gt = dataset.raster.GetGeoTransform()
    x0, dx, x_rot, y0, y_rot, dy = gt
    if x_rot or y_rot or dx <= 0 or dy >= 0:
        raise ProductError(
            "_crop_to_bbox requires a north-up grid (no rotation, dx > 0, "
            f"dy < 0); got geotransform {gt}"
        )
    cols, rows = dataset.raster.RasterXSize, dataset.raster.RasterYSize
    # The window is hand-rolled rather than read_array(bbox=, bbox_rounding=
    # "cover") because we also need the resolved pixel offsets to rebuild the
    # output geotransform, which read_array(bbox=) does not return; the
    # floor-near / ceil-far snap and [0, cols/rows] clamp match its "cover"
    # rounding. eps absorbs FP noise when a bbox edge lands on a pixel boundary:
    # 1e-9 is far below a UTM pixel (>= 10 m) yet above the ~2e-9 noise of
    # differencing two ~1e7 UTM northings.
    eps = 1e-9
    xoff = min(max(math.floor((minx - x0) / dx + eps), 0), cols)
    x_far = min(max(math.ceil((maxx - x0) / dx - eps), 0), cols)
    yoff = min(max(math.floor((y0 - maxy) / -dy + eps), 0), rows)
    y_far = min(max(math.ceil((y0 - miny) / -dy - eps), 0), rows)
    xsize, ysize = x_far - xoff, y_far - yoff
    if xsize <= 0 or ysize <= 0:
        raise ProductError(f"bbox {bbox} does not overlap the product extent")

    array = np.asarray(dataset.read_array(window=[xoff, yoff, xsize, ysize]))
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    out = Dataset.create_from_array(
        arr=array,
        geo=(x0 + xoff * dx, dx, 0.0, y0 + yoff * dy, 0.0, dy),
        epsg=dataset.epsg,
        no_data_value=[0.0 if v is None else v for v in dataset.no_data_value],
    )
    out.band_names = list(dataset.band_names)
    return out


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
    except (RuntimeError, ValueError, TypeError):
        # nodata is advisory; a rejecting setter must not fail the read, but a
        # programming error (wrong attr / key) still surfaces.
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
        from pyramids_eo.sensors.readers.harmonise import harmonise

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
