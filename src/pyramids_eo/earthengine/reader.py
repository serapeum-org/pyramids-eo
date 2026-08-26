"""Turnkey Google Earth Engine reader → pyramids ``Dataset`` / ``DatasetCollection``.

Three turnkey entry points pull Earth Engine data into pyramids, mirroring how
``DatasetCollection.from_stac`` / ``from_featureserver`` wrap their sources:

* :func:`from_earthengine` — a single EE ``Image`` asset → ``Dataset``; or, given
  ``start`` / ``end`` + ``reducer``, an ``ImageCollection`` reduced client-side to
  a single composite ``Dataset``.
* :func:`collection_from_earthengine` — an ``ImageCollection`` over a date range →
  ``DatasetCollection``, one aligned ``Dataset`` per scene.

Everything is built on the bundled GDAL ``EEDAI`` (raster) and ``EEDA`` (catalog)
drivers — no ``earthengine-api`` dependency — with auth carried by
:class:`~pyramids_eo.earthengine.credentials.EarthEngineCredentials` (Application
Default Credentials). Server-side EE reducers are intentionally out of scope:
compositing happens on the client over the fetched scene stack. No Earth Engine
objects leak into the public return types. See serapeum-org/pyramids-eo#13.
"""

from __future__ import annotations

# isort: off
import pyramids as _pyramids_bootstrap  # noqa: F401  (activates the bundled osgeo)
# isort: on

import gc
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
from osgeo import gdal, osr
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.merge import merge_rasters

from pyramids_eo.earthengine.credentials import CredentialsLike, EarthEngineCredentials
from pyramids_eo.errors import ReaderError

BBox = tuple[float, float, float, float]

#: GDAL connection prefixes for the Earth Engine Data API drivers.
_EEDAI_PREFIX = "EEDAI:"
_EEDA_PREFIX = "EEDA:"

#: Default output CRS when the caller does not specify one.
_DEFAULT_CRS = "EPSG:4326"

#: Statistical client-side reducers for the ``ImageCollection`` → composite path.
#: Each maps to a nodata-aware (masked) and a plain NumPy reduction over the scene
#: axis.
_STAT_REDUCERS: dict[str, tuple] = {
    "median": (np.ma.median, np.median),
    "mean": (np.ma.mean, np.mean),
    "min": (np.ma.min, np.min),
    "max": (np.ma.max, np.max),
    "sum": (np.ma.sum, np.sum),
}

#: All supported reducer names (statistical + the special ``mode`` / ``mosaic``).
REDUCERS: frozenset[str] = frozenset(_STAT_REDUCERS) | {"mode", "mosaic"}

#: Reducers that only ever emit values already present in the stack, so the result
#: can safely be cast back to the input dtype. ``mean`` / ``median`` (fractional)
#: and ``sum`` (can exceed the range) deliberately keep their widened dtype.
_VALUE_PRESERVING_REDUCERS: frozenset[str] = frozenset({"min", "max", "mode", "mosaic"})

#: Resampling algorithms accepted by the reader, mapped to their GDAL constants.
_RESAMPLERS: dict[str, int] = {
    "nearest": gdal.GRA_NearestNeighbour,
    "bilinear": gdal.GRA_Bilinear,
    "cubic": gdal.GRA_Cubic,
    "average": gdal.GRA_Average,
    "mode": gdal.GRA_Mode,
}


def _resample_alg(resample: str) -> int:
    """Resolve a resampling name to its GDAL constant.

    Args:
        resample: One of :data:`_RESAMPLERS` (``"nearest"`` / ``"bilinear"`` /
            ``"cubic"`` / ``"average"`` / ``"mode"``).

    Returns:
        The GDAL ``GRA_*`` resampling constant.

    Raises:
        ValueError: ``resample`` is not a known algorithm.
    """
    if resample not in _RESAMPLERS:
        raise ValueError(
            f"Unknown resample {resample!r}; choose from {sorted(_RESAMPLERS)}."
        )
    return _RESAMPLERS[resample]


class _Scene(NamedTuple):
    """A single discovered ``ImageCollection`` scene.

    Attributes:
        connection: The scene's ``EEDAI:`` connection string (from the EEDA
            catalog's ``gdal_dataset`` field), ready to open with the raster driver.
        time: The scene's acquisition ``startTime`` as reported by the catalog.
    """

    connection: str
    time: str


def _open_eedai(
    asset_or_connection: str,
    *,
    bands: list[str] | None,
    credentials: EarthEngineCredentials,
) -> Dataset:
    """Open an Earth Engine ``Image`` (or scene) through the GDAL EEDAI driver.

    This is a network seam: tests monkeypatch it with a local fixture raster so CI
    needs no live Earth Engine account. The bare ``gdal.OpenEx`` is required — it is
    the only call that takes the EEDAI open options — but its result is wrapped as a
    pyramids ``Dataset`` right here, so no caller handles a raw ``gdal.Dataset``.

    Args:
        asset_or_connection: An EE image asset id (e.g. ``"USGS/SRTMGL1_003"``) or a
            full ``EEDAI:`` connection string (e.g. a scene's ``gdal_dataset``).
        bands: Optional band names to request (EEDAI ``BANDS`` open option).
        credentials: Resolved credentials whose config authorises the read.

    Returns:
        The opened whole-asset pyramids ``Dataset`` (window/reproject happens later),
        carrying the credential ``gdal_env`` for any deferred read.

    Raises:
        ReaderError: The driver could not open the asset.
    """
    connection = (
        asset_or_connection
        if asset_or_connection.startswith(_EEDAI_PREFIX)
        else _EEDAI_PREFIX + asset_or_connection
    )
    # Pin the block size the block-aligned read in `_materialize` relies on, so a
    # future driver default cannot silently reintroduce cross-block reads.
    #
    # Pin a lossless transport encoding. The driver's ``AUTO`` default selects a
    # codec from the band count/dtype and, for a multi-band Byte transfer *above a
    # size threshold*, picks a lossy image codec (PNG/JPEG) that silently returns
    # pixels that are not the asset's. The 256-px block read here stays below that
    # threshold, so reads are lossless today — but pinning ``GEO_TIFF`` makes the
    # guarantee explicit and independent of the block size, which the block-sizing
    # work will raise (a larger transfer would re-cross the threshold under ``AUTO``).
    # ``GEO_TIFF`` is byte-identical to ``NPY`` on Byte and Int16 assets and smaller
    # on the wire than raw ``NPY``.
    open_options: list[str] = [
        f"BLOCK_SIZE={_EEDAI_BLOCK}",
        f"PIXEL_ENCODING={_EEDAI_PIXEL_ENCODING}",
    ]
    if bands:
        open_options.append("BANDS=" + ",".join(bands))
    with credentials.activate():
        src = gdal.OpenEx(
            connection,
            gdal.OF_RASTER | gdal.OF_VERBOSE_ERROR,
            open_options=open_options,
        )
    if src is None:
        raise ReaderError(
            f"Earth Engine asset {asset_or_connection!r} could not be opened via "
            f"EEDAI: {gdal.GetLastErrorMsg() or 'no detail'}"
        )
    return Dataset(src, gdal_env=credentials.gdal_env())


#: EEDAI serves the raster in 256-pixel blocks. Reading strictly within one block
#: per ``RasterIO`` call is the only reliably-correct EEDAI read: a window that
#: spans blocks raises "Access window out of range", and the driver's overviews
#: are corrupt, so a multi-block read or an overview-backed downsample returns
#: garbage. We therefore materialise the native window one block at a time.
_EEDAI_BLOCK = 256

#: Lossless EEDAI transport encoding pinned on every open. The driver's ``AUTO``
#: default silently picks a lossy image codec for multi-band Byte reads; ``GEO_TIFF``
#: is lossless for every dtype (verified byte-identical to ``NPY``).
_EEDAI_PIXEL_ENCODING = "GEO_TIFF"


def _materialize(ee: Dataset, bbox: BBox, crs: str) -> Dataset:
    """Read the EEDAI window covering ``bbox`` into a clean native-res ``Dataset``.

    EEDAI cannot serve a window that crosses a 256-px block boundary, and its
    overviews are unreliable — so ``gdal.Warp``-ing an EEDAI dataset directly
    corrupts the result. This reads the source at native resolution one
    block-aligned tile at a time (each ``RasterIO`` stays inside a single block),
    stitches the tiles into one array, and lets pyramids build the georeferenced
    copy that :func:`_window` can safely warp.

    Args:
        ee: The opened EEDAI source as a pyramids ``Dataset`` (from
            :func:`_open_eedai`).
        bbox: AOI ``(min_x, min_y, max_x, max_y)`` in ``crs``.
        crs: The CRS ``bbox`` is expressed in.

    Returns:
        A pyramids ``Dataset`` in the source CRS covering ``bbox`` (padded one pixel
        for resampling), holding correct native-resolution pixels for every band.
    """
    geotransform = ee.geotransform
    x0, y0, x1, y1 = _native_pixel_window(ee, bbox, crs)
    # Include the geotransform rotation/shear cross-terms so the sub-window origin
    # is correct even for a non-north-up source (north-up assets have gt[2]=gt[4]=0).
    subwindow_geo = (
        geotransform[0] + x0 * geotransform[1] + y0 * geotransform[2],
        geotransform[1],
        geotransform[2],
        geotransform[3] + x0 * geotransform[4] + y0 * geotransform[5],
        geotransform[4],
        geotransform[5],
    )
    # An ImageCollection's scenes share one per-band nodata, so the first band's is
    # representative for the whole asset.
    nodata = ee.no_data_value[0]
    # The block-aligned native read is the one step that must stay on raw GDAL bands
    # (the EEDAI corruption workaround), so it takes the underlying ``.raster``.
    data = _read_native_blocks(ee.raster, x0, y0, x1, y1)
    # Hand the block-stitched native pixels to pyramids to build the georeferenced
    # raster — no manual MEM driver or per-band juggling. The source projection is
    # passed through as WKT so a non-EPSG Earth Engine grid round-trips exactly, and
    # ``no_data_value=None`` leaves the bands without a nodata sentinel.
    return Dataset.create_from_array(
        data, geo=subwindow_geo, epsg=ee.crs, no_data_value=nodata
    )


def _native_pixel_window(
    ee: Dataset, bbox: BBox, crs: str
) -> tuple[int, int, int, int]:
    """Map an AOI ``bbox`` in ``crs`` to a block-padded native pixel window.

    The AOI boundary (corners + edge midpoints) is reprojected into the source CRS
    and mapped to pixel coordinates with the dataset's own ``rowcol`` (which handles
    rotated grids); the enclosing window is padded one pixel for resampling and
    clamped to the source extent.

    Args:
        ee: The opened EEDAI source wrapped as a pyramids ``Dataset``.
        bbox: AOI ``(min_x, min_y, max_x, max_y)`` in ``crs``.
        crs: The CRS ``bbox`` is expressed in.

    Returns:
        The pixel window ``(x0, y0, x1, y1)`` — top-left inclusive, bottom-right
        exclusive.

    Raises:
        ReaderError: The geotransform is non-invertible, or the AOI misses the asset.
    """
    geotransform = ee.geotransform
    # A zero affine determinant means the grid cannot be inverted to pixel indices.
    if (
        abs(geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4])
        < 1e-15
    ):
        raise ReaderError("Earth Engine asset has a non-invertible geotransform.")
    source_srs = osr.SpatialReference()
    source_srs.ImportFromWkt(ee.crs)
    source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target_srs = osr.SpatialReference()
    target_srs.SetFromUserInput(crs)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    min_x, min_y, max_x, max_y = bbox
    # Sample the AOI boundary (corners + edge midpoints) rather than just the four
    # corners, so a reprojected / distorted envelope is captured — for a large or
    # high-distortion AOI the true envelope can bulge past the corners on an edge.
    steps = [index / 4 for index in range(5)]
    boundary = (
        [(min_x + s * (max_x - min_x), min_y) for s in steps]
        + [(min_x + s * (max_x - min_x), max_y) for s in steps]
        + [(min_x, min_y + s * (max_y - min_y)) for s in steps]
        + [(max_x, min_y + s * (max_y - min_y)) for s in steps]
    )
    if not source_srs.IsSame(target_srs):
        transform = osr.CoordinateTransformation(target_srs, source_srs)
        boundary = [transform.TransformPoint(x, y)[:2] for x, y in boundary]

    # Map the boundary to pixel coordinates with the dataset's own rowcol. It floors
    # to whole pixels, so pad the far edge by an extra pixel to cover the remainder.
    rows_, cols_ = ee.rowcol([x for x, _ in boundary], [y for _, y in boundary])
    x0 = max(0, int(np.min(cols_)) - 1)
    y0 = max(0, int(np.min(rows_)) - 1)
    x1 = min(ee.columns, int(np.max(cols_)) + 2)
    y1 = min(ee.rows, int(np.max(rows_)) + 2)
    if x1 <= x0 or y1 <= y0:
        raise ReaderError(f"AOI {bbox} does not intersect the Earth Engine asset.")
    return x0, y0, x1, y1


def _read_native_blocks(
    src: gdal.Dataset, x0: int, y0: int, x1: int, y1: int
) -> np.ndarray:
    """Read a native pixel window one block-aligned tile at a time and stitch it.

    Each ``RasterIO`` is kept inside a single EEDAI block (a read that crosses a
    256-px block boundary corrupts the result), so the window is walked block by
    block and assembled into one ``(bands, rows, cols)`` array.

    Args:
        src: The opened EEDAI source dataset.
        x0: Left pixel of the window (inclusive).
        y0: Top pixel of the window (inclusive).
        x1: Right pixel of the window (exclusive).
        y1: Bottom pixel of the window (exclusive).

    Returns:
        The stitched native-resolution pixels, shaped ``(bands, y1 - y0, x1 - x0)``.

    Raises:
        ReaderError: A block read returned ``None``.
    """
    width, height = x1 - x0, y1 - y0
    band_count = src.RasterCount
    block = _EEDAI_BLOCK
    data: np.ndarray | None = None
    for band_index in range(band_count):
        source_band = src.GetRasterBand(band_index + 1)
        for by in range((y0 // block) * block, y1, block):
            for bx in range((x0 // block) * block, x1, block):
                rx0, ry0 = max(bx, x0), max(by, y0)
                rx1, ry1 = min(bx + block, x1), min(by + block, y1)
                tile = source_band.ReadAsArray(rx0, ry0, rx1 - rx0, ry1 - ry0)
                if tile is None:
                    raise ReaderError(
                        "Earth Engine block read failed at "
                        f"({rx0}, {ry0}): {gdal.GetLastErrorMsg() or 'no detail'}"
                    )
                if data is None:
                    data = np.empty((band_count, height, width), dtype=tile.dtype)
                data[band_index, ry0 - y0 : ry1 - y0, rx0 - x0 : rx1 - x0] = tile
    assert data is not None  # band_count >= 1 over a non-empty window
    return data


def _window(
    source: Dataset,
    *,
    bbox: BBox,
    crs: str,
    scale: float | None,
    shape: tuple[int, int] | None,
    resample: str = "nearest",
) -> Dataset:
    """Read ``source`` over ``bbox`` in ``crs`` at the requested resolution/shape.

    The EEDAI window is first materialised block-aligned into a clean native-res
    ``Dataset`` (:func:`_materialize`) — reading EEDAI directly through ``gdal.Warp``
    corrupts the result — and that copy is then warped to the target grid. The warp
    is the one GDAL primitive kept here on purpose: it reprojects to an arbitrary
    ``crs`` and lands the exact ``bbox`` + ``shape`` grid in a single pass, which
    pyramids' step-wise reproject/resample does not reproduce pixel-for-pixel.

    Args:
        source: The source EEDAI ``Dataset`` to window.
        bbox: Output bounds ``(min_x, min_y, max_x, max_y)`` in ``crs``.
        crs: Target CRS (and the CRS ``bbox`` is expressed in).
        scale: Output pixel size in ``crs`` units, or ``None``.
        shape: Output ``(rows, cols)``, or ``None``.
        resample: Resampling algorithm for the warp — see :func:`_resample_alg`.

    Returns:
        The warped pyramids ``Dataset``.

    Raises:
        ReaderError: The read or warp failed.
    """
    native = _materialize(source, bbox, crs)
    warp_kwargs: dict[str, object] = {
        "format": "MEM",
        "outputBounds": list(bbox),
        "outputBoundsSRS": crs,
        "dstSRS": crs,
        "resampleAlg": _resample_alg(resample),
    }
    if shape is not None:
        rows, cols = shape
        warp_kwargs["width"] = cols
        warp_kwargs["height"] = rows
    elif scale is not None:
        warp_kwargs["xRes"] = scale
        warp_kwargs["yRes"] = scale

    out = gdal.Warp("", native.raster, **warp_kwargs)
    if out is None:
        raise ReaderError(
            f"Earth Engine read failed while windowing to {bbox} in {crs}: "
            f"{gdal.GetLastErrorMsg() or 'no detail'}"
        )
    return Dataset(out)


def _iso(value: str) -> str:
    """Normalise a date/datetime string to an ISO datetime for the lower bound.

    A bare date (``"2024-06-01"``) gains a midnight time component so it compares
    correctly against the catalog's ``startTime`` datetimes.

    Args:
        value: An ISO date or datetime string.

    Returns:
        An ISO datetime string (with a ``T`` time component).
    """
    return value if "T" in value else f"{value}T00:00:00"


def _end_clause(end: str) -> str:
    """Build the inclusive upper-bound filter fragment for ``startTime``.

    A bare ``end`` date resolves to a **next-day, exclusive** midnight bound
    (``startTime < <end+1day>T00:00:00``) so every scene acquired on the end date is
    kept regardless of its sub-second/timezone serialisation — avoiding the lexical
    edge of comparing against an ``end-of-day`` literal. An explicit ``end``
    datetime is treated as an inclusive instant (``startTime <= <end>``).

    Args:
        end: An ISO date or datetime string (already validated).

    Returns:
        An OGR attribute-filter fragment bounding ``startTime`` from above.
    """
    if "T" in end:
        return f"startTime <= '{end}'"
    next_day = (datetime.fromisoformat(end) + timedelta(days=1)).date().isoformat()
    return f"startTime < '{next_day}T00:00:00'"


def _require_iso(label: str, value: str) -> None:
    """Reject a non-ISO ``start``/``end`` before it reaches the catalog filter.

    Validating the date/datetime (it is interpolated into an OGR attribute-filter
    string) both catches typos early and blocks a stray quote from altering the
    filter.

    Args:
        label: The parameter name, for the error message.
        value: The candidate ISO date/datetime string.

    Raises:
        ReaderError: ``value`` is not a valid ISO date/datetime.
    """
    try:
        datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ReaderError(
            f"{label!r} must be an ISO date or datetime, got {value!r}."
        ) from exc


def _bbox_to_4326(bbox: BBox, crs: str) -> BBox:
    """Reproject an AOI to EPSG:4326 for the EEDA catalog spatial filter.

    The EEDA catalog is queried in lon/lat, so a ``bbox`` expressed in another CRS
    is transformed to its EPSG:4326 envelope.

    Args:
        bbox: AOI ``(min_x, min_y, max_x, max_y)`` in ``crs``.
        crs: The CRS ``bbox`` is expressed in.

    Returns:
        The AOI's envelope in EPSG:4326 lon/lat.
    """
    if crs.upper() in (_DEFAULT_CRS, "WGS84"):
        return bbox
    source = osr.SpatialReference()
    source.SetFromUserInput(crs)
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)
    min_x, min_y, max_x, max_y = bbox
    corners = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
    xs, ys = [], []
    for x, y in corners:
        lon, lat, *_ = transform.TransformPoint(x, y)
        xs.append(lon)
        ys.append(lat)
    return (min(xs), min(ys), max(xs), max(ys))


def _geometry_bounds(geometry: object) -> BBox:
    """Return the ``(min_x, min_y, max_x, max_y)`` envelope of a polygon AOI.

    Args:
        geometry: A polygon AOI exposing ``total_bounds`` (a geopandas
            ``GeoDataFrame`` / ``GeoSeries``).

    Returns:
        The geometry's bounding box.

    Raises:
        ReaderError: The geometry does not expose ``total_bounds``.
    """
    bounds = getattr(geometry, "total_bounds", None)
    if bounds is None:
        raise ReaderError(
            "Pass a 'bbox', or a 'geometry' exposing 'total_bounds' (e.g. a "
            "geopandas GeoDataFrame)."
        )
    return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))


def _geometry_in_crs(geometry: object, crs: str) -> object:
    """Reproject a polygon AOI to ``crs`` so its bounds and cutline align.

    A geopandas geometry that carries its own CRS is reprojected to ``crs`` (a
    no-op when already in it). A geometry without a ``crs`` / ``to_crs`` is
    assumed to already be in ``crs`` and returned unchanged.

    Args:
        geometry: The polygon AOI.
        crs: The target CRS (the reader's ``crs`` argument).

    Returns:
        The geometry expressed in ``crs``.
    """
    if getattr(geometry, "crs", None) is not None and hasattr(geometry, "to_crs"):
        return geometry.to_crs(crs)
    return geometry


def _apply_geometry(dataset: Dataset, geometry: object | None) -> Dataset:
    """Clip ``dataset`` to a polygon cutline, or return it unchanged.

    Args:
        dataset: The windowed dataset.
        geometry: A polygon mask (geopandas ``GeoDataFrame`` / pyramids
            ``FeatureCollection``), or ``None`` for no clip.

    Returns:
        The clipped dataset, or ``dataset`` when ``geometry`` is ``None``.
    """
    if geometry is None:
        return dataset
    return dataset.crop(mask=geometry)


def _retain_credentials(obj: object, credentials: EarthEngineCredentials) -> object:
    """Pin ``credentials`` onto a returned object so its resources outlive it.

    An inline-JSON :class:`EarthEngineCredentials` owns a temp key file removed by
    a finalizer when the credentials are collected. The returned
    ``Dataset``/``DatasetCollection`` only captures ``gdal_env()`` (the path
    string), so without this the file would vanish while the object still names
    it. Attaching the credentials keeps them (and the file) alive for the object's
    lifetime.
    """
    obj._ee_credentials = credentials  # type: ignore[attr-defined]
    return obj


def _discover_scenes(
    asset_id: str,
    *,
    start: str,
    end: str,
    bbox_4326: BBox,
    credentials: EarthEngineCredentials,
) -> list[_Scene]:
    """Discover ``ImageCollection`` scenes over a date range + AOI via EEDA.

    This is a network seam: tests monkeypatch it so CI needs no live account.

    Args:
        asset_id: EE ``ImageCollection`` id (e.g. ``"COPERNICUS/S2_SR_HARMONIZED"``).
        start: Inclusive ISO start of the acquisition window.
        end: Inclusive ISO end of the acquisition window.
        bbox_4326: AOI envelope in EPSG:4326 lon/lat.
        credentials: Resolved credentials whose config authorises the query.

    Returns:
        Scenes intersecting the AOI within the window, sorted by acquisition time.

    Raises:
        ReaderError: The EEDA collection could not be opened, or ``start`` / ``end``
            is not a valid ISO date/datetime.
    """
    _require_iso("start", start)
    _require_iso("end", end)
    with credentials.activate():
        catalog = gdal.OpenEx(
            _EEDA_PREFIX,
            gdal.OF_VECTOR | gdal.OF_VERBOSE_ERROR,
            open_options=[f"COLLECTION={asset_id}"],
        )
    if catalog is None:
        raise ReaderError(
            f"Earth Engine collection {asset_id!r} could not be opened via EEDA: "
            f"{gdal.GetLastErrorMsg() or 'no detail'}"
        )
    layer = catalog.GetLayer(0)
    # Select by acquisition time (``startTime``): lower bound at the start date's
    # midnight, upper bound as a next-day-exclusive midnight for a bare end date so
    # every scene acquired on the end date is kept regardless of serialisation.
    layer.SetAttributeFilter(f"startTime >= '{_iso(start)}' AND {_end_clause(end)}")
    min_x, min_y, max_x, max_y = bbox_4326
    layer.SetSpatialFilterRect(min_x, min_y, max_x, max_y)
    scenes: list[_Scene] = []
    for feature in layer:
        connection = feature.GetFieldAsString("gdal_dataset")
        if connection:
            scenes.append(_Scene(connection, feature.GetFieldAsString("startTime")))
    return sorted(scenes, key=lambda s: (s.time, s.connection))


def _read_scenes_aligned(
    scenes: list[_Scene],
    *,
    bbox: BBox,
    crs: str,
    scale: float | None,
    shape: tuple[int, int] | None,
    bands: list[str] | None,
    credentials: EarthEngineCredentials,
    resample: str = "nearest",
) -> list[Dataset]:
    """Read every scene windowed to one common grid.

    When neither ``scale`` nor ``shape`` is given, the first scene's native
    windowed size fixes the grid and the rest are resampled to match — so the
    stack is always alignable into a cube / reducible into a composite.

    Args:
        scenes: The scenes to read.
        bbox: AOI ``(min_x, min_y, max_x, max_y)`` in ``crs``.
        crs: Target CRS.
        scale: Output pixel size in ``crs`` units, or ``None``.
        shape: Output ``(rows, cols)``, or ``None``.
        bands: Band names to request, or ``None`` for all.
        credentials: Resolved credentials.
        resample: Resampling algorithm for the warp.

    Returns:
        One windowed pyramids ``Dataset`` per scene, all on the same grid.
    """
    windowed: list[Dataset] = []
    target_shape = shape
    for scene in scenes:
        src = _open_eedai(scene.connection, bands=bands, credentials=credentials)
        # Release the EEDAI source handle whether the window succeeds or raises;
        # the windowed result is a self-contained in-memory copy that no longer
        # needs it.
        try:
            if target_shape is None and scale is None:
                first = _window(
                    src, bbox=bbox, crs=crs, scale=None, shape=None, resample=resample
                )
                target_shape = (first.rows, first.columns)
                windowed.append(first)
            else:
                windowed.append(
                    _window(
                        src,
                        bbox=bbox,
                        crs=crs,
                        scale=scale,
                        shape=target_shape,
                        resample=resample,
                    )
                )
        finally:
            src = None
    return windowed


def _reduce_mosaic(stack: np.ndarray, nodata: float | None) -> np.ndarray:
    """Mosaic reducer: per pixel, the first non-nodata value down the scene axis."""
    if nodata is None:
        return np.asarray(stack[0])
    masked = np.ma.masked_equal(stack, nodata)
    result = np.ma.masked_all(stack.shape[1:], dtype=stack.dtype)
    for index in range(stack.shape[0]):
        fill = result.mask & ~np.ma.getmaskarray(masked)[index]
        result[fill] = masked[index][fill]
    return np.ma.filled(result, nodata)


def _reduce_mode(stack: np.ndarray, nodata: float | None) -> np.ndarray:
    """Mode reducer: per pixel, the most frequent (non-nodata) value.

    Ties resolve to the smallest value. Windows are small, so a per-pixel
    ``Counter`` over the scene axis is fast enough.
    """
    from collections import Counter

    def _pick(values: np.ndarray) -> object:
        kept = [v for v in values.tolist() if nodata is None or v != nodata]
        if not kept:
            # ``kept`` is only empty when every sample equals ``nodata`` (with
            # ``nodata`` unset, the filter keeps everything).
            return nodata
        counts = Counter(kept)
        top = max(counts.values())
        return min(v for v, c in counts.items() if c == top)

    return np.apply_along_axis(_pick, 0, stack)


def _reduce(stack: np.ndarray, reducer: str, nodata: float | None) -> np.ndarray:
    """Reduce a scene stack over its leading (scene) axis.

    Args:
        stack: Array shaped ``(scenes, ...)`` — the aligned scene stack.
        reducer: One of :data:`REDUCERS` (``"median"`` / ``"mean"`` / ``"min"`` /
            ``"max"`` / ``"sum"`` / ``"mode"`` / ``"mosaic"``).
        nodata: Nodata value to mask out before reducing, or ``None``.

    Returns:
        The reduced array (scene axis removed). Value-preserving reducers
        (``min`` / ``max`` / ``mode`` / ``mosaic``) keep the stack dtype;
        ``mean`` / ``median`` keep NumPy's (floating) result dtype so they are
        not truncated, and ``sum`` keeps its widened accumulator dtype so it does
        not overflow.

    Raises:
        ValueError: ``reducer`` is not a known reducer.
    """
    if reducer in _STAT_REDUCERS:
        masked_fn, plain_fn = _STAT_REDUCERS[reducer]
        if nodata is None:
            reduced = plain_fn(stack, axis=0)
        else:
            reduced = np.ma.filled(
                masked_fn(np.ma.masked_equal(stack, nodata), axis=0), nodata
            )
    elif reducer == "mosaic":
        reduced = _reduce_mosaic(stack, nodata)
    elif reducer == "mode":
        reduced = _reduce_mode(stack, nodata)
    else:
        raise ValueError(
            f"Unknown reducer {reducer!r}; choose from {sorted(REDUCERS)}."
        )
    reduced = np.asarray(reduced)
    if reducer in _VALUE_PRESERVING_REDUCERS:
        # These only ever return values already present in the stack, so casting
        # back to its dtype is lossless (and keeps e.g. an int band integral).
        return reduced.astype(stack.dtype)
    # mean/median (fractional) and sum (can exceed the stack range) return float64
    # so they are neither truncated nor overflowed (float64 represents integer
    # sums exactly up to 2**53, well beyond any realistic pixel total).
    return reduced.astype(np.float64)


def _composite(
    windowed: list[Dataset],
    reducer: str,
    credentials: EarthEngineCredentials,
    geometry: object | None = None,
) -> Dataset:
    """Reduce aligned scenes into a single composite ``Dataset``.

    The per-band nodata is taken from the **first** scene and applied to every
    band of the composite — an ``ImageCollection``'s scenes share a per-band nodata
    by construction, so this holds for real collections.

    Args:
        windowed: Aligned windowed scene datasets (all on the same grid).
        reducer: The client-side reducer to apply over the scene axis.
        credentials: Resolved credentials (carried onto the returned ``Dataset``).
        geometry: Optional polygon cutline to clip the composite to.

    Returns:
        A single composite pyramids ``Dataset``.

    Raises:
        ReaderError: The scenes have mismatched band counts.
    """
    template = windowed[0]
    band_count = template.band_count
    for scene in windowed[1:]:
        if scene.band_count != band_count:
            raise ReaderError(
                "Earth Engine scenes have mismatched band counts "
                f"({band_count} vs {scene.band_count}); cannot composite."
            )
    nodatas = [template.no_data_value[index] for index in range(band_count)]
    stack = np.stack([scene.read_array() for scene in windowed], axis=0)
    if stack.ndim == 4:
        # (scenes, bands, rows, cols) — reduce each band with its own nodata.
        reduced = np.stack(
            [
                _reduce(stack[:, band], reducer, nodatas[band])
                for band in range(band_count)
            ],
            axis=0,
        )
    else:
        reduced = _reduce(stack, reducer, nodatas[0])
    # Scenes of one ImageCollection share a per-band nodata, so collapse the list to
    # a single sentinel (or ``None`` for no nodata) and let pyramids build the raster.
    band_nodata = nodatas[0] if len(set(nodatas)) == 1 else nodatas
    composite = Dataset.create_from_array(
        reduced,
        geo=template.geotransform,
        epsg=template.epsg,
        no_data_value=band_nodata,
    )
    return _apply_geometry(composite, geometry)


def _composite_read(
    asset_id: str,
    *,
    bands: list[str] | None,
    bbox: BBox | None,
    crs: str,
    scale: float | None,
    shape: tuple[int, int] | None,
    resample: str,
    start: str | None,
    end: str | None,
    reducer: str | None,
    geometry: object | None,
    credentials: EarthEngineCredentials,
    path: str | Path | None = None,
) -> Dataset:
    """Reduce an ``ImageCollection`` over a date range into one composite ``Dataset``.

    The ``ImageCollection`` branch of :func:`from_earthengine`; see it for the
    argument semantics. When ``path`` is given the composite is written there and a
    file-backed ``Dataset`` reading it is returned.

    Raises:
        ValueError: ``reducer`` is missing or the ``start``/``end``/``bbox`` trio is
            incomplete.
        ReaderError: The date range + AOI matched no scenes.
    """
    if reducer is None:
        raise ValueError(
            "'start'/'end' select an ImageCollection; pass 'reducer' for a single "
            "composite, or use collection_from_earthengine() for a DatasetCollection."
        )
    if start is None or end is None or bbox is None:
        raise ValueError(
            "The composite mode requires 'start', 'end', and a 'bbox' or 'geometry'."
        )
    scenes = _discover_scenes(
        asset_id,
        start=start,
        end=end,
        bbox_4326=_bbox_to_4326(bbox, crs),
        credentials=credentials,
    )
    if not scenes:
        raise ReaderError(
            f"No Earth Engine scenes for {asset_id!r} in [{start}, {end}] over {bbox}."
        )
    # Keep the credential config in effect across the scene reads (the EEDAI pixel
    # fetch is the ``gdal.Warp`` inside `_read_scenes_aligned`), then restore it.
    with credentials.activate():
        windowed = _read_scenes_aligned(
            scenes,
            bbox=bbox,
            crs=crs,
            scale=scale,
            shape=shape,
            bands=bands,
            credentials=credentials,
            resample=resample,
        )
    composite = _composite(windowed, reducer, credentials, geometry)
    if path is not None:
        composite.to_file(str(path))
        composite = Dataset.read_file(str(path))
    return _retain_credentials(composite, credentials)


def _single_image_read(
    asset_id: str,
    *,
    bands: list[str] | None,
    bbox: BBox | None,
    crs: str,
    scale: float | None,
    shape: tuple[int, int] | None,
    resample: str,
    geometry: object | None,
    credentials: EarthEngineCredentials,
    tile_size: int | None = None,
    path: str | Path | None = None,
) -> Dataset:
    """Read a single EE ``Image`` asset into a ``Dataset``.

    The single-``Image`` branch of :func:`from_earthengine`; see it for the
    argument semantics.

    Raises:
        ReaderError: A windowing option is set without a ``bbox``, or the asset
            could not be opened / windowed.
    """
    if bbox is None:
        if (
            scale is not None
            or shape is not None
            or crs != _DEFAULT_CRS
            or resample != "nearest"
        ):
            raise ReaderError(
                "A 'bbox' is required to window an Earth Engine asset when "
                "'crs', 'scale', 'shape', or 'resample' is set (assets are "
                "global/huge)."
            )
        dataset = _open_eedai(asset_id, bands=bands, credentials=credentials)
        # The whole-asset Dataset is read lazily, so pixel reads happen after this
        # returns — outside any `activate()` block. Install the credential config
        # process-wide so those deferred EEDAI reads still authenticate. This is the
        # one path that mutates global GDAL config (see the note in the docstring).
        for config_key, config_value in credentials.gdal_env().items():
            gdal.SetConfigOption(config_key, config_value)
        return _retain_credentials(dataset, credentials)

    # Keep the credential config in effect across the open AND the windowing read
    # (the EEDAI pixel fetch is the block-aligned RasterIO inside `_window`), then
    # restore it — no process-global leak for the windowed path.
    with credentials.activate():
        src = _open_eedai(asset_id, bands=bands, credentials=credentials)
        try:
            if tile_size is not None:
                # Stream a large window to disk one tile at a time (bounded memory),
                # reusing the single open EEDAI handle for every tile. from_earthengine
                # guarantees a path accompanies tile_size.
                if path is None:  # pragma: no cover - guaranteed by from_earthengine
                    raise ReaderError("A tiled read requires a 'path'.")
                merged = _tiled_windowed_read(
                    src,
                    bbox=bbox,
                    crs=crs,
                    scale=scale,
                    shape=shape,
                    resample=resample,
                    tile_size=tile_size,
                    path=path,  # required by from_earthengine when tile_size is set
                )
                return _retain_credentials(merged, credentials)
            windowed_single = _window(
                src, bbox=bbox, crs=crs, scale=scale, shape=shape, resample=resample
            )
        finally:
            src = None  # release the EEDAI source whether the window succeeds or not
    # ``windowed_single`` is a fully-materialised in-memory Dataset (the warp read
    # every pixel eagerly), so it needs no credential env for any deferred read.
    windowed_dataset = _apply_geometry(windowed_single, geometry)
    if path is not None:
        # Persist the windowed result and hand back a file-backed Dataset.
        windowed_dataset.to_file(str(path))
        windowed_dataset = Dataset.read_file(str(path))
    return _retain_credentials(windowed_dataset, credentials)


def _tile_edges(size: int, tile_size: int) -> list[tuple[int, int]]:
    """Split ``[0, size)`` into consecutive ``(start, end)`` blocks of ``tile_size``.

    Args:
        size: Total number of pixels along the axis.
        tile_size: Maximum block length; the final block may be shorter.

    Returns:
        ``(start, end)`` pairs (end exclusive) covering ``[0, size)`` in order.

    Examples:
        - A grid that does not divide evenly keeps a short final block:
            ```python
            >>> from pyramids_eo.earthengine.reader import _tile_edges
            >>> _tile_edges(10, 4)
            [(0, 4), (4, 8), (8, 10)]

            ```
        - An exact multiple splits into equal blocks:
            ```python
            >>> _tile_edges(6, 3)
            [(0, 3), (3, 6)]

            ```
    """
    return [(i, min(i + tile_size, size)) for i in range(0, size, tile_size)]


def _nodata_tile(
    source: Dataset,
    sub_bbox: BBox,
    shape: tuple[int, int],
    crs: str,
    nodata: float | None,
) -> Dataset:
    """Build an all-nodata tile for a sub-window fully outside the asset footprint.

    Reproduces what the un-tiled warp puts there: the source's band count and dtype,
    filled with its nodata (or ``0`` when the source has none, matching the warp's
    default fill), on the tile's exact grid.

    Args:
        source: The opened EEDAI source ``Dataset`` (for band count and dtype).
        sub_bbox: The tile's bounds ``(min_x, min_y, max_x, max_y)`` in ``crs``.
        shape: The tile's ``(rows, cols)``.
        crs: Target CRS.
        nodata: The source nodata to fill with, or ``None`` for a no-nodata source.

    Returns:
        An in-memory pyramids ``Dataset`` covering ``sub_bbox`` filled with nodata.
    """
    rows, cols = shape
    min_x, min_y, max_x, max_y = sub_bbox
    fill = nodata if nodata is not None else 0
    array = np.full((source.band_count, rows, cols), fill, dtype=source.numpy_dtype[0])
    geo = (min_x, (max_x - min_x) / cols, 0.0, max_y, 0.0, -(max_y - min_y) / rows)
    return Dataset.create_from_array(array, geo=geo, epsg=crs, no_data_value=nodata)


def _tile_grid(
    bbox: BBox, scale: float | None, shape: tuple[int, int] | None
) -> tuple[int, int, float, float]:
    """Compute the tiled output grid ``(rows, cols, cell_x, cell_y)`` over ``bbox``.

    A ``shape`` read fits the grid exactly to the bbox (as ``gdal.Warp`` does with
    width/height + outputBounds). A ``scale`` read keeps the pixel size == ``scale``
    and sizes the grid the way ``gdal.Warp`` does with ``xRes``/``yRes`` — round-half-up
    via GDAL's exact ``int((extent + res/2) / res)`` (bit-identical), extent =
    origin + n*scale (which may extend past the bbox) — reproducing the un-tiled
    scale read exactly.

    Args:
        bbox: Output bounds ``(min_x, min_y, max_x, max_y)``.
        scale: Output pixel size, or ``None`` when ``shape`` is set.
        shape: Output ``(rows, cols)``, or ``None`` when ``scale`` is set.

    Returns:
        ``(rows, cols, cell_x, cell_y)`` for the full output grid.
    """
    min_x, min_y, max_x, max_y = bbox
    if shape is not None:
        rows, cols = shape
        return rows, cols, (max_x - min_x) / cols, (max_y - min_y) / rows
    if scale is None:  # pragma: no cover - from_earthengine guarantees scale/shape
        raise ReaderError("A tiled scale read requires a 'scale'.")
    cols = max(1, int(((max_x - min_x) + scale / 2) / scale))
    rows = max(1, int(((max_y - min_y) + scale / 2) / scale))
    return rows, cols, scale, scale


def _mosaic_tiles(tile_paths: list[str], path: str, nodata: float | None) -> None:
    """Mosaic grid-aligned tile files into ``path`` with correct nodata handling.

    Non-overlapping, grid-aligned tiles fully cover the window, so the merge is exact
    placement. The source nodata is carried through (treated as transparent and
    stamped on the output); when the source has none, it is unset on the mosaic
    (``"none"``) to match the un-tiled read, with a 0 fill that never triggers GDAL's
    "cannot represent nan" cast warning. A float source with a NaN nodata takes the
    same with-nodata branch (``n=init=no_data_value=nan``), relying on GDAL's
    NaN-aware nodata matching; EE assets are effectively always integer nodata.

    Args:
        tile_paths: The temporary tile raster paths to mosaic.
        path: Destination raster path.
        nodata: The shared source nodata, or ``None`` for a no-nodata source.
    """
    if nodata is not None:
        merge_rasters(
            tile_paths,
            path,
            no_data_value=nodata,
            n=nodata,
            init=nodata,
            method="first",
        )
    else:
        merge_rasters(
            tile_paths, path, no_data_value="none", n=0, init=0, method="first"
        )


def _tiled_windowed_read(
    source: Dataset,
    *,
    bbox: BBox,
    crs: str,
    scale: float | None,
    shape: tuple[int, int] | None,
    resample: str,
    tile_size: int,
    path: str | Path,
) -> Dataset:
    """Read a large window as grid-aligned tiles and mosaic them to ``path``.

    The output grid over ``bbox`` is split into blocks of at most ``tile_size``
    pixels per side. Each block is read through the normal windowed path (its own
    block-aligned EEDAI materialise + warp), written to a temporary raster and
    released, then the tiles are mosaicked with pyramids ``merge_rasters`` into
    ``path``. Because every tile is warped (nearest) to its exact grid-aligned
    sub-window, the mosaic reproduces the equivalent un-tiled ``nearest`` read
    exactly. ``resample`` other than ``"nearest"`` is rejected upstream, since an
    interpolating kernel would sample across a tile seam.

    Memory/cost notes: the per-tile step (not the whole output) is what is bounded,
    but each tile's :func:`_materialize` still reads the tile's **native-resolution**
    window into memory, so peak memory is governed by that native window — pick
    ``tile_size`` relative to native resolution. The z-order ``merge_rasters`` opens
    all tile files at once, and the read is O(n_tiles) independent
    materialise/warp/write round-trips (re-fetching the block-aligned + 1-px pad at
    every seam), so very large tile counts are correct but not free.

    Args:
        source: The opened EEDAI source ``Dataset`` (reused for every tile).
        bbox: Output bounds ``(min_x, min_y, max_x, max_y)`` in ``crs``.
        crs: Target CRS (and the CRS ``bbox`` is expressed in).
        scale: Output pixel size in ``crs`` units, or ``None`` when ``shape`` is set.
        shape: Output ``(rows, cols)``, or ``None`` when ``scale`` is set.
        resample: Resampling algorithm for each tile's warp (always ``"nearest"``).
        tile_size: Maximum tile size in pixels per side.
        path: Destination raster path for the mosaic.

    Returns:
        A file-backed pyramids ``Dataset`` reading the mosaic at ``path``.
    """
    min_x, _, _, max_y = bbox  # only the top-left anchors the tile grid
    rows, cols, cell_x, cell_y = _tile_grid(bbox, scale, shape)

    # All tiles inherit the source's per-band nodata, so read it once here.
    nodata = source.no_data_value[0]
    tmp_dir = tempfile.mkdtemp(prefix="ee_tiles_")
    tile_paths: list[str] = []
    any_covered = False
    try:
        for row0, row1 in _tile_edges(rows, tile_size):
            for col0, col1 in _tile_edges(cols, tile_size):
                sub_bbox = (
                    min_x + col0 * cell_x,
                    max_y - row1 * cell_y,
                    min_x + col1 * cell_x,
                    max_y - row0 * cell_y,
                )
                tile_shape = (row1 - row0, col1 - col0)
                try:
                    tile = _window(
                        source,
                        bbox=sub_bbox,
                        crs=crs,
                        scale=None,
                        shape=tile_shape,
                        resample=resample,
                    )
                    any_covered = True
                except ReaderError as exc:
                    if "does not intersect" not in str(exc):
                        raise
                    # A tile fully outside the asset footprint: emit an all-nodata
                    # tile, matching how the un-tiled warp nodata-fills that region
                    # (the un-tiled read clamps the window and fills the overhang).
                    tile = _nodata_tile(source, sub_bbox, tile_shape, crs, nodata)
                tile_path = os.path.join(tmp_dir, f"tile_{row0}_{col0}.tif")
                tile.to_file(tile_path)
                tile.close()  # release the GDAL handle so the temp file can be removed
                tile_paths.append(tile_path)
        if not any_covered:
            # No tile intersected the asset — the whole AOI is off the footprint,
            # the same case the un-tiled read rejects.
            raise ReaderError(f"AOI {bbox} does not intersect the Earth Engine asset.")
        _mosaic_tiles(tile_paths, str(path), nodata)
    finally:
        # merge_rasters opens the tile files (and holds them via GC-managed handles);
        # force their release before removing the temp dir, or Windows leaves the
        # ``ee_tiles_*`` directory behind (a delete-while-open failure that
        # ``ignore_errors`` would otherwise hide and accumulate).
        gc.collect()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return Dataset.read_file(str(path))


def _validate_read_request(
    *,
    scale: float | None,
    shape: tuple[int, int] | None,
    resample: str,
    path: str | Path | None,
    bbox: BBox | None,
    geometry: object | None,
    tile_size: int | None,
    reducer: str | None,
    start: str | None,
    end: str | None,
) -> None:
    """Validate a :func:`from_earthengine` option combination before any network call.

    Args:
        scale: Requested output pixel size, or ``None``.
        shape: Requested output ``(rows, cols)``, or ``None``.
        resample: Resampling algorithm name.
        path: Output raster path, or ``None``.
        bbox: AOI bounds, or ``None``.
        geometry: Polygon AOI, or ``None``.
        tile_size: Oversize tile size, or ``None``.
        reducer: Composite reducer, or ``None``.
        start: Composite start date, or ``None``.
        end: Composite end date, or ``None``.

    Raises:
        ValueError: An incompatible or incomplete option combination (or an unknown
            ``resample`` name).
    """
    if scale is not None and shape is not None:
        raise ValueError("Pass at most one of 'scale' or 'shape', not both.")
    _resample_alg(resample)  # rejects an unknown resample name up front
    if path is not None and bbox is None and geometry is None:
        raise ValueError(
            "'path' needs a 'bbox' or 'geometry'; the whole-asset read is lazy."
        )
    if tile_size is None:
        return
    if reducer is not None or start is not None or end is not None:
        raise ValueError(
            "'tile_size' is for the single-image raw read, not an "
            "ImageCollection composite."
        )
    if tile_size <= 0:
        raise ValueError("'tile_size' must be a positive number of pixels.")
    if geometry is not None:
        raise ValueError("'tile_size' cannot be combined with a polygon 'geometry'.")
    if scale is None and shape is None:
        raise ValueError(
            "'tile_size' needs 'scale' or 'shape' to define the output grid."
        )
    if path is None:
        raise ValueError("'tile_size' needs 'path' to stream the mosaic to disk.")
    if resample != "nearest":
        # Tiles are warped independently, so an interpolating kernel samples missing
        # neighbours across a tile seam and the mosaic no longer matches the un-tiled
        # read. Only nearest (each output pixel from one source pixel) is seam-exact.
        raise ValueError(
            "'tile_size' supports only resample='nearest'; an interpolating "
            "resampler would differ from the un-tiled read at tile seams."
        )


def from_earthengine(
    asset_id: str,  # NOSONAR(S107) - a flat keyword reader API (windowing/composite/output options) is intentional; consolidating would break the released scale=/shape= surface
    *,
    bands: list[str] | None = None,
    bbox: BBox | None = None,
    geometry: object | None = None,
    crs: str = _DEFAULT_CRS,
    scale: float | None = None,
    shape: tuple[int, int] | None = None,
    resample: str = "nearest",
    start: str | None = None,
    end: str | None = None,
    reducer: str | None = None,
    credentials: CredentialsLike = None,
    tile_size: int | None = None,
    path: str | Path | None = None,
) -> Dataset:
    """Read an Earth Engine ``Image`` (or reduced ``ImageCollection``) into a ``Dataset``.

    Two modes:

    * **Single ``Image``** (default): opens ``asset_id`` through the GDAL EEDAI
      driver and, when a ``bbox`` is given, warps it to ``crs`` at the requested
      resolution.
    * **``ImageCollection`` composite**: when ``start`` / ``end`` + ``reducer`` are
      given, discovers the collection's scenes over the AOI + date range (EEDA),
      reads them aligned, and reduces them **client-side** into one composite
      ``Dataset``. Server-side EE reducers are out of scope.

    Args:
        asset_id: EE image asset id (single mode) or ``ImageCollection`` id
            (composite mode), e.g. ``"USGS/SRTMGL1_003"`` /
            ``"COPERNICUS/S2_SR_HARMONIZED"``.
        bands: Band names to request; ``None`` reads every band.
        bbox: AOI ``(min_x, min_y, max_x, max_y)`` in ``crs``. Required to
            materialise a window (single mode) and for the composite mode, unless
            a ``geometry`` is given (its envelope is used as the ``bbox``).
        geometry: Optional polygon AOI (a geopandas ``GeoDataFrame`` / pyramids
            ``FeatureCollection``). A geometry carrying its own CRS is reprojected
            to ``crs``; one without is assumed to already be in ``crs``. Its
            envelope drives the read window and the result is then clipped to the
            polygon cutline. Takes the place of ``bbox`` when ``bbox`` is omitted.
        crs: Target CRS (and the CRS ``bbox`` is expressed in). Defaults to
            ``"EPSG:4326"``.
        scale: Output pixel size in ``crs`` units. Mutually exclusive with ``shape``.
        shape: Output ``(rows, cols)``. Mutually exclusive with ``scale``.
        resample: Resampling algorithm used when the native window is warped to the
            output grid — one of ``"nearest"`` (default), ``"bilinear"``,
            ``"cubic"``, ``"average"``, ``"mode"``. The default is nearest-neighbour;
            for continuous imagery that is downsampled, ``"average"`` or
            ``"bilinear"`` give a more representative result. (``"mode"`` here is a
            *resampling* algorithm — distinct from the ``"mode"`` composite
            ``reducer``.)
        start: Inclusive ISO start of the acquisition window (composite mode).
        end: Inclusive ISO end of the acquisition window (composite mode).
        reducer: Client-side reducer for the composite mode — one of ``"median"``,
            ``"mean"``, ``"min"``, ``"max"``, ``"sum"``, ``"mode"``, ``"mosaic"``.
            ``mode`` is computed per pixel in Python, so it is slower than the
            others over a large AOI.
        credentials: An
            :class:`~pyramids_eo.earthengine.credentials.EarthEngineCredentials`, a
            path to a service-account JSON key, or ``None`` for ADC.
        tile_size: Maximum tile size (pixels per side) for an oversize read. When
            set, the output grid is split into grid-aligned tiles of at most this
            size, each read and written to disk in turn, then mosaicked into
            ``path`` — bounding the peak of the per-tile warp/write step rather than
            materialising the whole output at once. The mosaic reproduces the
            equivalent un-tiled ``nearest`` read exactly. Single-``Image`` raw reads
            only: requires a ``bbox``, a ``path``, ``scale`` or ``shape``, and the
            default ``resample="nearest"`` (interpolating resamplers differ from the
            un-tiled read at tile seams); cannot be combined with a ``geometry``
            cutline or the composite mode. Peak memory is still governed by each
            tile's **native-resolution** window (see the Performance note), so choose
            ``tile_size`` relative to the asset's native resolution.
        path: Output raster path. When given, the result is written there and a
            file-backed ``Dataset`` reading it is returned instead of an in-memory
            one; required when ``tile_size`` is set, and honoured for the single-image
            and composite paths alike. Needs a ``bbox`` or ``geometry`` (the
            whole-asset read is lazy).

    Returns:
        A pyramids :class:`~pyramids.dataset.Dataset` — the windowed image or the
        reduced composite (file-backed when ``path`` is given).

    Note:
        The windowed and composite paths scope the credential config to the read
        and restore it afterward. The **no-bbox lazy wrap** is the exception: its
        pixels are read after this returns, so a service-account credential is
        installed into the **process-global** GDAL config with no restore. That
        means a later no-bbox call with a *different* service account overwrites it
        (an earlier still-open lazy ``Dataset`` would then read with the newer
        credential), and the option leaks into unrelated GDAL work. Prefer passing a
        ``bbox``/``geometry`` when using a service-account key; ADC mode is
        unaffected. See also the thread-safety note on
        :meth:`EarthEngineCredentials.activate`.

    Note:
        **Performance.** The EEDAI driver's overviews are unreliable, so the reader
        always fetches the AOI at the asset's **native resolution** (block by
        block) and downsamples locally with ``resample`` (nearest by default). A
        small ``shape``/``scale`` output from a fine-resolution asset over a wide
        AOI therefore still transfers the full native window — e.g. a 32x32 read of
        10 m Sentinel-2 over a 0.1° box pulls ~1100x1100 native pixels. It is
        correct but can be slow/data-heavy; keep the AOI tight for fine-resolution
        assets.

    Raises:
        ValueError: ``scale`` and ``shape`` are both given; ``start`` / ``end`` are
            given without a ``reducer`` (use :func:`collection_from_earthengine` for
            a ``DatasetCollection``); the composite mode is missing
            ``start`` / ``end`` / ``bbox``; ``path`` is given without a
            ``bbox`` / ``geometry``; or ``tile_size`` is invalid or set without its
            required ``path`` / ``scale`` or ``shape`` (or combined with a composite
            or a ``geometry``).
        ReaderError: The asset could not be opened or windowed, or the composite
            date range + AOI matched no scenes.

    Examples:
        - Read a small SRTM window (requires Earth Engine credentials, so skipped
          offline):
            ```python
            >>> from pyramids_eo import from_earthengine
            >>> ds = from_earthengine(  # doctest: +SKIP
            ...     "USGS/SRTMGL1_003",
            ...     bbox=(86.9, 27.9, 87.0, 28.0),
            ... )

            ```
        - A median composite over an ``ImageCollection`` date range (skipped
          offline):
            ```python
            >>> from pyramids_eo import from_earthengine
            >>> composite = from_earthengine(  # doctest: +SKIP
            ...     "COPERNICUS/S2_SR_HARMONIZED",
            ...     bbox=(86.9, 27.9, 87.0, 28.0),
            ...     start="2024-06-01",
            ...     end="2024-06-30",
            ...     reducer="median",
            ... )

            ```
        - Clip to a polygon AOI instead of a bbox (skipped offline):
            ```python
            >>> import geopandas as gpd  # doctest: +SKIP
            >>> from pyramids_eo import from_earthengine
            >>> aoi = gpd.read_file("basin.geojson")  # doctest: +SKIP
            >>> ds = from_earthengine("USGS/SRTMGL1_003", geometry=aoi)  # doctest: +SKIP

            ```
        - Stream an oversize window to disk in 1024-px tiles (skipped offline):
            ```python
            >>> from pyramids_eo import from_earthengine
            >>> ds = from_earthengine(  # doctest: +SKIP
            ...     "USGS/SRTMGL1_003",
            ...     bbox=(86.0, 27.0, 88.0, 29.0),
            ...     scale=0.0003,
            ...     tile_size=1024,
            ...     path="srtm_big.tif",
            ... )

            ```
        - ``tile_size`` without a ``path`` is rejected before any read:
            ```python
            >>> from pyramids_eo import from_earthengine
            >>> from_earthengine(
            ...     "USGS/SRTMGL1_003",
            ...     bbox=(86.9, 27.9, 87.0, 28.0),
            ...     shape=(4096, 4096),
            ...     tile_size=1024,
            ... )
            Traceback (most recent call last):
                ...
            ValueError: 'tile_size' needs 'path' to stream the mosaic to disk.

            ```
        - Passing both ``scale`` and ``shape`` is rejected before any read:
            ```python
            >>> from pyramids_eo import from_earthengine
            >>> from_earthengine(
            ...     "USGS/SRTMGL1_003",
            ...     bbox=(86.9, 27.9, 87.0, 28.0),
            ...     scale=0.01,
            ...     shape=(5, 5),
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Pass at most one of 'scale' or 'shape', not both.

            ```
    """
    _validate_read_request(
        scale=scale,
        shape=shape,
        resample=resample,
        path=path,
        bbox=bbox,
        geometry=geometry,
        tile_size=tile_size,
        reducer=reducer,
        start=start,
        end=end,
    )

    creds = EarthEngineCredentials.coerce(credentials)
    if geometry is not None:
        geometry = _geometry_in_crs(geometry, crs)
        if bbox is None:
            bbox = _geometry_bounds(geometry)

    if reducer is not None or start is not None or end is not None:
        return _composite_read(
            asset_id,
            bands=bands,
            bbox=bbox,
            crs=crs,
            scale=scale,
            shape=shape,
            resample=resample,
            start=start,
            end=end,
            reducer=reducer,
            geometry=geometry,
            credentials=creds,
            path=path,
        )
    return _single_image_read(
        asset_id,
        bands=bands,
        bbox=bbox,
        crs=crs,
        scale=scale,
        shape=shape,
        resample=resample,
        geometry=geometry,
        credentials=creds,
        tile_size=tile_size,
        path=path,
    )


def collection_from_earthengine(
    asset_id: str,
    *,
    start: str,
    end: str,
    bbox: BBox | None = None,
    geometry: object | None = None,
    bands: list[str] | None = None,
    crs: str = _DEFAULT_CRS,
    scale: float | None = None,
    shape: tuple[int, int] | None = None,
    resample: str = "nearest",
    credentials: CredentialsLike = None,
) -> DatasetCollection:
    """Read an Earth Engine ``ImageCollection`` into a ``DatasetCollection``.

    Discovers the collection's scenes over the AOI + date range via the GDAL EEDA
    catalog driver, then reads each one (EEDAI) windowed to a common grid — one
    aligned :class:`~pyramids.dataset.Dataset` per scene, in acquisition order.

    Args:
        asset_id: EE ``ImageCollection`` id (e.g. ``"COPERNICUS/S2_SR_HARMONIZED"``).
        start: Inclusive ISO start of the acquisition window.
        end: Inclusive ISO end of the acquisition window.
        bbox: AOI ``(min_x, min_y, max_x, max_y)`` in ``crs``. Required (to bound
            scene discovery) unless a ``geometry`` is given.
        geometry: Optional polygon AOI (a geopandas ``GeoDataFrame`` / pyramids
            ``FeatureCollection``). A geometry carrying its own CRS is reprojected
            to ``crs``; one without is assumed to already be in ``crs``. Its
            envelope bounds scene discovery and each scene is clipped to the
            polygon cutline. Takes the place of ``bbox`` when ``bbox`` is omitted.
        bands: Band names to request; ``None`` reads every band.
        crs: Target CRS (and the CRS ``bbox`` is expressed in). Defaults to
            ``"EPSG:4326"``.
        scale: Output pixel size in ``crs`` units. Mutually exclusive with ``shape``.
            When both are omitted the first scene's native windowed grid is used
            for every scene.
        shape: Output ``(rows, cols)``. Mutually exclusive with ``scale``.
        resample: Resampling algorithm for the per-scene warp — one of
            ``"nearest"`` (default), ``"bilinear"``, ``"cubic"``, ``"average"``,
            ``"mode"``. See :func:`from_earthengine`.
        credentials: An
            :class:`~pyramids_eo.earthengine.credentials.EarthEngineCredentials`, a
            path to a service-account JSON key, or ``None`` for ADC.

    Returns:
        A pyramids :class:`~pyramids.dataset.DatasetCollection`, one timestep per
        scene, with the acquisition times as its time axis.

    Raises:
        ValueError: ``scale`` and ``shape`` are both given, or neither ``bbox``
            nor ``geometry`` is given.
        ReaderError: The catalog could not be opened, or the date range + AOI
            matched no scenes.

    Examples:
        - Read a Sentinel-2 collection over a date range (skipped offline):
            ```python
            >>> from pyramids_eo import collection_from_earthengine
            >>> collection = collection_from_earthengine(  # doctest: +SKIP
            ...     "COPERNICUS/S2_SR_HARMONIZED",
            ...     start="2024-06-01",
            ...     end="2024-06-10",
            ...     bbox=(86.9, 27.9, 87.0, 28.0),
            ... )

            ```
    """
    if scale is not None and shape is not None:
        raise ValueError("Pass at most one of 'scale' or 'shape', not both.")
    _resample_alg(resample)  # validate up front, before any network call
    if geometry is not None:
        geometry = _geometry_in_crs(geometry, crs)
    if bbox is None:
        if geometry is None:
            raise ValueError("Pass a 'bbox' or a 'geometry'.")
        bbox = _geometry_bounds(geometry)

    creds = EarthEngineCredentials.coerce(credentials)
    scenes = _discover_scenes(
        asset_id,
        start=start,
        end=end,
        bbox_4326=_bbox_to_4326(bbox, crs),
        credentials=creds,
    )
    if not scenes:
        raise ReaderError(
            f"No Earth Engine scenes for {asset_id!r} in [{start}, {end}] over {bbox}."
        )
    # Keep the credential config in effect across the scene reads, then restore it.
    with creds.activate():
        windowed = _read_scenes_aligned(
            scenes,
            bbox=bbox,
            crs=crs,
            scale=scale,
            shape=shape,
            bands=bands,
            credentials=creds,
            resample=resample,
        )
    env = creds.gdal_env()
    # ``windowed`` scenes are already fully-materialised pyramids Datasets (the warp
    # read every pixel eagerly), so they need no re-wrapping or credential env.
    datasets = [
        _retain_credentials(_apply_geometry(scene, geometry), creds)
        for scene in windowed
    ]
    collection = DatasetCollection(
        datasets[0],
        time_length=len(datasets),
        datasets=datasets,
        time=[scene.time for scene in scenes],
        gdal_env=env,
    )
    return _retain_credentials(collection, creds)
