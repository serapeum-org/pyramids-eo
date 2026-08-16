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

import math
from typing import NamedTuple

import numpy as np
from osgeo import gdal, gdal_array
from pyramids.dataset import Dataset, DatasetCollection

from pyramids_eo.earthengine.credentials import CredentialsLike, EarthEngineCredentials
from pyramids_eo.errors import ReaderError

BBox = tuple[float, float, float, float]

#: GDAL connection prefixes for the Earth Engine Data API drivers.
_EEDAI_PREFIX = "EEDAI:"
_EEDA_PREFIX = "EEDA:"

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
) -> gdal.Dataset:
    """Open an Earth Engine ``Image`` (or scene) through the GDAL EEDAI driver.

    This is a network seam: tests monkeypatch it with a local fixture raster so CI
    needs no live Earth Engine account.

    Args:
        asset_or_connection: An EE image asset id (e.g. ``"USGS/SRTMGL1_003"``) or a
            full ``EEDAI:`` connection string (e.g. a scene's ``gdal_dataset``).
        bands: Optional band names to request (EEDAI ``BANDS`` open option).
        credentials: Resolved credentials whose config authorises the read.

    Returns:
        The opened GDAL dataset (whole asset; window/reproject happens later).

    Raises:
        ReaderError: The driver could not open the asset.
    """
    connection = (
        asset_or_connection
        if asset_or_connection.startswith(_EEDAI_PREFIX)
        else _EEDAI_PREFIX + asset_or_connection
    )
    open_options: list[str] = []
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
    return src


def _window(
    src: gdal.Dataset,
    *,
    bbox: BBox,
    crs: str,
    scale: float | None,
    shape: tuple[int, int] | None,
) -> gdal.Dataset:
    """Warp ``src`` to ``bbox`` in ``crs`` at the requested resolution/shape.

    Returns an in-memory (``MEM``) GDAL dataset so the result carries no on-disk or
    ``/vsimem`` lifetime.

    Args:
        src: The source GDAL dataset to window.
        bbox: Output bounds ``(min_x, min_y, max_x, max_y)`` in ``crs``.
        crs: Target CRS (and the CRS ``bbox`` is expressed in).
        scale: Output pixel size in ``crs`` units, or ``None``.
        shape: Output ``(rows, cols)``, or ``None``.

    Returns:
        The warped in-memory GDAL dataset.

    Raises:
        ReaderError: The warp failed.
    """
    warp_kwargs: dict[str, object] = {
        "format": "MEM",
        "outputBounds": list(bbox),
        "outputBoundsSRS": crs,
        "dstSRS": crs,
    }
    if shape is not None:
        rows, cols = shape
        warp_kwargs["width"] = cols
        warp_kwargs["height"] = rows
    elif scale is not None:
        warp_kwargs["xRes"] = scale
        warp_kwargs["yRes"] = scale

    out = gdal.Warp("", src, **warp_kwargs)
    if out is None:
        raise ReaderError(
            f"Earth Engine read failed while windowing to {bbox} in {crs}: "
            f"{gdal.GetLastErrorMsg() or 'no detail'}"
        )
    return out


def _tiled_window(
    src: gdal.Dataset,
    *,
    bbox: BBox,
    crs: str,
    scale: float | None,
    shape: tuple[int, int] | None,
    tile_size: int,
) -> gdal.Dataset | None:
    """Read a large window as a grid of tiles and mosaic them, or return ``None``.

    Returns ``None`` when the target grid is unknown (no ``scale``/``shape``) or
    already within ``tile_size`` on both axes — the caller then does a single
    :func:`_window`. Otherwise the AOI is split into ``<= tile_size`` px tiles,
    each read via :func:`_window`, and the tiles are mosaicked (via ``gdal.Warp``)
    onto the full grid. EEDAI has no hard request cap, so ``tile_size`` bounds
    local memory/practicality, not an Earth Engine limit.

    Args:
        src: The opened EEDAI source dataset.
        bbox: AOI ``(min_x, min_y, max_x, max_y)`` in ``crs``.
        crs: Target CRS.
        scale: Output pixel size in ``crs`` units, or ``None``.
        shape: Output ``(rows, cols)``, or ``None``.
        tile_size: Maximum pixels per axis per tile.

    Returns:
        The mosaicked in-memory GDAL dataset, or ``None`` if tiling does not apply.

    Raises:
        ReaderError: The mosaic warp failed.
    """
    min_x, min_y, max_x, max_y = bbox
    if shape is not None:
        rows, cols = shape
        px_x = (max_x - min_x) / cols
        px_y = (max_y - min_y) / rows
    elif scale is not None:
        px_x = px_y = scale
        cols = math.ceil((max_x - min_x) / scale)
        rows = math.ceil((max_y - min_y) / scale)
    else:
        return None
    if max(rows, cols) <= tile_size:
        return None

    tiles: list[gdal.Dataset] = []
    for iy in range(math.ceil(rows / tile_size)):
        tile_max_y = max_y - iy * tile_size * px_y
        tile_min_y = max(min_y, tile_max_y - tile_size * px_y)
        for ix in range(math.ceil(cols / tile_size)):
            tile_min_x = min_x + ix * tile_size * px_x
            tile_max_x = min(max_x, tile_min_x + tile_size * px_x)
            tiles.append(
                _window(
                    src,
                    bbox=(tile_min_x, tile_min_y, tile_max_x, tile_max_y),
                    crs=crs,
                    scale=None,
                    shape=None,
                )
            )
    mosaic = gdal.Warp(
        "",
        tiles,
        format="MEM",
        outputBounds=[min_x, min_y, max_x, max_y],
        outputBoundsSRS=crs,
        dstSRS=crs,
        xRes=px_x,
        yRes=px_y,
    )
    if mosaic is None:
        raise ReaderError(
            f"Earth Engine tiled read failed to mosaic to {bbox} in {crs}: "
            f"{gdal.GetLastErrorMsg() or 'no detail'}"
        )
    return mosaic


def _iso(value: str, *, end_of_day: bool = False) -> str:
    """Normalise a date/datetime string to an ISO datetime for catalog filtering.

    A bare date (``"2024-06-01"``) gains a time component so it compares correctly
    against the catalog's ``startTime`` datetimes: midnight for a lower bound, or
    end-of-day for an inclusive upper bound (so scenes acquired any time on the end
    date are kept).

    Args:
        value: An ISO date or datetime string.
        end_of_day: When ``value`` is a bare date, use ``23:59:59.999`` instead of
            midnight — for an inclusive ``end`` bound.

    Returns:
        An ISO datetime string (with a ``T`` time component).
    """
    if "T" in value:
        return value
    return f"{value}T23:59:59.999" if end_of_day else f"{value}T00:00:00"


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
    if crs.upper() in ("EPSG:4326", "WGS84"):
        return bbox
    from osgeo import osr

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
        ReaderError: The EEDA collection could not be opened.
    """
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
    # Select by acquisition time (``startTime``) inclusively on both dates: a bare
    # ``end`` date resolves to end-of-day so scenes acquired that day are kept, and
    # a scene whose interval extends past ``end`` is not dropped for that reason.
    layer.SetAttributeFilter(
        f"startTime >= '{_iso(start)}' AND startTime <= '{_iso(end, end_of_day=True)}'"
    )
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
) -> list[gdal.Dataset]:
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

    Returns:
        One windowed in-memory GDAL dataset per scene, all on the same grid.
    """
    windowed: list[gdal.Dataset] = []
    target_shape = shape
    for scene in scenes:
        src = _open_eedai(scene.connection, bands=bands, credentials=credentials)
        if target_shape is None and scale is None:
            first = _window(src, bbox=bbox, crs=crs, scale=None, shape=None)
            target_shape = (first.RasterYSize, first.RasterXSize)
            windowed.append(first)
        else:
            windowed.append(
                _window(src, bbox=bbox, crs=crs, scale=scale, shape=target_shape)
            )
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


def _build_like(
    template: gdal.Dataset, array: np.ndarray, nodata: float | None
) -> gdal.Dataset:
    """Build an in-memory GDAL dataset holding ``array`` on ``template``'s grid.

    Args:
        template: The dataset whose geotransform / projection / dtype to copy.
        array: The pixel data, shaped ``(rows, cols)`` or ``(bands, rows, cols)``.
        nodata: Nodata value to stamp on each band, or ``None``.

    Returns:
        A ``MEM`` GDAL dataset georeferenced like ``template``.
    """
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    n_bands, rows, cols = array.shape
    # Derive the band dtype from the array (not the template) so a float mean /
    # median or a widened int sum is stored without truncation or overflow.
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype)
    out = gdal.GetDriverByName("MEM").Create("", cols, rows, n_bands, dtype)
    out.SetGeoTransform(template.GetGeoTransform())
    out.SetProjection(template.GetProjection())
    for index in range(n_bands):
        band = out.GetRasterBand(index + 1)
        band.WriteArray(array[index])
        if nodata is not None:
            band.SetNoDataValue(nodata)
    return out


def _composite(
    windowed: list[gdal.Dataset],
    reducer: str,
    credentials: EarthEngineCredentials,
    geometry: object | None = None,
) -> Dataset:
    """Reduce aligned scenes into a single composite ``Dataset``.

    Args:
        windowed: Aligned windowed scene datasets (all on the same grid).
        reducer: The client-side reducer to apply over the scene axis.
        credentials: Resolved credentials (carried onto the returned ``Dataset``).
        geometry: Optional polygon cutline to clip the composite to.

    Returns:
        A single composite pyramids ``Dataset``.
    """
    stack = np.stack([scene.ReadAsArray() for scene in windowed], axis=0)
    nodata = windowed[0].GetRasterBand(1).GetNoDataValue()
    reduced = _reduce(stack, reducer, nodata)
    composite = Dataset(
        _build_like(windowed[0], reduced, nodata), gdal_env=credentials.gdal_env()
    )
    return _apply_geometry(composite, geometry)


def from_earthengine(
    asset_id: str,
    *,
    bands: list[str] | None = None,
    bbox: BBox | None = None,
    geometry: object | None = None,
    crs: str = "EPSG:4326",
    scale: float | None = None,
    shape: tuple[int, int] | None = None,
    tile_size: int | None = None,
    start: str | None = None,
    end: str | None = None,
    reducer: str | None = None,
    credentials: CredentialsLike = None,
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
        tile_size: Optional maximum pixels per axis for a single read (single
            ``Image`` mode). When the target window is larger, the AOI is read as a
            grid of ``<= tile_size`` px tiles and mosaicked locally. EEDAI has no
            hard request cap, so this bounds local memory, not an EE limit. Needs
            ``scale`` or ``shape`` to know the target grid.
        start: Inclusive ISO start of the acquisition window (composite mode).
        end: Inclusive ISO end of the acquisition window (composite mode).
        reducer: Client-side reducer for the composite mode — one of ``"median"``,
            ``"mean"``, ``"min"``, ``"max"``, ``"sum"``, ``"mode"``, ``"mosaic"``.
        credentials: An
            :class:`~pyramids_eo.earthengine.credentials.EarthEngineCredentials`, a
            path to a service-account JSON key, or ``None`` for ADC.

    Returns:
        A pyramids :class:`~pyramids.dataset.Dataset` — the windowed image or the
        reduced composite.

    Raises:
        ValueError: ``scale`` and ``shape`` are both given; or ``start`` / ``end``
            are given without a ``reducer`` (use
            :func:`collection_from_earthengine` for a ``DatasetCollection``); or the
            composite mode is missing ``start`` / ``end`` / ``bbox``.
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
    if scale is not None and shape is not None:
        raise ValueError("Pass at most one of 'scale' or 'shape', not both.")

    creds = EarthEngineCredentials.coerce(credentials)
    if geometry is not None:
        geometry = _geometry_in_crs(geometry, crs)
        if bbox is None:
            bbox = _geometry_bounds(geometry)

    if reducer is not None or start is not None or end is not None:
        if reducer is None:
            raise ValueError(
                "'start'/'end' select an ImageCollection; pass 'reducer' for a "
                "single composite, or use collection_from_earthengine() for a "
                "DatasetCollection."
            )
        if start is None or end is None or bbox is None:
            raise ValueError(
                "The composite mode requires 'start', 'end', and a 'bbox' or "
                "'geometry'."
            )
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
        windowed = _read_scenes_aligned(
            scenes,
            bbox=bbox,
            crs=crs,
            scale=scale,
            shape=shape,
            bands=bands,
            credentials=creds,
        )
        return _retain_credentials(
            _composite(windowed, reducer, creds, geometry), creds
        )

    src = _open_eedai(asset_id, bands=bands, credentials=creds)
    if bbox is None:
        if scale is not None or shape is not None or crs != "EPSG:4326":
            raise ReaderError(
                "A 'bbox' is required to window an Earth Engine asset when "
                "'crs', 'scale', or 'shape' is set (assets are global/huge)."
            )
        # The whole-asset wrap is read lazily, so pixel reads happen after this
        # returns — outside any `activate()` block. Install the credential config
        # process-wide so those deferred EEDAI reads still authenticate.
        for config_key, config_value in creds.gdal_env().items():
            gdal.SetConfigOption(config_key, config_value)
        return _retain_credentials(Dataset(src, gdal_env=creds.gdal_env()), creds)

    windowed_single = None
    if tile_size is not None:
        windowed_single = _tiled_window(
            src, bbox=bbox, crs=crs, scale=scale, shape=shape, tile_size=tile_size
        )
    if windowed_single is None:
        windowed_single = _window(src, bbox=bbox, crs=crs, scale=scale, shape=shape)
    return _retain_credentials(
        _apply_geometry(Dataset(windowed_single, gdal_env=creds.gdal_env()), geometry),
        creds,
    )


def collection_from_earthengine(
    asset_id: str,
    *,
    start: str,
    end: str,
    bbox: BBox | None = None,
    geometry: object | None = None,
    bands: list[str] | None = None,
    crs: str = "EPSG:4326",
    scale: float | None = None,
    shape: tuple[int, int] | None = None,
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
    windowed = _read_scenes_aligned(
        scenes,
        bbox=bbox,
        crs=crs,
        scale=scale,
        shape=shape,
        bands=bands,
        credentials=creds,
    )
    env = creds.gdal_env()
    datasets = [
        _retain_credentials(
            _apply_geometry(Dataset(scene, gdal_env=env), geometry), creds
        )
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
