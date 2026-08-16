"""Turnkey Google Earth Engine reader → pyramids ``Dataset``.

:func:`from_earthengine` pulls a single Earth Engine ``Image`` asset into a
pyramids :class:`~pyramids.dataset.Dataset` over an AOI, mirroring how
``DatasetCollection.from_stac`` / ``from_featureserver`` give turnkey access to
their sources. It is built entirely on the bundled GDAL ``EEDAI`` driver — no
``earthengine-api`` dependency — with auth carried by
:class:`~pyramids_eo.earthengine.credentials.EarthEngineCredentials`
(Application Default Credentials).

``ImageCollection`` support (``collection_from_earthengine`` and the client-side
``reducer`` composite) lands on top of this same backend in a later step; see
serapeum-org/pyramids-eo#13.
"""

from __future__ import annotations

# isort: off
import pyramids as _pyramids_bootstrap  # noqa: F401  (activates the bundled osgeo)
# isort: on

from osgeo import gdal
from pyramids.dataset import Dataset

from pyramids_eo.earthengine.credentials import CredentialsLike, EarthEngineCredentials
from pyramids_eo.errors import ReaderError

BBox = tuple[float, float, float, float]

#: GDAL connection prefix for the Earth Engine Data API *image* (raster) driver.
_EEDAI_PREFIX = "EEDAI:"


def _open_eedai(
    asset_id: str,
    *,
    bands: list[str] | None,
    credentials: EarthEngineCredentials,
) -> gdal.Dataset:
    """Open an Earth Engine ``Image`` asset through the GDAL EEDAI driver.

    This is the single network seam of the reader: tests monkeypatch it with a
    local fixture raster so CI needs no live Earth Engine account.

    Args:
        asset_id: Earth Engine image asset id (e.g. ``"USGS/SRTMGL1_003"``).
        bands: Optional band names to request (EEDAI ``BANDS`` open option).
        credentials: Resolved credentials whose config authorises the read.

    Returns:
        The opened GDAL dataset (whole asset; window/reproject happens later).

    Raises:
        ReaderError: The driver could not open the asset.
    """
    open_options: list[str] = []
    if bands:
        open_options.append("BANDS=" + ",".join(bands))
    with credentials.activate():
        src = gdal.OpenEx(
            _EEDAI_PREFIX + asset_id,
            gdal.OF_RASTER | gdal.OF_VERBOSE_ERROR,
            open_options=open_options,
        )
    if src is None:
        raise ReaderError(
            f"Earth Engine asset {asset_id!r} could not be opened via EEDAI: "
            f"{gdal.GetLastErrorMsg() or 'no detail'}"
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

    Returns an in-memory (``MEM``) GDAL dataset so the result carries no
    on-disk or ``/vsimem`` lifetime.
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


def from_earthengine(
    asset_id: str,
    *,
    bands: list[str] | None = None,
    bbox: BBox | None = None,
    crs: str = "EPSG:4326",
    scale: float | None = None,
    shape: tuple[int, int] | None = None,
    credentials: CredentialsLike = None,
) -> Dataset:
    """Read a single Earth Engine ``Image`` asset into a pyramids ``Dataset``.

    Opens the asset through the GDAL EEDAI driver and, when an AOI is given,
    warps it to ``bbox`` in ``crs`` at the requested resolution — returning a
    standard pyramids :class:`~pyramids.dataset.Dataset` with no Earth Engine
    objects leaking into the return type.

    Args:
        asset_id: Earth Engine image asset id, e.g. ``"USGS/SRTMGL1_003"`` or a
            single ``COPERNICUS/S2_SR_HARMONIZED`` scene id.
        bands: Band names to request; ``None`` reads every band.
        bbox: AOI as ``(min_x, min_y, max_x, max_y)`` in ``crs``. Earth Engine
            assets are global/huge, so a ``bbox`` is required to materialise a
            window; omit it only to wrap the lazy full-asset dataset.
        crs: Target CRS for the output (and the CRS ``bbox`` is expressed in).
            Defaults to ``"EPSG:4326"``.
        scale: Output pixel size in ``crs`` units (for ``EPSG:4326`` that is
            degrees, not metres). Mutually exclusive with ``shape``. When both
            are omitted the source resolution is kept.
        shape: Output ``(rows, cols)``. Mutually exclusive with ``scale``.
        credentials: An
            :class:`~pyramids_eo.earthengine.credentials.EarthEngineCredentials`,
            a path to a service-account JSON key, or ``None`` for Application
            Default Credentials.

    Returns:
        A pyramids :class:`~pyramids.dataset.Dataset` for the requested window.

    Raises:
        ValueError: ``scale`` and ``shape`` are both given, or a windowing
            option (``crs`` other than default / ``scale`` / ``shape``) is given
            without a ``bbox``.
        ReaderError: The asset could not be opened or windowed.

    Examples:
        Read a small SRTM window (requires Earth Engine credentials, so skipped
        offline):

            >>> from pyramids_eo import from_earthengine
            >>> ds = from_earthengine(  # doctest: +SKIP
            ...     "USGS/SRTMGL1_003",
            ...     bbox=(86.9, 27.9, 87.0, 28.0),
            ... )
    """
    if scale is not None and shape is not None:
        raise ValueError("Pass at most one of 'scale' or 'shape', not both.")

    creds = EarthEngineCredentials.coerce(credentials)
    src = _open_eedai(asset_id, bands=bands, credentials=creds)

    if bbox is None:
        if scale is not None or shape is not None or crs != "EPSG:4326":
            raise ReaderError(
                "A 'bbox' is required to window an Earth Engine asset when "
                "'crs', 'scale', or 'shape' is set (assets are global/huge)."
            )
        return Dataset(src, gdal_env=creds.gdal_env())

    windowed = _window(src, bbox=bbox, crs=crs, scale=scale, shape=shape)
    return Dataset(windowed, gdal_env=creds.gdal_env())
