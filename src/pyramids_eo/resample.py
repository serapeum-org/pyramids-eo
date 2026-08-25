"""Resample a composite onto an explicit target grid.

`to_area` places a georeferenced composite onto an exact target grid — a CRS, an
extent, and a pixel width/height — with a chosen interpolation method, in a
single GDAL warp. This is the one step of the day/night chain with no pyramids
equivalent: `Dataset.to_crs` lets GDAL pick the grid, and `Dataset.align` needs a
pre-built template `Dataset` already on the grid. A direct warp pins
`-t_srs` / `-te` / `-ts` / `-r` together and lands the exact grid in one pass —
the same primitive the Earth Engine reader keeps for the same reason.
"""

from __future__ import annotations

from typing import Any

import pyramids as _pyramids_bootstrap  # noqa: F401  (activates the bundled osgeo)
from osgeo import gdal
from pyramids.dataset import Dataset

#: Supported resampling methods mapped to their GDAL algorithm.
_RESAMPLERS: dict[str, int] = {
    "nearest": gdal.GRA_NearestNeighbour,
    "bilinear": gdal.GRA_Bilinear,
    "cubic": gdal.GRA_Cubic,
    "cubicspline": gdal.GRA_CubicSpline,
    "lanczos": gdal.GRA_Lanczos,
    "average": gdal.GRA_Average,
    "mode": gdal.GRA_Mode,
}


def to_area(
    dataset: Any,
    crs: Any,
    extent: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    method: str = "bilinear",
) -> Dataset:
    """Warp a composite onto an exact target grid (CRS + extent + size).

    Args:
        dataset: The source composite — a pyramids `Dataset` (needs a `.raster`
            handle to warp; an ndarray has no georeferencing to place).
        crs: Target CRS — any string GDAL accepts, including an EPSG code or a
            PROJ4 string with no EPSG (e.g. a `+proj=nsper ...` perspective).
        extent: Output bounds `(min_x, min_y, max_x, max_y)` in `crs` units.
        width: Output width in pixels.
        height: Output height in pixels.
        method: Resampling method — one of `to_area`'s supported names
            (`"bilinear"` default, `"nearest"`, `"cubic"`, ...).

    Returns:
        A pyramids `Dataset` on exactly that grid — CRS `crs`, extent `extent`,
        shape `(bands, height, width)` — with band count, dtype and nodata
        preserved from `dataset`.

    Raises:
        ValueError: When `dataset` has no raster handle, `extent` is malformed,
            `width`/`height` are not positive, or `method` is unknown.
        RuntimeError: When the GDAL warp fails.
    """
    raster = getattr(dataset, "raster", None)
    if raster is None:
        raise ValueError(
            "to_area needs a georeferenced pyramids Dataset (with a raster "
            "handle); an ndarray has no grid to place."
        )
    if method not in _RESAMPLERS:
        raise ValueError(f"method must be one of {sorted(_RESAMPLERS)}; got {method!r}")
    if len(extent) != 4:
        raise ValueError(f"extent must be (min_x, min_y, max_x, max_y); got {extent!r}")
    min_x, min_y, max_x, max_y = (float(v) for v in extent)
    if not (min_x < max_x and min_y < max_y):
        raise ValueError(
            f"extent must have min_x < max_x and min_y < max_y; got {extent!r}"
        )
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive; got {width}x{height}")

    # An int EPSG code must be a string GDAL's SRS parser accepts. `outputBounds`
    # is left in the target CRS (dstSRS), so no separate `-te_srs` is needed.
    dst_srs = f"EPSG:{crs}" if isinstance(crs, int) else str(crs)
    warp_kwargs: dict[str, Any] = {
        "format": "MEM",
        "outputBounds": [min_x, min_y, max_x, max_y],
        "dstSRS": dst_srs,
        "width": int(width),
        "height": int(height),
        "resampleAlg": _RESAMPLERS[method],
    }
    nodata = _scalar_nodata(getattr(dataset, "no_data_value", None))
    if nodata is not None:
        warp_kwargs["srcNodata"] = nodata
        warp_kwargs["dstNodata"] = nodata

    out = gdal.Warp("", raster, **warp_kwargs)
    if out is None:
        raise RuntimeError(
            f"gdal.Warp failed for to_area: {gdal.GetLastErrorMsg() or 'no detail'}"
        )
    return Dataset(out)


def _scalar_nodata(no_data_value: Any) -> Any:
    """Return a single nodata value from a scalar or per-band nodata.

    Args:
        no_data_value: A scalar nodata, a per-band list/tuple, or `None`.

    Returns:
        The scalar nodata (the first band's, for a list), or `None` when there
        is none to carry through.
    """
    if isinstance(no_data_value, (list, tuple)):
        return no_data_value[0] if no_data_value else None
    return no_data_value
