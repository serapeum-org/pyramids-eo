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

import numbers
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
            `width`/`height` are not whole positive pixel counts, `method` is
            unknown, or `crs` is a bool.
        RuntimeError: When the GDAL warp fails.

    Examples:
        - Warp a small WGS84 raster onto an exact 4x4 grid:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids_eo.resample import to_area
            >>> src = Dataset.from_array(
            ...     np.ones((2, 2)),
            ...     geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
            ... )
            >>> out = to_area(src, 4326, (0.0, 0.0, 2.0, 2.0), 4, 4)
            >>> out.read_array().shape
            (4, 4)

            ```
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
    if int(width) != width or int(height) != height:
        raise ValueError(
            f"width and height must be whole numbers of pixels; got {width}x{height}"
        )
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive; got {width}x{height}")

    # An integer EPSG code must be a string GDAL's SRS parser accepts — including
    # a NumPy integer (e.g. an epsg looked up from an array), which is not a
    # Python `int`. A `bool` is also `Integral` but is never a real code, so
    # reject it rather than mapping `True` to "EPSG:1". `outputBounds` is left in
    # the target CRS (dstSRS), so no separate `-te_srs` is needed.
    if isinstance(crs, bool):
        raise ValueError(
            f"crs must be an EPSG code or CRS string, not a bool; got {crs!r}"
        )
    dst_srs = f"EPSG:{int(crs)}" if isinstance(crs, numbers.Integral) else str(crs)
    warp_kwargs: dict[str, Any] = {
        "format": "MEM",
        "outputBounds": [min_x, min_y, max_x, max_y],
        "dstSRS": dst_srs,
        "width": int(width),
        "height": int(height),
        "resampleAlg": _RESAMPLERS[method],
    }
    nodata = _warp_nodata(getattr(dataset, "no_data_value", None))
    if nodata is not None:
        warp_kwargs["srcNodata"] = nodata
        warp_kwargs["dstNodata"] = nodata

    out = gdal.Warp("", raster, **warp_kwargs)
    if out is None:
        raise RuntimeError(
            f"gdal.Warp failed for to_area: {gdal.GetLastErrorMsg() or 'no detail'}"
        )
    return Dataset(out)


def _warp_nodata(no_data_value: Any) -> Any:
    """Return a GDAL `srcNodata`/`dstNodata` value from a scalar or per-band nodata.

    Args:
        no_data_value: A scalar nodata, a per-band list/tuple (which may hold
            `None` entries), or `None`.

    Returns:
        The scalar for one band, a space-separated per-band string for several
        bands (each band's own value preserved, the form GDAL accepts), or
        `None` when there is no nodata to carry through.
    """
    if isinstance(no_data_value, (list, tuple)):
        values = [value for value in no_data_value if value is not None]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return " ".join(str(value) for value in values)
    return no_data_value
