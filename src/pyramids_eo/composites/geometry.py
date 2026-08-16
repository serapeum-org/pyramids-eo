"""Solar geometry for day/night compositing.

`solar_zenith_angle` is the pyramids-eo port of satpy's `get_cos_sza`: the
per-pixel solar zenith angle (SZA) from a UTC time and a lon/lat grid, computed
directly with the NOAA solar-position algorithm over NumPy — **no pyorbital / no
PyTroll dependency**. The SZA drives the day/night cross-fade (`day_night_blend`),
which keys off the Sun's *geometric* position rather than how dark a pixel looks
(the property that renders an eclipse shadow as day, not night).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np


def _to_utc(time: _dt.datetime) -> _dt.datetime:
    """Return `time` as a timezone-aware UTC datetime.

    A naive datetime is assumed to already be UTC; an aware one is converted.

    Args:
        time: The observation time.

    Returns:
        The same instant expressed in UTC.

    Raises:
        TypeError: When `time` is not a `datetime.datetime`.
    """
    if not isinstance(time, _dt.datetime):
        raise TypeError("time must be a datetime.datetime")
    if time.tzinfo is None:
        return time.replace(tzinfo=_dt.UTC)
    return time.astimezone(_dt.UTC)


def solar_zenith_angle(
    time: _dt.datetime,
    *,
    lat: Any = None,
    lon: Any = None,
    grid: Any = None,
) -> np.ndarray:
    """Per-pixel solar zenith angle (degrees) for a UTC time and a lon/lat grid.

    Port of satpy's `get_cos_sza`, computed directly with the NOAA solar-position
    algorithm (no pyorbital). Provide the geographic coordinates either as `lat`
    and `lon` arrays/scalars, or as a `grid` (a pyramids `Dataset` in EPSG:4326,
    whose `lat` / `lon` cell-centre axes are meshed to 2-D).

    The returned angle is the zenith angle in **degrees** (0 = Sun overhead,
    90 = on the horizon / terminator, 180 = antisolar). `day_night_blend` takes
    this and applies `cos(deg2rad(...))` internally.

    Args:
        time: Observation time. A naive datetime is treated as UTC; an aware one
            is converted to UTC.
        lat: Latitude(s) in degrees north — scalar or array. Mutually exclusive
            with `grid`; must be paired with `lon`.
        lon: Longitude(s) in degrees east — scalar or array, broadcast against
            `lat`. Mutually exclusive with `grid`; must be paired with `lat`.
        grid: A pyramids `Dataset` (EPSG:4326) supplying `lat` / `lon` cell-centre
            axes. Mutually exclusive with `lat` / `lon`.

    Returns:
        The solar zenith angle in degrees. Shape follows the broadcast of `lat`
        and `lon`, or `(grid.rows, grid.columns)` for a `grid`.

    Raises:
        ValueError: When neither `grid` nor both `lat` and `lon` are given, when
            both are given, or when `grid` is not geographic (EPSG:4326).

    Examples:
        - The Sun is nearly overhead at (0degN, 0degE) at equinox noon:
            ```python
            >>> import datetime as dt
            >>> from pyramids_eo.composites import solar_zenith_angle
            >>> t = dt.datetime(2024, 3, 20, 12, 0, tzinfo=dt.timezone.utc)
            >>> bool(solar_zenith_angle(t, lat=0.0, lon=0.0) < 5)
            True

            ```
        - The antisolar point is in deep night (SZA near 180deg):
            ```python
            >>> bool(solar_zenith_angle(t, lat=0.0, lon=180.0) > 175)
            True

            ```
    """
    if grid is not None:
        if lat is not None or lon is not None:
            raise ValueError("pass either `grid` or (`lat`, `lon`), not both")
        epsg = getattr(grid, "epsg", None)
        if epsg is not None and int(epsg) != 4326:
            raise ValueError(
                f"grid must be geographic (EPSG:4326); got EPSG:{epsg}. "
                "Reproject it with to_crs(4326) first."
            )
        lon2d, lat2d = np.meshgrid(
            np.asarray(grid.lon, dtype=float), np.asarray(grid.lat, dtype=float)
        )
    else:
        if lat is None or lon is None:
            raise ValueError("provide `grid`, or both `lat` and `lon`")
        lat2d, lon2d = np.broadcast_arrays(
            np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        )

    utc = _to_utc(time)
    day_of_year = utc.timetuple().tm_yday
    hour = utc.hour + utc.minute / 60 + utc.second / 3600 + utc.microsecond / 3.6e9

    # NOAA fractional-year angle (radians) and its harmonics.
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1 + (hour - 12) / 24)
    # Equation of time (minutes) and solar declination (radians).
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    # True solar time (minutes) per pixel: UTC clock time + equation of time +
    # 4 min per degree of east longitude (timezone offset is 0 for UTC).
    true_solar_time = hour * 60 + eqtime + 4.0 * lon2d
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)

    lat_rad = np.deg2rad(lat2d)
    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(
        decl
    ) * np.cos(hour_angle)
    return np.asarray(
        np.rad2deg(np.arccos(np.clip(cos_zenith, -1.0, 1.0))), dtype=float
    )
