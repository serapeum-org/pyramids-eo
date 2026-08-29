"""Solar geometry for day/night compositing.

`solar_zenith_angle` gives the per-pixel solar zenith angle (SZA) from a UTC
time and a lon/lat grid, computed directly with the NOAA solar-position
algorithm over NumPy. The SZA drives the day/night cross-fade
(`day_night_blend`),
which keys off the Sun's *geometric* position rather than how dark a pixel looks
(the property that renders an eclipse shadow as day, not night).

`sunz_correct` and `sunz_reduce` consume that angle: the first divides a solar
band by `cos(sza)` (capped, so it does not blow up at the terminator), the second
tapers the signal back down toward the terminator so deep shadow renders dark
rather than as a washed-out floor. Together they mirror the `sunz_corrected` and
`sunz_reduced` modifiers used on every solar prerequisite of a reference
true-colour composite.

For atmospheric corrections, `solar_zenith_azimuth` and `satellite_zenith_azimuth`
add the viewing geometry — the Sun's and a geostationary satellite's zenith and
azimuth per pixel — and `relative_azimuth` folds the two azimuths into the
Sun-satellite difference every correction model expects. All azimuths use one
convention: **degrees clockwise from north** (`[0, 360)`).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np

from pyramids_eo.composites._common import _as_array, _wrap_like

#: Earth equatorial radius (km, WGS84) and geostationary orbital radius (km).
_R_EARTH_KM = 6378.137
_R_GEO_KM = 42164.0


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


def _resolve_coords(lat: Any, lon: Any, grid: Any) -> tuple[np.ndarray, np.ndarray]:
    """Resolve `(lon2d, lat2d)` degree meshes from a lat/lon pair or a grid.

    Args:
        lat: Latitude(s) in degrees north, paired with `lon`.
        lon: Longitude(s) in degrees east, paired with `lat`.
        grid: A pyramids `Dataset` (EPSG:4326), mutually exclusive with lat/lon.

    Returns:
        The broadcast `(lon2d, lat2d)` coordinate arrays in degrees.

    Raises:
        ValueError: When neither `grid` nor both `lat`/`lon` are given, both are
            given, or `grid` is not geographic (EPSG:4326).
    """
    if grid is not None:
        if lat is not None or lon is not None:
            raise ValueError("pass either `grid` or (`lat`, `lon`), not both")
        epsg = getattr(grid, "epsg", None)
        if epsg is None or int(epsg) != 4326:
            raise ValueError(
                f"grid must be geographic (EPSG:4326); got EPSG:{epsg}. A grid "
                "with no EPSG (e.g. geostationary) is not lon/lat — reproject it "
                "with to_crs(4326) first."
            )
        return np.meshgrid(
            np.asarray(grid.lon, dtype=float), np.asarray(grid.lat, dtype=float)
        )
    if lat is None or lon is None:
        raise ValueError("provide `grid`, or both `lat` and `lon`")
    lat2d, lon2d = np.broadcast_arrays(
        np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    )
    return lon2d, lat2d


def _solar_position(
    time: _dt.datetime, lat2d: np.ndarray, lon2d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel solar `(zenith, azimuth)` in degrees via the NOAA algorithm.

    The NOAA declination / equation-of-time / hour-angle terms are computed once
    and drive both outputs, so `solar_zenith_angle` and `solar_zenith_azimuth`
    share this single implementation.

    Args:
        time: Observation time (converted to UTC).
        lat2d: Latitudes in degrees north.
        lon2d: Longitudes in degrees east.

    Returns:
        `(zenith_deg, azimuth_deg)` — zenith `0` (Sun overhead) .. `180`
        (antisolar); azimuth in **degrees clockwise from north**, `[0, 360)`.
    """
    utc = _to_utc(time)
    day_of_year = utc.timetuple().tm_yday
    hour = utc.hour + utc.minute / 60 + utc.second / 3600 + utc.microsecond / 3.6e9

    # NOAA fractional-year angle (radians) and its harmonics.
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1 + (hour - 12) / 24)
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
    true_solar_time = hour * 60 + eqtime + 4.0 * lon2d
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)

    lat_rad = np.deg2rad(lat2d)
    cos_zenith = np.clip(
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle),
        -1.0,
        1.0,
    )
    zenith = np.arccos(cos_zenith)

    # NOAA azimuth, degrees clockwise from north (0=N, 90=E, 180=S, 270=W).
    denom = np.cos(lat_rad) * np.sin(zenith)
    cos_az = np.divide(
        np.sin(lat_rad) * cos_zenith - np.sin(decl),
        denom,
        out=np.zeros_like(cos_zenith),
        where=denom != 0,
    )
    az = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    azimuth = np.where(hour_angle > 0, (az + 180.0) % 360.0, (540.0 - az) % 360.0)

    return (
        np.asarray(np.degrees(zenith), dtype=float),
        np.asarray(azimuth, dtype=float),
    )


def solar_zenith_angle(
    time: _dt.datetime,
    *,
    lat: Any = None,
    lon: Any = None,
    grid: Any = None,
) -> np.ndarray:
    """Per-pixel solar zenith angle (degrees) for a UTC time and a lon/lat grid.

    Computed directly with the NOAA solar-position algorithm. Provide the
    geographic coordinates either as `lat`
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
    lon2d, lat2d = _resolve_coords(lat, lon, grid)
    return _solar_position(time, lat2d, lon2d)[0]


def cos_solar_zenith_angle(
    time: _dt.datetime,
    *,
    lat: Any = None,
    lon: Any = None,
    grid: Any = None,
) -> np.ndarray:
    """Per-pixel cosine of the solar zenith angle.

    A thin wrapper over `solar_zenith_angle` returning `cos(SZA)` directly — the
    form the readers' and `radiance_to_reflectance`'s `cos_sza` arguments expect
    (note `solar_zenith_angle` itself returns the angle in *degrees*). Same
    arguments as `solar_zenith_angle`.

    Args:
        time: Observation time (a naive datetime is treated as UTC).
        lat: Latitude(s) in degrees north, paired with `lon`.
        lon: Longitude(s) in degrees east, paired with `lat`.
        grid: A pyramids `Dataset` grid (EPSG:4326), mutually exclusive with
            `lat` / `lon`.

    Returns:
        The cosine of the solar zenith angle, same shape as the coordinates.
    """
    return np.cos(np.deg2rad(solar_zenith_angle(time, lat=lat, lon=lon, grid=grid)))


def solar_zenith_azimuth(
    time: _dt.datetime,
    *,
    lat: Any = None,
    lon: Any = None,
    grid: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel solar `(zenith, azimuth)` in degrees.

    Same coordinate inputs as `solar_zenith_angle` (a `lat`/`lon` pair, or an
    EPSG:4326 `grid`). The returned zenith is identical to `solar_zenith_angle`;
    the azimuth is the NOAA solar azimuth in **degrees clockwise from north**
    (`[0, 360)`: 0 = north, 90 = east, 180 = south, 270 = west) — the same
    convention as `satellite_zenith_azimuth`, so the two feed `relative_azimuth`.

    Args:
        time: Observation time (a naive datetime is treated as UTC).
        lat: Latitude(s) in degrees north, paired with `lon`.
        lon: Longitude(s) in degrees east, paired with `lat`.
        grid: A pyramids `Dataset` grid (EPSG:4326), mutually exclusive with
            `lat` / `lon`.

    Returns:
        `(zenith, azimuth)` arrays in degrees.

    Raises:
        ValueError: Same coordinate-argument errors as `solar_zenith_angle`.

    Examples:
        - At equinox noon on the Greenwich meridian the Sun is due south:
            ```python
            >>> import datetime as dt
            >>> from pyramids_eo.composites import solar_zenith_azimuth
            >>> t = dt.datetime(2024, 3, 20, 12, 0, tzinfo=dt.timezone.utc)
            >>> _, az = solar_zenith_azimuth(t, lat=45.0, lon=0.0)
            >>> bool(170.0 < az < 190.0)
            True

            ```
    """
    lon2d, lat2d = _resolve_coords(lat, lon, grid)
    return _solar_position(time, lat2d, lon2d)


def satellite_zenith_azimuth(
    lat: Any = None,
    lon: Any = None,
    *,
    grid: Any = None,
    sat_lon: float = 0.0,
    sat_radius_km: float = _R_GEO_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel `(zenith, azimuth)` of a geostationary satellite, in degrees.

    Closed-form and time-invariant (a geostationary satellite sits at a fixed
    sub-satellite longitude). The zenith is `0` at the sub-satellite point and
    rises toward `90` at the limb; the azimuth is in **degrees clockwise from
    north** (`[0, 360)`), the same convention as `solar_zenith_azimuth`, so the
    two feed `relative_azimuth`.

    Args:
        lat: Latitude(s) in degrees north, paired with `lon`.
        lon: Longitude(s) in degrees east, paired with `lat`.
        grid: A pyramids `Dataset` grid (EPSG:4326), mutually exclusive with
            `lat` / `lon`.
        sat_lon: Sub-satellite longitude in degrees east (default 0.0, Meteosat
            prime).
        sat_radius_km: Satellite orbital **radius from Earth's centre** in km
            (default 42164, geostationary) — this is a radius, not an altitude.

    Returns:
        `(zenith, azimuth)` arrays in degrees.

    Raises:
        ValueError: Same coordinate-argument errors as `solar_zenith_angle`.

    Examples:
        - The sub-satellite point sees the satellite at the local zenith (0deg):
            ```python
            >>> from pyramids_eo.composites import satellite_zenith_azimuth
            >>> vza, _ = satellite_zenith_azimuth(0.0, 0.0, sat_lon=0.0)
            >>> bool(vza < 0.01)
            True

            ```
    """
    lon2d, lat2d = _resolve_coords(lat, lon, grid)
    la = np.deg2rad(lat2d)
    dlon = np.deg2rad(lon2d - sat_lon)
    cos_psi = np.clip(np.cos(la) * np.cos(dlon), -1.0, 1.0)
    psi = np.arccos(cos_psi)
    distance = np.sqrt(
        _R_EARTH_KM**2 + sat_radius_km**2 - 2.0 * _R_EARTH_KM * sat_radius_km * cos_psi
    )
    zenith = np.degrees(
        np.arcsin(np.clip(sat_radius_km * np.sin(psi) / distance, -1.0, 1.0))
    )
    azimuth = np.degrees(np.arctan2(np.sin(-dlon), -np.sin(la) * np.cos(dlon))) % 360.0
    return np.asarray(zenith, dtype=float), np.asarray(azimuth, dtype=float)


def relative_azimuth(sun_az: Any, sat_az: Any) -> np.ndarray:
    """Sun-satellite azimuth difference, folded into `[0, 180]` degrees.

    The absolute difference of two azimuths (each in `[0, 360)`, clockwise from
    north) folded to `[0, 180]` — the relative-azimuth input every atmospheric
    correction expects. Order-independent.

    Args:
        sun_az: Solar azimuth(s) in degrees (e.g. from `solar_zenith_azimuth`).
        sat_az: Satellite azimuth(s) in degrees (e.g. from
            `satellite_zenith_azimuth`).

    Returns:
        The relative azimuth in `[0, 180]` degrees, broadcast over the inputs.

    Examples:
        - Opposed azimuths (0deg and 180deg) give the maximum, 180deg:
            ```python
            >>> from pyramids_eo.composites import relative_azimuth
            >>> float(relative_azimuth(10.0, 190.0))
            180.0

            ```
        - Azimuths 350deg and 10deg are only 20deg apart (wrap-around):
            ```python
            >>> from pyramids_eo.composites import relative_azimuth
            >>> float(relative_azimuth(350.0, 10.0))
            20.0

            ```
    """
    diff = (
        np.abs(np.asarray(sun_az, dtype=float) - np.asarray(sat_az, dtype=float))
        % 360.0
    )
    return np.asarray(np.where(diff > 180.0, 360.0 - diff, diff), dtype=float)


def sunz_correct(
    band: Any,
    sza: Any,
    *,
    correction_limit: float = 88.0,
    max_sza: float | None = 95.0,
) -> Any:
    """Correct a solar band for the sun-zenith angle (divide by `cos(sza)`, capped).

    A plain `1 / cos(sza)` diverges at the terminator, so the correction is capped:
    below `correction_limit` the factor is `1 / cos(sza)`; from `correction_limit`
    to `max_sza` it tapers from `1 / cos(correction_limit)` down to `0` by an
    inverted-`log2` ramp; beyond `max_sza` (and where `sza` is NaN) the factor is
    `0`. The band is multiplied by this factor. This mirrors the reference
    `sunz_corrected` modifier (`SunZenithCorrector`), whose defaults are
    `correction_limit=88`, `max_sza=95`.

    Args:
        band: The solar band (reflectance) — array-like or a pyramids `Dataset`.
        sza: Per-pixel solar zenith angle in **degrees** (e.g. from
            `solar_zenith_angle`), broadcastable against `band`.
        correction_limit: SZA (degrees), in `[0, 90)`, at/beyond which the
            `1 / cos` factor is capped and starts to taper. Also the largest
            correction applied (`1 / cos(correction_limit)`).
        max_sza: SZA (degrees) at which the correction reaches `0`, or `None` to
            hold the factor constant at `1 / cos(correction_limit)` beyond the
            limit (no taper). Must exceed `correction_limit`.

    Returns:
        The corrected band (`band` times the per-pixel factor). A pyramids
        `Dataset` (carrying `band`'s geotransform + CRS) when `band` is a
        `Dataset`, otherwise an ndarray (a 0-d array for scalar inputs). The
        *factor* is `0` beyond `max_sza` and where `sza` is NaN, but a NaN band
        value stays NaN (`0 * NaN`), so off-disk nodata is preserved, not blacked.

    Raises:
        ValueError: When `correction_limit` is not in `[0, 90)`, or `max_sza` is
            not greater than `correction_limit`.

    Examples:
        - Overhead sun is unchanged; a 60deg zenith doubles the signal (`1/cos60`):
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import sunz_correct
            >>> sunz_correct(np.array([1.0, 1.0]), np.array([0.0, 60.0])).round(3).tolist()
            [1.0, 2.0]

            ```
        - The correction stays finite at the terminator (no `1/0` blow-up):
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import sunz_correct
            >>> bool(np.isfinite(sunz_correct(np.array([1.0]), np.array([90.0]))).all())
            True

            ```
    """
    if not 0.0 <= correction_limit < 90.0:
        raise ValueError(
            f"correction_limit must be in [0, 90) degrees, got {correction_limit}"
        )
    if max_sza is not None and max_sza <= correction_limit:
        raise ValueError(
            f"max_sza ({max_sza}) must be greater than correction_limit "
            f"({correction_limit})"
        )
    arr = _as_array(band)
    angle = _as_array(sza)
    cos_zen = np.cos(np.deg2rad(angle))
    limit_cos = float(np.cos(np.deg2rad(correction_limit)))

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = 1.0 / cos_zen
    if max_sza is not None:
        ramp = (angle - correction_limit) / (max_sza - correction_limit)
        with np.errstate(invalid="ignore"):
            grad_factor = 1.0 - np.log2(ramp + 1.0)
        grad_factor = np.clip(grad_factor, 0.0, None)
    else:
        grad_factor = np.ones_like(cos_zen)

    corr = np.where(cos_zen > limit_cos, corr, grad_factor / limit_cos)
    corr = np.where(np.isnan(cos_zen), 0.0, corr)
    return _wrap_like(arr * corr, band)


def sunz_reduce(
    band: Any,
    sza: Any,
    *,
    correction_limit: float = 80.0,
    max_sza: float | None = 90.0,
    strength: float = 1.3,
) -> Any:
    """Taper a solar band's signal toward the terminator so deep shadow reads dark.

    Below `correction_limit` the signal is unchanged; from `correction_limit` to
    `max_sza` it is multiplied by a factor that ramps from `1` down to `0` by an
    inverted-`log2` curve sharpened by a `strength` sigmoid; at/beyond `max_sza`
    (and where `sza` is NaN) the factor is `0`. This mirrors the reference
    `sunz_reduced` modifier (`SunZenithReducer`), whose defaults are
    `correction_limit=80`, `max_sza=90`, `strength=1.3`.

    Args:
        band: The solar band (reflectance) — array-like or a pyramids `Dataset`.
        sza: Per-pixel solar zenith angle in **degrees** (e.g. from
            `solar_zenith_angle`), broadcastable against `band`.
        correction_limit: SZA (degrees), `>= 0`, below which the signal is
            unchanged.
        max_sza: SZA (degrees) at which the signal is fully reduced to `0`. Must
            exceed `correction_limit`; required (unlike `sunz_correct`, `None` is
            rejected).
        strength: Sigmoid power sharpening the reduction ramp; `1.0` leaves it as
            the plain inverted-`log2` curve. Must be `> 0`.

    Returns:
        The reduced band (`band` times the per-pixel factor). A pyramids `Dataset`
        (carrying `band`'s geotransform + CRS) when `band` is a `Dataset`,
        otherwise an ndarray (a 0-d array for scalar inputs). The *factor* is `0`
        at/beyond `max_sza` and where `sza` is NaN, but a NaN band value stays NaN
        (`0 * NaN`), so off-disk nodata is preserved, not blacked.

    Raises:
        ValueError: When `max_sza` is `None` or not greater than
            `correction_limit`, `correction_limit < 0`, or `strength <= 0`.

    Examples:
        - Unchanged below the limit, fully reduced to `0` at `max_sza`:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import sunz_reduce
            >>> sunz_reduce(np.array([1.0, 1.0]), np.array([70.0, 90.0])).round(3).tolist()
            [1.0, 0.0]

            ```
        - Inside the band the signal is dimmed (a shadow, not a grey floor):
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import sunz_reduce
            >>> bool(sunz_reduce(np.array([1.0]), np.array([85.0]))[0] < 1.0)
            True

            ```
    """
    if max_sza is None:
        raise ValueError("max_sza is required for sunz_reduce (got None)")
    if correction_limit < 0:
        raise ValueError(f"correction_limit must be >= 0, got {correction_limit}")
    if max_sza <= correction_limit:
        raise ValueError(
            f"max_sza ({max_sza}) must be greater than correction_limit "
            f"({correction_limit})"
        )
    if strength <= 0:
        raise ValueError(f"strength must be > 0, got {strength}")
    arr = _as_array(band)
    angle = _as_array(sza)

    ramp = np.clip((angle - correction_limit) / (max_sza - correction_limit), 0.0, 1.0)
    reduction = 1.0 - np.log2(ramp + 1.0)
    reduction = reduction**strength / (
        reduction**strength + (1.0 - reduction) ** strength
    )
    corr = np.where(angle < correction_limit, 1.0, reduction)
    corr = np.where(np.isnan(angle), 0.0, corr)
    return _wrap_like(arr * corr, band)
