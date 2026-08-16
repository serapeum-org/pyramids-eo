"""Solar-zenith-angle day/night blending.

`day_night_blend` is the pyramids-eo port of satpy's `DayNightCompositor`: it
cross-fades a *day* image and a *night* image by the per-pixel solar zenith
angle (SZA), producing the smooth twilight transition of the
`true_color_with_night_ir` look. The blend keys off the Sun's geometric
position (from `solar_zenith_angle`), not on how dark a pixel looks — which is
why an eclipse shadow is rendered as day, not night.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids_eo.composites._common import _as_array, _wrap_like

_MODES = ("day_night", "day_only", "night_only")


def day_weight(sza: Any, lim_low: float = 78.0, lim_high: float = 88.0) -> np.ndarray:
    """Per-pixel day weight in `[0, 1]` from the solar zenith angle.

    The weight is 1 where the Sun is high (`sza <= lim_low`, full day), 0 where
    it is low (`sza >= lim_high`, full night), and a smooth cos-space ramp
    through the twilight band between them — the DayNightCompositor curve.

    Args:
        sza: Solar zenith angle in degrees (scalar or array), e.g. from
            `solar_zenith_angle`.
        lim_low: SZA (degrees) at/below which it is fully day. Default 78.
        lim_high: SZA (degrees) at/above which it is fully night. Default 88.

    Returns:
        The day weight, same shape as `sza`, clipped to `[0, 1]`.

    Raises:
        ValueError: When `lim_low >= lim_high`.

    Examples:
        - Overhead Sun is full day, horizon is full night, midway is ~0.5:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import day_weight
            >>> day_weight(np.array([0.0, 83.0, 90.0])).round(2).tolist()
            [1.0, 0.5, 0.0]

            ```
    """
    if lim_low >= lim_high:
        raise ValueError(
            f"lim_low ({lim_low}) must be < lim_high ({lim_high}) in SZA degrees"
        )
    low = np.cos(np.deg2rad(lim_low))
    high = np.cos(np.deg2rad(lim_high))
    coszen = np.cos(np.deg2rad(np.asarray(sza, dtype=float)))
    weight = (coszen - min(low, high)) / abs(low - high)
    return np.asarray(np.clip(weight, 0.0, 1.0), dtype=float)


def day_night_blend(
    day: Any,
    night: Any,
    sza: Any,
    *,
    lim_low: float = 78.0,
    lim_high: float = 88.0,
    mode: str = "day_night",
) -> Any:
    """Cross-fade a day and a night image by solar zenith angle.

    Port of satpy's `DayNightCompositor`. Computes a per-pixel day weight from
    `sza` (see `day_weight`) and mixes: `day * weight + night * (1 - weight)`.
    `day` / `night` may be `(H, W)` or `(band, H, W)` arrays, or pyramids
    `Dataset` objects; the weight broadcasts across bands.

    Args:
        day: The day image — array-like or a pyramids `Dataset`.
        night: The night image — same shape/type family as `day`. Ignored when
            `mode="day_only"`.
        sza: Per-pixel solar zenith angle in degrees (`(H, W)`), from
            `solar_zenith_angle`.
        lim_low: SZA at/below which it is fully day (default 78).
        lim_high: SZA at/above which it is fully night (default 88).
        mode: `"day_night"` (blend, default), `"day_only"` (`day * weight`), or
            `"night_only"` (`night * (1 - weight)`).

    Returns:
        The blended image. A pyramids `Dataset` (carrying `day`'s / `night`'s
        geotransform + CRS) when either input is a `Dataset`, otherwise an
        ndarray.

    Raises:
        ValueError: When `mode` is unknown or `lim_low >= lim_high`.

    Examples:
        - A day (1s) / night (0s) pair collapses to the day weight per pixel:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import day_night_blend
            >>> day = np.ones((2, 2))
            >>> night = np.zeros((2, 2))
            >>> sza = np.array([[0.0, 83.0], [88.0, 180.0]])
            >>> day_night_blend(day, night, sza).round(2).tolist()
            [[1.0, 0.5], [0.0, 0.0]]

            ```
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")

    weight = day_weight(_as_array(sza), lim_low=lim_low, lim_high=lim_high)
    day_arr = _as_array(day)
    if weight.ndim == 2 and day_arr.ndim == 3:
        weight = weight[np.newaxis, ...]

    if mode == "day_only":
        out = day_arr * weight
    elif mode == "night_only":
        out = _as_array(night) * (1.0 - weight)
    else:
        out = day_arr * weight + _as_array(night) * (1.0 - weight)
    return _wrap_like(out, day, night)
