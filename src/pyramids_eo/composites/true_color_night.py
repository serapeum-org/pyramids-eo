"""Assemble the `true_color_with_night_ir` day/night image.

This wires the compositing primitives into the `true_color_with_night_ir`
chain, entirely in pyramids-eo:

```
true_color_with_night_ir = day_night_blend(
    day        = true_color,
    night      = alpha_overlay(night_ir, static_image(BlackMarble)),
    sza        = solar_zenith_angle(...),
)
night_ir = stack(ir_38, ir_105, ir_123) -> RGB + alpha
```

`night_ir` builds the RGBA night-cloud image (the IR-stack step);
`true_color_with_night_ir` overlays it on the city-lights background and
cross-fades against the day image by solar zenith angle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids_eo.composites._common import _as_array, _coverage, _wrap_like
from pyramids_eo.composites.blend import day_night_blend
from pyramids_eo.composites.overlay import alpha_overlay


def night_ir(red: Any, green: Any, blue: Any, *, alpha: Any = None) -> Any:
    """Stack three IR bands into an RGBA night-cloud image.

    Stack the three IR bands (e.g. `ir_38`, `ir_105`, `ir_123`) as RGB and
    attach an alpha channel. The alpha controls where the
    background (city lights) shows through — pass a cloud-derived alpha
    (opaque over cold cloud tops, transparent over clear sky) to reproduce the
    reference look; the default is fully opaque (alpha = 1).

    Args:
        red: The IR band mapped to red (e.g. `ir_38`) — array-like or `Dataset`.
        green: The IR band mapped to green (e.g. `ir_105`).
        blue: The IR band mapped to blue (e.g. `ir_123`).
        alpha: Per-pixel alpha in `[0, 1]`, or `None` for fully opaque.

    Returns:
        A `(4, H, W)` RGBA image — a pyramids `Dataset` when any input is a
        `Dataset`, otherwise an ndarray.
    """
    r = _as_array(red)
    g = _as_array(green)
    b = _as_array(blue)
    a = np.ones_like(r) if alpha is None else _as_array(alpha)
    rgba = np.stack([r, g, b, a], axis=0)
    return _wrap_like(rgba, red, green, blue)


def true_color_with_night_ir(
    day: Any,
    night_ir_rgba: Any,
    background: Any,
    sza: Any,
    *,
    lim_low: float = 78.0,
    lim_high: float = 88.0,
    keep_alpha: bool = False,
) -> Any:
    """Compose the full `true_color_with_night_ir` day/night image.

    Overlays the RGBA night-IR clouds on the city-lights `background`
    (`alpha_overlay`), then cross-fades that night image against the `day`
    true-colour image by solar zenith angle (`day_night_blend`).

    Args:
        day: The day image — true-colour RGB `(3, H, W)` array-like or `Dataset`.
        night_ir_rgba: The RGBA night-IR clouds from `night_ir`.
        background: The RGB city-lights background (e.g. Black Marble warped to
            the grid via `static_image`).
        sza: Per-pixel solar zenith angle in degrees, from `solar_zenith_angle`.
        lim_low: SZA at/below which it is fully day (default 78).
        lim_high: SZA at/above which it is fully night (default 88).
        keep_alpha: When `True`, append a coverage / alpha band to the `(3, H, W)`
            result, giving `(4, H, W)`. Coverage is derived from the two
            *satellite-derived* inputs before the global background is merged in —
            the `day` true-colour image (valid on the day side of the disk) and
            the `night_ir_rgba` clouds (valid on the night side) — so it marks the
            whole sensor disk, including dark-but-valid night pixels, and is `0`
            off-disk. Default `False` returns the 3-band image unchanged.

    Returns:
        The composed day/night image — a pyramids `Dataset` when the inputs carry
        one, otherwise an ndarray. `(4, H, W)` when `keep_alpha=True`, else
        `(3, H, W)`.

    Examples:
        - Compose a day/night frame from the primitives:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import night_ir, true_color_with_night_ir
            >>> day = np.ones((3, 1, 2))
            >>> clouds = night_ir(np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)))
            >>> bg = np.zeros((3, 1, 2))
            >>> true_color_with_night_ir(day, clouds, bg, np.zeros((1, 2))).shape
            (3, 1, 2)

            ```
        - Keep a coverage band for edge feathering (4-band RGBA):
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import night_ir, true_color_with_night_ir
            >>> day = np.ones((3, 1, 2))
            >>> clouds = night_ir(np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)))
            >>> bg = np.zeros((3, 1, 2))
            >>> out = true_color_with_night_ir(
            ...     day, clouds, bg, np.zeros((1, 2)), keep_alpha=True
            ... )
            >>> out.shape
            (4, 1, 2)

            ```
    """
    night = alpha_overlay(night_ir_rgba, background)
    blended = day_night_blend(day, night, sza, lim_low=lim_low, lim_high=lim_high)
    if not keep_alpha:
        return blended

    # Coverage from the satellite-derived inputs *before* the global background
    # is merged in: the day true-colour image covers the day side of the disk,
    # the night-IR clouds (RGB, ignoring their own alpha) cover the night side.
    # Their union is the sensor disk; off-disk both are NaN, so alpha is 0 even
    # though the blended RGB there is the (finite) background.
    coverage = _coverage(day) | _coverage(_as_array(night_ir_rgba)[:3])
    alpha = coverage.astype(float)
    blended_arr = _as_array(blended)
    rgb = blended_arr if blended_arr.ndim >= 3 else blended_arr[np.newaxis, ...]
    rgba = np.concatenate([rgb, alpha[np.newaxis, ...]], axis=0)
    return _wrap_like(rgba, day, night_ir_rgba, background)
