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

from pyramids_eo.composites._common import _as_array, _wrap_like
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

    Returns:
        The composed day/night image — a pyramids `Dataset` when the inputs carry
        one, otherwise an ndarray.
    """
    night = alpha_overlay(night_ir_rgba, background)
    return day_night_blend(day, night, sza, lim_low=lim_low, lim_high=lim_high)
