"""Day/night compositing for EO imagery.

pyramids-eo's port of satpy's day/night composite chain (the
`true_color_with_night_ir` look), implemented over NumPy + pyramids-gis with no
satpy / PyTroll dependency. So far:

* `solar_zenith_angle` — per-pixel solar zenith angle (satpy `get_cos_sza`
  equivalent), the geometry the day/night blend keys off.
* `day_night_blend` / `day_weight` — the SZA-weighted cross-fade of a day and a
  night image (satpy `DayNightCompositor`).
* `alpha_overlay` — the "over" composite of an RGBA foreground on an RGB(A)
  background (satpy `BackgroundCompositor`).

The remaining primitive (`static_image`) lands alongside these as its sub-issue
is implemented.
"""

from __future__ import annotations

from pyramids_eo.composites.blend import day_night_blend, day_weight
from pyramids_eo.composites.geometry import solar_zenith_angle
from pyramids_eo.composites.overlay import alpha_overlay

__all__ = [
    "alpha_overlay",
    "day_night_blend",
    "day_weight",
    "solar_zenith_angle",
]
