"""Day/night compositing for EO imagery.

pyramids-eo's port of satpy's day/night composite chain (the
`true_color_with_night_ir` look), implemented over NumPy + pyramids-gis with no
satpy / PyTroll dependency. So far:

* `solar_zenith_angle` — per-pixel solar zenith angle (satpy `get_cos_sza`
  equivalent), the geometry the day/night blend keys off.
* `day_night_blend` / `day_weight` — the SZA-weighted cross-fade of a day and a
  night image (satpy `DayNightCompositor`).

The remaining primitives (`alpha_overlay`, `static_image`) land alongside these
as their sub-issues are implemented.
"""

from __future__ import annotations

from pyramids_eo.composites.blend import day_night_blend, day_weight
from pyramids_eo.composites.geometry import solar_zenith_angle

__all__ = [
    "day_night_blend",
    "day_weight",
    "solar_zenith_angle",
]
