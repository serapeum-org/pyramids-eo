"""Day/night compositing for EO imagery.

pyramids-eo's port of satpy's day/night composite chain (the
`true_color_with_night_ir` look), implemented over NumPy + pyramids-gis with no
satpy / PyTroll dependency. So far:

* `solar_zenith_angle` — per-pixel solar zenith angle (satpy `get_cos_sza`
  equivalent), the geometry the day/night blend keys off.

The remaining primitives (`day_night_blend`, `alpha_overlay`, `static_image`)
land alongside this as their sub-issues are implemented.
"""

from __future__ import annotations

from pyramids_eo.composites.geometry import solar_zenith_angle

__all__ = [
    "solar_zenith_angle",
]
