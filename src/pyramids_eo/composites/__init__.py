"""Day/night compositing for EO imagery.

The pyramids-eo day/night composite chain (the `true_color_with_night_ir`
look), implemented over NumPy + pyramids-gis with no third-party compositing
dependency. So far:

* `solar_zenith_angle` / `cos_solar_zenith_angle` — per-pixel solar zenith angle
  (degrees) and its cosine (the `cos_sza` form the readers expect), the geometry
  the day/night blend keys off.
* `sunz_correct` / `sunz_reduce` — sun-zenith correction (divide by `cos(sza)`,
  capped) and reduction (taper the signal toward the terminator), the
  `sunz_corrected` / `sunz_reduced` pair applied to a true-colour composite's
  solar bands so deep shadow renders dark rather than as a washed-out floor.
* `solar_zenith_azimuth` / `satellite_zenith_azimuth` / `relative_azimuth` — the
  solar and geostationary viewing geometry (zenith + azimuth, degrees clockwise
  from north) that an atmospheric correction needs.
* `rayleigh_correct` — a local, closed-form single-scattering Rayleigh correction
  (no third-party dependency) that removes the blue molecular-scattering veil,
  usable per band through `true_color`'s `rayleigh=` hook.
* `day_night_blend` / `day_weight` — the SZA-weighted cross-fade of a day and a
  night image.
* `alpha_overlay` — the "over" composite of an RGBA foreground on an RGB(A)
  background.
* `static_image` — load a georeferenced background image (e.g. Black Marble),
  caching a remote URL and warping it to a target grid.
* `true_color` — true-colour RGB from calibrated reflectance bands with a CIMSS
  synthetic green (no Rayleigh).
* `night_ir` / `true_color_with_night_ir` — assemble the full day/night image
  (RGBA IR clouds over city lights, cross-faded against the day image by SZA).

Together these compose the `true_color_with_night_ir` day/night look.
"""

from __future__ import annotations

from pyramids_eo.composites.background import static_image
from pyramids_eo.composites.blend import day_night_blend, day_weight
from pyramids_eo.composites.geometry import (
    cos_solar_zenith_angle,
    relative_azimuth,
    satellite_zenith_azimuth,
    solar_zenith_angle,
    solar_zenith_azimuth,
    sunz_correct,
    sunz_reduce,
)
from pyramids_eo.composites.overlay import alpha_overlay
from pyramids_eo.composites.rayleigh import rayleigh_correct
from pyramids_eo.composites.true_color import true_color
from pyramids_eo.composites.true_color_night import (
    night_ir,
    true_color_with_night_ir,
)

__all__ = [
    "alpha_overlay",
    "cos_solar_zenith_angle",
    "day_night_blend",
    "day_weight",
    "night_ir",
    "rayleigh_correct",
    "relative_azimuth",
    "satellite_zenith_azimuth",
    "solar_zenith_angle",
    "solar_zenith_azimuth",
    "static_image",
    "sunz_correct",
    "sunz_reduce",
    "true_color",
    "true_color_with_night_ir",
]
