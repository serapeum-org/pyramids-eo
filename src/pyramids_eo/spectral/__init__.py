"""Spectral transforms for EO rasters.

Operations that rotate or combine a sensor's reflectance bands into a
derived, physically meaningful space, using the per-sensor knowledge in
`pyramids_eo.sensors.registry`:

* `tasseled_cap` — the Kauth-Thomas brightness / greenness / wetness transform,
  applied to a multi-band reflectance `Dataset` with the sensor's coefficients.
"""

from __future__ import annotations

from pyramids_eo.spectral.tasseled_cap import tasseled_cap

__all__ = [
    "tasseled_cap",
]
