"""Sensor metadata registry.

Maps a sensor's bands to their wavelengths, native resolution, radiometric kind,
and (nominal) calibration constants, backed by YAML tables under
`sensors/registry/data/`. Also exposes the radiometric calibration functions:

* `get_sensor` / `Sensor` / `Channel` — the channel metadata tables.
* `radiance_to_reflectance` — solar-channel radiance → reflectance.
* `radiance_to_brightness_temperature` — thermal-channel radiance → brightness
  temperature (EUMETSAT inverse Planck).
"""

from __future__ import annotations

from pyramids_eo.sensors.registry.calibration import (
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)
from pyramids_eo.sensors.registry.sensors import Channel, Sensor, get_sensor

__all__ = [
    "Channel",
    "Sensor",
    "get_sensor",
    "radiance_to_brightness_temperature",
    "radiance_to_reflectance",
]
