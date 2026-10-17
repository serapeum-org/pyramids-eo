"""Sensor metadata registry.

Maps a sensor's bands to their wavelengths, native resolution, radiometric kind,
and (nominal) calibration constants, backed by YAML tables under
`sensors/registry/data/`. Also exposes the radiometric calibration functions and
the tasseled-cap coefficient tables:

* `get_sensor` / `Sensor` / `Channel` — the channel metadata tables.
* `radiance_to_reflectance` — solar-channel radiance → reflectance.
* `radiance_to_brightness_temperature` — thermal-channel radiance → brightness
  temperature (EUMETSAT inverse Planck).
* `get_tasseled_cap` / `TasseledCapCoefficients` /
  `available_tasseled_cap_sensors` — the per-sensor tasseled-cap coefficient
  tables (consumed by `pyramids_eo.spectral.tasseled_cap`).
"""

from __future__ import annotations

from pyramids_eo.sensors.registry.calibration import (
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)
from pyramids_eo.sensors.registry.sensors import Channel, Sensor, get_sensor
from pyramids_eo.sensors.registry.tasseled_cap import (
    TasseledCapCoefficients,
    available_tasseled_cap_sensors,
    get_tasseled_cap,
)

__all__ = [
    "Channel",
    "Sensor",
    "TasseledCapCoefficients",
    "available_tasseled_cap_sensors",
    "get_sensor",
    "get_tasseled_cap",
    "radiance_to_brightness_temperature",
    "radiance_to_reflectance",
]
