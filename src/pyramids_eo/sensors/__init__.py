"""Sensor data-access layer: format readers + the calibration/sensor registry.

`pyramids_eo.sensors` groups the L1 data-access tier that turns instrument
granules into calibrated, geolocated pyramids `Dataset`s:

* `registry` — radiometric calibration (radiance -> reflectance / brightness
  temperature) plus the bundled FCI/SEVIRI sensor and channel tables.
* `readers` — `read_fci`, `read_seviri`, `harmonise`, and `open_fci_l1c_chunk`.

These sit below `pyramids_eo.composites`, which consumes the calibrated bands
they produce (composites import nothing from here — the dependency runs one way).
"""

from __future__ import annotations

from pyramids_eo.sensors.readers import (
    harmonise,
    open_fci_l1c_chunk,
    read_fci,
    read_fci_l1c,
    read_seviri,
)
from pyramids_eo.sensors.registry import (
    Channel,
    Sensor,
    get_sensor,
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)

__all__ = [
    "Channel",
    "Sensor",
    "get_sensor",
    "harmonise",
    "open_fci_l1c_chunk",
    "radiance_to_brightness_temperature",
    "radiance_to_reflectance",
    "read_fci",
    "read_fci_l1c",
    "read_seviri",
]
