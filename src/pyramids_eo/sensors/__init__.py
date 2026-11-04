"""Sensor data-access layer: format readers + the calibration/sensor registry.

`pyramids_eo.sensors` groups the L1 data-access tier that turns instrument
granules into calibrated, geolocated pyramids `Dataset`s:

* `registry` — radiometric calibration (radiance -> reflectance / brightness
  temperature), the bundled FCI/SEVIRI/OLCI/SLSTR sensor and channel tables, and
  the per-sensor tasseled-cap coefficient tables (`get_tasseled_cap`).
* `readers` — `read_fci`, `read_seviri`, `read_ahi`, `read_olci`, `read_slstr`,
  `harmonise`,
  and `open_fci_l1c_chunk`.

These sit below `pyramids_eo.composites`, which consumes the calibrated bands
they produce (composites import nothing from here — the dependency runs one way).
"""

from __future__ import annotations

from pyramids_eo.sensors.readers import (
    HsdHeader,
    available_channels,
    available_olci_channels,
    available_slstr_channels,
    band_from_filename,
    band_name,
    harmonise,
    open_fci_l1c_chunk,
    parse_ahi_hsd,
    parse_hsd_header,
    parse_seviri_native,
    product_type,
    read_ahi,
    read_fci,
    read_fci_l1c,
    read_olci,
    read_seviri,
    read_slstr,
)
from pyramids_eo.sensors.registry import (
    Channel,
    Sensor,
    TasseledCapCoefficients,
    available_tasseled_cap_sensors,
    get_sensor,
    get_tasseled_cap,
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)

__all__ = [
    "available_channels",
    "available_olci_channels",
    "available_slstr_channels",
    "available_tasseled_cap_sensors",
    "band_from_filename",
    "band_name",
    "Channel",
    "get_sensor",
    "get_tasseled_cap",
    "harmonise",
    "HsdHeader",
    "open_fci_l1c_chunk",
    "parse_ahi_hsd",
    "parse_hsd_header",
    "parse_seviri_native",
    "product_type",
    "radiance_to_brightness_temperature",
    "radiance_to_reflectance",
    "read_ahi",
    "read_fci",
    "read_fci_l1c",
    "read_olci",
    "read_seviri",
    "read_slstr",
    "Sensor",
    "TasseledCapCoefficients",
]
