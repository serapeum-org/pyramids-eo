"""Shared helpers for the instrument readers."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids_eo.errors import CalibrationError
from pyramids_eo.registry import (
    get_sensor,
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)


def calibrate_channel(
    radiance: np.ndarray,
    channel: str,
    sensor: str,
    sun_earth_distance: float,
    cos_sza: Any,
) -> np.ndarray:
    """Calibrate raw radiance for `channel` to a physical quantity.

    Looks the channel up in the registry and applies the conversion its
    radiometric kind needs: reflectance for a solar channel, brightness
    temperature for a thermal one.

    Args:
        radiance: Raw radiance array.
        channel: Channel identifier (registry key).
        sensor: Sensor name for the registry lookup.
        sun_earth_distance: Sun-earth distance (AU) for a solar channel.
        cos_sza: Cosine of the solar zenith angle, or `None`.

    Returns:
        Reflectance (solar) or brightness temperature (thermal).

    Raises:
        CalibrationError: When the channel lacks the constants its kind needs.
        UnknownSensorError: When the sensor / channel is not in the registry.
    """
    ch = get_sensor(sensor).get_channel(channel)
    if ch.kind == "solar":
        if ch.solar_irradiance is None:
            raise CalibrationError(f"solar channel {channel!r} has no solar_irradiance")
        return radiance_to_reflectance(
            radiance, ch.solar_irradiance, sun_earth_distance, cos_sza
        )
    if ch.central_wavenumber_cm1 is None:
        raise CalibrationError(
            f"thermal channel {channel!r} has no central_wavenumber_cm1"
        )
    return radiance_to_brightness_temperature(
        radiance, ch.central_wavenumber_cm1, ch.alpha or 1.0, ch.beta or 0.0
    )
