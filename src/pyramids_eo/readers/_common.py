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
    *,
    coeffs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Calibrate raw radiance for `channel` to a physical quantity.

    Applies the conversion the channel's radiometric kind needs: reflectance for
    a solar channel, brightness temperature for a thermal one. Each constant is
    taken from `coeffs` (the per-granule metadata) when present there, otherwise
    from the bundled registry table as a fallback — so a reader that carries the
    granule's own coefficients gets granule-accurate output, while the registry
    supplies nominal values when it does not.

    Args:
        radiance: Raw radiance array.
        channel: Channel identifier (registry key).
        sensor: Sensor name for the registry lookup.
        sun_earth_distance: Sun-earth distance (AU) for a solar channel.
        cos_sza: Cosine of the solar zenith angle, or `None`.
        coeffs: Per-granule calibration coefficients that override the registry —
            any of `kind`, `solar_irradiance`, `central_wavenumber_cm1`, `alpha`,
            `beta`. Missing keys fall back to the registry channel.

    Returns:
        Reflectance (solar) or brightness temperature (thermal).

    Raises:
        CalibrationError: When the channel lacks the constants its kind needs.
        UnknownSensorError: When the sensor / channel is not in the registry.
    """
    ch = get_sensor(sensor).get_channel(channel)
    overrides = coeffs or {}

    def _coef(key: str, fallback: Any) -> Any:
        return overrides[key] if key in overrides else fallback

    if _coef("kind", ch.kind) == "solar":
        solar_irradiance = _coef("solar_irradiance", ch.solar_irradiance)
        if solar_irradiance is None:
            raise CalibrationError(f"solar channel {channel!r} has no solar_irradiance")
        return radiance_to_reflectance(
            radiance, solar_irradiance, sun_earth_distance, cos_sza
        )

    central_wavenumber = _coef("central_wavenumber_cm1", ch.central_wavenumber_cm1)
    if central_wavenumber is None:
        raise CalibrationError(
            f"thermal channel {channel!r} has no central_wavenumber_cm1"
        )
    alpha = _coef("alpha", ch.alpha)
    beta = _coef("beta", ch.beta)
    return radiance_to_brightness_temperature(
        radiance,
        central_wavenumber,
        alpha if alpha is not None else 1.0,
        beta if beta is not None else 0.0,
    )
