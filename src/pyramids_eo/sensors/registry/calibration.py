"""Radiometric calibration for EO instrument channels.

Two conversions turn raw L1 radiances into physical quantities:

* **solar channels** → top-of-atmosphere bidirectional reflectance factor (BRF),
  via `radiance_to_reflectance`;
* **thermal (IR) channels** → brightness temperature, via
  `radiance_to_brightness_temperature` (the EUMETSAT effective-radiance inverse
  Planck with the per-channel `alpha` / `beta` correction).

The math is exact; the per-channel constants come from the file metadata where
available, falling back to the bundled registry tables (see
`pyramids_eo.sensors.registry.get_sensor`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids_eo.errors import CalibrationError

#: First radiation constant `2 h c^2` in mW m-2 sr-1 (cm-1)-4 (wavenumber form).
C1 = 1.19104e-5
#: Second radiation constant `h c / k` in K (cm-1)-1 (wavenumber form).
C2 = 1.43877


def radiance_to_reflectance(
    radiance: Any,
    solar_irradiance: float,
    sun_earth_distance: float = 1.0,
    cos_sza: Any = None,
) -> np.ndarray:
    """Convert solar-channel radiance to top-of-atmosphere reflectance.

    Computes the bidirectional reflectance factor
    `rho = pi * L * d^2 / E0`, optionally normalised by the cosine of the solar
    zenith angle (`/ cos_sza`) to give a sun-angle-corrected BRF.

    `L` and `E0` must be in **consistent** spectral units for the ratio to be
    dimensionless — for MSG/MTG L1 that is the wavenumber form (`L` in
    `mW m-2 sr-1 (cm-1)-1`, `E0` the band-effective solar irradiance in
    `mW m-2 (cm-1)-1`), which the FCI/SEVIRI registry tables provide.

    Args:
        radiance: Spectral radiance `L`, scalar or array.
        solar_irradiance: Band-effective solar irradiance `E0`, same spectral
            grid as `L`.
        sun_earth_distance: Sun-earth distance `d` in astronomical units
            (default 1.0).
        cos_sza: Cosine of the solar zenith angle for the sun-angle correction,
            or `None` to skip it.

    Returns:
        The reflectance, same shape as `radiance`.

    Raises:
        CalibrationError: When `solar_irradiance` is not positive.
    """
    if solar_irradiance <= 0:
        raise CalibrationError(
            f"solar_irradiance must be positive; got {solar_irradiance}"
        )
    reflectance = (
        np.pi * np.asarray(radiance, dtype=float) * sun_earth_distance**2
    ) / solar_irradiance
    if cos_sza is not None:
        cos = np.asarray(cos_sza, dtype=float)
        # At/below the terminator (cos_sza <= 0) the sun-angle correction is
        # undefined; map those pixels to NaN instead of emitting inf + a warning.
        with np.errstate(invalid="ignore", divide="ignore"):
            reflectance = np.where(cos > 0, reflectance / cos, np.nan)
    return np.asarray(reflectance, dtype=float)


def radiance_to_brightness_temperature(
    radiance: Any,
    central_wavenumber_cm1: float,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Convert thermal-channel radiance to brightness temperature (kelvin).

    Applies the EUMETSAT effective-radiance inverse Planck function
    `Tb* = C2 * nu / ln(1 + C1 * nu^3 / L)` followed by the per-channel band
    correction `Tb = (Tb* - beta) / alpha`, with `nu` the channel central
    wavenumber (cm-1) and `L` in mW m-2 sr-1 (cm-1)-1.

    Args:
        radiance: Spectral radiance `L` (mW m-2 sr-1 (cm-1)-1), scalar or array.
            Non-positive values yield NaN.
        central_wavenumber_cm1: Channel central wavenumber `nu` (cm-1).
        alpha: Band-correction slope (default 1.0).
        beta: Band-correction offset in kelvin (default 0.0).

    Returns:
        The brightness temperature in kelvin, same shape as `radiance`.

    Raises:
        CalibrationError: When `central_wavenumber_cm1` is not positive or
            `alpha` is zero.
    """
    if central_wavenumber_cm1 <= 0:
        raise CalibrationError(
            f"central_wavenumber_cm1 must be positive; got {central_wavenumber_cm1}"
        )
    if alpha == 0:
        raise CalibrationError("alpha must be non-zero")
    nu = central_wavenumber_cm1
    radiance_arr = np.asarray(radiance, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tb_star = C2 * nu / np.log1p(C1 * nu**3 / radiance_arr)
        tb_star = np.where(radiance_arr > 0, tb_star, np.nan)
    return np.asarray((tb_star - beta) / alpha, dtype=float)
