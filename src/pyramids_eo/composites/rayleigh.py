"""Local Rayleigh (molecular-scattering) correction for solar bands.

A self-contained, closed-form **single-scattering** Rayleigh correction over
NumPy — no third-party radiative-transfer dependency. It removes the blue
molecular-scattering veil a no-Rayleigh true colour leaves over the disc, using
the viewing geometry from `geometry.py` (`solar_zenith_azimuth`,
`satellite_zenith_azimuth`, `relative_azimuth`).

The model is:

* Rayleigh optical thickness `tau_r(lambda)` from the Hansen & Travis (1974)
  closed form, scaled linearly by surface pressure.
* The Rayleigh phase function `P(Theta) = 3/4 (1 + cos^2 Theta)` at the
  Sun-satellite scattering angle.
* The single-scattering path reflectance
  `rho_r = P(Theta) / (4 (mu_s + mu_v)) * (1 - exp(-tau_r (1/mu_s + 1/mu_v)))`,
  which stays bounded toward the terminator / limb.

`rho_r` is subtracted from the (already reflectance-scaled `[0, 1]`) band. Because
`tau_r` goes as `lambda^-4`, blue is corrected several times more than red — the
per-band selectivity the composite needs. This is a single-scattering
approximation (no multiple scattering, aerosol, or surface coupling), so it is
lighter and slightly less accurate than a full radiative-transfer table, but it
carries no dependency and reproduces the bulk of the blue-veil removal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids_eo.composites._common import _as_array, _wrap_like

#: Standard sea-level pressure (hPa), the reference for the pressure scaling.
_P0_HPA = 1013.25


def rayleigh_optical_depth(
    wavelength_um: float, pressure_hpa: float = _P0_HPA
) -> float:
    """Rayleigh optical thickness at a wavelength, scaled by surface pressure.

    Uses the Hansen & Travis (1974) closed form
    `tau0 = 0.008569 l^-4 (1 + 0.0113 l^-2 + 0.00013 l^-4)` (`l` in micrometres,
    sea level), scaled linearly by `pressure_hpa / 1013.25`.

    Args:
        wavelength_um: Band centre wavelength in micrometres.
        pressure_hpa: Surface pressure in hPa (default sea level, 1013.25).

    Returns:
        The (dimensionless) Rayleigh optical thickness.

    Raises:
        ValueError: When `wavelength_um` or `pressure_hpa` is not positive.

    Examples:
        - Blue scatters far more than red (the `lambda^-4` law):
            ```python
            >>> from pyramids_eo.composites.rayleigh import rayleigh_optical_depth
            >>> blue = rayleigh_optical_depth(0.444)
            >>> red = rayleigh_optical_depth(0.640)
            >>> bool(blue > 3 * red)
            True

            ```
    """
    if wavelength_um <= 0:
        raise ValueError(f"wavelength_um must be > 0, got {wavelength_um}")
    if pressure_hpa <= 0:
        raise ValueError(f"pressure_hpa must be > 0, got {pressure_hpa}")
    lam = wavelength_um
    tau0 = 0.008569 * lam**-4 * (1.0 + 0.0113 * lam**-2 + 0.00013 * lam**-4)
    return tau0 * (pressure_hpa / _P0_HPA)


def rayleigh_reflectance(
    wavelength_um: float,
    sza: Any,
    vza: Any,
    azidiff: Any,
    *,
    pressure_hpa: float = _P0_HPA,
) -> np.ndarray:
    """Single-scattering Rayleigh path reflectance for the viewing geometry.

    Args:
        wavelength_um: Band centre wavelength in micrometres.
        sza: Solar zenith angle(s) in degrees (e.g. from `solar_zenith_azimuth`).
        vza: Satellite (view) zenith angle(s) in degrees (from
            `satellite_zenith_azimuth`).
        azidiff: Sun-satellite relative azimuth in degrees (from
            `relative_azimuth`).
        pressure_hpa: Surface pressure in hPa (default sea level).

    Returns:
        The per-pixel Rayleigh path reflectance in `[0, 1]`, `0` on the night side
        (`sza >= 90`) and NaN where a geometry input is NaN.

    Raises:
        ValueError: When `wavelength_um` or `pressure_hpa` is not positive (via
            :func:`rayleigh_optical_depth`).

    Examples:
        - The path reflectance is a small positive fraction, larger for blue:
            ```python
            >>> from pyramids_eo.composites.rayleigh import rayleigh_reflectance
            >>> blue = rayleigh_reflectance(0.444, sza=40.0, vza=30.0, azidiff=60.0)
            >>> red = rayleigh_reflectance(0.640, sza=40.0, vza=30.0, azidiff=60.0)
            >>> bool(0.0 < float(blue) < 0.3)
            True
            >>> bool(float(blue) > float(red))
            True

            ```
        - There is no single-scatter path on the night side (`sza >= 90`):
            ```python
            >>> from pyramids_eo.composites.rayleigh import rayleigh_reflectance
            >>> float(rayleigh_reflectance(0.444, sza=120.0, vza=30.0, azidiff=60.0))
            0.0

            ```
    """
    tau = rayleigh_optical_depth(wavelength_um, pressure_hpa)
    sza_r = np.deg2rad(_as_array(sza))
    vza_r = np.deg2rad(_as_array(vza))
    raz_r = np.deg2rad(_as_array(azidiff))

    mu_s = np.cos(sza_r)
    mu_v = np.cos(vza_r)
    # Rayleigh phase function at the scattering angle. cos(Theta) is
    # -mu_s*mu_v - sin(sza)*sin(vza)*cos(azidiff); azidiff = 0 (co-azimuth) is
    # back-scatter, where cos(Theta) = -1 and the phase is maximal. cos**2 is even,
    # so only the azimuth *sign* is immaterial -- the term's own sign is not.
    cos_scat = -mu_s * mu_v - np.sin(sza_r) * np.sin(vza_r) * np.cos(raz_r)
    phase = 0.75 * (1.0 + cos_scat**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        transmittance = 1.0 - np.exp(-tau * (1.0 / mu_s + 1.0 / mu_v))
        refl = phase / (4.0 * (mu_s + mu_v)) * transmittance
    # Only the sunlit side has a single-scatter path; a NaN geometry stays NaN.
    refl = np.where(mu_s > 0, refl, 0.0)
    invalid = np.isnan(mu_s) | np.isnan(mu_v) | np.isnan(raz_r)
    return np.asarray(np.where(invalid, np.nan, refl), dtype=float)


def rayleigh_correct(
    band: Any,
    *,
    wavelength_um: float,
    sza: Any,
    vza: Any,
    azidiff: Any,
    pressure_hpa: float = _P0_HPA,
) -> Any:
    """Rayleigh-correct a solar band by subtracting the molecular path reflectance.

    Computes the single-scattering Rayleigh reflectance (see
    :func:`rayleigh_reflectance`) for the band's wavelength and geometry and
    subtracts it from the band, clipping the result at `0`. Because the
    correction grows as `wavelength^-4`, blue is corrected far more than red.
    Designed to be used as the `true_color` `rayleigh=` hook (per band):

        rayleigh=lambda band, *, role: rayleigh_correct(
            band, wavelength_um=WL[role], sza=sza, vza=vza, azidiff=azidiff)

    Args:
        band: The solar band reflectance in `[0, 1]` — array-like or a pyramids
            `Dataset`.
        wavelength_um: The band's centre wavelength in micrometres (e.g. from the
            sensor registry `Channel.wavelength_um`).
        sza: Solar zenith angle(s) in degrees.
        vza: Satellite (view) zenith angle(s) in degrees.
        azidiff: Sun-satellite relative azimuth in degrees.
        pressure_hpa: Surface pressure in hPa (default sea level).

    Returns:
        The corrected band. A pyramids `Dataset` (carrying `band`'s geotransform +
        CRS) when `band` is a `Dataset`, otherwise an ndarray.

    Examples:
        - Correcting reduces a band, and blue drops more than red:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites.rayleigh import rayleigh_correct
            >>> geom = dict(sza=40.0, vza=30.0, azidiff=60.0)
            >>> blue = rayleigh_correct(np.array([0.6]), wavelength_um=0.444, **geom)
            >>> red = rayleigh_correct(np.array([0.6]), wavelength_um=0.640, **geom)
            >>> bool(blue[0] < red[0] < 0.6)
            True

            ```
    """
    arr = _as_array(band)
    path = rayleigh_reflectance(
        wavelength_um, sza, vza, azidiff, pressure_hpa=pressure_hpa
    )
    return _wrap_like(np.clip(arr - path, 0.0, None), band)
