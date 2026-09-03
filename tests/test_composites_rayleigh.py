"""Unit tests for the local Rayleigh correction (offline, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from pyramids_eo.composites import rayleigh_correct
from pyramids_eo.composites.rayleigh import (
    rayleigh_optical_depth,
    rayleigh_reflectance,
)

_GEOM = {"sza": 40.0, "vza": 30.0, "azidiff": 60.0}


class TestRayleighOpticalDepth:
    """`rayleigh_optical_depth` is the Hansen & Travis lambda^-4 closed form."""

    def test_blue_scatters_far_more_than_red(self):
        """The lambda^-4 law makes blue several times thicker than red."""
        blue = rayleigh_optical_depth(0.444)
        red = rayleigh_optical_depth(0.640)
        assert blue > 3 * red, f"blue {blue} should be >> red {red}"

    def test_decreases_monotonically_with_wavelength(self):
        """Optical depth falls monotonically from blue to NIR."""
        taus = [rayleigh_optical_depth(w) for w in (0.444, 0.510, 0.640, 0.865)]
        assert all(a > b for a, b in zip(taus, taus[1:])), f"not monotone: {taus}"

    def test_scales_linearly_with_pressure(self):
        """Halving the pressure halves the optical depth."""
        full = rayleigh_optical_depth(0.510, pressure_hpa=1013.25)
        half = rayleigh_optical_depth(0.510, pressure_hpa=1013.25 / 2)
        assert half == pytest.approx(full / 2, rel=1e-9), "pressure scaling wrong"

    def test_non_positive_wavelength_raises(self):
        """A non-positive wavelength is rejected."""
        with pytest.raises(ValueError, match="wavelength_um"):
            rayleigh_optical_depth(0.0)

    def test_non_positive_pressure_raises(self):
        """A non-positive pressure is rejected."""
        with pytest.raises(ValueError, match="pressure_hpa"):
            rayleigh_optical_depth(0.5, pressure_hpa=-1.0)


class TestRayleighReflectance:
    """`rayleigh_reflectance` is the bounded single-scattering path reflectance."""

    def test_positive_and_bounded(self):
        """The reflectance is a small positive fraction (well under 1)."""
        rho = rayleigh_reflectance(0.444, **_GEOM)
        assert 0.0 < float(rho) < 0.3, f"reflectance out of range: {rho}"

    def test_blue_exceeds_red(self):
        """At one geometry blue has a larger path reflectance than red."""
        blue = rayleigh_reflectance(0.444, **_GEOM)
        red = rayleigh_reflectance(0.640, **_GEOM)
        assert float(blue) > float(red), f"blue {blue} should exceed red {red}"

    def test_zero_on_the_night_side(self):
        """No sunlight (sza >= 90) means no single-scatter path (reflectance 0)."""
        rho = rayleigh_reflectance(0.444, sza=120.0, vza=30.0, azidiff=60.0)
        assert float(rho) == pytest.approx(0.0), f"night reflectance not 0: {rho}"

    def test_finite_toward_the_limb(self):
        """The bounded form stays finite at a grazing view angle."""
        rho = rayleigh_reflectance(0.444, sza=85.0, vza=88.0, azidiff=60.0)
        assert np.isfinite(rho).all(), f"reflectance not finite near the limb: {rho}"

    def test_nan_geometry_propagates(self):
        """A NaN geometry pixel yields NaN (nodata), not a spurious value."""
        rho = rayleigh_reflectance(0.444, sza=np.nan, vza=30.0, azidiff=60.0)
        assert np.isnan(rho), f"NaN geometry should give NaN: {rho}"

    def test_clamped_to_unit_interval_toward_the_limb(self):
        """The single-scattering form is unbounded near the limb but is clamped to <= 1."""
        rho = rayleigh_reflectance(0.444, sza=89.0, vza=89.0, azidiff=0.0)
        assert 0.0 <= float(rho) <= 1.0, f"reflectance not clamped to [0, 1]: {rho}"

    def test_off_disc_view_is_zero(self):
        """An off-disc view (vza >= 90) has no single-scatter path (reflectance 0)."""
        rho = rayleigh_reflectance(0.444, sza=40.0, vza=95.0, azidiff=60.0)
        assert float(rho) == pytest.approx(0.0), f"off-disc reflectance not 0: {rho}"

    def test_peaks_at_back_scatter(self):
        """Reflectance peaks at back-scatter (azidiff=0) and falls as the azimuth opens to 180."""
        geom = {"sza": 40.0, "vza": 50.0}  # non-negligible sin(sza)*sin(vza)
        back = rayleigh_reflectance(0.444, azidiff=0.0, **geom)
        side = rayleigh_reflectance(0.444, azidiff=90.0, **geom)
        opposed = rayleigh_reflectance(0.444, azidiff=180.0, **geom)
        assert float(back) > float(side) > float(opposed), (
            f"azimuth dependence wrong: back={back}, side={side}, opposed={opposed}"
        )

    def test_matches_independent_reference(self):
        """The reflectance equals a hand-composed phase * bounded-transmittance value."""
        wl, sza, vza, azidiff = 0.510, 40.0, 30.0, 60.0
        tau = rayleigh_optical_depth(wl)
        mu_s, mu_v = np.cos(np.deg2rad(sza)), np.cos(np.deg2rad(vza))
        cos_theta = -mu_s * mu_v - (
            np.sin(np.deg2rad(sza))
            * np.sin(np.deg2rad(vza))
            * np.cos(np.deg2rad(azidiff))
        )
        phase = 0.75 * (1.0 + cos_theta**2)
        trans = 1.0 - np.exp(-tau * (1.0 / mu_s + 1.0 / mu_v))
        expected = phase / (4.0 * (mu_s + mu_v)) * trans
        rho = rayleigh_reflectance(wl, sza=sza, vza=vza, azidiff=azidiff)
        assert float(rho) == pytest.approx(float(expected)), (
            f"reflectance {rho} != independent reference {expected}"
        )


class TestRayleighCorrect:
    """`rayleigh_correct` subtracts the path reflectance from a band."""

    def test_correction_reduces_the_band(self):
        """The corrected band is lower than the input (haze removed)."""
        out = rayleigh_correct(np.array([0.6]), wavelength_um=0.444, **_GEOM)
        assert float(out[0]) < 0.6, f"correction did not reduce the band: {out}"

    def test_blue_dropped_more_than_red(self):
        """Blue loses more than red — the per-band selectivity Rayleigh needs."""
        blue = rayleigh_correct(np.array([0.6]), wavelength_um=0.444, **_GEOM)
        red = rayleigh_correct(np.array([0.6]), wavelength_um=0.640, **_GEOM)
        assert float(blue[0]) < float(red[0]) < 0.6, f"blue {blue} not < red {red}"

    def test_equals_band_minus_reflectance(self):
        """The result is exactly `band - rayleigh_reflectance`, clipped at 0."""
        band = np.array([0.6])
        out = rayleigh_correct(band, wavelength_um=0.510, **_GEOM)
        expected = 0.6 - rayleigh_reflectance(0.510, **_GEOM)
        assert float(out[0]) == pytest.approx(float(expected)), f"mismatch: {out}"

    def test_clips_at_zero(self):
        """A dark band never goes negative after subtracting the path."""
        out = rayleigh_correct(np.array([0.001]), wavelength_um=0.444, **_GEOM)
        assert float(out[0]) >= 0.0, f"correction produced a negative value: {out}"

    def test_night_side_band_unchanged(self):
        """On the night side the correction is 0, so the band is unchanged."""
        out = rayleigh_correct(
            np.array([0.3]), wavelength_um=0.444, sza=120.0, vza=30.0, azidiff=60.0
        )
        assert float(out[0]) == pytest.approx(0.3), f"night band changed: {out}"

    def test_works_as_a_true_color_hook(self):
        """It plugs into true_color's role-aware rayleigh= hook per band."""
        from pyramids_eo.composites import true_color

        wl = {"red": 0.640, "green": 0.510, "blue": 0.444, "nir": 0.865}

        def correct(band, *, role):
            return rayleigh_correct(band, wavelength_um=wl[role], **_GEOM)

        plain = true_color(
            np.full((1, 1), 0.6), np.full((1, 1), 0.6), np.full((1, 1), 0.6)
        )
        corrected = true_color(
            np.full((1, 1), 0.6),
            np.full((1, 1), 0.6),
            np.full((1, 1), 0.6),
            rayleigh=correct,
        )
        assert corrected[2].item() < plain[2].item(), (
            "blue should be reduced in-composite"
        )

    def test_dataset_in_dataset_out(self):
        """A Dataset band yields a georeferenced Dataset."""
        ds = Dataset.from_array(
            np.full((2, 2), 0.5),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        )
        out = rayleigh_correct(ds, wavelength_um=0.444, **_GEOM)
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS not preserved, got {out.epsg}"

    def test_does_not_mutate_input(self):
        """The correction is pure — the input band is left unchanged."""
        band = np.array([0.6, 0.6])
        rayleigh_correct(band, wavelength_um=0.444, **_GEOM)
        assert np.array_equal(band, [0.6, 0.6]), "band was mutated"

    def test_broadcasts_over_a_disc(self):
        """A 2-D band with per-pixel geometry corrects pixel-by-pixel."""
        band = np.full((2, 2), 0.6)
        sza = np.array([[20.0, 50.0], [70.0, 100.0]])  # (1, 1) is night
        vza = np.full((2, 2), 30.0)
        azidiff = np.full((2, 2), 60.0)
        out = rayleigh_correct(
            band, wavelength_um=0.444, sza=sza, vza=vza, azidiff=azidiff
        )
        assert out.shape == (2, 2), f"shape not preserved: {out.shape}"
        assert float(out[0, 0]) < 0.6, "day pixel should be corrected"
        assert float(out[1, 1]) == pytest.approx(0.6), "night pixel should be unchanged"

    def test_higher_pressure_corrects_more(self):
        """A higher surface pressure (thicker atmosphere) removes more haze."""
        low = rayleigh_correct(
            np.array([0.6]), wavelength_um=0.444, pressure_hpa=500.0, **_GEOM
        )
        high = rayleigh_correct(
            np.array([0.6]), wavelength_um=0.444, pressure_hpa=1013.25, **_GEOM
        )
        assert float(high[0]) < float(low[0]), "higher pressure should correct more"
