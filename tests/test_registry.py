"""Unit tests for `pyramids_eo.registry` (calibration + sensor tables)."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids_eo.errors import CalibrationError, UnknownSensorError
from pyramids_eo.registry import (
    Channel,
    Sensor,
    get_sensor,
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)
from pyramids_eo.registry.calibration import C1, C2


def _forward_planck(nu: float, temperature: float) -> float:
    """Forward Planck radiance (mW m-2 sr-1 (cm-1)-1) for a wavenumber + kelvin."""
    return C1 * nu**3 / np.expm1(C2 * nu / temperature)


class TestRadianceToReflectance:
    """Solar-channel radiance -> reflectance."""

    def test_basic_formula(self):
        """rho = pi * L * d^2 / E0 for the simple (no sun-angle) case."""
        out = radiance_to_reflectance(100.0, solar_irradiance=200.0)
        assert out == pytest.approx(np.pi * 100.0 / 200.0), f"got {out}"

    def test_sun_earth_distance_scales_squared(self):
        """A sun-earth distance d scales the reflectance by d^2."""
        out = radiance_to_reflectance(
            10.0, solar_irradiance=50.0, sun_earth_distance=2.0
        )
        assert out == pytest.approx(np.pi * 10.0 * 4.0 / 50.0), f"got {out}"

    def test_cos_sza_normalisation(self):
        """Passing cos_sza divides the reflectance by it."""
        base = radiance_to_reflectance(10.0, solar_irradiance=50.0)
        out = radiance_to_reflectance(10.0, solar_irradiance=50.0, cos_sza=0.5)
        assert out == pytest.approx(base / 0.5), f"got {out}"

    def test_array_input(self):
        """Array radiance yields an array of the same shape."""
        out = radiance_to_reflectance(np.array([10.0, 20.0]), solar_irradiance=50.0)
        assert out.shape == (2,), f"got shape {out.shape}"

    def test_non_positive_irradiance_raises(self):
        """A non-positive solar irradiance is rejected."""
        with pytest.raises(CalibrationError, match="solar_irradiance"):
            radiance_to_reflectance(10.0, solar_irradiance=0.0)


class TestRadianceToBrightnessTemperature:
    """Thermal-channel radiance -> brightness temperature (inverse Planck)."""

    @pytest.mark.parametrize("temperature", [220.0, 280.0, 320.0])
    def test_planck_round_trip(self, temperature):
        """Inverting the forward Planck radiance recovers the temperature.

        Args:
            temperature: The kelvin value to round-trip.

        Test scenario:
            With alpha=1, beta=0 the inverse Planck exactly undoes the forward
            Planck for a representative IR channel.
        """
        nu = 930.647
        radiance = _forward_planck(nu, temperature)
        out = radiance_to_brightness_temperature(radiance, nu)
        assert float(out) == pytest.approx(temperature, abs=1e-6), f"got {out}"

    def test_alpha_beta_band_correction(self):
        """The (Tb* - beta) / alpha correction is applied after the Planck inverse."""
        nu = 930.647
        radiance = _forward_planck(nu, 280.0)
        corrected = radiance_to_brightness_temperature(
            radiance, nu, alpha=0.99, beta=0.5
        )
        assert float(corrected) == pytest.approx((280.0 - 0.5) / 0.99, abs=1e-6)

    def test_non_positive_radiance_is_nan(self):
        """Zero or negative radiance yields NaN, not a crash."""
        out = radiance_to_brightness_temperature(np.array([0.0, -1.0, 5.0]), 930.647)
        assert np.isnan(out[0]) and np.isnan(out[1]), f"got {out}"
        assert np.isfinite(out[2]), "positive radiance should be finite"

    def test_non_positive_wavenumber_raises(self):
        """A non-positive central wavenumber is rejected."""
        with pytest.raises(CalibrationError, match="central_wavenumber"):
            radiance_to_brightness_temperature(5.0, -1.0)

    def test_zero_alpha_raises(self):
        """A zero alpha is rejected (avoids division by zero)."""
        with pytest.raises(CalibrationError, match="alpha"):
            radiance_to_brightness_temperature(5.0, 930.647, alpha=0.0)


class TestGetSensor:
    """`get_sensor` loads channel tables from the bundled YAML."""

    def test_fci_loads_expected_channels(self):
        """The FCI table exposes its solar and thermal channels."""
        fci = get_sensor("fci")
        assert isinstance(fci, Sensor) and fci.name == "fci"
        assert "ir_105" in fci.channels and "vis_06" in fci.channels

    def test_seviri_loads(self):
        """The SEVIRI table loads and carries the IR_108 Planck constants."""
        ch = get_sensor("seviri").get_channel("IR_108")
        assert ch.kind == "thermal", f"expected thermal, got {ch.kind}"
        assert ch.central_wavenumber_cm1 == pytest.approx(930.647)

    def test_case_insensitive_name(self):
        """Sensor names are matched case-insensitively."""
        assert get_sensor("FCI").name == "fci", "sensor name should be normalised"

    def test_result_is_cached(self):
        """Repeated lookups return the same cached object."""
        assert get_sensor("fci") is get_sensor("fci"), "sensor should be cached"

    def test_unknown_sensor_raises(self):
        """An unknown sensor name lists the available tables."""
        with pytest.raises(UnknownSensorError, match="unknown sensor"):
            get_sensor("nope")

    def test_channel_names_sorted(self):
        """`channel_names` returns the identifiers sorted."""
        names = get_sensor("seviri").channel_names
        assert names == sorted(names), "channel names should be sorted"

    def test_solar_channel_has_irradiance(self):
        """A solar channel carries a solar irradiance and no wavenumber."""
        vis = get_sensor("fci").get_channel("vis_06")
        assert vis.kind == "solar" and vis.solar_irradiance is not None
        assert vis.central_wavenumber_cm1 is None


class TestSensorChannel:
    """`Sensor.get_channel` and the `Channel` record."""

    def test_get_channel_returns_record(self):
        """A known channel returns its Channel record."""
        ch = get_sensor("fci").get_channel("ir_105")
        assert isinstance(ch, Channel) and ch.name == "ir_105"

    def test_unknown_channel_raises(self):
        """An unknown channel lists the known ones."""
        with pytest.raises(UnknownSensorError, match="has no channel"):
            get_sensor("fci").get_channel("does_not_exist")

    def test_channel_is_frozen(self):
        """Channel records are immutable."""
        ch = get_sensor("fci").get_channel("ir_105")
        with pytest.raises((AttributeError, TypeError)):
            ch.name = "changed"  # type: ignore[misc]
