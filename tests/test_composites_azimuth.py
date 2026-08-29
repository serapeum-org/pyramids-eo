"""Unit tests for the solar/satellite azimuth geometry (offline, deterministic)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.composites import (
    relative_azimuth,
    satellite_zenith_azimuth,
    solar_zenith_angle,
    solar_zenith_azimuth,
)

_EQUINOX_NOON = dt.datetime(2024, 3, 20, 12, 0, tzinfo=dt.timezone.utc)


class TestSolarZenithAzimuth:
    """`solar_zenith_azimuth` returns the NOAA (zenith, azimuth) pair."""

    def test_zenith_matches_solar_zenith_angle_exactly(self):
        """The zenith is byte-identical to the standalone `solar_zenith_angle`."""
        lat = np.array([[-30.0, 0.0], [30.0, 60.0]])
        lon = np.array([[0.0, 45.0], [90.0, 135.0]])
        zen, _ = solar_zenith_azimuth(_EQUINOX_NOON, lat=lat, lon=lon)
        expected = solar_zenith_angle(_EQUINOX_NOON, lat=lat, lon=lon)
        assert np.array_equal(zen, expected), "zenith diverged from solar_zenith_angle"

    def test_azimuth_in_0_360(self):
        """The azimuth stays within [0, 360)."""
        lat = np.linspace(-80, 80, 9)
        lon = np.linspace(-170, 170, 9)
        _, az = solar_zenith_azimuth(_EQUINOX_NOON, lat=lat, lon=lon)
        assert np.all((az >= 0.0) & (az < 360.0)), f"azimuth out of range: {az}"

    def test_noon_sun_is_due_south_in_north(self):
        """At equinox noon on lon 0, a north-hemisphere Sun is ~due south."""
        _, az = solar_zenith_azimuth(_EQUINOX_NOON, lat=45.0, lon=0.0)
        assert 170.0 < float(az) < 190.0, f"expected ~180deg (south), got {az}"

    def test_grid_signature(self):
        """A `grid=` Dataset (EPSG:4326) is accepted, like solar_zenith_angle."""
        grid = Dataset.create_from_array(
            np.zeros((2, 2)), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
        )
        zen, az = solar_zenith_azimuth(_EQUINOX_NOON, grid=grid)
        assert zen.shape == (2, 2), f"grid zenith shape wrong: {zen.shape}"
        assert az.shape == (2, 2), f"grid azimuth shape wrong: {az.shape}"

    def test_grid_and_latlon_together_raise(self):
        """Passing both a grid and lat/lon is rejected."""
        grid = Dataset.create_from_array(
            np.zeros((1, 1)), top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326
        )
        with pytest.raises(ValueError, match="either"):
            solar_zenith_azimuth(_EQUINOX_NOON, lat=0.0, lon=0.0, grid=grid)

    def test_non_geographic_grid_raises(self):
        """A grid that is not EPSG:4326 is rejected."""
        grid = Dataset.create_from_array(
            np.zeros((1, 1)), top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=3857
        )
        with pytest.raises(ValueError, match="EPSG:4326"):
            solar_zenith_azimuth(_EQUINOX_NOON, grid=grid)

    def test_missing_coordinates_raise(self):
        """Neither a grid nor both lat and lon is rejected."""
        with pytest.raises(ValueError, match="provide"):
            solar_zenith_azimuth(_EQUINOX_NOON, lat=0.0)


class TestSatelliteZenithAzimuth:
    """`satellite_zenith_azimuth` gives the geostationary viewing geometry."""

    def test_zero_zenith_at_subsatellite_point(self):
        """The sub-satellite point sees the satellite at the local zenith (0deg)."""
        vza, _ = satellite_zenith_azimuth(0.0, 0.0, sat_lon=0.0)
        assert float(vza) == pytest.approx(0.0, abs=1e-6), f"sub-sat vza != 0: {vza}"

    def test_zenith_rises_off_nadir(self):
        """A point away from the sub-satellite point is seen off-nadir (0 < vza < 90)."""
        vza, _ = satellite_zenith_azimuth(30.0, 20.0, sat_lon=0.0)
        assert 0.0 < float(vza) < 90.0, f"off-nadir vza out of range: {vza}"

    def test_zenith_increases_with_distance(self):
        """Zenith grows monotonically as the ground point moves off the sub-point."""
        vza, _ = satellite_zenith_azimuth(
            np.array([0.0, 20.0, 40.0]), np.array([0.0, 0.0, 0.0]), sat_lon=0.0
        )
        assert np.all(np.diff(vza) > 0), f"zenith not increasing with distance: {vza}"

    def test_azimuth_south_when_north_of_subpoint(self):
        """A point due north of the sub-point sees the satellite due south (~180)."""
        _, az = satellite_zenith_azimuth(30.0, 0.0, sat_lon=0.0)
        assert 175.0 < float(az) < 185.0, f"expected ~180deg (south), got {az}"

    def test_azimuth_west_when_east_of_subpoint(self):
        """A point due east of the sub-point sees the satellite due west (~270)."""
        _, az = satellite_zenith_azimuth(0.0, 15.0, sat_lon=0.0)
        assert 265.0 < float(az) < 275.0, f"expected ~270deg (west), got {az}"

    def test_azimuth_in_0_360(self):
        """The azimuth stays within [0, 360)."""
        lat, lon = np.meshgrid(np.linspace(-60, 60, 7), np.linspace(-60, 60, 7))
        _, az = satellite_zenith_azimuth(lat, lon, sat_lon=0.0)
        assert np.all((az >= 0.0) & (az < 360.0)), "satellite azimuth out of range"

    def test_grid_signature(self):
        """A `grid=` Dataset is accepted."""
        grid = Dataset.create_from_array(
            np.zeros((2, 2)), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
        )
        vza, az = satellite_zenith_azimuth(grid=grid, sat_lon=0.0)
        assert vza.shape == (2, 2) and az.shape == (2, 2), "grid shapes wrong"

    def test_higher_orbit_reduces_off_nadir_zenith(self):
        """A larger orbital radius sees the same point closer to nadir."""
        geo, _ = satellite_zenith_azimuth(30.0, 20.0, sat_radius_km=42164.0)
        higher, _ = satellite_zenith_azimuth(30.0, 20.0, sat_radius_km=60000.0)
        assert float(higher) < float(geo), "a higher orbit should reduce the zenith"


class TestRelativeAzimuth:
    """`relative_azimuth` folds the Sun-satellite difference into [0, 180]."""

    def test_opposed_is_180(self):
        """Azimuths 180deg apart give the maximum relative azimuth."""
        assert float(relative_azimuth(10.0, 190.0)) == pytest.approx(180.0)

    def test_wraps_around_360(self):
        """350deg and 10deg are only 20deg apart across the 0/360 seam."""
        assert float(relative_azimuth(350.0, 10.0)) == pytest.approx(20.0)

    def test_within_0_180(self):
        """Any pair folds into [0, 180]."""
        rng = np.random.default_rng(0)
        a, b = rng.uniform(0, 360, 100), rng.uniform(0, 360, 100)
        out = relative_azimuth(a, b)
        assert np.all((out >= 0.0) & (out <= 180.0)), "relative azimuth out of range"

    def test_order_independent(self):
        """The difference is symmetric in its two arguments."""
        a, b = np.array([12.0, 200.0]), np.array([300.0, 5.0])
        assert np.allclose(relative_azimuth(a, b), relative_azimuth(b, a))


class TestConventionAlignment:
    """The solar and satellite azimuths share one clockwise-from-north convention."""

    def test_sun_and_satellite_both_south_gives_zero_relative(self):
        """At noon a 45N point sees both Sun and a sub-0 satellite due south."""
        _, sun_az = solar_zenith_azimuth(_EQUINOX_NOON, lat=45.0, lon=0.0)
        _, sat_az = satellite_zenith_azimuth(45.0, 0.0, sat_lon=0.0)
        azidiff = relative_azimuth(sun_az, sat_az)
        assert float(azidiff) < 10.0, (
            f"conventions misaligned: sun={sun_az}, sat={sat_az}, diff={azidiff}"
        )
