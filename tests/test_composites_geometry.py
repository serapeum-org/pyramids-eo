"""Unit tests for `pyramids_eo.composites.solar_zenith_angle` (offline, deterministic)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from pyramids_eo.composites import cos_solar_zenith_angle, solar_zenith_angle
from pyramids_eo.composites.geometry import _to_utc

# Equinox solar noon at Greenwich: the subsolar point sits near (0degN, 0degE),
# so SZA is ~0 there and ~180 at the antipode. A fixed instant keeps it deterministic.
_EQUINOX_NOON = dt.datetime(2024, 3, 20, 12, 0, 0, tzinfo=dt.timezone.utc)


class TestSolarZenithAngle:
    """The NOAA solar-position port returns per-pixel zenith angles in degrees."""

    def test_subsolar_point_is_near_zero(self):
        """At equinox noon the Sun is nearly overhead at (0, 0), so SZA ~ 0."""
        assert float(solar_zenith_angle(_EQUINOX_NOON, lat=0.0, lon=0.0)) < 5.0

    def test_antisolar_point_is_near_180(self):
        """The antipode of the subsolar point is in deep night (SZA ~ 180)."""
        assert float(solar_zenith_angle(_EQUINOX_NOON, lat=0.0, lon=180.0)) > 175.0

    def test_terminator_is_near_90(self):
        """Ninety degrees of longitude from the subsolar meridian sits on the terminator."""
        sza = float(solar_zenith_angle(_EQUINOX_NOON, lat=0.0, lon=90.0))
        assert 80.0 < sza < 100.0

    def test_sza_increases_away_from_subsolar_point(self):
        """SZA grows monotonically as longitude moves off the subsolar meridian."""
        lons = np.array([0.0, 30.0, 60.0, 90.0])
        szas = solar_zenith_angle(_EQUINOX_NOON, lat=0.0, lon=lons)
        assert np.all(np.diff(szas) > 0)

    def test_array_inputs_broadcast(self):
        """lat/lon broadcast to a common shape."""
        lat = np.array([[0.0], [10.0]])
        lon = np.array([[0.0, 20.0, 40.0]])
        out = solar_zenith_angle(_EQUINOX_NOON, lat=lat, lon=lon)
        assert out.shape == (2, 3), f"expected (2, 3), got {out.shape}"

    def test_scalar_inputs_return_zero_dim_array(self):
        """Scalar lat/lon yield a 0-d float array (a single angle)."""
        out = solar_zenith_angle(_EQUINOX_NOON, lat=0.0, lon=0.0)
        assert out.shape == (), f"expected scalar (0-d) result, got shape {out.shape}"
        assert np.isfinite(float(out)), f"result should be finite, got {out}"

    @pytest.mark.parametrize(
        "when, north_is_sunnier",
        [
            (dt.datetime(2024, 6, 21, 12, 0, tzinfo=dt.timezone.utc), True),
            (dt.datetime(2024, 12, 21, 12, 0, tzinfo=dt.timezone.utc), False),
        ],
    )
    def test_declination_sign_by_season(self, when, north_is_sunnier):
        """Solar declination tilts the Sun north in June and south in December.

        Args:
            when: A solstice instant (UTC).
            north_is_sunnier: True when +23degN should have the smaller SZA.

        Test scenario:
            At identical time/longitude, +23degN vs -23degN differ only by
            `2*sin(23deg)*sin(decl)` in cos(zenith); the sign of `decl` (positive
            in June, negative in December) decides which hemisphere is sunnier,
            independent of the hour angle.
        """
        north = float(solar_zenith_angle(when, lat=23.44, lon=0.0))
        south = float(solar_zenith_angle(when, lat=-23.44, lon=0.0))
        if north_is_sunnier:
            assert north < south, f"June: north SZA {north} should be < south {south}"
        else:
            assert north > south, (
                f"December: north SZA {north} should be > south {south}"
            )

    def test_result_bounds(self):
        """Every SZA lies within the physical 0..180 range."""
        lat = np.linspace(-80, 80, 9)
        lon = np.linspace(-180, 180, 9)
        lon2d, lat2d = np.meshgrid(lon, lat)
        out = solar_zenith_angle(_EQUINOX_NOON, lat=lat2d, lon=lon2d)
        assert out.min() >= 0.0, f"SZA below 0 degrees: {out.min()}"
        assert out.max() <= 180.0, f"SZA above 180 degrees: {out.max()}"

    def test_naive_time_treated_as_utc(self):
        """A naive datetime yields the same result as the equivalent UTC-aware one."""
        naive = dt.datetime(2024, 3, 20, 12, 0, 0)
        assert float(solar_zenith_angle(naive, lat=10.0, lon=20.0)) == pytest.approx(
            float(solar_zenith_angle(_EQUINOX_NOON, lat=10.0, lon=20.0))
        )

    def test_timezone_aware_time_converted_to_utc(self):
        """A +02:00 wall-clock time equals its UTC instant."""
        plus2 = dt.datetime(
            2024, 3, 20, 14, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=2))
        )
        assert float(solar_zenith_angle(plus2, lat=10.0, lon=20.0)) == pytest.approx(
            float(solar_zenith_angle(_EQUINOX_NOON, lat=10.0, lon=20.0))
        )

    def test_non_datetime_time_raises(self):
        """A non-datetime `time` is rejected."""
        with pytest.raises(TypeError, match="datetime"):
            solar_zenith_angle("2024-03-20", lat=0.0, lon=0.0)

    def test_missing_coordinates_raises(self):
        """Neither grid nor lat/lon is an error."""
        with pytest.raises(ValueError, match="provide"):
            solar_zenith_angle(_EQUINOX_NOON)

    def test_lat_without_lon_raises(self):
        """lat without lon is an error."""
        with pytest.raises(ValueError, match="provide"):
            solar_zenith_angle(_EQUINOX_NOON, lat=0.0)


class TestCosSolarZenithAngle:
    """`cos_solar_zenith_angle` returns the cosine of the SZA."""

    def test_equals_cos_of_sza(self):
        """The result equals cos(deg2rad(solar_zenith_angle))."""
        sza = solar_zenith_angle(_EQUINOX_NOON, lat=10.0, lon=20.0)
        cos = cos_solar_zenith_angle(_EQUINOX_NOON, lat=10.0, lon=20.0)
        assert float(cos) == pytest.approx(np.cos(np.deg2rad(float(sza))))

    def test_subsolar_point_is_near_one(self):
        """cos(SZA) is ~1 where the Sun is overhead."""
        assert float(cos_solar_zenith_angle(_EQUINOX_NOON, lat=0.0, lon=0.0)) > 0.99

    def test_grid_path(self):
        """The grid path returns a (rows, columns) array."""
        grid = Dataset.from_array(
            np.zeros((2, 3)),
            geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=4326),
        )
        assert cos_solar_zenith_angle(_EQUINOX_NOON, grid=grid).shape == (2, 3)


class TestToUtc:
    """Tests for the `_to_utc` time-normalisation helper."""

    def test_naive_datetime_stamped_as_utc(self):
        """A naive datetime keeps its wall clock and gains a UTC tzinfo.

        Test scenario:
            `datetime(2024, 3, 20, 12)` with no tzinfo returns the same fields
            tagged UTC (naive input is assumed to already be UTC).
        """
        out = _to_utc(dt.datetime(2024, 3, 20, 12, 0, 0))
        assert out.tzinfo is dt.timezone.utc, f"expected UTC tzinfo, got {out.tzinfo}"
        assert out.hour == 12, f"wall-clock hour should be unchanged, got {out.hour}"

    def test_aware_datetime_converted_to_utc(self):
        """An offset datetime is converted to the equivalent UTC instant.

        Test scenario:
            14:00 at +02:00 becomes 12:00 UTC.
        """
        aware = dt.datetime(
            2024, 3, 20, 14, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=2))
        )
        out = _to_utc(aware)
        assert out.hour == 12, f"expected 12:00 UTC, got {out.hour}:00"
        assert out.utcoffset() == dt.timedelta(0), "result should be UTC"

    def test_already_utc_is_unchanged(self):
        """A UTC datetime round-trips to the same instant."""
        out = _to_utc(_EQUINOX_NOON)
        assert out == _EQUINOX_NOON, f"UTC input should be unchanged, got {out}"

    def test_non_datetime_raises_type_error(self):
        """A non-datetime argument raises TypeError naming the expected type."""
        with pytest.raises(TypeError, match="datetime") as exc:
            _to_utc("2024-03-20")
        assert "datetime" in str(exc.value), f"unexpected message: {exc.value}"


class TestSolarZenithAngleGrid:
    """The `grid` path derives lat/lon from a pyramids Dataset in EPSG:4326."""

    @staticmethod
    def _grid(epsg: int = 4326) -> Dataset:
        return Dataset.from_array(
            np.zeros((2, 3)),
            geo_ref=GeoReference(top_left_corner=(0.0, 0.0), cell_size=1.0, epsg=epsg),
        )

    def test_grid_returns_row_col_shape(self):
        """A grid yields a (rows, columns) array."""
        out = solar_zenith_angle(_EQUINOX_NOON, grid=self._grid())
        assert out.shape == (2, 3)

    def test_grid_matches_meshgrid_of_lat_lon(self):
        """The grid path equals meshing the dataset's lat/lon axes explicitly."""
        grid = self._grid()
        lon2d, lat2d = np.meshgrid(grid.lon, grid.lat)
        assert np.allclose(
            solar_zenith_angle(_EQUINOX_NOON, grid=grid),
            solar_zenith_angle(_EQUINOX_NOON, lat=lat2d, lon=lon2d),
        )

    def test_non_geographic_grid_raises(self):
        """A projected (non-4326) grid is rejected with a clear message."""
        grid = self._grid(epsg=3857)
        with pytest.raises(ValueError, match="geographic"):
            solar_zenith_angle(_EQUINOX_NOON, grid=grid)

    def test_grid_without_epsg_raises(self):
        """A grid with no EPSG (e.g. geostationary) is rejected, not assumed 4326."""
        import types

        fake = types.SimpleNamespace(
            epsg=None, lon=np.array([0.0, 1.0]), lat=np.array([0.0, -1.0])
        )
        with pytest.raises(ValueError, match="geographic"):
            solar_zenith_angle(_EQUINOX_NOON, grid=fake)

    def test_grid_and_latlon_together_raises(self):
        """Passing both grid and lat/lon is an error."""
        grid = self._grid()
        with pytest.raises(ValueError, match="not both"):
            solar_zenith_angle(_EQUINOX_NOON, grid=grid, lat=0.0, lon=0.0)
