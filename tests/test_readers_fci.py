"""Unit tests for `pyramids_eo.readers.read_fci` (offline; synthetic chunks)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.errors import CalibrationError, ReaderError, UnknownSensorError
from pyramids_eo.readers import read_fci
from pyramids_eo.readers.fci import _default_open_chunk
from pyramids_eo.registry import (
    Channel,
    Sensor,
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)
from pyramids_eo.registry import sensors as _sensors


def _chunk(arr: np.ndarray, tlc=(0.0, 4.0)) -> Dataset:
    """A pyramids Dataset chunk holding raw radiance."""
    return Dataset.create_from_array(arr, top_left_corner=tlc, cell_size=1.0, epsg=4326)


class TestReadFci:
    """`read_fci` stitches chunks row-wise and calibrates via the registry."""

    def test_empty_chunks_raise(self):
        """No chunks is a ReaderError."""
        with pytest.raises(ReaderError, match="no chunks"):
            read_fci([], "ir_105")

    def test_stitches_rows_in_order(self):
        """Chunk arrays are concatenated along the row axis in the given order."""
        top = _chunk(np.full((2, 3), 5.0), tlc=(0.0, 4.0))
        bottom = _chunk(np.full((2, 3), 9.0), tlc=(0.0, 2.0))
        out = read_fci([top, bottom], "ir_105", calibrate=False)
        assert out.shape[-2:] == (4, 3), f"expected 4 stitched rows, got {out.shape}"
        arr = out.read_array()
        assert np.allclose(arr[:2], 5.0) and np.allclose(arr[2:], 9.0), (
            "row order wrong"
        )

    def test_thermal_channel_calibrated_to_bt(self):
        """A thermal channel is calibrated to brightness temperature."""
        radiance = np.full((2, 2), 80.0)
        out = read_fci([_chunk(radiance)], "ir_105")
        ch = _sensors.get_sensor("fci").get_channel("ir_105")
        expected = radiance_to_brightness_temperature(
            radiance, ch.central_wavenumber_cm1, ch.alpha, ch.beta
        )
        assert np.allclose(out.read_array(), expected), "BT calibration mismatch"

    def test_solar_channel_calibrated_to_reflectance(self):
        """A solar channel is calibrated to reflectance."""
        radiance = np.full((2, 2), 120.0)
        out = read_fci([_chunk(radiance)], "vis_06")
        ch = _sensors.get_sensor("fci").get_channel("vis_06")
        expected = radiance_to_reflectance(radiance, ch.solar_irradiance)
        assert np.allclose(out.read_array(), expected), "reflectance mismatch"

    def test_calibrate_false_returns_raw(self):
        """With calibrate=False the stitched raw radiance is returned."""
        radiance = np.full((2, 2), 42.0)
        out = read_fci([_chunk(radiance)], "ir_105", calibrate=False)
        assert np.allclose(out.read_array(), 42.0), "raw radiance should pass through"

    def test_output_declares_nan_nodata(self):
        """The calibrated output declares NaN as its nodata (not the -9999 default)."""
        out = read_fci([_chunk(np.ones((2, 2)))], "ir_105")
        assert np.isnan(out.no_data_value[0]), (
            f"nodata should be NaN: {out.no_data_value}"
        )

    def test_geolocation_from_northernmost_chunk(self):
        """The result carries the northernmost chunk's CRS + geotransform."""
        north = _chunk(np.ones((2, 2)), tlc=(0.0, 4.0))
        out = read_fci([north, _chunk(np.ones((2, 2)), tlc=(0.0, 2.0))], "ir_105")
        assert out.epsg == 4326, f"CRS not preserved, got {out.epsg}"
        assert out.geotransform == north.geotransform, "geotransform not from north"

    def test_reverse_order_chunks_still_geolocate_correctly(self):
        """Chunks passed south-first are reordered north -> south (M1 footgun)."""
        south = _chunk(np.full((2, 3), 9.0), tlc=(0.0, 2.0))
        north = _chunk(np.full((2, 3), 5.0), tlc=(0.0, 4.0))
        out = read_fci([south, north], "ir_105", calibrate=False)
        arr = out.read_array()
        assert np.allclose(arr[:2], 5.0) and np.allclose(arr[2:], 9.0), "not reordered"
        assert out.geotransform == north.geotransform, (
            "origin should be the north chunk"
        )

    def test_mixed_crs_chunks_raise(self):
        """Chunks with different CRS are rejected."""
        a = _chunk(np.ones((2, 2)), tlc=(0.0, 4.0))
        b = Dataset.create_from_array(
            np.ones((2, 2)), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=3857
        )
        with pytest.raises(ReaderError, match="mixed CRS"):
            read_fci([a, b], "ir_105")

    def test_mixed_cell_size_chunks_raise(self):
        """Chunks with different cell sizes are rejected."""
        a = _chunk(np.ones((2, 2)), tlc=(0.0, 4.0))
        b = Dataset.create_from_array(
            np.ones((2, 2)), top_left_corner=(0.0, 2.0), cell_size=2.0, epsg=4326
        )
        with pytest.raises(ReaderError, match="cell size"):
            read_fci([a, b], "ir_105")

    def test_mixed_column_count_chunks_raise(self):
        """Chunks with different widths are rejected."""
        a = _chunk(np.ones((2, 3)), tlc=(0.0, 4.0))
        b = _chunk(np.ones((2, 2)), tlc=(0.0, 2.0))
        with pytest.raises(ReaderError, match="column count"):
            read_fci([a, b], "ir_105")

    def test_non_contiguous_chunks_raise(self):
        """A vertical gap between chunks is rejected."""
        top = _chunk(np.ones((2, 2)), tlc=(0.0, 4.0))
        gapped = _chunk(np.ones((2, 2)), tlc=(0.0, -5.0))
        with pytest.raises(ReaderError, match="contiguous"):
            read_fci([top, gapped], "ir_105")

    def test_unknown_channel_raises(self):
        """An unknown channel surfaces UnknownSensorError from the registry."""
        with pytest.raises(UnknownSensorError, match="has no channel"):
            read_fci([_chunk(np.ones((2, 2)))], "not_a_channel")

    def test_coeffs_override_solar_irradiance(self):
        """Per-granule coeffs override the registry solar irradiance."""
        radiance = np.full((2, 2), 100.0)
        out = read_fci([_chunk(radiance)], "vis_06", coeffs={"solar_irradiance": 500.0})
        assert np.allclose(out.read_array(), radiance_to_reflectance(radiance, 500.0))

    def test_coeffs_override_thermal_constants(self):
        """Per-granule coeffs override the registry Planck constants."""
        radiance = np.full((2, 2), 80.0)
        out = read_fci(
            [_chunk(radiance)],
            "ir_105",
            coeffs={"central_wavenumber_cm1": 900.0, "alpha": 0.99, "beta": 0.5},
        )
        expected = radiance_to_brightness_temperature(radiance, 900.0, 0.99, 0.5)
        assert np.allclose(out.read_array(), expected), "coeffs not preferred"

    def test_coeffs_alpha_zero_not_coerced_to_one(self):
        """A coeffs alpha of 0 surfaces the invalid-alpha error (not silently 1.0)."""
        with pytest.raises(CalibrationError, match="alpha"):
            read_fci([_chunk(np.ones((2, 2)))], "ir_105", coeffs={"alpha": 0.0})

    def test_open_chunk_used_for_non_dataset(self):
        """A non-Dataset chunk is opened via the injected open_chunk callable."""
        captured = {}

        def _opener(path, channel):
            captured["path"], captured["channel"] = path, channel
            return _chunk(np.full((2, 2), 7.0))

        out = read_fci(["chunk0.nc"], "ir_105", calibrate=False, open_chunk=_opener)
        assert captured == {"path": "chunk0.nc", "channel": "ir_105"}, captured
        assert np.allclose(out.read_array(), 7.0), "opened chunk not used"

    def test_solar_channel_missing_irradiance_raises(self, monkeypatch):
        """A solar channel without solar_irradiance raises CalibrationError."""
        broken = Sensor(
            name="fci",
            channels={"x": Channel("x", 0.6, 1000, "solar", solar_irradiance=None)},
        )
        monkeypatch.setattr(
            "pyramids_eo.readers._common.get_sensor", lambda name: broken
        )
        with pytest.raises(CalibrationError, match="solar_irradiance"):
            read_fci([_chunk(np.ones((2, 2)))], "x")

    def test_thermal_channel_missing_wavenumber_raises(self, monkeypatch):
        """A thermal channel without a central wavenumber raises CalibrationError."""
        broken = Sensor(
            name="fci",
            channels={
                "y": Channel("y", 10.5, 2000, "thermal", central_wavenumber_cm1=None)
            },
        )
        monkeypatch.setattr(
            "pyramids_eo.readers._common.get_sensor", lambda name: broken
        )
        with pytest.raises(CalibrationError, match="central_wavenumber"):
            read_fci([_chunk(np.ones((2, 2)))], "y")


class TestDefaultOpenChunk:
    """The default NetCDF opener delegates to pyramids.netcdf.NetCDF."""

    def test_reads_variable_via_netcdf(self, monkeypatch):
        """`_default_open_chunk` reads the file and pulls the named variable."""
        marker = object()

        class _FakeNC:
            def get_variable(self, name):
                assert name == "ir_105", f"unexpected variable {name}"
                return marker

        import pyramids.netcdf as _ncmod

        monkeypatch.setattr(
            _ncmod.NetCDF, "read_file", classmethod(lambda cls, p: _FakeNC())
        )
        assert _default_open_chunk("chunk.nc", "ir_105") is marker, (
            "variable not returned"
        )
