"""Unit tests for `pyramids_eo.readers.read_seviri` (offline; parser injected)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.errors import CalibrationError, ReaderError, UnknownSensorError
from pyramids_eo.readers import read_seviri
from pyramids_eo.readers.seviri import _default_parse
from pyramids_eo.registry import (
    Channel,
    Sensor,
    get_sensor,
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)


def _ds(arr: np.ndarray, tlc=(0.0, 4.0)) -> Dataset:
    """A pyramids Dataset holding raw radiance."""
    return Dataset.create_from_array(arr, top_left_corner=tlc, cell_size=1.0, epsg=4326)


class TestReadSeviri:
    """`read_seviri` calibrates and geolocates a single channel."""

    def test_none_source_raises(self):
        """A missing source is a ReaderError."""
        with pytest.raises(ReaderError, match="source is required"):
            read_seviri(None, "IR_108")

    def test_thermal_channel_calibrated_to_bt(self):
        """A thermal channel is calibrated to brightness temperature."""
        radiance = np.full((2, 2), 80.0)
        out = read_seviri(_ds(radiance), "IR_108")
        ch = get_sensor("seviri").get_channel("IR_108")
        expected = radiance_to_brightness_temperature(
            radiance, ch.central_wavenumber_cm1, ch.alpha, ch.beta
        )
        assert np.allclose(out.read_array(), expected), "BT calibration mismatch"

    def test_solar_channel_calibrated_to_reflectance(self):
        """A solar channel is calibrated to reflectance."""
        radiance = np.full((2, 2), 120.0)
        out = read_seviri(_ds(radiance), "VIS006")
        ch = get_sensor("seviri").get_channel("VIS006")
        expected = radiance_to_reflectance(radiance, ch.solar_irradiance)
        assert np.allclose(out.read_array(), expected), "reflectance mismatch"

    def test_calibrate_false_returns_raw(self):
        """With calibrate=False the raw radiance is returned."""
        out = read_seviri(_ds(np.full((2, 2), 42.0)), "IR_108", calibrate=False)
        assert np.allclose(out.read_array(), 42.0), "raw radiance should pass through"

    def test_geolocation_preserved(self):
        """The result carries the source CRS + geotransform."""
        src = _ds(np.ones((2, 2)))
        out = read_seviri(src, "IR_108")
        assert out.epsg == 4326 and out.geotransform == src.geotransform

    def test_parse_used_for_non_dataset(self):
        """A non-Dataset source is decoded via the injected `parse` callable."""
        captured = {}

        def _parser(path, channel):
            captured["path"], captured["channel"] = path, channel
            return _ds(np.full((2, 2), 7.0))

        out = read_seviri("scene.nat", "IR_108", calibrate=False, parse=_parser)
        assert captured == {"path": "scene.nat", "channel": "IR_108"}, captured
        assert np.allclose(out.read_array(), 7.0), "parsed radiance not used"

    def test_unknown_channel_raises(self):
        """An unknown channel surfaces UnknownSensorError."""
        with pytest.raises(UnknownSensorError, match="has no channel"):
            read_seviri(_ds(np.ones((2, 2))), "NOPE")

    def test_missing_calibration_constant_raises(self, monkeypatch):
        """A channel missing its constants raises CalibrationError."""
        broken = Sensor(
            name="seviri",
            channels={
                "z": Channel("z", 3.9, 3000, "thermal", central_wavenumber_cm1=None)
            },
        )
        monkeypatch.setattr(
            "pyramids_eo.readers._common.get_sensor", lambda name: broken
        )
        with pytest.raises(CalibrationError, match="central_wavenumber"):
            read_seviri(_ds(np.ones((2, 2))), "z")


class TestDefaultParse:
    """The default `.nat` parser is a documented not-implemented placeholder."""

    def test_default_parse_not_implemented(self):
        """Calling the default parser raises NotImplementedError with guidance."""
        with pytest.raises(NotImplementedError, match="parse"):
            _default_parse("scene.nat", "IR_108")

    def test_non_dataset_without_parse_raises(self):
        """A non-Dataset source with no parser hits the not-implemented default."""
        with pytest.raises(NotImplementedError):
            read_seviri("scene.nat", "IR_108")
