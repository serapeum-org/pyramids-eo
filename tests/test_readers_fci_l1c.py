"""Unit + live tests for `pyramids_eo.sensors.readers.fci_l1c` (real FCI L1C)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers import fci_l1c
from pyramids_eo.sensors.readers.fci_l1c import (
    _granule_coeffs,
    _measured_group,
    _satellite_height,
    _scalar,
    _unpack_radiance,
    read_fci_l1c,
    read_fci_l1c_chunk,
)
from pyramids_eo.sensors.registry import radiance_to_brightness_temperature

#: A minimal geostationary CRS whose PROJ form carries `+h=35786400`.
GEOS_WKT = (
    'PROJCS["Geostationary_Satellite",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
    'UNIT["degree",0.0174532925199433]],PROJECTION["Geostationary_Satellite"],'
    'PARAMETER["central_meridian",0],PARAMETER["satellite_height",35786400],'
    'UNIT["metre",1],AXIS["X",EAST],AXIS["Y",NORTH]]'
)
_THERMAL = {
    "kind": "thermal",
    "central_wavenumber_cm1": 950.0,
    "alpha": 0.999,
    "beta": 0.36,
}


class _FakeBand:
    """A GDAL raster band exposing a packed array and its metadata."""

    def __init__(self, raw, meta):
        self._raw, self._meta = raw, meta

    def ReadAsArray(self):
        return self._raw

    def GetMetadata(self):
        return self._meta


class _FakeRaster:
    """A GDAL raster carrying one band, a geotransform, and a CRS."""

    def __init__(self, band, gt, proj):
        self._band, self._gt, self._proj = band, gt, proj

    def GetRasterBand(self, index):
        return self._band

    def GetGeoTransform(self):
        return self._gt

    def GetProjection(self):
        return self._proj


class _FakeArray:
    """A GDAL MDArray returning a fixed value."""

    def __init__(self, value):
        self._value = value

    def ReadAsArray(self):
        return np.array(self._value)


class _FakeGroup:
    """A GDAL multidim group over a `{name: value}` variable map."""

    def __init__(self, arrays, names=None):
        self._arrays = arrays
        self._names = list(arrays) if names is None else names

    def GetMDArrayNames(self):
        return self._names

    def OpenMDArray(self, name):
        if name not in self._arrays:
            raise RuntimeError(f"no variable {name}")
        return _FakeArray(self._arrays[name])


class TestUnpackRadiance:
    """`_unpack_radiance` unpacks uint16 counts and masks invalid pixels."""

    def test_unpacks_and_masks(self, monkeypatch):
        """Counts unpack via scale/offset; fill and out-of-range map to NaN."""
        raw = np.array([[100, 65535], [4096, 500]], dtype="uint16")
        meta = {
            "scale_factor": "0.5",
            "add_offset": "-10.0",
            "_FillValue": "65535",
            "valid_range": "{0, 4095}",
        }
        raster = _FakeRaster(_FakeBand(raw, meta), (0.1, -1e-5, 0, 0.2, 0, 1e-5), "WKT")
        import osgeo.gdal as _g

        monkeypatch.setattr(_g, "Open", lambda sub: raster)
        rad, gt, crs = _unpack_radiance("f.nc", "ir_105")
        assert rad[0, 0] == pytest.approx(40.0), "100*0.5-10 should unpack to 40"
        assert np.isnan(rad[0, 1]), "the _FillValue count should be NaN"
        assert np.isnan(rad[1, 0]), "a count above valid_range max should be NaN"
        assert rad[1, 1] == pytest.approx(240.0), "500*0.5-10 should unpack to 240"
        assert gt[1] == -1e-5 and crs == "WKT", "geotransform + CRS should pass through"

    def test_no_masking_without_fill_or_range(self, monkeypatch):
        """With no valid_range and no _FillValue, no pixel is masked to NaN."""
        raw = np.array([[10, 20]], dtype="uint16")
        raster = _FakeRaster(
            _FakeBand(raw, {"scale_factor": "1.0", "add_offset": "0.0"}), (0,) * 6, "W"
        )
        import osgeo.gdal as _g

        monkeypatch.setattr(_g, "Open", lambda sub: raster)
        rad, _gt, _crs = _unpack_radiance("f.nc", "ir_105")
        assert not np.isnan(rad).any(), "nothing should be masked"
        assert rad[0, 1] == pytest.approx(20.0), "counts should still unpack"


class TestMeasuredGroup:
    """`_measured_group` returns the channel group or None for a trailer."""

    def _patch(self, monkeypatch, group):
        class _Root:
            def OpenGroupFromFullname(self, path):
                if group is None:
                    raise RuntimeError("no group")
                return group

        class _DS:
            def GetRootGroup(self):
                return _Root()

        import osgeo.gdal as _g

        monkeypatch.setattr(_g, "OpenEx", lambda p, flags: _DS())

    def test_returns_group(self, monkeypatch):
        """A present channel group is returned with its owning dataset."""
        g = _FakeGroup({"effective_radiance": 1})
        self._patch(monkeypatch, g)
        _dataset, group = _measured_group("f.nc", "ir_105")
        assert group is g

    def test_missing_group_returns_none(self, monkeypatch):
        """A file without the group (trailer) yields (None, None)."""
        self._patch(monkeypatch, None)
        _dataset, group = _measured_group("trail.nc", "ir_105")
        assert group is None


class TestScalar:
    """`_scalar` reads a 0-D variable, or None when absent."""

    def test_reads_value(self):
        """A present scalar variable is read as a float."""
        assert (
            _scalar(_FakeGroup({"start_position_row": 140}), "start_position_row")
            == 140.0
        )

    def test_absent_returns_none(self):
        """A missing scalar variable yields None."""
        assert _scalar(_FakeGroup({}), "missing") is None

    def test_none_mdarray_returns_none(self):
        """A group whose OpenMDArray returns None yields None."""

        class _G:
            def OpenMDArray(self, name):
                return None

        assert _scalar(_G(), "x") is None


class TestGranuleCoeffs:
    """`_granule_coeffs` picks solar vs thermal from the group variables."""

    def test_solar(self, monkeypatch):
        """A finite solar irradiance selects the solar path."""
        monkeypatch.setattr(
            fci_l1c, "_scalar", lambda g, n: 1580.0 if "solar" in n else None
        )
        assert _granule_coeffs(object()) == {
            "kind": "solar",
            "solar_irradiance": 1580.0,
        }

    def test_thermal_when_irradiance_is_fill(self, monkeypatch):
        """A fill-valued solar irradiance selects the thermal Planck triple."""
        values = {
            "channel_effective_solar_irradiance": 9.97e36,
            "radiance_to_bt_conversion_coefficient_wavenumber": 950.0,
            "radiance_to_bt_conversion_coefficient_a": 0.999,
            "radiance_to_bt_conversion_coefficient_b": 0.36,
        }
        monkeypatch.setattr(fci_l1c, "_scalar", lambda g, n: values.get(n))
        coeffs = _granule_coeffs(object())
        assert coeffs["kind"] == "thermal", "fill irradiance should route to thermal"
        assert coeffs["central_wavenumber_cm1"] == 950.0, "wavenumber should be read"
        assert coeffs["alpha"] == 0.999 and coeffs["beta"] == 0.36


class TestReadFciL1cChunk:
    """`read_fci_l1c_chunk` assembles a chunk record or skips a trailer."""

    def test_assembles_record(self, monkeypatch):
        """A chunk with radiance yields a full record."""
        monkeypatch.setattr(
            fci_l1c,
            "_measured_group",
            lambda p, c: (object(), _FakeGroup({"effective_radiance": 1})),
        )
        monkeypatch.setattr(
            fci_l1c,
            "_unpack_radiance",
            lambda p, c: (np.ones((2, 2)), (0.1,) * 6, "WKT"),
        )
        monkeypatch.setattr(fci_l1c, "_scalar", lambda g, n: 140.0)
        monkeypatch.setattr(fci_l1c, "_granule_coeffs", lambda g: _THERMAL)
        rec = read_fci_l1c_chunk("f.nc", "ir_105")
        assert rec["start_row"] == 140.0 and rec["coeffs"] == _THERMAL
        assert rec["crs"] == "WKT" and rec["radiance"].shape == (2, 2)

    def test_none_group_skips(self, monkeypatch):
        """A file without the channel group is skipped (None)."""
        monkeypatch.setattr(fci_l1c, "_measured_group", lambda p, c: (None, None))
        assert read_fci_l1c_chunk("trail.nc", "ir_105") is None

    def test_group_without_radiance_skips(self, monkeypatch):
        """A group lacking effective_radiance is skipped (None)."""
        monkeypatch.setattr(
            fci_l1c,
            "_measured_group",
            lambda p, c: (object(), _FakeGroup({}, names=["x"])),
        )
        assert read_fci_l1c_chunk("trail.nc", "ir_105") is None


class TestSatelliteHeight:
    """`_satellite_height` extracts +h from a geostationary CRS WKT."""

    def test_reads_height(self):
        """The geostationary WKT yields its satellite height in metres."""
        assert _satellite_height(GEOS_WKT) == pytest.approx(35786400.0)

    def test_missing_height_raises(self):
        """A non-geostationary CRS (no +h) raises ReaderError."""
        wkt = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
            '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        )
        with pytest.raises(ReaderError, match="satellite height"):
            _satellite_height(wkt)


class TestReadFciL1c:
    """`read_fci_l1c` orders, validates, stitches and calibrates chunks."""

    @staticmethod
    def _chunk(radiance, start, end, coeffs=None):
        # A geotransform whose Y origin follows start_row, so adjacent chunks are
        # geospatially contiguous (north-up: gt[5] < 0, origin = start * gt[5]).
        gt5 = -1e-5
        return {
            "radiance": radiance,
            "start_row": start,
            "end_row": end,
            "coeffs": coeffs or _THERMAL,
            "geotransform": (0.1, gt5, 0.0, start * gt5, 0.0, gt5),
            "crs": GEOS_WKT,
        }

    def _patch(self, monkeypatch, mapping):
        monkeypatch.setattr(fci_l1c, "read_fci_l1c_chunk", lambda p, c: mapping[p])
        monkeypatch.setattr(fci_l1c, "_satellite_height", lambda wkt: 1.0e5)

    def test_orders_geospatially_and_skips_trailer(self, monkeypatch):
        """Chunks are ordered north->south by geotransform Y; a trailer is dropped."""
        mapping = {
            "b.nc": self._chunk(np.full((2, 3), 9.0), 142, 143),
            "trail.nc": None,
            "a.nc": self._chunk(np.full((2, 3), 5.0), 140, 141),
        }
        self._patch(monkeypatch, mapping)
        out = read_fci_l1c(["b.nc", "trail.nc", "a.nc"], "ir_105", calibrate=False)
        arr = out.read_array()
        assert arr.shape == (4, 3), f"expected 4 stitched rows, got {arr.shape}"
        assert np.allclose(arr[:2], 5.0), (
            "the northernmost (largest gt[3]) chunk is on top"
        )
        assert np.allclose(arr[2:], 9.0), "the southern chunk is below"

    def test_non_contiguous_raises(self, monkeypatch):
        """A vertical gap between chunks is rejected."""
        mapping = {
            "a.nc": self._chunk(np.ones((2, 3)), 140, 141),
            "b.nc": self._chunk(np.ones((2, 3)), 279, 280),
        }
        self._patch(monkeypatch, mapping)
        with pytest.raises(ReaderError, match="contiguous"):
            read_fci_l1c(["a.nc", "b.nc"], "ir_105")

    def test_no_chunks_raises(self, monkeypatch):
        """When every path is a trailer, a ReaderError is raised."""
        self._patch(monkeypatch, {"trail.nc": None})
        with pytest.raises(ReaderError, match="no chunk"):
            read_fci_l1c(["trail.nc"], "ir_105")

    def test_calibrates_with_granule_coeffs(self, monkeypatch):
        """Calibration uses the per-granule Planck coefficients."""
        radiance = np.full((2, 3), 80.0)
        self._patch(monkeypatch, {"a.nc": self._chunk(radiance, 140, 141)})
        out = read_fci_l1c(["a.nc"], "ir_105")
        expected = radiance_to_brightness_temperature(radiance, 950.0, 0.999, 0.36)
        assert np.allclose(out.read_array(), expected), "BT should use granule coeffs"

    def test_geotransform_scaled_by_height(self, monkeypatch):
        """The metre geotransform is the angular one times the satellite height."""
        self._patch(monkeypatch, {"a.nc": self._chunk(np.ones((2, 3)), 140, 141)})
        out = read_fci_l1c(["a.nc"], "ir_105", calibrate=False)
        assert out.geotransform[1] == pytest.approx(-1e-5 * 1.0e5), (
            "px should be scaled"
        )
        assert out.epsg is None, "a geostationary grid has no EPSG code"

    def test_missing_position_raises(self, monkeypatch):
        """A chunk without a row position is a ReaderError, not a TypeError."""
        chunk = self._chunk(np.ones((2, 3)), 140, 141)
        chunk["start_row"] = None
        self._patch(monkeypatch, {"a.nc": chunk})
        with pytest.raises(ReaderError, match="start/end_position_row"):
            read_fci_l1c(["a.nc"], "ir_105")

    def test_mixed_column_count_raises(self, monkeypatch):
        """Chunks with different widths are rejected."""
        mapping = {
            "a.nc": self._chunk(np.ones((2, 3)), 140, 141),
            "b.nc": self._chunk(np.ones((2, 4)), 142, 143),
        }
        self._patch(monkeypatch, mapping)
        with pytest.raises(ReaderError, match="column count"):
            read_fci_l1c(["a.nc", "b.nc"], "ir_105")

    def test_mixed_crs_raises(self, monkeypatch):
        """Chunks from a different CRS are rejected."""
        other = self._chunk(np.ones((2, 3)), 142, 143)
        other["crs"] = "OTHER_WKT"
        mapping = {"a.nc": self._chunk(np.ones((2, 3)), 140, 141), "b.nc": other}
        self._patch(monkeypatch, mapping)
        with pytest.raises(ReaderError, match="mixed CRS"):
            read_fci_l1c(["a.nc", "b.nc"], "ir_105")


@pytest.mark.live
def test_read_fci_l1c_real_granule():
    """End-to-end decode of real FCI L1C chunks into ir_105 brightness temperature.

    Skips unless a directory of real FDHSI chunks is provided via
    `FCI_FIXTURES_DIR` (the marker, not the env var, gates whether this runs).
    """
    fixtures = Path(os.environ.get("FCI_FIXTURES_DIR", "tests/data/fci_l1c"))
    paths = sorted(fixtures.glob("*.nc"))
    if len(paths) < 2:
        pytest.skip("real FCI L1C fixtures not available (set FCI_FIXTURES_DIR)")
    scene = read_fci_l1c(paths, "ir_105")
    array = np.asarray(scene.read_array(), dtype=float)
    finite = array[np.isfinite(array)]
    assert 180.0 < finite.min() and finite.max() < 340.0, "BT outside a physical range"
    assert "Geostationary" in str(scene.crs), (
        "result should carry the geostationary CRS"
    )
    assert abs(scene.geotransform[1]) == pytest.approx(2000.0, abs=1.0), (
        "ir_105 is 2 km"
    )
    assert scene.geotransform[5] < 0, "the stitched grid must be north-up (gt[5] < 0)"
    assert np.isnan(scene.no_data_value[0]), "nodata should be NaN"
