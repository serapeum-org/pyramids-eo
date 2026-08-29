"""Unit + live tests for `pyramids_eo.sensors.readers.fci_l1c` (real FCI L1C)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers import fci_l1c
from pyramids_eo.sensors.readers._common import resolve_channels
from pyramids_eo.sensors.readers.fci_l1c import (
    _granule_coeffs,
    _measured_group,
    _satellite_height,
    _scalar,
    _unpack_radiance,
    _valid_bounds,
    available_channels,
    read_fci_l1c,
    read_fci_l1c_chunk,
    read_fci_l1c_chunks,
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
        assert gt[1] == -1e-5, "the geotransform should pass through"
        assert crs == "WKT", "the CRS should pass through"

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

    def test_masks_below_valid_min(self, monkeypatch):
        """A count below the valid_range minimum is masked to NaN."""
        raw = np.array([[5, 50]], dtype="uint16")
        meta = {"scale_factor": "1.0", "add_offset": "0.0", "valid_range": "{10, 4095}"}
        raster = _FakeRaster(_FakeBand(raw, meta), (0,) * 6, "W")
        import osgeo.gdal as _g

        monkeypatch.setattr(_g, "Open", lambda sub: raster)
        rad, _gt, _crs = _unpack_radiance("f.nc", "ir_105")
        assert np.isnan(rad[0, 0]), "a count below valid_min should be NaN"
        assert rad[0, 1] == pytest.approx(50.0), "an in-range count should survive"

    def test_missing_scale_raises(self, monkeypatch):
        """Missing scale_factor/add_offset raises ReaderError, not KeyError."""
        raster = _FakeRaster(_FakeBand(np.ones((1, 1), "uint16"), {}), (0,) * 6, "W")
        import osgeo.gdal as _g

        monkeypatch.setattr(_g, "Open", lambda sub: raster)
        with pytest.raises(ReaderError, match="scale_factor"):
            _unpack_radiance("f.nc", "ir_105")

    def test_none_raster_raises(self, monkeypatch):
        """A None from gdal.Open (exceptions off) raises ReaderError."""
        import osgeo.gdal as _g

        monkeypatch.setattr(_g, "Open", lambda sub: None)
        with pytest.raises(ReaderError, match="cannot open"):
            _unpack_radiance("f.nc", "ir_105")

    def test_open_runtimeerror_raises(self, monkeypatch):
        """A RuntimeError from gdal.Open (exceptions on) raises ReaderError."""
        import osgeo.gdal as _g

        def _raise(sub):
            raise RuntimeError("not a supported file format")

        monkeypatch.setattr(_g, "Open", _raise)
        with pytest.raises(ReaderError, match="cannot open"):
            _unpack_radiance("f.nc", "ir_105")


class TestValidBounds:
    """`_valid_bounds` parses (min, max) from a valid_range string."""

    def test_parses_pair(self):
        """A two-number range yields its (min, max)."""
        assert _valid_bounds("{10, 4095}") == (10.0, 4095.0)

    def test_missing_or_degenerate_yields_none(self):
        """None or a string without two numbers yields (None, None)."""
        assert _valid_bounds(None) == (None, None)
        assert _valid_bounds("{}") == (None, None)
        assert _valid_bounds("{5}") == (None, None)

    def test_scientific_and_negative(self):
        """Scientific-notation and negative bounds parse correctly."""
        assert _valid_bounds("{-1.5e2, 4.0e3}") == (-150.0, 4000.0)


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

    def test_open_failure_raises(self, monkeypatch):
        """A RuntimeError from OpenEx (exceptions on) raises ReaderError."""
        import osgeo.gdal as _g

        def _raise(path, flags):
            raise RuntimeError("not recognized as a supported file format")

        monkeypatch.setattr(_g, "OpenEx", _raise)
        with pytest.raises(ReaderError, match="cannot open"):
            _measured_group("corrupt.nc", "ir_105")

    def test_none_dataset_raises(self, monkeypatch):
        """A None from OpenEx (exceptions off) raises ReaderError."""
        import osgeo.gdal as _g

        monkeypatch.setattr(_g, "OpenEx", lambda path, flags: None)
        with pytest.raises(ReaderError, match="cannot open"):
            _measured_group("corrupt.nc", "ir_105")


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
        assert coeffs["alpha"] == 0.999, "alpha should be read"
        assert coeffs["beta"] == 0.36, "beta should be read"

    def test_thermal_omits_absent_coeffs(self, monkeypatch):
        """A missing thermal coefficient is omitted so the registry fallback engages."""
        values = {
            "channel_effective_solar_irradiance": 9.97e36,
            "radiance_to_bt_conversion_coefficient_a": 0.999,
        }
        monkeypatch.setattr(fci_l1c, "_scalar", lambda g, n: values.get(n))
        coeffs = _granule_coeffs(object())
        assert coeffs == {"kind": "thermal", "alpha": 0.999}, (
            f"absent wavenumber/beta should be omitted, got {coeffs}"
        )

    def test_no_coeffs_leaves_kind_unset(self, monkeypatch):
        """With neither irradiance nor thermal coefficients, no kind is claimed."""
        monkeypatch.setattr(fci_l1c, "_scalar", lambda g, n: None)
        assert _granule_coeffs(object()) == {}, (
            "a channel with no granule coeffs must let the registry decide its kind"
        )


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
        assert rec["start_row"] == 140.0, "start_row should be read"
        assert rec["coeffs"] == _THERMAL, "coeffs should be set"
        assert rec["crs"] == "WKT", "crs should be set"
        assert rec["radiance"].shape == (2, 2), "radiance should be set"

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

    def test_implausible_height_raises(self):
        """A satellite height far from geostationary is rejected."""
        from osgeo import osr

        srs = osr.SpatialReference()
        srs.ImportFromProj4("+proj=geos +lon_0=0 +h=100 +ellps=WGS84 +units=m")
        wkt = srs.ExportToWkt()
        with pytest.raises(ReaderError, match="implausible"):
            _satellite_height(wkt)


class TestReadFciL1c:
    """`read_fci_l1c` orders, validates, stitches and calibrates chunks."""

    @staticmethod
    def _chunk(radiance, top_y, *, start=1, end=1, coeffs=None):
        # The geotransform Y origin (top_y) is independent of the row index, so a
        # test can make the two disagree (as they do on a real FCI granule).
        gt5 = -1e-5
        return {
            "radiance": radiance,
            "start_row": start,
            "end_row": end,
            "coeffs": coeffs or _THERMAL,
            "geotransform": (0.1, gt5, 0.0, top_y, 0.0, gt5),
            "crs": GEOS_WKT,
        }

    def _patch(self, monkeypatch, mapping):
        # read_fci_l1c decodes each chunk once for the whole channel set via
        # read_fci_l1c_chunks -> {channel: record}; the fake returns the mapped
        # record (or None) for each requested channel.
        monkeypatch.setattr(
            fci_l1c,
            "read_fci_l1c_chunks",
            lambda p, channels: {ch: mapping[p] for ch in channels},
        )
        monkeypatch.setattr(fci_l1c, "_satellite_height", lambda wkt: 1.0e5)

    def test_orders_by_geotransform_not_row_index(self, monkeypatch):
        """Ordering follows geotransform Y even when start_row disagrees (the flip)."""
        # On a real granule the NORTH chunk (larger gt[3]) has the LARGER start_row,
        # so ordering by start_row would flip the scene; ordering by gt[3] must not.
        mapping = {
            "b.nc": self._chunk(np.full((2, 3), 9.0), -2e-5, start=140),  # south
            "trail.nc": None,
            "a.nc": self._chunk(np.full((2, 3), 5.0), 0.0, start=279),  # north
        }
        self._patch(monkeypatch, mapping)
        out = read_fci_l1c(["b.nc", "trail.nc", "a.nc"], "ir_105", calibrate=False)
        arr = out.read_array()
        assert arr.shape == (4, 3), f"expected 4 stitched rows, got {arr.shape}"
        assert np.allclose(arr[:2], 5.0), (
            "the north chunk (largest gt[3]) is on top, despite its larger start_row"
        )
        assert np.allclose(arr[2:], 9.0), "the south chunk is below"

    def test_non_contiguous_raises(self, monkeypatch):
        """A vertical gap between chunks is rejected."""
        mapping = {
            "a.nc": self._chunk(np.ones((2, 3)), 0.0),
            "b.nc": self._chunk(np.ones((2, 3)), -0.5),  # far below -> gap
        }
        self._patch(monkeypatch, mapping)
        with pytest.raises(ReaderError, match="contiguous"):
            read_fci_l1c(["a.nc", "b.nc"], "ir_105")

    def test_overlap_raises(self, monkeypatch):
        """A vertical overlap between chunks is rejected."""
        mapping = {
            "a.nc": self._chunk(np.ones((2, 3)), 0.0),
            "b.nc": self._chunk(np.ones((2, 3)), -1e-5),  # inside a's span -> overlap
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
        self._patch(monkeypatch, {"a.nc": self._chunk(radiance, 0.0)})
        out = read_fci_l1c(["a.nc"], "ir_105")
        expected = radiance_to_brightness_temperature(radiance, 950.0, 0.999, 0.36)
        assert np.allclose(out.read_array(), expected), "BT should use granule coeffs"

    def test_geotransform_scaled_by_height(self, monkeypatch):
        """The metre geotransform is the angular one times the satellite height."""
        self._patch(monkeypatch, {"a.nc": self._chunk(np.ones((2, 3)), 0.0)})
        out = read_fci_l1c(["a.nc"], "ir_105", calibrate=False)
        # The y axis is not sign-normalised, so it shows the raw height scaling.
        assert out.geotransform[5] == pytest.approx(-1e-5 * 1.0e5), (
            "the y pixel height should be the angular one scaled by the height"
        )
        # The x pixel width is scaled by the same height, then normalised positive.
        assert abs(out.geotransform[1]) == pytest.approx(1e-5 * 1.0e5), (
            "the x pixel width magnitude should be the angular one scaled by height"
        )
        assert out.epsg is None, "a geostationary grid has no EPSG code"

    def test_x_axis_normalised_to_positive_width(self, monkeypatch):
        """A negative angular x width is reconciled to a positive, west-anchored one.

        FCI's x is a scanning azimuth angle whose sign runs opposite to PROJ geos x,
        so the scaled geotransform comes out with a negative x pixel width anchored
        on the eastern limb. The reader must flip that to a positive width and
        re-anchor the origin on the western limb, leaving the pixels untouched, so a
        warp does not mirror the scene east-west (issue #56).
        """
        radiance = np.arange(6, dtype=float).reshape(2, 3)
        self._patch(monkeypatch, {"a.nc": self._chunk(radiance, 0.0)})
        out = read_fci_l1c(["a.nc"], "ir_105", calibrate=False)
        geo = out.geotransform
        # angular (0.1, -1e-5, 0, 0, 0, -1e-5) * height 1e5, width 3 -> re-anchored.
        assert geo[1] > 0, f"x pixel width must be positive, got {geo[1]}"
        assert geo[1] == pytest.approx(1.0), "x width is |−1e-5| * 1e5"
        assert geo[0] == pytest.approx(0.1 * 1.0e5 + (-1.0) * 3), (
            "origin re-anchored on the western limb: origin_x + pixel_w * ncols"
        )
        assert geo[5] < 0, "the y axis stays north-up (negative pixel height)"
        assert np.array_equal(out.read_array(), radiance), (
            "the array must be returned unchanged — only the geotransform is fixed"
        )

    def test_positive_x_width_passes_through(self, monkeypatch):
        """A geotransform already east-increasing is scaled but not re-anchored."""
        record = {
            "radiance": np.ones((2, 3)),
            "start_row": 1,
            "end_row": 1,
            "coeffs": _THERMAL,
            "geotransform": (0.1, 1e-5, 0.0, 0.0, 0.0, -1e-5),  # positive x width
            "crs": GEOS_WKT,
        }
        self._patch(monkeypatch, {"a.nc": record})
        out = read_fci_l1c(["a.nc"], "ir_105", calibrate=False)
        geo = out.geotransform
        assert geo[1] == pytest.approx(1e-5 * 1.0e5), (
            "positive x width is left positive"
        )
        assert geo[0] == pytest.approx(0.1 * 1.0e5), (
            "an already-positive width is not re-anchored (origin is the scaled one)"
        )

    def test_rotated_grid_raises(self, monkeypatch):
        """A non-zero geotransform rotation term is rejected, not silently emitted."""
        record = {
            "radiance": np.ones((2, 3)),
            "start_row": 1,
            "end_row": 1,
            "coeffs": _THERMAL,
            "geotransform": (0.1, -1e-5, 0.1, 0.0, 0.0, -1e-5),  # non-zero row_rot
            "crs": GEOS_WKT,
        }
        self._patch(monkeypatch, {"a.nc": record})
        with pytest.raises(ReaderError, match="rotated grid"):
            read_fci_l1c(["a.nc"], "ir_105", calibrate=False)

    def test_zero_x_width_raises(self, monkeypatch):
        """A degenerate zero-width x geotransform is rejected."""
        record = {
            "radiance": np.ones((2, 3)),
            "start_row": 1,
            "end_row": 1,
            "coeffs": _THERMAL,
            "geotransform": (0.1, 0.0, 0.0, 0.0, 0.0, -1e-5),  # zero x width
            "crs": GEOS_WKT,
        }
        self._patch(monkeypatch, {"a.nc": record})
        with pytest.raises(ReaderError, match="zero-width"):
            read_fci_l1c(["a.nc"], "ir_105", calibrate=False)

    def test_mixed_column_count_raises(self, monkeypatch):
        """Chunks with different widths are rejected."""
        mapping = {
            "a.nc": self._chunk(np.ones((2, 3)), 0.0),
            "b.nc": self._chunk(np.ones((2, 4)), -2e-5),
        }
        self._patch(monkeypatch, mapping)
        with pytest.raises(ReaderError, match="column count"):
            read_fci_l1c(["a.nc", "b.nc"], "ir_105")

    def test_mixed_crs_raises(self, monkeypatch):
        """Chunks from a different CRS are rejected."""
        other = self._chunk(np.ones((2, 3)), -2e-5)
        other["crs"] = "OTHER_WKT"
        mapping = {"a.nc": self._chunk(np.ones((2, 3)), 0.0), "b.nc": other}
        self._patch(monkeypatch, mapping)
        with pytest.raises(ReaderError, match="mixed CRS"):
            read_fci_l1c(["a.nc", "b.nc"], "ir_105")

    def test_mixed_x_origin_raises(self, monkeypatch):
        """Chunks that disagree on x origin are rejected (would mis-stitch in x)."""
        other = self._chunk(np.ones((2, 3)), -2e-5)
        other["geotransform"] = (0.5, -1e-5, 0.0, -2e-5, 0.0, -1e-5)  # x origin 0.5
        mapping = {"a.nc": self._chunk(np.ones((2, 3)), 0.0), "b.nc": other}
        self._patch(monkeypatch, mapping)
        with pytest.raises(ReaderError, match="mixed x origin"):
            read_fci_l1c(["a.nc", "b.nc"], "ir_105")

    def test_mixed_cell_size_raises(self, monkeypatch):
        """Chunks with a different cell size are rejected."""
        other = self._chunk(np.ones((2, 3)), -2e-5)
        other["geotransform"] = (0.1, -2e-5, 0.0, -2e-5, 0.0, -1e-5)
        mapping = {"a.nc": self._chunk(np.ones((2, 3)), 0.0), "b.nc": other}
        self._patch(monkeypatch, mapping)
        with pytest.raises(ReaderError, match="cell size"):
            read_fci_l1c(["a.nc", "b.nc"], "ir_105")

    def test_mixed_coeffs_raises(self, monkeypatch):
        """Chunks with different calibration coefficients are rejected."""
        other = self._chunk(
            np.ones((2, 3)),
            -2e-5,
            coeffs={"kind": "thermal", "central_wavenumber_cm1": 800.0},
        )
        mapping = {"a.nc": self._chunk(np.ones((2, 3)), 0.0), "b.nc": other}
        self._patch(monkeypatch, mapping)
        with pytest.raises(ReaderError, match="calibration coefficients"):
            read_fci_l1c(["a.nc", "b.nc"], "ir_105")

    def test_south_up_grid_raises(self, monkeypatch):
        """A south-up grid (gt[5] >= 0) is rejected with a clear message."""
        chunk = self._chunk(np.ones((2, 3)), 0.0)
        chunk["geotransform"] = (0.1, -1e-5, 0.0, 0.0, 0.0, 1e-5)  # gt[5] > 0
        self._patch(monkeypatch, {"a.nc": chunk})
        with pytest.raises(ReaderError, match="north-up"):
            read_fci_l1c(["a.nc"], "ir_105")

    def test_channels_returns_dict(self, monkeypatch):
        """`channels=[...]` returns a dict with one Dataset per requested channel."""
        chunk = self._chunk(np.full((2, 3), 80.0), 0.0)
        self._patch(monkeypatch, {"a.nc": chunk})
        out = read_fci_l1c(["a.nc"], channels=["ir_105", "vis_06"])
        assert isinstance(out, dict), f"expected a dict, got {type(out)}"
        assert set(out) == {"ir_105", "vis_06"}, "one entry per requested channel"
        assert out["ir_105"].read_array().shape == (2, 3), "each entry is a Dataset"

    def test_channels_dict_equals_single(self, monkeypatch):
        """A dict entry equals the single-channel result for that channel."""
        chunk = self._chunk(np.full((2, 3), 80.0), 0.0)
        self._patch(monkeypatch, {"a.nc": chunk})
        single = read_fci_l1c(["a.nc"], "ir_105")
        multi = read_fci_l1c(["a.nc"], channels=["ir_105"])
        assert np.allclose(multi["ir_105"].read_array(), single.read_array()), (
            "channels=[...] must match calling per channel"
        )

    def test_neither_channel_nor_channels_raises(self):
        """Passing neither `channel` nor `channels` is rejected."""
        with pytest.raises(ReaderError, match="exactly one"):
            read_fci_l1c(["a.nc"])

    def test_both_channel_and_channels_raises(self):
        """Passing both `channel` and `channels` is rejected."""
        with pytest.raises(ReaderError, match="exactly one"):
            read_fci_l1c(["a.nc"], "ir_105", channels=["ir_105"])

    def test_channels_carry_their_own_data(self, monkeypatch):
        """Each dict entry carries its own channel's radiance (no cross-channel mix-up)."""
        records = {
            "ir_105": self._chunk(np.full((2, 3), 10.0), 0.0),
            "vis_06": self._chunk(np.full((2, 3), 20.0), 0.0),
        }
        monkeypatch.setattr(
            fci_l1c,
            "read_fci_l1c_chunks",
            lambda p, channels: {ch: records[ch] for ch in channels},
        )
        monkeypatch.setattr(fci_l1c, "_satellite_height", lambda wkt: 1.0e5)
        out = read_fci_l1c(["a.nc"], channels=["ir_105", "vis_06"], calibrate=False)
        assert np.allclose(out["ir_105"].read_array(), 10.0), (
            "ir_105 keeps its own data"
        )
        assert np.allclose(out["vis_06"].read_array(), 20.0), (
            "vis_06 keeps its own data"
        )

    def test_channel_absent_in_one_chunk(self, monkeypatch):
        """A channel absent from one chunk is stitched only from the chunks carrying it."""
        rec_a = self._chunk(np.full((2, 3), 5.0), 0.0)
        rec_b = self._chunk(np.full((2, 3), 5.0), -2e-5)  # contiguous below rec_a
        mapping = {
            "a.nc": {"ir_105": rec_a, "vis_06": rec_a},
            "b.nc": {"ir_105": rec_b, "vis_06": None},  # vis_06 absent from b.nc
        }
        monkeypatch.setattr(
            fci_l1c, "read_fci_l1c_chunks", lambda p, channels: mapping[p]
        )
        monkeypatch.setattr(fci_l1c, "_satellite_height", lambda wkt: 1.0e5)
        out = read_fci_l1c(
            ["a.nc", "b.nc"], channels=["ir_105", "vis_06"], calibrate=False
        )
        assert out["ir_105"].read_array().shape[0] == 4, (
            "ir_105 stitched from both chunks"
        )
        assert out["vis_06"].read_array().shape[0] == 2, (
            "vis_06 only from the chunk with it"
        )


class TestResolveChannels:
    """`resolve_channels` normalises the channel / channels arguments."""

    def test_single_channel(self):
        """A lone `channel` returns a one-item list flagged single."""
        assert resolve_channels("ir_105", None, "read_fci") == (["ir_105"], True)

    def test_channels_sequence(self):
        """A `channels` sequence returns the list flagged not-single."""
        assert resolve_channels(None, ["ir_105", "vis_06"], "read_fci") == (
            ["ir_105", "vis_06"],
            False,
        )

    def test_neither_raises(self):
        """Neither argument is an error."""
        with pytest.raises(ReaderError, match="exactly one"):
            resolve_channels(None, None, "read_fci")

    def test_both_raises(self):
        """Both arguments together are an error."""
        with pytest.raises(ReaderError, match="exactly one"):
            resolve_channels("ir_105", ["ir_105"], "read_fci")

    def test_empty_channels_raises(self):
        """An empty `channels` sequence is an error."""
        with pytest.raises(ReaderError, match="empty"):
            resolve_channels(None, [], "read_fci")

    def test_bare_string_channels_raises(self):
        """A bare string for `channels` is rejected, not split char-by-char."""
        with pytest.raises(ReaderError, match="sequence of channel names"):
            resolve_channels(None, "ir_105", "read_fci")

    def test_duplicate_channels_raises(self):
        """Duplicate channel names are rejected rather than silently de-duplicated."""
        with pytest.raises(ReaderError, match="duplicate channels"):
            resolve_channels(None, ["ir_105", "vis_06", "ir_105"], "read_fci")


class TestReadFciL1cChunks:
    """`read_fci_l1c_chunks` decodes several channels from one chunk open."""

    def test_opens_root_once_for_the_set(self, monkeypatch):
        """The chunk's multidim root is opened once for the whole channel set."""
        opens = []

        class _Root:
            def OpenGroupFromFullname(self, path):
                if "ir_105" in path:
                    return _FakeGroup({"effective_radiance": 1})
                raise RuntimeError("no group")  # vis_06 absent

        def _fake_open_root(path):
            opens.append(path)
            return object(), _Root()

        monkeypatch.setattr(fci_l1c, "_open_root", _fake_open_root)
        monkeypatch.setattr(fci_l1c, "_scalar", lambda g, n: 140.0)
        monkeypatch.setattr(fci_l1c, "_granule_coeffs", lambda g: _THERMAL)
        monkeypatch.setattr(
            fci_l1c, "_unpack_radiance", lambda p, c: (np.ones((2, 2)), (0.1,) * 6, "W")
        )
        records = read_fci_l1c_chunks("f.nc", ["ir_105", "vis_06"])
        assert opens == ["f.nc"], "the root is opened exactly once, not per channel"
        assert records["ir_105"]["coeffs"] == _THERMAL, "present channel decoded"
        assert records["vis_06"] is None, "an absent channel maps to None"

    def test_group_without_radiance_is_none(self, monkeypatch):
        """A channel group lacking effective_radiance maps to None."""

        class _Root:
            def OpenGroupFromFullname(self, path):
                return _FakeGroup({}, names=["x"])

        monkeypatch.setattr(fci_l1c, "_open_root", lambda p: (object(), _Root()))
        assert read_fci_l1c_chunks("f.nc", ["ir_105"]) == {"ir_105": None}


class TestAvailableChannels:
    """`available_channels` lists the channels carrying radiance."""

    class _Measured:
        def __init__(self, has_radiance):
            self._names = ["effective_radiance"] if has_radiance else ["x"]

        def GetMDArrayNames(self):
            return self._names

    class _Channel:
        def __init__(self, name):
            self._name = name

        def OpenGroup(self, sub):
            if self._name == "meta":
                raise RuntimeError("no measured group")
            return TestAvailableChannels._Measured(self._name == "ir_105")

    class _Data:
        def GetGroupNames(self):
            return ["vis_06", "ir_105", "meta"]

        def OpenGroup(self, name):
            return TestAvailableChannels._Channel(name)

    class _Root:
        def OpenGroup(self, name):
            if name != "data":
                raise RuntimeError("no data group")
            return TestAvailableChannels._Data()

    def test_lists_channels_with_radiance(self, monkeypatch):
        """Only channels whose measured group carries effective_radiance are listed."""
        monkeypatch.setattr(fci_l1c, "_open_root", lambda p: (object(), self._Root()))
        assert available_channels("f.nc") == ["ir_105"], "sorted, radiance-only"

    def test_accepts_iterable_and_unions(self, monkeypatch):
        """A chunk iterable is accepted and the channel sets are unioned."""
        monkeypatch.setattr(fci_l1c, "_open_root", lambda p: (object(), self._Root()))
        assert available_channels(["a.nc", "b.nc"]) == ["ir_105"], "union across chunks"

    def test_chunk_without_data_group_is_skipped(self, monkeypatch):
        """A chunk whose root has no `data` group contributes no channels."""

        class _EmptyRoot:
            def OpenGroup(self, name):
                raise RuntimeError("no data group")

        monkeypatch.setattr(fci_l1c, "_open_root", lambda p: (object(), _EmptyRoot()))
        assert available_channels("trailer.nc") == [], "no data group -> no channels"

    def test_none_channel_group_is_skipped(self, monkeypatch):
        """With GDAL exceptions off a None channel group is skipped, not an error."""

        class _NoneData:
            def GetGroupNames(self):
                return ["ir_105"]

            def OpenGroup(self, name):
                return None  # exceptions-disabled: a missing subgroup returns None

        class _Root:
            def OpenGroup(self, name):
                return _NoneData()

        monkeypatch.setattr(fci_l1c, "_open_root", lambda p: (object(), _Root()))
        assert available_channels("f.nc") == [], "a None group must not raise"


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
    assert finite.min() > 180.0, f"BT min too low: {finite.min()}"
    assert finite.max() < 340.0, f"BT max too high: {finite.max()}"
    assert "Geostationary" in str(scene.crs), (
        "result should carry the geostationary CRS"
    )
    assert scene.geotransform[1] == pytest.approx(2000.0, abs=1.0), (
        "ir_105 is 2 km, with a positive x pixel width (east-increasing, not "
        "mirrored — issue #56)"
    )
    assert scene.geotransform[5] < 0, "the stitched grid must be north-up (gt[5] < 0)"
    assert np.isnan(scene.no_data_value[0]), "nodata should be NaN"


@pytest.mark.live
def test_read_fci_l1c_real_granule_multichannel():
    """Multi-channel decode of real FCI L1C chunks matches the per-channel result.

    Skips unless real FDHSI chunks are provided via `FCI_FIXTURES_DIR`.
    """
    fixtures = Path(os.environ.get("FCI_FIXTURES_DIR", "tests/data/fci_l1c"))
    paths = sorted(fixtures.glob("*.nc"))
    if len(paths) < 2:
        pytest.skip("real FCI L1C fixtures not available (set FCI_FIXTURES_DIR)")

    channels = ["vis_06", "ir_105"]
    available = available_channels(paths)
    assert set(channels) <= set(available), f"requested channels not in {available}"

    bands = read_fci_l1c(paths, channels=channels)
    assert set(bands) == set(channels), "one entry per requested channel"
    for name in channels:
        single = np.asarray(read_fci_l1c(paths, name).read_array(), dtype=float)
        got = np.asarray(bands[name].read_array(), dtype=float)
        assert np.allclose(got, single, equal_nan=True), (
            f"{name}: channels=[...] must match the per-channel decode"
        )
