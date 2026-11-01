"""Unit tests for `pyramids_eo.sensors.readers._sentinel3`, the shared `.SEN3` helpers.

The granule-level behaviour is covered against real NetCDF fixtures in
`test_readers_olci` / `test_readers_slstr`. These tests target the branches those cannot
reach, by injecting a fake GDAL raster: the open-failure paths, and the CF unpacking /
masking cases GDAL's netCDF *writer* cannot produce (a `valid_range` attribute, a
non-numeric `scale_factor`, a non-numeric `_FillValue`).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._sentinel3 import (
    SEN3_SUFFIX,
    geolocation_domain,
    granule_file,
    read_packed_variable,
    resolve_granule,
    subdataset,
)


class _FakeBand:
    """A stand-in for a GDAL band exposing one array and one metadata dict."""

    def __init__(self, array, meta):
        self._array = array
        self._meta = meta

    def ReadAsArray(self):  # noqa: N802 - GDAL's API spelling
        """Return the band's array."""
        return self._array

    def GetMetadata(self):  # noqa: N802 - GDAL's API spelling
        """Return the band's attribute dict."""
        return dict(self._meta)


class _FakeRaster:
    """A stand-in for a GDAL dataset holding a single `_FakeBand`."""

    def __init__(self, band):
        self._band = band

    def GetRasterBand(self, index):  # noqa: N802 - GDAL's API spelling
        """Return the single fake band regardless of `index`."""
        return self._band


def _patch_open(monkeypatch, result):
    """Point `gdal.Open` at `result` — a raster, or an exception to raise."""
    from osgeo import gdal

    def _open(_source):
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(gdal, "Open", _open)


def _read(monkeypatch, array, meta):
    """Read `array` + `meta` through `read_packed_variable` via a fake GDAL raster."""
    _patch_open(monkeypatch, _FakeRaster(_FakeBand(array, meta)))
    return read_packed_variable("g.nc", "v", reader="read_test")


class TestSubdataset:
    """Tests for `subdataset`."""

    def test_builds_the_gdal_connection_string(self):
        """The variable is addressed with GDAL's NETCDF:"file":var form."""
        assert subdataset("a/b.nc", "longitude") == 'NETCDF:"a/b.nc":longitude'


class TestResolveGranule:
    """Tests for `resolve_granule`."""

    def test_a_directory_resolves_to_itself(self, tmp_path):
        """A granule directory is returned unchanged."""
        granule = tmp_path / f"g{SEN3_SUFFIX}"
        granule.mkdir()
        assert resolve_granule(granule, "read_test") == granule

    def test_a_file_resolves_to_its_parent(self, tmp_path):
        """A file inside a granule resolves to the granule directory."""
        granule = tmp_path / f"g{SEN3_SUFFIX}"
        granule.mkdir()
        band = granule / "Oa08_radiance.nc"
        band.write_bytes(b"")
        assert resolve_granule(band, "read_test") == granule

    def test_a_missing_source_raises(self, tmp_path):
        """A path that is neither a directory nor a file is rejected."""
        with pytest.raises(ReaderError, match="not a Sentinel-3 granule"):
            resolve_granule(tmp_path / "missing", "read_test")


class TestGranuleFile:
    """Tests for `granule_file`."""

    def test_returns_an_existing_file(self, tmp_path):
        """An existing file in the granule is returned."""
        (tmp_path / "geo_coordinates.nc").write_bytes(b"")
        assert granule_file(tmp_path, "geo_coordinates.nc", "read_test").name == (
            "geo_coordinates.nc"
        )

    def test_missing_file_raises_naming_it(self, tmp_path):
        """A missing file is reported by name so the cause is obvious."""
        with pytest.raises(ReaderError, match="geo_coordinates.nc"):
            granule_file(tmp_path, "geo_coordinates.nc", "read_test")


class TestGeolocationDomain:
    """Tests for `geolocation_domain`."""

    def test_defaults_to_a_full_resolution_per_pixel_grid(self):
        """The default domain references longitude/latitude at a stride of one."""
        domain = geolocation_domain("g.nc")
        assert domain["X_DATASET"] == 'NETCDF:"g.nc":longitude'
        assert domain["Y_DATASET"] == 'NETCDF:"g.nc":latitude'
        assert domain["PIXEL_STEP"] == "1"
        assert domain["LINE_STEP"] == "1"
        assert domain["SRS"] == "EPSG:4326"

    def test_custom_variables_and_stride(self):
        """SLSTR-style suffixed variables and a tie-point stride are carried through."""
        domain = geolocation_domain(
            "geodetic_in.nc",
            longitude_variable="longitude_in",
            latitude_variable="latitude_in",
            pixel_step=16,
            line_step=4,
        )
        assert domain["X_DATASET"].endswith(":longitude_in")
        assert (domain["PIXEL_STEP"], domain["LINE_STEP"]) == ("16", "4")


class TestReadPackedVariableErrors:
    """Tests for `read_packed_variable`'s failure paths."""

    def test_gdal_runtime_error_becomes_a_reader_error(self, monkeypatch):
        """A GDAL RuntimeError (exceptions enabled) is wrapped as ReaderError."""
        _patch_open(monkeypatch, RuntimeError("no such subdataset"))
        with pytest.raises(ReaderError, match="cannot open"):
            read_packed_variable("g.nc", "v", reader="read_test")

    def test_gdal_none_becomes_a_reader_error(self, monkeypatch):
        """A None from GDAL (exceptions disabled) is wrapped as ReaderError."""
        _patch_open(monkeypatch, None)
        with pytest.raises(ReaderError, match="cannot open"):
            read_packed_variable("g.nc", "v", reader="read_test")

    def test_non_numeric_scale_factor_raises(self, monkeypatch):
        """A non-numeric scale_factor is a clear ReaderError, not a ValueError."""
        with pytest.raises(ReaderError, match="non-numeric scale_factor"):
            _read(
                monkeypatch,
                np.array([[1, 2]], dtype="uint16"),
                {"scale_factor": "abc", "add_offset": "0"},
            )


class TestReadPackedVariableUnpacking:
    """Tests for `read_packed_variable`'s unpacking and masking."""

    def test_applies_scale_and_offset(self, monkeypatch):
        """Values are unpacked as raw * scale_factor + add_offset."""
        values, _ = _read(
            monkeypatch,
            np.array([[10, 20]], dtype="uint16"),
            {"scale_factor": "0.5", "add_offset": "-1.0"},
        )
        assert np.allclose(values, [[4.0, 9.0]]), values

    def test_scale_only_defaults_the_offset_to_zero(self, monkeypatch):
        """A scale_factor with no add_offset multiplies without shifting."""
        values, _ = _read(
            monkeypatch, np.array([[10]], dtype="uint16"), {"scale_factor": "2"}
        )
        assert np.allclose(values, [[20.0]]), values

    def test_offset_only_defaults_the_scale_to_one(self, monkeypatch):
        """An add_offset with no scale_factor shifts without scaling."""
        values, _ = _read(
            monkeypatch, np.array([[10]], dtype="uint16"), {"add_offset": "5"}
        )
        assert np.allclose(values, [[15.0]]), values

    def test_no_packing_reads_values_as_is(self, monkeypatch):
        """A variable with no packing attributes (e.g. geolocation) is read unchanged."""
        values, _ = _read(
            monkeypatch, np.array([[1.5, 2.5]], dtype="float32"), {"units": "degrees"}
        )
        assert np.allclose(values, [[1.5, 2.5]]), values

    def test_fill_value_is_masked(self, monkeypatch):
        """The _FillValue pixel becomes NaN."""
        values, _ = _read(
            monkeypatch,
            np.array([[10, 65535]], dtype="uint16"),
            {"_FillValue": "65535"},
        )
        assert np.isfinite(values[0, 0]) and np.isnan(values[0, 1]), values

    def test_non_numeric_fill_value_is_ignored(self, monkeypatch):
        """A non-numeric _FillValue cannot match, so it masks nothing rather than failing."""
        values, _ = _read(
            monkeypatch, np.array([[10, 20]], dtype="uint16"), {"_FillValue": "nodata"}
        )
        assert np.all(np.isfinite(values)), values

    def test_values_below_valid_range_are_masked(self, monkeypatch):
        """Counts under valid_min are masked to NaN."""
        values, _ = _read(
            monkeypatch,
            np.array([[1, 50]], dtype="uint16"),
            {"valid_range": "{10, 100}"},
        )
        assert np.isnan(values[0, 0]) and np.isfinite(values[0, 1]), values

    def test_values_above_valid_range_are_masked(self, monkeypatch):
        """Counts over valid_max are masked to NaN."""
        values, _ = _read(
            monkeypatch,
            np.array([[50, 500]], dtype="uint16"),
            {"valid_range": "{10, 100}"},
        )
        assert np.isfinite(values[0, 0]) and np.isnan(values[0, 1]), values

    def test_malformed_valid_range_masks_nothing(self, monkeypatch):
        """An unparseable valid_range is ignored rather than masking everything."""
        values, _ = _read(
            monkeypatch,
            np.array([[1, 500]], dtype="uint16"),
            {"valid_range": "junk"},
        )
        assert np.all(np.isfinite(values)), values

    def test_returns_the_variable_metadata(self, monkeypatch):
        """The variable's attribute dict comes back alongside the values."""
        _, meta = _read(
            monkeypatch, np.array([[1]], dtype="uint16"), {"units": "K", "long_name": "bt"}
        )
        assert meta["units"] == "K" and meta["long_name"] == "bt", meta
