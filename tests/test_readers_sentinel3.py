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
from pyramids_eo.sensors.readers._sentinel3 import (  # noqa: I001
    PLACEHOLDER_GEOTRANSFORM,
    SEN3_SUFFIX,
    geolocation_domain,
    granule_file,
    read_packed_variable,
    resolve_granule,
    subdataset,
    wrap_swath,
)
from tests._sentinel3_granule import write_geolocation


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


def _warp_geotransform(domain, shape=(2, 3)):
    """Warp a dummy raster through `domain` and return the output geotransform."""
    from osgeo import gdal

    height, width = shape
    source = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Float32)
    source.GetRasterBand(1).WriteArray(
        np.arange(width * height, dtype="float32").reshape(height, width)
    )
    source.SetMetadata(dict(domain), "GEOLOCATION")
    warped = gdal.Warp("", source, format="MEM", geoloc=True)
    return warped.GetGeoTransform()


class TestGeolocationDomain:
    """Tests for `geolocation_domain`."""

    def test_emits_the_required_gdal_key_set(self, tmp_path):
        """Every key GDAL requires to build a geolocation transformer is present.

        Test scenario:
            GDAL hard-requires X_BAND, Y_BAND, PIXEL_OFFSET, LINE_OFFSET,
            PIXEL_STEP and LINE_STEP with no defaults on the metadata-domain path
            and raises "Missing some geolocation fields" without any of them.
            `SRS` is optional but must be WKT, and
            GEOREFERENCING_CONVENTION defaults to TOP_LEFT_CORNER, which is why
            PIXEL_CENTER has to be stated for per-pixel lon/lat.
        """
        geo = write_geolocation(tmp_path / "geo_coordinates.nc")
        domain = geolocation_domain(geo, reader="read_test")
        required = {
            "X_BAND",
            "Y_BAND",
            "PIXEL_OFFSET",
            "LINE_OFFSET",
            "PIXEL_STEP",
            "LINE_STEP",
        }
        assert required <= set(domain), f"missing required keys: {sorted(domain)}"
        assert domain["GEOREFERENCING_CONVENTION"] == "PIXEL_CENTER", domain
        assert (domain["PIXEL_STEP"], domain["LINE_STEP"]) == ("1", "1"), domain

    def test_srs_is_parseable_wkt_for_a_geographic_crs(self, tmp_path):
        """`SRS` must be WKT, because GDAL reads the key with importFromWkt.

        Test scenario:
            An authority string such as "EPSG:4326" fails importFromWkt
            *silently*, leaving GDAL's
            bGeographicSRSWithMinus180Plus180LongRange flag false and skipping
            the antimeridian handling a polar-orbiting swath needs. Pinning the
            parse, rather than the string, is what catches that.
        """
        from osgeo import osr

        geo = write_geolocation(tmp_path / "geo_coordinates.nc")
        domain = geolocation_domain(geo, reader="read_test")
        reference = osr.SpatialReference()
        assert reference.ImportFromWkt(domain["SRS"]) == 0, (
            f"SRS is not parseable WKT: {domain['SRS']!r}"
        )
        assert reference.IsGeographic(), "the geolocation SRS must be geographic"

    def test_packed_geolocation_warps_to_degrees(self, tmp_path):
        """A real warp through the domain lands in degrees, not packed units.

        Test scenario:
            Sentinel-3 stores longitude/latitude CF-packed as int32
            micro-degrees (scale_factor 1e-06), and GDAL's geolocation
            transformer reads geolocation arrays raw through RasterIO without
            applying scale/offset. Handing it the bare subdataset therefore warps
            to micro-degrees -- silently. This is the end-to-end assertion that
            the unscaling VRT puts real degrees in front of GDAL.
        """
        geo = write_geolocation(tmp_path / "geo_coordinates.nc", packed=True)
        domain = geolocation_domain(geo, reader="read_test")
        origin_x, _, _, origin_y, _, _ = _warp_geotransform(domain)
        assert 3.0 < origin_x < 6.0, f"longitude origin not in degrees: {origin_x}"
        assert 50.0 < origin_y < 53.0, f"latitude origin not in degrees: {origin_y}"

    def test_unpacked_geolocation_is_referenced_directly(self, tmp_path):
        """An already-unpacked geolocation variable needs no VRT indirection."""
        geo = write_geolocation(tmp_path / "geo_coordinates.nc", packed=False)
        domain = geolocation_domain(geo, reader="read_test")
        assert domain["X_DATASET"] == f'NETCDF:"{geo}":longitude', domain
        origin_x = _warp_geotransform(domain)[0]
        assert 3.0 < origin_x < 6.0, f"longitude origin not in degrees: {origin_x}"

    def test_custom_variables_and_stride(self, tmp_path):
        """SLSTR-style suffixed variables and a tie-point stride are carried through."""
        geo = write_geolocation(
            tmp_path / "geodetic_in.nc",
            longitude_variable="longitude_in",
            latitude_variable="latitude_in",
        )
        domain = geolocation_domain(
            geo,
            reader="read_test",
            longitude_variable="longitude_in",
            latitude_variable="latitude_in",
            pixel_step=16,
            line_step=4,
        )
        assert "longitude_in" in domain["X_DATASET"], domain
        assert (domain["PIXEL_STEP"], domain["LINE_STEP"]) == ("16", "4"), domain

    def test_a_missing_geolocation_variable_raises(self, tmp_path):
        """A geolocation file lacking the variable fails in the reader, not in a warp.

        Test scenario:
            Only the annotation *file*'s existence used to be checked, so a
            truncated or renamed-variable granule produced a well-formed,
            completely unusable domain whose failure surfaced much later inside
            the caller's warp.
        """
        from tests._netcdf3 import write_netcdf

        geo = write_netcdf(
            tmp_path / "geo_coordinates.nc",
            {"longitude": (np.zeros((2, 3), dtype="int32"), {})},
        )
        with pytest.raises(ReaderError, match="latitude"):
            geolocation_domain(geo, reader="read_test")


#: A minimal domain for the `wrap_swath` tests, which only store and return it.
_DOMAIN = {"X_DATASET": "x.vrt", "X_BAND": "1", "Y_DATASET": "y.vrt", "Y_BAND": "1"}


class TestWrapSwath:
    """Tests for `wrap_swath`, which builds the swath `Dataset`."""

    def test_declares_nan_nodata_and_the_placeholder_grid(self):
        """A swath Dataset declares NaN nodata, the placeholder grid, and no EPSG.

        Test scenario:
            A swath has no affine mapping to the ground, so the geotransform must be
            the documented placeholder rather than a plausible-looking geographic one
            that downstream code might trust.
        """
        dataset = wrap_swath(np.array([[1.0, 2.0]]), dict(_DOMAIN))
        assert dataset.geotransform == PLACEHOLDER_GEOTRANSFORM, (
            f"expected the placeholder grid, got {dataset.geotransform}"
        )
        assert dataset.epsg is None, f"a swath has no EPSG, got {dataset.epsg}"
        assert np.isnan(dataset.no_data_value[0]), (
            f"nodata should be NaN, got {dataset.no_data_value}"
        )

    def test_attaches_the_domain(self):
        """The GEOLOCATION domain is set on the Dataset's GDAL handle."""
        dataset = wrap_swath(np.array([[1.0]]), dict(_DOMAIN))
        assert dataset.raster.GetMetadata("GEOLOCATION")["X_DATASET"] == "x.vrt"

    def test_does_not_mutate_the_callers_domain(self):
        """The caller's domain dict is copied, not adopted and later mutated."""
        domain = dict(_DOMAIN)
        before = dict(domain)
        dataset = wrap_swath(np.array([[1.0]]), domain)
        dataset.raster.SetMetadata({"sensor": "olci"})
        assert domain == before, f"caller's domain was mutated: {domain}"

    def test_an_empty_domain_leaves_no_geolocation(self):
        """An empty domain (geolocation=False) attaches no GEOLOCATION metadata."""
        dataset = wrap_swath(np.array([[1.0]]), {})
        assert not dataset.raster.GetMetadata("GEOLOCATION"), "domain should be empty"


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
        with pytest.raises(ReaderError, match="not a finite number"):
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

    def test_sentinel3_valid_min_valid_max_are_masked(self, monkeypatch):
        """Counts outside the separate `valid_min`/`valid_max` scalars are masked.

        Test scenario:
            This is the spelling the Sentinel-3 L1 products actually use --
            `valid_range` appears in neither the OLCI nor the SLSTR
            specification. Reading only `valid_range` made this masking dead
            code, so out-of-range radiances and brightness temperatures passed
            through as valid.
        """
        values, _ = _read(
            monkeypatch,
            np.array([[1, 50, 500]], dtype="int32"),
            {"valid_min": "10", "valid_max": "100"},
        )
        assert np.isnan(values[0, 0]) and np.isnan(values[0, 2]), values
        assert values[0, 1] == 50.0, values

    def test_a_single_sentinel3_bound_is_honoured(self, monkeypatch):
        """`valid_min` without `valid_max` masks only below, and vice versa."""
        low_only, _ = _read(
            monkeypatch, np.array([[1, 500]], dtype="int32"), {"valid_min": "10"}
        )
        assert np.isnan(low_only[0, 0]) and low_only[0, 1] == 500.0, low_only
        high_only, _ = _read(
            monkeypatch, np.array([[1, 500]], dtype="int32"), {"valid_max": "100"}
        )
        assert high_only[0, 0] == 1.0 and np.isnan(high_only[0, 1]), high_only

    def test_packed_geolocation_bounds_mask_the_fill(self, monkeypatch):
        """A real OLCI latitude variable unpacks to degrees with its fill masked.

        Test scenario:
            Exercises the published packing together: `int32` micro-degrees with
            `scale_factor` 1e-06, `valid_min`/`valid_max` of ±90000000 in *packed*
            units, and `_FillValue` -2147483648.
        """
        values, _ = _read(
            monkeypatch,
            np.array([[-2147483648, 45000000]], dtype="int32"),
            {
                "scale_factor": "1e-06",
                "valid_min": "-90000000",
                "valid_max": "90000000",
                "_FillValue": "-2147483648",
            },
        )
        assert np.isnan(values[0, 0]), f"the fill value must mask: {values}"
        assert values[0, 1] == pytest.approx(45.0), values

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
    def test_non_finite_packing_raises(self, monkeypatch, bad):
        """A non-finite scale_factor raises instead of voiding or de-masking the array.

        Test scenario:
            `float("nan")` and `float("inf")` parse happily, so the old numeric
            guard let them through: a NaN scale silently made every pixel
            invalid, and an infinite one produced infinities that evade the
            Dataset's NaN nodata and are reported as valid data.
        """
        with pytest.raises(ReaderError, match="not a finite number"):
            _read(
                monkeypatch,
                np.array([[1, 2]], dtype="int32"),
                {"scale_factor": bad},
            )

    def test_reversed_valid_range_masks_nothing(self, monkeypatch):
        """A reversed `valid_range` masks nothing rather than every pixel."""
        values, _ = _read(
            monkeypatch, np.array([[1, 2]], dtype="int32"), {"valid_range": "{4, 1}"}
        )
        assert np.all(np.isfinite(values)), f"a reversed range must not mask: {values}"

    def test_reversed_valid_min_max_masks_nothing(self, monkeypatch):
        """A reversed `valid_min`/`valid_max` pair likewise masks nothing."""
        values, _ = _read(
            monkeypatch,
            np.array([[1, 2]], dtype="int32"),
            {"valid_min": "100", "valid_max": "10"},
        )
        assert np.all(np.isfinite(values)), f"a reversed pair must not mask: {values}"

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
            monkeypatch,
            np.array([[1]], dtype="uint16"),
            {"units": "K", "long_name": "bt"},
        )
        assert meta["units"] == "K" and meta["long_name"] == "bt", meta
