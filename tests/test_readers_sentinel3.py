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
from pyramids_eo.sensors.readers._common import validity_bounds
from pyramids_eo.sensors.readers._sentinel3 import (  # noqa: I001
    PLACEHOLDER_GEOTRANSFORM,
    SEN3_SUFFIX,
    geolocation_domain,
    geolocation_source,
    granule_file,
    read_packed_variable,
    resolve_granule,
    subdataset,
    wrap_swath,
)
from tests._netcdf3 import write_netcdf
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
        """`SRS` must be a parseable, geographic coordinate system description.

        Test scenario:
            Measured on GDAL 3.13, the key is imported permissively -- this WKT and
            a bare "EPSG:4326" give an identical geotransform, while an
            unparseable value is rejected loudly ("Failed to import coordinate
            system"). So the requirement is simply that it parses and is
            geographic; what the key actually buys is a CRS on the warped output,
            which `TestGeolocationSrsGivesTheOutputACrs` asserts. The
            antimeridian rationale this test once cited was never reproduced and
            is not why the key is emitted.
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
        assert np.isfinite(values[0, 0]), f"the valid pixel must survive: {values}"
        assert np.isnan(values[0, 1]), f"the fill pixel must be masked: {values}"

    def test_non_numeric_fill_value_is_ignored(self, monkeypatch):
        """A non-numeric _FillValue masks nothing rather than failing the read.

        It warns while doing so -- see `TestNonNumericFillValueWarns` -- because
        the lost masking is a real gap in the result's nodata.
        """
        with pytest.warns(UserWarning, match="non-numeric _FillValue"):
            values, _ = _read(
                monkeypatch,
                np.array([[10, 20]], dtype="uint16"),
                {"_FillValue": "nodata"},
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
        assert np.isnan(values[0, 0]), f"the below-range pixel must mask: {values}"
        assert np.isnan(values[0, 2]), f"the above-range pixel must mask: {values}"
        assert values[0, 1] == 50.0, values

    def test_a_single_sentinel3_bound_is_honoured(self, monkeypatch):
        """`valid_min` without `valid_max` masks only below, and vice versa."""
        low_only, _ = _read(
            monkeypatch, np.array([[1, 500]], dtype="int32"), {"valid_min": "10"}
        )
        assert np.isnan(low_only[0, 0]), f"below valid_min must mask: {low_only}"
        assert low_only[0, 1] == 500.0, f"no upper bound was given: {low_only}"
        high_only, _ = _read(
            monkeypatch, np.array([[1, 500]], dtype="int32"), {"valid_max": "100"}
        )
        assert high_only[0, 0] == 1.0, f"no lower bound was given: {high_only}"
        assert np.isnan(high_only[0, 1]), f"above valid_max must mask: {high_only}"

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
        assert np.isnan(values[0, 0]), f"the below-range pixel must mask: {values}"
        assert np.isfinite(values[0, 1]), f"the in-range pixel must survive: {values}"

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
        assert meta["units"] == "K", f"units should round trip: {meta}"
        assert meta["long_name"] == "bt", f"long_name should round trip: {meta}"

    def test_bounds_are_compared_before_unpacking(self, monkeypatch):
        """The validity limits are applied to the raw counts, not the unpacked values.

        Test scenario:
            With `scale_factor` 0.5 and `add_offset` -1.0, packed bounds of
            [10, 100] and raw counts of 20 and 150, the two readings disagree:
            comparing *unpacked* values would mask 20 (it unpacks to 9.0, under
            10) and keep 150 (it unpacks to 74.0, inside the pair). CF requires
            the packed comparison, so the opposite must happen. A fixture whose
            bounds straddle both readings is the only way to tell them apart --
            which is why a scale of 1.0 would prove nothing here.
        """
        values, _ = _read(
            monkeypatch,
            np.array([[20, 150]], dtype="int32"),
            {
                "scale_factor": "0.5",
                "add_offset": "-1.0",
                "valid_min": "10",
                "valid_max": "100",
            },
        )
        assert values[0, 0] == pytest.approx(9.0), (
            f"raw 20 is inside the packed bounds and must survive: {values}"
        )
        assert np.isnan(values[0, 1]), (
            f"raw 150 is above the packed valid_max and must mask: {values}"
        )

    def test_a_fill_value_outside_the_valid_range_still_masks(self, monkeypatch):
        """A `_FillValue` that the validity limits already exclude masks exactly once.

        Test scenario:
            Sentinel-3 fill sentinels sit well outside the variable's valid
            range, so both rules fire on the same pixel. The mask is a boolean
            or, so the pixel must be NaN and its neighbours untouched -- not, for
            instance, the whole row.
        """
        values, _ = _read(
            monkeypatch,
            np.array([[-32768, 20, 30]], dtype="int32"),
            {
                "scale_factor": "0.5",
                "add_offset": "-1.0",
                "_FillValue": "-32768",
                "valid_min": "0",
                "valid_max": "1000",
            },
        )
        assert np.isnan(values[0, 0]), f"the fill pixel must mask: {values}"
        assert np.all(np.isfinite(values[0, 1:])), (
            f"only the fill pixel may mask: {values}"
        )


class TestValidityBoundsEdges:
    """Tests for how `validity_bounds` resolves a half-usable attribute pair."""

    @pytest.mark.parametrize(
        "meta, expected",
        [
            # One bound junk: the other must stay in force rather than both being
            # discarded, because the bounds are independent CF scalars.
            ({"valid_min": "10", "valid_max": "junk"}, (10.0, None)),
            ({"valid_min": "junk", "valid_max": "100"}, (None, 100.0)),
            # A non-finite bound is as unusable as a non-numeric one.
            ({"valid_min": "10", "valid_max": "nan"}, (10.0, None)),
            ({"valid_min": "-inf", "valid_max": "100"}, (None, 100.0)),
            # Both unusable: fall through to the FCI spelling if it is there.
            ({"valid_min": "junk", "valid_max": "junk"}, (None, None)),
            (
                {"valid_min": "junk", "valid_max": "junk", "valid_range": "{0, 10}"},
                (0.0, 10.0),
            ),
        ],
    )
    def test_one_unusable_bound_leaves_the_other_in_force(self, meta, expected):
        """A junk or non-finite bound is dropped alone, not together with its partner.

        Args:
            meta: The variable's attribute dict.
            expected: The `(valid_min, valid_max)` pair that must be resolved.

        Test scenario:
            Dropping both on one bad value would silently stop masking the side
            that *was* readable; dropping neither would compare against a NaN or
            an infinity, which masks everything or nothing at random.
        """
        assert validity_bounds(meta) == expected, (
            f"{meta} should resolve to {expected}, got {validity_bounds(meta)}"
        )

    def test_the_sentinel3_scalars_win_over_a_combined_valid_range(self):
        """`valid_min`/`valid_max` take precedence over an also-present `valid_range`.

        Test scenario:
            `valid_range` is MTG-FCI's spelling and only a fallback here. A
            granule carrying both must be masked by the Sentinel-3 pair, so the
            two sets are deliberately disjoint in this fixture.
        """
        resolved = validity_bounds(
            {"valid_min": "10", "valid_max": "100", "valid_range": "{0, 4095}"}
        )
        assert resolved == (10.0, 100.0), (
            f"the separate scalars must win over valid_range, got {resolved}"
        )


def _warp(domain, shape):
    """Warp a dummy raster of `shape` through `domain`; return its geotransform + size."""
    from osgeo import gdal

    height, width = shape
    source = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Float32)
    source.GetRasterBand(1).WriteArray(
        np.arange(width * height, dtype="float32").reshape(height, width)
    )
    source.SetMetadata(dict(domain), "GEOLOCATION")
    warped = gdal.Warp("", source, format="MEM", geoloc=True)
    return warped.GetGeoTransform(), (warped.RasterXSize, warped.RasterYSize)


def _geolocation_file(path, longitude, latitude, attributes):
    """Write a geolocation file carrying `longitude` and `latitude` with `attributes`."""
    return write_netcdf(
        path,
        {
            "longitude": (longitude, dict(attributes)),
            "latitude": (latitude, dict(attributes)),
        },
    )


class TestUnscalingVrt:
    """Tests for the `/vsimem` VRT that hands GDAL unpacked degrees."""

    def test_an_offset_without_a_scale_is_applied_with_a_unit_ratio(self, tmp_path):
        """A variable carrying only `add_offset` is shifted, not multiplied by zero.

        Test scenario:
            The VRT's `ScaleRatio` and `ScaleOffset` are filled from
            `GetScale()` / `GetOffset()`, either of which a product may omit. An
            absent `scale_factor` must default to 1.0: a 0.0 ratio would flatten
            every coordinate to the offset, putting the whole swath on one point.
        """
        raw = np.array([[0, 1000000, 2000000]], dtype="int32")
        geo = _geolocation_file(
            tmp_path / "geo_coordinates.nc", raw, raw, {"add_offset": 4.0}
        )
        source = geolocation_source(geo, "longitude", reader="read_test")
        assert source.endswith(".vrt"), f"an offset alone needs a VRT: {source}"
        from osgeo import gdal

        values = np.asarray(gdal.Open(source).ReadAsArray(), dtype=float)
        assert np.allclose(values, raw.astype(float) + 4.0), (
            f"an offset-only variable must be shifted by 4.0, got {values}"
        )

    def test_a_scale_and_an_offset_are_both_applied(self, tmp_path):
        """A variable with both packing attributes unpacks as raw * scale + offset.

        Test scenario:
            The geolocation path is tested mostly against the published
            scale-only packing, so a product adding an `add_offset` would
            exercise a combination nothing pins. Both must reach the VRT.
        """
        raw = np.array([[0, 1000000, 2000000]], dtype="int32")
        geo = _geolocation_file(
            tmp_path / "geo_coordinates.nc",
            raw,
            raw,
            {"scale_factor": 1e-06, "add_offset": 4.0},
        )
        source = geolocation_source(geo, "longitude", reader="read_test")
        from osgeo import gdal

        values = np.asarray(gdal.Open(source).ReadAsArray(), dtype=float)
        assert np.allclose(values, [[4.0, 5.0, 6.0]]), (
            f"expected raw * 1e-06 + 4.0, got {values}"
        )

    def test_two_granules_sharing_a_variable_name_get_distinct_vrts(self, tmp_path):
        """One `/vsimem` entry per source, so two granules never share a VRT.

        Test scenario:
            Every OLCI granule names its geolocation variables `longitude` and
            `latitude`, and the VRT lives in a process-wide `/vsimem` namespace.
            Keying it on the variable name alone would make the second granule
            read of a session silently reuse the first granule's coordinates.
        """
        first = tmp_path / "a.SEN3"
        second = tmp_path / "b.SEN3"
        first.mkdir()
        second.mkdir()
        one = geolocation_domain(
            write_geolocation(first / "geo_coordinates.nc"), reader="read_test"
        )
        two = geolocation_domain(
            write_geolocation(second / "geo_coordinates.nc", shape=(3, 4)),
            reader="read_test",
        )
        assert one["X_DATASET"] != two["X_DATASET"], (
            f"two granules share a VRT name: {one['X_DATASET']}"
        )
        from osgeo import gdal

        assert gdal.Open(one["X_DATASET"]).RasterYSize == 2, "granule A is 2 rows"
        assert gdal.Open(two["X_DATASET"]).RasterYSize == 3, "granule B is 3 rows"

    def test_the_two_variables_of_one_file_get_distinct_vrts(self, tmp_path):
        """Longitude and latitude live in one file but need separate VRTs.

        Test scenario:
            A shared name would make X and Y resolve to the same array, i.e. a
            swath placed along the line longitude == latitude.
        """
        geo = write_geolocation(tmp_path / "geo_coordinates.nc")
        domain = geolocation_domain(geo, reader="read_test")
        assert domain["X_DATASET"] != domain["Y_DATASET"], domain
        from osgeo import gdal

        longitudes = np.asarray(gdal.Open(domain["X_DATASET"]).ReadAsArray(), float)
        latitudes = np.asarray(gdal.Open(domain["Y_DATASET"]).ReadAsArray(), float)
        assert 3.0 < longitudes.min() < 6.0, f"X is not longitude: {longitudes}"
        assert 50.0 < latitudes.min() < 53.0, f"Y is not latitude: {latitudes}"

    def test_a_second_read_of_one_granule_reuses_the_vrt_and_still_warps(
        self, tmp_path
    ):
        """Re-reading a granule rewrites the same `/vsimem` entry without corrupting it.

        Test scenario:
            The VRT name is derived from the source so repeated reads share one
            entry, which means the second read overwrites a file the first read
            may still have cached. The warp has to keep working, and land in the
            same place.
        """
        geo = write_geolocation(tmp_path / "geo_coordinates.nc")
        first = geolocation_domain(geo, reader="read_test")
        before = _warp(first, (2, 3))
        second = geolocation_domain(geo, reader="read_test")
        assert second["X_DATASET"] == first["X_DATASET"], (
            "a repeated read should reuse the one /vsimem entry"
        )
        assert _warp(second, (2, 3)) == before, (
            "the second read must warp identically to the first"
        )

    @pytest.mark.parametrize("shape", [(1, 4), (4, 1), (1, 1)])
    def test_a_degenerate_grid_still_yields_readable_degrees(self, tmp_path, shape):
        """A one-row, one-column or one-pixel grid produces a valid, unpacked VRT.

        Args:
            shape: The degenerate `(rows, columns)` geolocation grid.

        Test scenario:
            The VRT states its own `rasterXSize` / `rasterYSize` and `SrcRect`,
            so a grid with a dimension of one is where an off-by-one in those
            would show. GDAL's own warp cannot back-map such a swath ("Invalid
            pixel size for backmap"), which is a GDAL limitation rather than a
            reader one -- the reader's contract is to *describe* the geolocation,
            so what is asserted is that the described arrays read back as the
            degrees the file holds.
        """
        rows, columns = shape
        longitude = np.round(
            np.tile(np.linspace(4.0, 5.0, columns), (rows, 1)) / 1e-06
        ).astype("int32")
        latitude = np.round(
            np.repeat(np.linspace(52.0, 51.0, rows)[:, None], columns, axis=1) / 1e-06
        ).astype("int32")
        geo = _geolocation_file(
            tmp_path / "geo_coordinates.nc",
            longitude,
            latitude,
            {"scale_factor": 1e-06},
        )
        domain = geolocation_domain(geo, reader="read_test")
        from osgeo import gdal

        raster = gdal.Open(domain["X_DATASET"])
        assert (raster.RasterXSize, raster.RasterYSize) == (columns, rows), (
            f"the VRT must keep the grid shape, got "
            f"{(raster.RasterXSize, raster.RasterYSize)}"
        )
        values = np.asarray(raster.ReadAsArray(), dtype=float).reshape(rows, columns)
        assert np.allclose(values, longitude * 1e-06), (
            f"a {rows}x{columns} grid unpacked wrongly:\n{values}"
        )


class TestGeolocationWarpExtent:
    """A warp through the domain must land on the granule's own footprint."""

    def test_a_packed_geolocation_warps_to_a_grid_the_size_of_the_swath(self, tmp_path):
        """The warped grid covers the granule's extent at roughly its own size.

        Test scenario:
            Asserting only the origin leaves a whole class of defect invisible:
            any coordinate the transformer reads out of range -- unscaled
            micro-degrees, or a sentinel value taken as real -- stretches the
            computed output bounds instead of failing, so the warp still
            "succeeds" while producing a grid orders of magnitude larger than the
            swath. Bounding the output size is what detects that.
        """
        geo = write_geolocation(tmp_path / "geo_coordinates.nc", shape=(8, 8))
        domain = geolocation_domain(geo, reader="read_test")
        (origin_x, _, _, origin_y, _, _), (width, height) = _warp(domain, (8, 8))
        assert 3.0 < origin_x < 6.0, f"longitude origin not in degrees: {origin_x}"
        assert 50.0 < origin_y < 53.0, f"latitude origin not in degrees: {origin_y}"
        stretched = (
            f"an 8x8 swath warped to {width}x{height}: a coordinate outside the "
            f"granule's footprint has stretched the output bounds"
        )
        assert width <= 32, stretched
        assert height <= 32, stretched


class TestGeolocationFillValue:
    """The geolocation fill value must not be consumed as a real coordinate."""

    @staticmethod
    def _granule(path, *, filled):
        """Write an 8x8 packed geolocation file, optionally with one filled pixel."""
        from tests._netcdf3 import write_netcdf

        scale, fill = 1e-06, -2147483648
        longitude = np.round(np.tile(np.linspace(4.0, 5.0, 8), (8, 1)) / scale)
        latitude = np.round(
            np.repeat(np.linspace(52.0, 51.0, 8)[:, None], 8, axis=1) / scale
        )
        longitude = longitude.astype("int32")
        latitude = latitude.astype("int32")
        if filled:
            longitude[0, 0] = fill
            latitude[0, 0] = fill
        attributes = {"scale_factor": scale, "_FillValue": fill}
        return write_netcdf(
            path,
            {
                "longitude": (longitude, dict(attributes)),
                "latitude": (latitude, dict(attributes)),
            },
        )

    def test_a_filled_geolocation_pixel_does_not_move_the_extent(self, tmp_path):
        """One filled geolocation pixel must not drag the warped extent off the globe.

        Test scenario:
            GDAL's geolocation transformer consults the band's nodata to decide
            which samples to skip. A VRT that carried the scaling but dropped the
            fill value let the sentinel through as a coordinate --
            `-2147483648 * 1e-06` is -2147.48 degrees -- which moved the origin
            from 3.93 to -4836.91 while `gdal.Warp` only *warned*, so a caller got
            a silently mis-georeferenced result rather than an error. Real
            products do ship fills here: SLSTR's oblique `geodetic_*o.nc` covers a
            narrower swath than the nadir grid.
        """
        from osgeo import gdal

        control = _warp_geotransform(
            geolocation_domain(
                self._granule(tmp_path / "clean.nc", filled=False), reader="read_test"
            ),
            shape=(8, 8),
        )
        filled = _warp_geotransform(
            geolocation_domain(
                self._granule(tmp_path / "filled.nc", filled=True), reader="read_test"
            ),
            shape=(8, 8),
        )
        assert filled[0] == pytest.approx(control[0], abs=1e-6), (
            f"a filled pixel moved the longitude origin: {filled[0]} vs {control[0]}"
        )
        assert filled[3] == pytest.approx(control[3], abs=1e-6), (
            f"a filled pixel moved the latitude origin: {filled[3]} vs {control[3]}"
        )
        assert gdal is not None  # the import is what exercises the warp path

    def test_the_vrt_declares_the_fill_in_both_domains(self, tmp_path):
        """The VRT carries the raw sentinel as source NODATA and the scaled one as band nodata."""
        from osgeo import gdal

        geo = self._granule(tmp_path / "geo.nc", filled=True)
        domain = geolocation_domain(geo, reader="read_test")
        xml = gdal.VSIFile(domain["X_DATASET"], "rb").read().decode("utf-8")
        assert "<NODATA>-2147483648" in xml, f"raw source fill missing: {xml}"
        assert "<NoDataValue>-2147.483648" in xml, f"scaled band fill missing: {xml}"

    def test_the_netcdf_default_fill_is_carried_too(self, tmp_path):
        """A variable with no explicit `_FillValue` still gets netCDF's type default.

        Test scenario:
            netCDF defines a per-type default fill (`NC_FILL_INT` is -2147483647)
            that marks never-written data, and GDAL reports it as the band's
            nodata even when the file declares no `_FillValue` attribute --
            measured, not assumed. Carrying it means an unwritten gap in a
            geolocation array is skipped by the transformer just as an explicit
            fill is, rather than being read as a coordinate.
        """
        from osgeo import gdal

        from tests._netcdf3 import write_netcdf

        values = np.round(np.tile(np.linspace(4.0, 5.0, 3), (2, 1)) / 1e-06)
        geo = write_netcdf(
            tmp_path / "geo.nc",
            {
                "longitude": (values.astype("int32"), {"scale_factor": 1e-06}),
                "latitude": (values.astype("int32"), {"scale_factor": 1e-06}),
            },
        )
        domain = geolocation_domain(geo, reader="read_test")
        xml = gdal.VSIFile(domain["X_DATASET"], "rb").read().decode("utf-8")
        assert "<NODATA>-2147483647" in xml, (
            f"netCDF's default int fill should be carried: {xml}"
        )

    def test_a_vrt_built_without_a_fill_declares_none(self):
        """`_unscaling_vrt` omits both nodata elements when given no fill value.

        Test scenario:
            Exercised directly because GDAL supplies a per-type default fill for
            every netCDF integer variable, so no real geolocation file reaches
            this path -- but the helper must still emit valid VRT when a caller
            has no sentinel to declare.
        """
        from osgeo import gdal

        from pyramids_eo.sensors.readers._sentinel3 import _unscaling_vrt

        path = _unscaling_vrt(
            'NETCDF:"nofill.nc":longitude', "longitude", 3, 2, 1e-06, 0.0, None
        )
        xml = gdal.VSIFile(path, "rb").read().decode("utf-8")
        assert "NODATA" not in xml, f"no fill given, so none should appear: {xml}"
        assert "<ScaleRatio>1e-06</ScaleRatio>" in xml, f"scaling must survive: {xml}"


class TestGeolocationSrsGivesTheOutputACrs:
    """The domain's `SRS` key is what puts a CRS on the warped result."""

    def test_the_warped_output_carries_a_geographic_crs(self, tmp_path):
        """With the SRS emitted, the warped raster states its coordinate system.

        Test scenario:
            The key was first justified by an antimeridian rationale that turned
            out not to hold on GDAL 3.13, and the geotransform is identical with
            or without it -- which made the key look inert. It is not: the warped
            output's projection is empty without it, so a caller gets a raster
            positioned correctly but not saying in what. This pins the real
            benefit, which no assertion covered.
        """
        from osgeo import gdal

        geo = write_geolocation(tmp_path / "geo_coordinates.nc")
        domain = geolocation_domain(geo, reader="read_test", shape=(2, 3))
        source = gdal.GetDriverByName("MEM").Create("", 3, 2, 1, gdal.GDT_Float32)
        source.GetRasterBand(1).WriteArray(np.arange(6, dtype="float32").reshape(2, 3))
        source.SetMetadata(dict(domain), "GEOLOCATION")
        with_srs = gdal.Warp("", source, format="MEM", geoloc=True).GetProjection()
        assert "WGS 84" in with_srs, f"the output should carry WGS 84: {with_srs!r}"

        stripped = dict(domain)
        del stripped["SRS"]
        source.SetMetadata(stripped, "GEOLOCATION")
        without_srs = gdal.Warp("", source, format="MEM", geoloc=True).GetProjection()
        assert without_srs == "", (
            f"without SRS the output should have no CRS, got {without_srs!r}"
        )


class TestValidityFallbackFillsOnlyTheMissingBound:
    """A single validity scalar must not suppress the combined-range fallback."""

    @pytest.mark.parametrize(
        "meta, expected",
        [
            ({"valid_min": "10", "valid_range": "{0, 50}"}, (10.0, 50.0)),
            ({"valid_max": "40", "valid_range": "{5, 99}"}, (5.0, 40.0)),
            ({"valid_range": "{5, 99}"}, (5.0, 99.0)),
            ({"valid_min": "10", "valid_max": "40"}, (10.0, 40.0)),
        ],
    )
    def test_each_bound_is_resolved_independently(self, meta, expected):
        """The scalar wins where present; the range supplies only what is missing.

        Args:
            meta: The variable's CF attributes.
            expected: The resolved `(valid_min, valid_max)`.

        Test scenario:
            One usable scalar used to skip the `valid_range` fallback entirely, so
            `{valid_min: 10, valid_range: "{0, 50}"}` resolved to `(10.0, None)` --
            silently dropping the upper bound and leaving saturated pixels unmasked.
        """
        from pyramids_eo.sensors.readers._common import validity_bounds

        assert validity_bounds(meta) == expected, f"{meta} should give {expected}"


class TestNonNumericFillValueWarns:
    """An unparseable `_FillValue` masks nothing, so it must not pass in silence."""

    def test_it_warns_that_masking_is_incomplete(self):
        """A non-numeric fill value warns rather than silently masking nothing.

        Test scenario:
            Before the CF decoder was shared, the FCI reader raised `ValueError`
            on an unparseable `_FillValue`; the lift turned that into a silent
            no-op, which is the one behaviour difference in a change billed as
            behaviour-preserving. Warning keeps the lost masking visible without
            failing a read that is otherwise fine.
        """
        from pyramids_eo.sensors.readers._common import invalid_mask

        with pytest.warns(UserWarning, match="non-numeric _FillValue"):
            mask = invalid_mask(
                np.array([[1, 2]], dtype="int32"), {"_FillValue": "nodata"}
            )
        assert not mask.any(), "nothing can be masked from a non-numeric fill"


class TestValidityBoundsPrecedenceMatrix:
    """The whole `valid_min` x `valid_max` x `valid_range` cross-product, not a sample.

    The other validity tests each pick a row. A sample cannot show that the
    precedence rule is uniform, and the two bounds are resolved independently by
    three interacting cases apiece (present, absent, unparseable) against four
    `valid_range` cases (absent, usable, reversed, unparseable). Every cell is
    stated below with its own expected pair, so a change to the rule has to change
    this table rather than slip through on the cells that happen to agree.
    """

    @pytest.mark.parametrize(
        "meta, expected",
        [
            ({}, (None, None)),
            ({"valid_range": "{0, 100}"}, (0.0, 100.0)),
            ({"valid_range": "{100, 0}"}, (None, None)),
            ({"valid_range": "junk"}, (None, None)),
            ({"valid_max": "50"}, (None, 50.0)),
            ({"valid_max": "50", "valid_range": "{0, 100}"}, (0.0, 50.0)),
            ({"valid_max": "50", "valid_range": "{100, 0}"}, (None, 50.0)),
            ({"valid_max": "50", "valid_range": "junk"}, (None, 50.0)),
            ({"valid_max": "junk"}, (None, None)),
            ({"valid_max": "junk", "valid_range": "{0, 100}"}, (0.0, 100.0)),
            ({"valid_max": "junk", "valid_range": "{100, 0}"}, (None, None)),
            ({"valid_max": "junk", "valid_range": "junk"}, (None, None)),
            ({"valid_min": "10"}, (10.0, None)),
            ({"valid_min": "10", "valid_range": "{0, 100}"}, (10.0, 100.0)),
            ({"valid_min": "10", "valid_range": "{100, 0}"}, (10.0, None)),
            ({"valid_min": "10", "valid_range": "junk"}, (10.0, None)),
            ({"valid_min": "10", "valid_max": "50"}, (10.0, 50.0)),
            (
                {"valid_min": "10", "valid_max": "50", "valid_range": "{0, 100}"},
                (10.0, 50.0),
            ),
            (
                {"valid_min": "10", "valid_max": "50", "valid_range": "{100, 0}"},
                (10.0, 50.0),
            ),
            (
                {"valid_min": "10", "valid_max": "50", "valid_range": "junk"},
                (10.0, 50.0),
            ),
            ({"valid_min": "10", "valid_max": "junk"}, (10.0, None)),
            (
                {"valid_min": "10", "valid_max": "junk", "valid_range": "{0, 100}"},
                (10.0, 100.0),
            ),
            (
                {"valid_min": "10", "valid_max": "junk", "valid_range": "{100, 0}"},
                (10.0, None),
            ),
            (
                {"valid_min": "10", "valid_max": "junk", "valid_range": "junk"},
                (10.0, None),
            ),
            ({"valid_min": "junk"}, (None, None)),
            ({"valid_min": "junk", "valid_range": "{0, 100}"}, (0.0, 100.0)),
            ({"valid_min": "junk", "valid_range": "{100, 0}"}, (None, None)),
            ({"valid_min": "junk", "valid_range": "junk"}, (None, None)),
            ({"valid_min": "junk", "valid_max": "50"}, (None, 50.0)),
            (
                {"valid_min": "junk", "valid_max": "50", "valid_range": "{0, 100}"},
                (0.0, 50.0),
            ),
            (
                {"valid_min": "junk", "valid_max": "50", "valid_range": "{100, 0}"},
                (None, 50.0),
            ),
            (
                {"valid_min": "junk", "valid_max": "50", "valid_range": "junk"},
                (None, 50.0),
            ),
            ({"valid_min": "junk", "valid_max": "junk"}, (None, None)),
            (
                {"valid_min": "junk", "valid_max": "junk", "valid_range": "{0, 100}"},
                (0.0, 100.0),
            ),
            (
                {"valid_min": "junk", "valid_max": "junk", "valid_range": "{100, 0}"},
                (None, None),
            ),
            (
                {"valid_min": "junk", "valid_max": "junk", "valid_range": "junk"},
                (None, None),
            ),
        ],
    )
    def test_every_combination_resolves_as_documented(self, meta, expected):
        """Each cell of the precedence matrix resolves to its documented pair.

        Args:
            meta: The variable's CF attribute dict for this cell.
            expected: The `(valid_min, valid_max)` the contract requires.

        Test scenario:
            The rule is: a usable scalar wins for its own bound, an unparseable one
            counts as absent, `valid_range` supplies only the bounds no scalar
            gave, a reversed range supplies neither, and a resolved pair that is
            itself reversed masks nothing. Spelling out all 36 cells is what makes
            each clause load-bearing instead of inferable from a friendly sample.
        """
        assert validity_bounds(meta) == expected, (
            f"{meta} should resolve to {expected}, got {validity_bounds(meta)}"
        )

    def test_a_resolved_pair_that_is_reversed_masks_nothing(self):
        """A min above a max, assembled from the two scalars, discards both bounds.

        Test scenario:
            The reversal guard runs after the fallback, so it has to catch a pair
            that only becomes reversed once assembled -- otherwise every pixel is
            masked with no diagnostic.
        """
        assert validity_bounds({"valid_min": "90", "valid_max": "10"}) == (None, None)
        assert validity_bounds({"valid_min": "90", "valid_range": "{0, 10}"}) == (
            None,
            None,
        ), "a scalar min above the range's max must discard both bounds"


class TestValidBoundsTakesTheFirstAndLastNumber:
    """`valid_range` is parsed as first-and-last, which a two-number sample hides."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("{0, 4095}", (0.0, 4095.0)),
            # Three numbers: the LAST is the upper bound, not the second. Every
            # other fixture uses a two-element range, where those coincide.
            ("{1, 2, 3}", (1.0, 3.0)),
            # The middle number is ignored outright: it neither bounds the range
            # nor makes an otherwise-ascending one look reversed.
            ("{1, 9, 2}", (1.0, 2.0)),
            # Reversal is judged on the ends, so this is discarded although its
            # first two numbers ascend.
            ("{9, 1, 2}", (None, None)),
            # GDAL renders a vector attribute without braces in some builds.
            ("0 50", (0.0, 50.0)),
            ("{-1.5e2, 3}", (-150.0, 3.0)),
            ("{3}", (None, None)),
        ],
    )
    def test_the_bounds_come_from_the_ends_of_the_number_list(self, text, expected):
        """The first and last numbers bound the range, whatever sits between them.

        Args:
            text: The raw `valid_range` attribute value.
            expected: The `(low, high)` pair it must yield.

        Test scenario:
            `{1, 2, 3}` distinguishes "second number" from "last number", and
            `{1, 9, 2}` against `{9, 1, 2}` shows the reversal check operates on
            the ends rather than on the first pair. A two-element range -- the only
            shape the rest of the suite uses -- cannot tell those readings apart.
        """
        from pyramids_eo.sensors.readers._common import valid_bounds

        assert valid_bounds(text) == expected, (
            f"{text!r} should parse to {expected}, got {valid_bounds(text)}"
        )


class TestLongitudeDrivesXAndLatitudeDrivesY:
    """The domain's X slot must carry longitude and its Y slot latitude.

    Nothing else warps a grid whose longitude and latitude ranges cannot be
    confused *and* compares it against the same grid with the two variables
    exchanged. The fixture's longitude spans 4..5 and its latitude 51..52, so a
    transposition moves the warped origin by ~47 degrees instead of being masked by
    a shared range.
    """

    @staticmethod
    def _origin(domain):
        """Warp an 8x8 raster through `domain` and return its `(origin_x, origin_y)`."""
        geotransform, _ = _warp(domain, (8, 8))
        return geotransform[0], geotransform[3]

    def test_the_warped_origin_lands_in_the_longitude_and_latitude_ranges(
        self, tmp_path
    ):
        """X comes out in the longitude range and Y in the latitude range."""
        geo = write_geolocation(tmp_path / "geo_coordinates.nc", shape=(8, 8))
        origin_x, origin_y = self._origin(geolocation_domain(geo, reader="read_test"))
        assert 3.0 < origin_x < 6.0, (
            f"X must be driven by longitude (4..5 here), got {origin_x}"
        )
        assert 50.0 < origin_y < 53.0, (
            f"Y must be driven by latitude (51..52 here), got {origin_y}"
        )

    def test_exchanging_the_two_variables_visibly_moves_the_origin(self, tmp_path):
        """Feeding latitude to X and longitude to Y relocates the result entirely.

        Test scenario:
            This is the decisive half: it proves the previous assertion is
            sensitive to the X/Y wiring rather than to anything else about the
            domain. Measured, the origin moves from (3.93, 52.07) to (50.86,
            5.14), so a reader that filled the two slots the wrong way round --
            which no key name or warp success would reveal -- could not pass both
            tests.
        """
        geo = write_geolocation(tmp_path / "geo_coordinates.nc", shape=(8, 8))
        correct = self._origin(geolocation_domain(geo, reader="read_test"))
        exchanged = self._origin(
            geolocation_domain(
                geo,
                reader="read_test",
                longitude_variable="latitude",
                latitude_variable="longitude",
            )
        )
        swapped = f"exchanging the variables must put latitude on X, got {exchanged}"
        assert 50.0 < exchanged[0] < 53.0, swapped
        assert 3.0 < exchanged[1] < 6.0, swapped
        assert abs(exchanged[0] - correct[0]) > 40.0, (
            f"the two wirings must be distinguishable: {correct} vs {exchanged}"
        )
