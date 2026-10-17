"""Unit tests for the tasseled-cap transform and its coefficient tables."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from pyramids_eo.errors import TransformError, UnknownSensorError
from pyramids_eo.sensors.registry import (
    TasseledCapCoefficients,
    available_tasseled_cap_sensors,
    get_tasseled_cap,
)
from pyramids_eo.sensors.registry import tasseled_cap as _coeff_module
from pyramids_eo.sensors.registry.tasseled_cap import _load
from pyramids_eo.spectral import tasseled_cap
from pyramids_eo.spectral.tasseled_cap import _invalid_pixels, _resolve_band_order

_GEO = (0.0, 10.0, 0.0, 0.0, 0.0, -10.0)
#: The canonical six-band role order every bundled sensor table uses.
_ROLES = ["blue", "green", "red", "nir", "swir1", "swir2"]


def _dataset(stack: np.ndarray, band_names: list[str] | None = None) -> Dataset:
    """Build an in-memory Dataset from a `(band, row, col)` stack.

    Args:
        stack: The band stack.
        band_names: Band names to set, or `None` to leave the defaults.

    Returns:
        A WGS84 Dataset carrying `stack`.
    """
    ds = Dataset.from_array(
        np.asarray(stack, dtype=float),
        geo_ref=GeoReference(geo=_GEO, epsg=4326),
        no_data_value=np.nan,
    )
    if band_names is not None:
        ds.band_names = band_names
    return ds


class TestGetTasseledCap:
    """`get_tasseled_cap` loads and validates the bundled coefficient tables."""

    @pytest.mark.parametrize("sensor", ["landsat5_tm", "landsat7_etm", "landsat8_oli"])
    def test_loads_each_bundled_sensor(self, sensor):
        """Each bundled sensor loads with six bands and the three components.

        Args:
            sensor: A bundled tasseled-cap sensor identifier.

        Test scenario:
            The table has the canonical six-band role order and defines
            brightness / greenness / wetness, each with one coefficient per band.
        """
        table = get_tasseled_cap(sensor)
        assert table.bands == tuple(_ROLES), f"{sensor} band order: {table.bands}"
        assert table.component_names == ["brightness", "greenness", "wetness"], (
            f"{sensor} components: {table.component_names}"
        )
        for name, row in table.components.items():
            assert len(row) == len(table.bands), (
                f"{sensor}.{name} has {len(row)} coeffs for {len(table.bands)} bands"
            )
        assert table.reference, f"{sensor} should cite a reference"

    def test_unknown_sensor_raises(self):
        """An unknown sensor raises UnknownSensorError listing what is available."""
        with pytest.raises(UnknownSensorError, match="unknown tasseled-cap sensor"):
            get_tasseled_cap("sentinel2_msi")

    def test_case_insensitive(self):
        """The sensor lookup is case-insensitive."""
        assert get_tasseled_cap("LANDSAT8_OLI") is get_tasseled_cap("landsat8_oli"), (
            "case-insensitive lookup should return the same cached table"
        )

    def test_cached_identity(self):
        """Repeated loads return the same cached, frozen object."""
        assert get_tasseled_cap("landsat5_tm") is get_tasseled_cap("landsat5_tm"), (
            "get_tasseled_cap should cache"
        )

    def test_available_sensors(self):
        """`available_tasseled_cap_sensors` lists the bundled Landsat tables."""
        assert available_tasseled_cap_sensors() == [
            "landsat5_tm",
            "landsat7_etm",
            "landsat8_oli",
        ], "the three bundled sensors, sorted"

    def test_role_of_canonical_alias_and_unknown(self):
        """`role_of` resolves canonical names and aliases, else returns None."""
        table = get_tasseled_cap("landsat8_oli")
        assert table.role_of("red") == "red", "canonical role name"
        assert table.role_of("B4") == "red", "OLI alias B4 is red"
        assert table.role_of("sr_b5") == "nir", "alias match is case-insensitive"
        assert table.role_of("thermal") is None, "an unknown name resolves to None"

    def test_coefficients_are_frozen(self):
        """The returned table is a frozen dataclass with read-only mappings."""
        table = get_tasseled_cap("landsat8_oli")
        assert isinstance(table, TasseledCapCoefficients), "frozen dataclass"
        with pytest.raises(TypeError):
            table.components["brightness"] = (0.0,) * 6


#: Golden published coefficients, keyed by (sensor, component), pinned to the
#: literals in the cited papers (Crist 1985 TM, Huang 2002 ETM+, Baig 2014 OLI).
#: A YAML edit that changes the science must change these on purpose.
_GOLDEN = {
    ("landsat5_tm", "brightness"): (0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303),
    ("landsat5_tm", "greenness"): (-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446),
    ("landsat5_tm", "wetness"): (0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109),
    ("landsat7_etm", "brightness"): (0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596),
    ("landsat7_etm", "greenness"): (
        -0.3344,
        -0.3544,
        -0.4556,
        0.6966,
        -0.0242,
        -0.2630,
    ),
    ("landsat7_etm", "wetness"): (0.2626, 0.2141, 0.0926, 0.0656, -0.7629, -0.5388),
    ("landsat8_oli", "brightness"): (0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872),
    ("landsat8_oli", "greenness"): (-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608),
    ("landsat8_oli", "wetness"): (0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559),
}


class TestGoldenCoefficients:
    """The bundled coefficients match the published literals, not just the wiring."""

    @pytest.mark.parametrize(("sensor", "component"), list(_GOLDEN))
    def test_coefficients_match_published_literals(self, sensor, component):
        """Each sensor/component row equals its hand-checked published literal.

        Args:
            sensor: A bundled sensor identifier.
            component: A tasseled-cap component name.

        Test scenario:
            The YAML row must equal the paper's coefficients exactly, so a
            transcription typo (the PR's core risk) fails here rather than
            silently shipping wrong science.
        """
        got = get_tasseled_cap(sensor).components[component]
        assert got == _GOLDEN[(sensor, component)], (
            f"{sensor}.{component} drifted from the published literal: {got}"
        )


class TestTasseledCapTransform:
    """`tasseled_cap` applies the coefficients to a reflectance band stack."""

    def test_named_bands_canonical_order(self):
        """An all-ones stack yields each component's coefficient-row sum.

        Test scenario:
            With every reflectance equal to 1.0, a component value is the sum of
            its coefficient row; bands are named in canonical order.
        """
        ds = _dataset(np.ones((6, 1, 1)), band_names=_ROLES)
        out = tasseled_cap(ds, sensor="landsat8_oli")
        assert out.band_names == ["brightness", "greenness", "wetness"], (
            f"named component bands: {out.band_names}"
        )
        table = get_tasseled_cap("landsat8_oli")
        got = np.asarray(out.read_array())[:, 0, 0]
        expected = [sum(table.components[c]) for c in out.band_names]
        assert np.allclose(got, expected), f"expected {expected}, got {got.tolist()}"

    def test_known_numeric_value(self):
        """Brightness of a distinct-per-band stack matches a hand computation."""
        stack = np.array([1, 2, 3, 4, 5, 6], dtype=float).reshape(6, 1, 1)
        ds = _dataset(stack, band_names=_ROLES)
        brightness = np.asarray(
            tasseled_cap(
                ds, sensor="landsat8_oli", components=["brightness"]
            ).read_array()
        )
        coeffs = get_tasseled_cap("landsat8_oli").components["brightness"]
        expected = sum(c * (i + 1) for i, c in enumerate(coeffs))
        assert brightness[0, 0] == pytest.approx(expected), (
            f"brightness {brightness[0, 0]} != {expected}"
        )

    def test_reorders_by_band_name(self):
        """Shuffled-but-named bands are reordered to the coefficient order.

        Test scenario:
            The same values as the canonical [1..6] stack, supplied in a shuffled
            band order with matching names, must give the identical result.
        """
        canonical = _dataset(
            np.array([1, 2, 3, 4, 5, 6], dtype=float).reshape(6, 1, 1),
            band_names=_ROLES,
        )
        shuffled_names = ["red", "blue", "green", "swir1", "nir", "swir2"]
        shuffled_values = [3, 1, 2, 5, 4, 6]  # same role->value mapping, reordered
        shuffled = _dataset(
            np.array(shuffled_values, dtype=float).reshape(6, 1, 1),
            band_names=shuffled_names,
        )
        a = np.asarray(tasseled_cap(canonical, sensor="landsat8_oli").read_array())
        b = np.asarray(tasseled_cap(shuffled, sensor="landsat8_oli").read_array())
        assert np.allclose(a, b), "reordering by name must match the canonical order"

    def test_alias_names_are_recognised(self):
        """Sensor band spellings (B2..B7 for OLI) are matched to roles."""
        stack = np.array([1, 2, 3, 4, 5, 6], dtype=float).reshape(6, 1, 1)
        oli_names = ["B2", "B3", "B4", "B5", "B6", "B7"]
        by_alias = np.asarray(
            tasseled_cap(_dataset(stack, oli_names), sensor="landsat8_oli").read_array()
        )
        by_role = np.asarray(
            tasseled_cap(_dataset(stack, _ROLES), sensor="landsat8_oli").read_array()
        )
        assert np.allclose(by_alias, by_role), (
            "aliases must resolve like canonical roles"
        )

    def test_unnamed_bands_use_positional_order(self):
        """Generic (unrecognised) band names fall back to positional order."""
        stack = np.array([1, 2, 3, 4, 5, 6], dtype=float).reshape(6, 1, 1)
        generic = _dataset(stack, [f"Band_{i}" for i in range(1, 7)])
        out = np.asarray(tasseled_cap(generic, sensor="landsat8_oli").read_array())
        coeffs = get_tasseled_cap("landsat8_oli").components["brightness"]
        expected = sum(c * (i + 1) for i, c in enumerate(coeffs))
        assert out[0, 0, 0] == pytest.approx(expected), "positional order assumed"

    def test_result_carries_georeference(self):
        """The output keeps the input's geotransform and declares NaN nodata."""
        ds = _dataset(np.ones((6, 2, 3)), band_names=_ROLES)
        out = tasseled_cap(ds, sensor="landsat8_oli")
        assert out.geotransform == _GEO, (
            f"geotransform should pass through: {out.geotransform}"
        )
        assert out.epsg == 4326, f"CRS should pass through, got {out.epsg}"
        assert np.isnan(out.no_data_value[0]), "nodata should be NaN"
        assert out.read_array().shape == (3, 2, 3), "one band per component"

    def test_input_nodata_is_masked_to_nan(self):
        """A pixel equal to the input nodata comes out NaN in every component.

        Test scenario:
            A two-pixel stack whose second pixel is the declared -9999 fill must
            emit NaN there (not a plausible finite value), while the valid pixel
            stays finite -- the issue the output's NaN nodata must not lie about.
        """
        stack = np.ones((6, 1, 2))
        stack[:, 0, 1] = -9999.0
        ds = Dataset.from_array(
            np.asarray(stack),
            geo_ref=GeoReference(geo=_GEO, epsg=4326),
            no_data_value=-9999,
        )
        ds.band_names = _ROLES
        out = np.asarray(tasseled_cap(ds, sensor="landsat8_oli").read_array())
        assert np.isnan(out[:, 0, 1]).all(), (
            f"fill pixel must be NaN, got {out[:, 0, 1]}"
        )
        assert np.isfinite(out[:, 0, 0]).all(), "the valid pixel must stay finite"

    def test_non_finite_input_is_masked(self):
        """A non-finite value in any input band masks that whole pixel."""
        stack = np.ones((6, 1, 2))
        stack[3, 0, 1] = np.nan  # nir NaN at the second pixel
        ds = _dataset(stack, band_names=_ROLES)
        out = np.asarray(tasseled_cap(ds, sensor="landsat8_oli").read_array())
        assert np.isnan(out[:, 0, 1]).all(), "a non-finite input band masks the pixel"
        assert np.isfinite(out[:, 0, 0]).all(), "the finite pixel survives"

    def test_generic_bandN_names_fall_back_positionally(self):
        """A generic band1..band6 stack is positional, not a partial-match raise.

        Test scenario:
            With the bare `bandN` aliases dropped (issue #117 review L4), a
            generically-numbered stack resolves no roles and is assumed already
            in canonical order, matching the canonically-named result, rather
            than partial-matching (band1 unknown for OLI) and raising.
        """
        stack = np.array([1, 2, 3, 4, 5, 6], dtype=float).reshape(6, 1, 1)
        generic = _dataset(stack, [f"band{i}" for i in range(1, 7)])
        out = np.asarray(tasseled_cap(generic, sensor="landsat8_oli").read_array())
        by_role = np.asarray(
            tasseled_cap(_dataset(stack, _ROLES), sensor="landsat8_oli").read_array()
        )
        assert np.allclose(out, by_role), "band1..band6 should fall back to positional"

    def test_wkt_only_crs_round_trips(self):
        """A non-EPSG (WKT-only) input CRS is carried onto the output.

        Test scenario:
            A geostationary grid reports `epsg is None` and a WKT `crs`; the
            result must keep that WKT rather than dropping or mangling the CRS.
        """
        from osgeo import osr

        srs = osr.SpatialReference()
        srs.ImportFromProj4("+proj=geos +lon_0=0 +h=35785831 +ellps=WGS84 +units=m")
        wkt = srs.ExportToWkt()
        ds = Dataset.from_array(
            np.ones((6, 1, 1)),
            geo_ref=GeoReference(geo=_GEO, epsg=4326),
            no_data_value=np.nan,
        )
        ds.crs = wkt
        ds.band_names = _ROLES
        out = tasseled_cap(ds, sensor="landsat8_oli")
        assert out.epsg is None, f"a geostationary grid has no EPSG, got {out.epsg}"
        assert "Geostationary" in out.crs, f"the WKT CRS must round-trip: {out.crs}"

    def test_warns_on_non_reflectance_magnitude(self):
        """A scaled-integer / DN stack (values far above 1) warns about the domain."""
        ds = _dataset(np.full((6, 1, 1), 8000.0), band_names=_ROLES)
        with pytest.warns(UserWarning, match="reflectance"):
            tasseled_cap(ds, sensor="landsat8_oli")

    def test_no_warning_for_reflectance_magnitude(self, recwarn):
        """A normal 0-1 reflectance stack emits no domain warning.

        Args:
            recwarn: pytest's recorded-warnings fixture.
        """
        ds = _dataset(np.full((6, 1, 1), 0.3), band_names=_ROLES)
        tasseled_cap(ds, sensor="landsat8_oli")
        domain_warnings = [w for w in recwarn.list if "reflectance" in str(w.message)]
        assert not domain_warnings, f"0-1 data should not warn: {domain_warnings}"

    def test_subset_and_order_of_components(self):
        """`components` selects and orders the output bands."""
        ds = _dataset(np.ones((6, 1, 1)), band_names=_ROLES)
        out = tasseled_cap(
            ds, sensor="landsat8_oli", components=["wetness", "brightness"]
        )
        assert out.band_names == ["wetness", "brightness"], (
            f"output band order follows components: {out.band_names}"
        )

    def test_empty_components_raises(self):
        """An empty `components` is rejected."""
        ds = _dataset(np.ones((6, 1, 1)), band_names=_ROLES)
        with pytest.raises(TransformError, match="`components` is empty"):
            tasseled_cap(ds, sensor="landsat8_oli", components=[])

    def test_unknown_component_raises(self):
        """A component the table does not define is rejected."""
        ds = _dataset(np.ones((6, 1, 1)), band_names=_ROLES)
        with pytest.raises(TransformError, match="no component"):
            tasseled_cap(ds, sensor="landsat8_oli", components=["fourth"])

    def test_wrong_band_count_raises(self):
        """A stack with the wrong number of bands is rejected."""
        ds = _dataset(np.ones((4, 1, 1)), band_names=None)
        with pytest.raises(TransformError, match="expects 6 bands"):
            tasseled_cap(ds, sensor="landsat8_oli")

    def test_non_stack_input_raises(self):
        """A single-band (2-D) raster is rejected -- a stack is required."""
        ds = _dataset(np.ones((1, 1)))
        with pytest.raises(TransformError, match="band, row, col"):
            tasseled_cap(ds, sensor="landsat8_oli")

    def test_non_dataset_input_raises(self):
        """A bare ndarray (no read_array) is rejected with a clear message."""
        with pytest.raises(TransformError, match="must be a pyramids Dataset"):
            tasseled_cap(np.ones((6, 1, 1)), sensor="landsat8_oli")

    def test_partially_named_bands_raises(self):
        """A stack with only some bands recognised is ambiguous and rejected."""
        names = ["blue", "green", "red", "nir", "swir1", "mystery"]
        ds = _dataset(np.ones((6, 1, 1)), band_names=names)
        with pytest.raises(TransformError, match="cannot match band"):
            tasseled_cap(ds, sensor="landsat8_oli")

    def test_named_bands_not_matching_sensor_raises(self):
        """Recognised names that are not the sensor's band set are rejected."""
        # 'red' twice, no 'swir2' -> a duplicated/missing role.
        names = ["blue", "green", "red", "nir", "swir1", "red"]
        ds = _dataset(np.ones((6, 1, 1)), band_names=names)
        with pytest.raises(TransformError, match="missing, extra, or duplicated"):
            tasseled_cap(ds, sensor="landsat8_oli")

    def test_unknown_sensor_raises(self):
        """An unknown sensor propagates UnknownSensorError from the loader."""
        ds = _dataset(np.ones((6, 1, 1)), band_names=_ROLES)
        with pytest.raises(UnknownSensorError):
            tasseled_cap(ds, sensor="modis")


class TestBundledData:
    """Integrity checks on the bundled coefficient YAML."""

    def test_every_sensor_block_is_well_formed(self):
        """Every bundled sensor has bands, three components, and aliases per band."""
        for sensor in _load():
            table = get_tasseled_cap(sensor)
            assert len(table.bands) == 6, f"{sensor}: expected 6 bands"
            assert set(table.components) == {"brightness", "greenness", "wetness"}, (
                f"{sensor}: components {set(table.components)}"
            )
            for role in table.bands:
                assert table.aliases.get(role), f"{sensor}.{role} has no aliases"

    def test_malformed_component_row_raises(self, monkeypatch):
        """A component row whose length != band count raises ValueError.

        Test scenario:
            A hand-built table whose `brightness` row has one coefficient too few
            must be rejected by the loader's integrity check, not loaded.
        """
        bad = {
            "malformed_sensor": {
                "bands": ["blue", "green", "red", "nir", "swir1", "swir2"],
                "components": {"brightness": [0.1, 0.2, 0.3]},  # 3 coeffs, 6 bands
            }
        }
        monkeypatch.setattr(_coeff_module, "_load", lambda: bad)
        with pytest.raises(ValueError, match="coefficients but the sensor has 6 bands"):
            get_tasseled_cap("malformed_sensor")


class TestResolveBandOrder:
    """`_resolve_band_order` chooses a reorder, positional, or raises."""

    def test_missing_band_names_use_positional(self):
        """`None` or empty band names fall back to positional order (returns None)."""
        table = get_tasseled_cap("landsat8_oli")
        assert _resolve_band_order(None, table) is None, "None -> positional"
        assert _resolve_band_order([], table) is None, "empty -> positional"


class TestInvalidPixels:
    """`_invalid_pixels` flags fill and non-finite pixels."""

    def test_without_nodata_only_non_finite_is_flagged(self):
        """With no declared nodata, only non-finite pixels are masked."""

        class _NoNodata:
            no_data_value = None

        stack = np.ones((6, 1, 2))
        stack[0, 0, 1] = np.nan
        mask = _invalid_pixels(stack, _NoNodata())
        assert mask.tolist() == [[False, True]], f"only the NaN pixel flagged: {mask}"
