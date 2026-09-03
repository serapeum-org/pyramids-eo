"""Unit tests for `pyramids_eo.sensors.readers.read_fci` (offline; synthetic chunks)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from pyramids_eo.errors import CalibrationError, ReaderError, UnknownSensorError
from pyramids_eo.sensors.readers import read_fci
from pyramids_eo.sensors.readers.fci import _default_open_chunk, open_fci_l1c_chunk
from pyramids_eo.sensors.registry import (
    Channel,
    Sensor,
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)
from pyramids_eo.sensors.registry import sensors as _sensors


def _chunk(arr: np.ndarray, tlc=(0.0, 4.0)) -> Dataset:
    """A pyramids Dataset chunk holding raw radiance."""
    return Dataset.from_array(
        arr, geo_ref=GeoReference(top_left_corner=tlc, cell_size=1.0, epsg=4326)
    )


_GEOS_WKT = (
    'PROJCS["geos",GEOGCS["sphere",DATUM["D",SPHEROID["S",6378169,295.488065897]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Geostationary_Satellite"],PARAMETER["central_meridian",{lon}],'
    'PARAMETER["satellite_height",35785831],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)


def _geos_chunk(arr: np.ndarray, tlc=(0.0, 4.0), lon=0) -> Dataset:
    """A chunk on a geostationary grid: epsg is None, the projection is on crs."""
    ds = Dataset.from_array(
        arr, geo_ref=GeoReference(top_left_corner=tlc, cell_size=1.0, epsg=None)
    )
    ds.crs = _GEOS_WKT.format(lon=lon)
    return ds


def _channel_opener(per_channel: dict[str, np.ndarray]):
    """An `open_chunk` yielding a distinct Dataset per channel from `per_channel`."""

    def _open(path, channel):
        return _chunk(per_channel[channel])

    return _open


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
        assert np.allclose(arr[:2], 5.0), "top rows should be the first chunk"
        assert np.allclose(arr[2:], 9.0), "bottom rows should be the second chunk"

    def test_geostationary_chunks_keep_their_crs(self):
        """A geostationary chunk has no EPSG code; its projection must survive."""
        top = _geos_chunk(np.full((2, 3), 5.0), tlc=(0.0, 4.0))
        bottom = _geos_chunk(np.full((2, 3), 9.0), tlc=(0.0, 2.0))
        assert top.epsg is None, "a geostationary chunk should report no EPSG code"
        out = read_fci([top, bottom], "ir_105", calibrate=False)
        assert out.crs, "read_fci dropped the geostationary CRS"
        assert "Geostationary" in out.crs, "read_fci lost the geostationary WKT"

    def test_chunks_from_different_satellite_positions_raise(self):
        """Two geostationary chunks at different sub-satellite longitudes differ.

        Both report epsg None, so the guard has to compare the resolved CRS --
        an .epsg comparison is None != None and would stitch them silently.
        """
        top = _geos_chunk(np.full((2, 3), 5.0), tlc=(0.0, 4.0), lon=0)
        bottom = _geos_chunk(np.full((2, 3), 9.0), tlc=(0.0, 2.0), lon=41.5)
        with pytest.raises(ReaderError, match="mixed CRS"):
            read_fci([top, bottom], "ir_105", calibrate=False)

    def test_chunks_sharing_a_geostationary_crs_are_accepted(self):
        """Matching geostationary chunks still stitch."""
        top = _geos_chunk(np.full((2, 3), 5.0), tlc=(0.0, 4.0), lon=0)
        bottom = _geos_chunk(np.full((2, 3), 9.0), tlc=(0.0, 2.0), lon=0)
        out = read_fci([top, bottom], "ir_105", calibrate=False)
        assert out.shape[-2:] == (4, 3), f"expected 4 stitched rows, got {out.shape}"

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

    def test_channels_returns_dict(self):
        """`channels=[...]` returns a dict with each channel's own calibrated data.

        The opener yields distinct radiance per channel, so a cross-channel mix-up
        (wrong key or wrong data) would fail — the entries are not aliased.
        """
        data = {"ir_105": np.full((2, 2), 80.0), "vis_06": np.full((2, 2), 120.0)}
        out = read_fci(
            ["c0.nc"], channels=["ir_105", "vis_06"], open_chunk=_channel_opener(data)
        )
        assert isinstance(out, dict), f"expected a dict, got {type(out)}"
        assert set(out) == {"ir_105", "vis_06"}, "one entry per requested channel"
        ir = _sensors.get_sensor("fci").get_channel("ir_105")
        assert np.allclose(
            out["ir_105"].read_array(),
            radiance_to_brightness_temperature(
                data["ir_105"], ir.central_wavenumber_cm1, ir.alpha, ir.beta
            ),
        ), "ir_105 entry should be its own brightness temperature"
        vis = _sensors.get_sensor("fci").get_channel("vis_06")
        assert np.allclose(
            out["vis_06"].read_array(),
            radiance_to_reflectance(data["vis_06"], vis.solar_irradiance),
        ), "vis_06 entry should be its own reflectance"

    def test_channels_dict_equals_single(self):
        """A dict entry equals the single-channel result for that channel."""
        opener = _channel_opener({"ir_105": np.full((2, 2), 80.0)})
        single = read_fci(["c0.nc"], "ir_105", open_chunk=opener)
        multi = read_fci(["c0.nc"], channels=["ir_105"], open_chunk=opener)
        assert np.allclose(multi["ir_105"].read_array(), single.read_array()), (
            "channels=[...] must match calling per channel"
        )

    def test_channels_with_pre_opened_dataset_raises(self):
        """`channels=[...]` with pre-opened Dataset chunks is rejected."""
        chunk = _chunk(np.ones((2, 2)))
        with pytest.raises(ReaderError, match="path-like chunks"):
            read_fci([chunk], channels=["ir_105", "vis_06"])

    def test_channels_with_coeffs_raises(self):
        """`coeffs` is a single-channel override; combining it with channels is rejected."""
        with pytest.raises(ReaderError, match="coeffs"):
            read_fci(
                ["c0.nc"],
                channels=["ir_105", "vis_06"],
                coeffs={"solar_irradiance": 100.0},
            )

    def test_neither_channel_nor_channels_raises(self):
        """Passing neither `channel` nor `channels` is rejected."""
        chunk = _chunk(np.ones((2, 2)))
        with pytest.raises(ReaderError, match="exactly one"):
            read_fci([chunk])

    def test_both_channel_and_channels_raises(self):
        """Passing both `channel` and `channels` is rejected."""
        chunk = _chunk(np.ones((2, 2)))
        with pytest.raises(ReaderError, match="exactly one"):
            read_fci([chunk], "ir_105", channels=["ir_105"])

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
        assert np.allclose(arr[:2], 5.0), "top rows should be the north chunk"
        assert np.allclose(arr[2:], 9.0), "bottom rows should be the south chunk"
        assert out.geotransform == north.geotransform, (
            "origin should be the north chunk"
        )

    def test_mixed_crs_chunks_raise(self):
        """Chunks with different CRS are rejected."""
        a = _chunk(np.ones((2, 2)), tlc=(0.0, 4.0))
        b = Dataset.from_array(
            np.ones((2, 2)),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=3857),
        )
        with pytest.raises(ReaderError, match="mixed CRS"):
            read_fci([a, b], "ir_105")

    def test_mixed_cell_size_chunks_raise(self):
        """Chunks with different cell sizes are rejected."""
        a = _chunk(np.ones((2, 2)), tlc=(0.0, 4.0))
        b = Dataset.from_array(
            np.ones((2, 2)),
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=2.0, epsg=4326),
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
        chunk = _chunk(np.ones((2, 2)))
        with pytest.raises(UnknownSensorError, match="has no channel"):
            read_fci([chunk], "not_a_channel")

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
        chunk = _chunk(np.ones((2, 2)))
        with pytest.raises(CalibrationError, match="alpha"):
            read_fci([chunk], "ir_105", coeffs={"alpha": 0.0})

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
            "pyramids_eo.sensors.readers._common.get_sensor", lambda name: broken
        )
        chunk = _chunk(np.ones((2, 2)))
        with pytest.raises(CalibrationError, match="solar_irradiance"):
            read_fci([chunk], "x")

    def test_thermal_channel_missing_wavenumber_raises(self, monkeypatch):
        """A thermal channel without a central wavenumber raises CalibrationError."""
        broken = Sensor(
            name="fci",
            channels={
                "y": Channel("y", 10.5, 2000, "thermal", central_wavenumber_cm1=None)
            },
        )
        monkeypatch.setattr(
            "pyramids_eo.sensors.readers._common.get_sensor", lambda name: broken
        )
        chunk = _chunk(np.ones((2, 2)))
        with pytest.raises(CalibrationError, match="central_wavenumber"):
            read_fci([chunk], "y")


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


class TestOpenFciL1cChunk:
    """`open_fci_l1c_chunk` reads the nested FCI L1C FDHSI radiance group."""

    @staticmethod
    def _patch_netcdf(monkeypatch, captured):
        """Patch NetCDF.read_file to a fake recording the requested variable."""

        class _FakeNC:
            def get_variable(self, name):
                captured["name"] = name
                return captured.get("dataset", object())

        import pyramids.netcdf as _ncmod

        def _fake_read_file(cls, p, **kw):
            captured["read_file_kw"] = kw
            return _FakeNC()

        monkeypatch.setattr(_ncmod.NetCDF, "read_file", classmethod(_fake_read_file))

    def test_reads_nested_group_qualified_variable(self, monkeypatch):
        """The opener requests `data/<channel>/measured/effective_radiance`."""
        captured = {}
        self._patch_netcdf(monkeypatch, captured)
        open_fci_l1c_chunk("chunk.nc", "ir_105")
        assert captured["name"] == "data/ir_105/measured/effective_radiance", (
            f"unexpected group path {captured.get('name')!r}"
        )

    def test_custom_radiance_group_template(self, monkeypatch):
        """A custom radiance_group template overrides the default path."""
        captured = {}
        self._patch_netcdf(monkeypatch, captured)
        open_fci_l1c_chunk("chunk.nc", "vis_06", radiance_group="state/{channel}/rad")
        assert captured["name"] == "state/vis_06/rad", captured

    def test_wires_through_read_fci_as_open_chunk(self, monkeypatch):
        """read_fci uses open_fci_l1c_chunk to open a non-Dataset chunk."""
        captured = {"dataset": _chunk(np.full((2, 3), 80.0))}
        self._patch_netcdf(monkeypatch, captured)
        out = read_fci(
            ["chunk0.nc"], "ir_105", calibrate=False, open_chunk=open_fci_l1c_chunk
        )
        assert np.allclose(out.read_array(), 80.0), "nested-opened radiance not used"
        assert captured["name"] == "data/ir_105/measured/effective_radiance", captured

    def test_channel_with_braces_inserted_literally(self, monkeypatch):
        """A channel value containing braces is inserted literally, not re-parsed."""
        captured = {}
        self._patch_netcdf(monkeypatch, captured)
        open_fci_l1c_chunk("chunk.nc", "ir_{x}")
        assert captured["name"] == "data/ir_{x}/measured/effective_radiance", captured

    def test_opens_file_as_multidimensional(self, monkeypatch):
        """The opener passes open_as_multi_dimensional=True for group navigation."""
        captured = {}
        self._patch_netcdf(monkeypatch, captured)
        open_fci_l1c_chunk("chunk.nc", "ir_105")
        assert captured["read_file_kw"].get("open_as_multi_dimensional") is True, (
            f"expected a multidimensional open, got {captured.get('read_file_kw')}"
        )

    @pytest.mark.parametrize(
        "template",
        ["data/{chan}/rad", "data/{}/rad", "data/{channel"],
    )
    def test_malformed_template_raises_reader_error(self, monkeypatch, template):
        """A malformed template raises ReaderError, not a raw str.format error."""
        captured = {}
        self._patch_netcdf(monkeypatch, captured)
        with pytest.raises(ReaderError, match="radiance_group"):
            open_fci_l1c_chunk("chunk.nc", "ir_105", radiance_group=template)
