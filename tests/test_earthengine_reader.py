"""Tests for :mod:`pyramids_eo.earthengine.reader`.

The EEDAI network open is the single seam (``_open_eedai``); the offline tests
either monkeypatch it with a synthetic in-memory raster or drive it with a faked
``gdal`` so CI needs no live Earth Engine account. A ``live`` test exercises the
real driver end-to-end and is deselected by the default ``-m "not live"`` run.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset, DatasetCollection

import pyramids_eo.earthengine.reader as ee_reader
from pyramids_eo import (
    collection_from_earthengine,
    estimate_earthengine_cost,
    from_earthengine,
)
from pyramids_eo.earthengine import EarthEngineCredentials
from pyramids_eo.earthengine.reader import _Scene
from pyramids_eo.errors import ReaderError

_BBOX = (86.9, 27.9, 87.0, 28.0)
# A smaller AOI for the live Sentinel-2 tests: at 10 m native this still spans
# several 256-px blocks (so it exercises the block-crossing read the fix targets)
# without reading the ~25 blocks the full ``_BBOX`` would.
_S2_BBOX = (86.90, 27.90, 86.94, 27.94)


def _to_utm45(bbox_4326):
    """Transform a ``(min_lon, min_lat, max_lon, max_lat)`` box to EPSG:32645 metres.

    Used by the projected-CRS tests to express an AOI in a projected space (UTM 45N)
    the way a consumer with a projected ``crs`` would.
    """
    from osgeo import osr

    def _srs(epsg):
        s = osr.SpatialReference()
        s.ImportFromEPSG(epsg)
        s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)  # x=lon, y=lat
        return s

    tx = osr.CoordinateTransformation(_srs(4326), _srs(32645))
    minx, miny, _ = tx.TransformPoint(bbox_4326[0], bbox_4326[1])
    maxx, maxy, _ = tx.TransformPoint(bbox_4326[2], bbox_4326[3])
    return (minx, miny, maxx, maxy)


def _synthetic_srtm(fill: int = 42):
    """Build a 200x200 EPSG:4326 Int16 raster over lon [86, 88], lat [27, 29].

    Args:
        fill: Constant value written to the single band.

    Returns:
        A north-up in-memory (``MEM``) GDAL dataset at 0.01-degree resolution,
        standing in for what the EEDAI driver would return.
    """
    from osgeo import gdal, osr

    src = gdal.GetDriverByName("MEM").Create("", 200, 200, 1, gdal.GDT_Int16)
    src.SetGeoTransform((86.0, 0.01, 0.0, 29.0, 0.0, -0.01))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src.SetProjection(srs.ExportToWkt())
    band = src.GetRasterBand(1)
    band.Fill(fill)
    band.SetNoDataValue(-32768)
    return src


@pytest.fixture
def patched_eedai(monkeypatch):
    """Replace the EEDAI open seam with the synthetic raster.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Yields:
        None. While active, ``_open_eedai`` returns a fresh synthetic raster and
        never touches the network.
    """

    def _fake_open(asset_id, *, bands, credentials, **_kwargs):  # noqa: ARG001
        return Dataset(_synthetic_srtm())

    monkeypatch.setattr(ee_reader, "_open_eedai", _fake_open)


class _FakeGdal:
    """Minimal stand-in for the ``gdal`` module used by ``_open_eedai``.

    Records the connection string and open options handed to ``OpenEx`` and
    returns a preconfigured result, so the driver seam can be tested offline.
    """

    OF_RASTER = 0x02
    OF_VERBOSE_ERROR = 0x40

    def __init__(self, open_result: object, *, last_error: str = "") -> None:
        self._open_result = open_result
        self._last_error = last_error
        self.calls: list[tuple[str, list[str]]] = []
        self.config: dict[str, str | None] = {}

    def OpenEx(self, conn, flags, open_options):  # noqa: N802, ARG002
        self.calls.append((conn, list(open_options)))
        return self._open_result

    def GetLastErrorMsg(self):  # noqa: N802
        return self._last_error

    def GetConfigOption(self, key, default=None):  # noqa: N802
        return self.config.get(key, default)

    def SetConfigOption(self, key, value):  # noqa: N802
        self.config[key] = value


class TestFromEarthengine:
    """Tests for the public :func:`from_earthengine` reader."""

    def test_windows_to_bbox(self, patched_eedai) -> None:
        """A bbox read returns a windowed pyramids ``Dataset``.

        Test scenario:
            A 0.1-degree bbox over a 0.01-degree source yields a ~10x10 EPSG:4326
            ``Dataset`` with no Earth Engine objects leaking out.
        """
        ds = from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX)
        assert isinstance(ds, Dataset), f"Expected a pyramids Dataset, got {type(ds)}"
        assert ds.epsg == 4326, f"Expected EPSG:4326, got {ds.epsg}"
        _bands, rows, cols = ds.shape
        assert rows == pytest.approx(10, abs=1), f"Expected ~10 rows, got {rows}"
        assert cols == pytest.approx(10, abs=1), f"Expected ~10 cols, got {cols}"

    def test_honours_shape(self, patched_eedai) -> None:
        """An explicit ``shape`` sets the exact output dimensions.

        Test scenario:
            ``shape=(5, 5)`` yields a single-band 5x5 dataset.
        """
        ds = from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, shape=(5, 5))
        assert ds.shape == (1, 5, 5), f"Expected (1, 5, 5), got {ds.shape}"

    def test_nodata_tags_returned_dataset_without_altering_pixels(
        self, patched_eedai
    ) -> None:
        """A ``nodata`` value is tagged on the result, pixels untouched (#63).

        Test scenario:
            The synthetic source is a constant ``42``; ``nodata=999`` (a value not in
            the data) is marked as no-data on the returned dataset while every pixel
            stays ``42`` — the sentinel is recognised, not written.
        """
        ds = from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, shape=(5, 5), nodata=999)
        assert ds.no_data_value[0] == 999, (
            f"Expected nodata 999, got {ds.no_data_value}"
        )
        assert np.all(np.asarray(ds.read_array()) == 42), (
            "nodata tagging altered pixels"
        )

    def test_nodata_default_leaves_source_value(self, patched_eedai) -> None:
        """Omitting ``nodata`` does not force a fill tag (#63).

        Test scenario:
            Without ``nodata`` the reader does not stamp 999; the result keeps whatever
            the read produced (never the caller's sentinel).
        """
        ds = from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, shape=(5, 5))
        assert ds.no_data_value[0] != 999

    def test_projected_crs_bbox_interpreted_in_projected_units(
        self, patched_eedai
    ) -> None:
        """A ``bbox`` under a projected ``crs`` reads that ground, in projected units (#66).

        Test scenario:
            A 4326 sub-window is transformed to EPSG:32645 (UTM 45N) metres and passed
            as ``bbox`` with ``crs="EPSG:32645"``; the output is delivered in
            EPSG:32645 with bounds matching the projected box (within a pixel), proving
            the coordinates were read as metres, not mistaken for lon/lat.
        """
        minx, miny, maxx, maxy = _to_utm45((86.92, 27.92, 86.98, 27.98))
        ds = from_earthengine(
            "USGS/SRTMGL1_003",
            crs="EPSG:32645",
            bbox=(minx, miny, maxx, maxy),
            shape=(10, 10),
        )
        assert ds.epsg == 32645, f"Expected EPSG:32645 output, got {ds.epsg}"
        out = ds.total_bounds
        assert out[0] == pytest.approx(minx, abs=1000), f"minx off: {out}"
        assert out[2] == pytest.approx(maxx, abs=1000), f"maxx off: {out}"
        assert out[1] == pytest.approx(miny, abs=1000), f"miny off: {out}"

    def test_projected_crs_geometry_interpreted_in_projected_units(
        self, patched_eedai
    ) -> None:
        """A ``geometry`` under a projected ``crs`` reads that ground (#66).

        Test scenario:
            A polygon built in EPSG:32645 over the same sub-window drives a projected
            read; the output is in EPSG:32645 and its bounds fall within the polygon's
            projected envelope, so a labelled geometry is honoured in its own CRS.
        """
        import geopandas as gpd
        from shapely.geometry import box

        minx, miny, maxx, maxy = _to_utm45((86.92, 27.92, 86.98, 27.98))
        gdf = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:32645")
        ds = from_earthengine(
            "USGS/SRTMGL1_003", crs="EPSG:32645", geometry=gdf, shape=(10, 10)
        )
        assert ds.epsg == 32645, f"Expected EPSG:32645 output, got {ds.epsg}"
        out = ds.total_bounds
        assert out[0] == pytest.approx(minx, abs=2000), f"minx off: {out}"
        assert out[2] == pytest.approx(maxx, abs=2000), f"maxx off: {out}"

    def test_honours_scale(self, patched_eedai) -> None:
        """An explicit ``scale`` sets the output pixel size in CRS units.

        Test scenario:
            ``scale=0.02`` over a 0.1-degree bbox yields ~5x5 pixels.
        """
        ds = from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, scale=0.02)
        _bands, rows, cols = ds.shape
        assert rows == pytest.approx(5, abs=1), (
            f"Expected ~5 rows at scale 0.02, got {rows}"
        )
        assert cols == pytest.approx(5, abs=1), (
            f"Expected ~5 cols at scale 0.02, got {cols}"
        )

    def test_scale_and_shape_mutually_exclusive(self, patched_eedai) -> None:
        """Passing both ``scale`` and ``shape`` raises ``ValueError``.

        Test scenario:
            The guard rejects the ambiguous combination before any read.
        """
        with pytest.raises(ValueError, match="scale.*shape") as exc_info:
            from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, scale=0.01, shape=(5, 5))
        assert "scale" in str(exc_info.value), f"Unexpected message: {exc_info.value}"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scale": 0.01},
            {"shape": (5, 5)},
            {"crs": "EPSG:3857"},
            {"resample": "cubic"},
        ],
    )
    def test_windowing_options_require_bbox(self, patched_eedai, kwargs) -> None:
        """Windowing options without a bbox raise ``ReaderError``.

        Args:
            kwargs: A single windowing option supplied without ``bbox``.

        Test scenario:
            ``scale`` / ``shape`` / non-default ``crs`` all demand a bbox because
            EE assets are global and cannot be materialised whole.
        """
        with pytest.raises(ReaderError, match="bbox") as exc_info:
            from_earthengine("USGS/SRTMGL1_003", **kwargs)
        assert "bbox" in str(exc_info.value), (
            f"Message should mention bbox: {exc_info.value}"
        )

    def test_no_bbox_wraps_full_asset(self, patched_eedai) -> None:
        """With no windowing options the full asset is wrapped lazily.

        Test scenario:
            No bbox/scale/shape and default CRS → the whole 200x200 asset is
            wrapped as a ``Dataset`` without warping.
        """
        ds = from_earthengine("USGS/SRTMGL1_003")
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert ds.shape == (1, 200, 200), f"Expected full (1, 200, 200), got {ds.shape}"

    def test_credentials_forwarded_to_open(self, monkeypatch, key_tmp) -> None:
        """Explicit credentials are coerced and passed to the open seam.

        Test scenario:
            A service-account path string is coerced and reaches ``_open_eedai``.
        """
        captured = {}

        def _fake_open(asset_id, *, bands, credentials, **_kwargs):  # noqa: ARG001
            captured["credentials"] = credentials
            return Dataset(_synthetic_srtm())

        monkeypatch.setattr(ee_reader, "_open_eedai", _fake_open)
        from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, credentials=str(key_tmp))
        assert isinstance(captured["credentials"], EarthEngineCredentials), (
            "credentials should be coerced to EarthEngineCredentials"
        )
        assert captured["credentials"].service_account_path == key_tmp, (
            f"Unexpected coerced path: {captured['credentials'].service_account_path}"
        )

    @pytest.mark.live
    def test_live_srtm(self) -> None:
        """End-to-end read of a public EE asset (needs ADC / service-account creds).

        Test scenario:
            The real EEDAI driver returns a non-empty EPSG:4326 window for SRTM.
        """
        ds = from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX)
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert ds.epsg == 4326, f"Expected EPSG:4326, got {ds.epsg}"
        _bands, rows, cols = ds.shape
        assert rows > 0, f"Expected a non-empty window, got {rows} rows"
        assert cols > 0, f"Expected a non-empty window, got {cols} cols"

    @pytest.mark.live
    def test_live_median_composite(self) -> None:
        """End-to-end median composite over a Sentinel-2 date range.

        Test scenario:
            The EEDA→EEDAI path plus a client-side median reducer yields a single
            aligned composite ``Dataset``.
        """
        ds = from_earthengine(
            "COPERNICUS/S2_SR_HARMONIZED",
            bbox=_BBOX,
            start="2024-06-01",
            end="2024-06-10",
            reducer="median",
            bands=["B4"],
            shape=(16, 16),
        )
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert ds.shape == (1, 16, 16), f"Expected (1, 16, 16), got {ds.shape}"


class TestCollectionFromEarthengineLive:
    """Live end-to-end test for :func:`collection_from_earthengine`."""

    @pytest.mark.live
    def test_live_collection(self) -> None:
        """End-to-end Sentinel-2 collection read over a date range.

        Test scenario:
            The EEDA catalog discovers scenes and each is read aligned into a
            ``DatasetCollection`` with one timestep per scene.
        """
        dc = collection_from_earthengine(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-10",
            bbox=_BBOX,
            bands=["B4"],
            shape=(16, 16),
        )
        assert isinstance(dc, DatasetCollection), (
            f"Expected a DatasetCollection, got {type(dc)}"
        )
        assert dc.time_length > 0, "Expected at least one scene in the window"
        assert dc.datasets[0].shape == (1, 16, 16), (
            f"Unexpected scene shape: {dc.datasets[0].shape}"
        )

    @pytest.mark.live
    def test_live_cost_estimate_from_catalog(self) -> None:
        """``estimate_earthengine_cost`` reports real scene dimensions (#61).

        Test scenario:
            A Sentinel-2 window's cost estimate is sourced from the EEDA catalog —
            24 bands, ~10980 px wide — without fetching any pixels.
        """
        cost = estimate_earthengine_cost(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-30",
            bbox=_S2_BBOX,
        )
        assert cost.scene_count > 0, "Expected at least one scene in the window"
        assert cost.max_band_count == 24, (
            f"Expected 24 bands, got {cost.max_band_count}"
        )
        assert cost.max_width >= 10000, (
            f"Expected a ~10980 px scene, got {cost.max_width}"
        )
        assert cost.min_pixel_size == 10.0, (
            f"Expected a 10 m finest band, got {cost.min_pixel_size}"
        )

    @pytest.mark.live
    def test_live_property_filter_narrows_selection(self) -> None:
        """A cloud-cover ``property_filter`` selects a subset of scenes (#62).

        Test scenario:
            Filtering ``CLOUDY_PIXEL_PERCENTAGE < 20`` never selects more scenes than
            the unfiltered window over the same date range + AOI.
        """
        window = dict(
            asset_id="COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-30",
            bbox=_S2_BBOX,
        )
        all_scenes = estimate_earthengine_cost(**window).scene_count
        clear = estimate_earthengine_cost(
            **window, property_filter="CLOUDY_PIXEL_PERCENTAGE < 20"
        ).scene_count
        assert clear <= all_scenes, (
            f"Filtered ({clear}) should not exceed unfiltered ({all_scenes})"
        )


@pytest.fixture
def key_tmp(tmp_path):
    """Provide a dummy service-account key path.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Path to a readable JSON file used as a stand-in service-account key.
    """
    key = tmp_path / "key.json"
    key.write_text("{}", encoding="utf-8")
    return key


class TestOpenEedai:
    """Tests for the private EEDAI open seam :func:`_open_eedai`."""

    def test_builds_connection_and_bands_option(self, monkeypatch) -> None:
        """The asset id becomes an ``EEDAI:`` URI with a ``BANDS`` open option.

        Test scenario:
            ``bands=["B4", "B3"]`` produces open option ``BANDS=B4,B3`` on the
            ``EEDAI:<asset>`` connection string.
        """
        opened = _synthetic_srtm()
        fake = _FakeGdal(opened)
        monkeypatch.setattr(ee_reader, "gdal", fake)
        creds = EarthEngineCredentials.application_default()

        result = ee_reader._open_eedai(
            "USGS/SRTMGL1_003", bands=["B4", "B3"], credentials=creds
        )

        assert isinstance(result, Dataset), (
            f"Should return a Dataset, got {type(result)}"
        )
        assert result.raster is opened, "Dataset should wrap the driver's open result"
        conn, options = fake.calls[0]
        assert conn == "EEDAI:USGS/SRTMGL1_003", f"Unexpected connection string: {conn}"
        assert options == [
            "BLOCK_SIZE=256",
            "PIXEL_ENCODING=GEO_TIFF",
            "BANDS=B4,B3",
        ], f"Unexpected open options: {options}"

    def test_no_bands_option_when_bands_none(self, monkeypatch) -> None:
        """No ``BANDS`` option is emitted when ``bands`` is ``None``.

        Test scenario:
            ``bands=None`` opens with only the pinned ``BLOCK_SIZE`` and lossless
            ``PIXEL_ENCODING`` options.
        """
        fake = _FakeGdal(_synthetic_srtm())
        monkeypatch.setattr(ee_reader, "gdal", fake)
        ee_reader._open_eedai(
            "USGS/SRTMGL1_003",
            bands=None,
            credentials=EarthEngineCredentials.application_default(),
        )
        _conn, options = fake.calls[0]
        assert options == ["BLOCK_SIZE=256", "PIXEL_ENCODING=GEO_TIFF"], (
            f"Unexpected open options: {options}"
        )

    def test_block_size_threads_to_open_option(self, monkeypatch) -> None:
        """A caller ``block_size`` sets the EEDAI ``BLOCK_SIZE`` open option (#60).

        Test scenario:
            ``block_size=1024`` opens with ``BLOCK_SIZE=1024`` instead of the default.
        """
        fake = _FakeGdal(_synthetic_srtm())
        monkeypatch.setattr(ee_reader, "gdal", fake)
        ee_reader._open_eedai(
            "USGS/SRTMGL1_003",
            bands=None,
            credentials=EarthEngineCredentials.application_default(),
            block_size=1024,
        )
        _conn, options = fake.calls[0]
        assert "BLOCK_SIZE=1024" in options, f"block_size not threaded: {options}"

    def test_block_size_rejects_non_positive(self) -> None:
        """A non-positive ``block_size`` raises ``ValueError`` (#60).

        Test scenario:
            ``block_size=0`` is rejected before any open.
        """
        with pytest.raises(ValueError, match="block_size"):
            ee_reader._open_eedai(
                "USGS/SRTMGL1_003",
                bands=None,
                credentials=EarthEngineCredentials.application_default(),
                block_size=0,
            )

    def test_pins_lossless_pixel_encoding(self, monkeypatch) -> None:
        """Every EEDAI open pins a lossless ``PIXEL_ENCODING`` (regression for #69).

        Test scenario:
            The driver's ``AUTO`` default can pick a lossy codec for multi-band Byte
            reads; ``_open_eedai`` must pin ``GEO_TIFF`` so pixels are never lossy.
        """
        fake = _FakeGdal(_synthetic_srtm())
        monkeypatch.setattr(ee_reader, "gdal", fake)
        ee_reader._open_eedai(
            "USGS/SRTMGL1_003",
            bands=["TCI_R", "TCI_G", "TCI_B"],
            credentials=EarthEngineCredentials.application_default(),
        )
        _conn, options = fake.calls[0]
        assert "PIXEL_ENCODING=GEO_TIFF" in options, (
            f"lossless PIXEL_ENCODING must be pinned; got {options}"
        )

    def test_raises_reader_error_on_open_failure(self, monkeypatch) -> None:
        """A ``None`` driver result raises ``ReaderError`` with the GDAL message.

        Test scenario:
            ``OpenEx`` returning ``None`` surfaces as ``ReaderError`` including
            the last GDAL error text.
        """
        fake = _FakeGdal(None, last_error="permission denied")
        monkeypatch.setattr(ee_reader, "gdal", fake)
        creds = EarthEngineCredentials.application_default()
        with pytest.raises(ReaderError, match="permission denied") as exc_info:
            ee_reader._open_eedai("USGS/SRTMGL1_003", bands=None, credentials=creds)
        assert "USGS/SRTMGL1_003" in str(exc_info.value), (
            f"Error should name the asset, got: {exc_info.value}"
        )


class TestWindow:
    """Tests for the private windowing helper :func:`_window`."""

    def test_reader_error_when_warp_returns_none(self, monkeypatch) -> None:
        """A failed warp (``None``) raises ``ReaderError``.

        Test scenario:
            After the native window is materialised, ``gdal.Warp`` returning
            ``None`` surfaces as a ``ReaderError`` naming the windowing target.
        """
        source = Dataset(_synthetic_srtm())
        monkeypatch.setattr(ee_reader.gdal, "Warp", lambda dest, src, **kw: None)
        with pytest.raises(ReaderError, match="windowing"):
            ee_reader._window(
                source, bbox=_BBOX, crs="EPSG:4326", scale=None, shape=None
            )


class _FakeFeature:
    """Minimal OGR feature stand-in exposing ``GetFieldAsString``."""

    def __init__(self, fields: dict[str, str]) -> None:
        self._fields = fields

    def GetFieldAsString(self, key):  # noqa: N802
        return self._fields.get(key, "")


class _FakeLayer:
    """Minimal OGR layer stand-in that records the filters it was given."""

    def __init__(self, features: list[_FakeFeature]) -> None:
        self._features = features
        self.attribute_filter: str | None = None
        self.spatial_rect: tuple | None = None

    def SetAttributeFilter(self, expr):  # noqa: N802
        self.attribute_filter = expr

    def SetSpatialFilterRect(self, *rect):  # noqa: N802
        self.spatial_rect = rect

    def __iter__(self):
        return iter(self._features)


class _FakeEeda:
    """Stand-in for ``gdal`` used by ``_discover_scenes`` (EEDA catalog open)."""

    OF_VECTOR = 0x04
    OF_VERBOSE_ERROR = 0x40

    def __init__(self, layer: _FakeLayer | None, *, last_error: str = "") -> None:
        self._layer = layer
        self._last_error = last_error
        self.open_calls: list[tuple[str, list[str]]] = []

    def OpenEx(self, conn, flags, open_options):  # noqa: N802, ARG002
        self.open_calls.append((conn, list(open_options)))
        if self._layer is None:
            return None

        class _Catalog:
            def __init__(self, layer):
                self._layer = layer

            def GetLayer(self, index):  # noqa: N802, ARG002
                return self._layer

        return _Catalog(self._layer)

    def GetLastErrorMsg(self):  # noqa: N802
        return self._last_error


@pytest.fixture
def three_scenes(monkeypatch):
    """Patch scene discovery to return three fixed scenes and stub the raster open.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Returns:
        The fill values used for the three synthetic scenes, in order.
    """
    scenes = [
        _Scene("EEDAI:scene/a", "2024-06-01T00:00:00"),
        _Scene("EEDAI:scene/b", "2024-06-02T00:00:00"),
        _Scene("EEDAI:scene/c", "2024-06-03T00:00:00"),
    ]
    fills = {"EEDAI:scene/a": 10, "EEDAI:scene/b": 20, "EEDAI:scene/c": 30}

    def _fake_discover(asset_id, *, start, end, bbox_4326, credentials, **_kwargs):  # noqa: ARG001
        return scenes

    def _fake_open(connection, *, bands, credentials, **_kwargs):  # noqa: ARG001
        return Dataset(_synthetic_srtm(fill=fills[connection]))

    monkeypatch.setattr(ee_reader, "_discover_scenes", _fake_discover)
    monkeypatch.setattr(ee_reader, "_open_eedai", _fake_open)
    return [10, 20, 30]


class TestFromEarthengineComposite:
    """Tests for the ImageCollection composite mode of :func:`from_earthengine`."""

    @pytest.mark.parametrize(
        "reducer, expected",
        [
            ("median", 20),
            ("mean", 20),
            ("min", 10),
            ("max", 30),
            ("sum", 60),
            ("mode", 10),
            ("mosaic", 10),
        ],
    )
    def test_reducer_composites_scene_stack(
        self, three_scenes, reducer, expected
    ) -> None:
        """Each reducer collapses the aligned scene stack to a single value.

        Args:
            three_scenes: Fixture patching discovery/open (fills 10, 20, 30).
            reducer: The reducer under test.
            expected: The value every output pixel should take.

        Test scenario:
            Three constant scenes (10/20/30) reduce to a single-band composite of
            the reducer's value across the whole window.
        """
        ds = from_earthengine(
            "COPERNICUS/S2_SR_HARMONIZED",
            bbox=_BBOX,
            start="2024-06-01",
            end="2024-06-30",
            reducer=reducer,
            shape=(4, 4),
        )
        assert ds.shape == (1, 4, 4), f"Expected (1, 4, 4), got {ds.shape}"
        values = ds.read_array()
        assert (values == expected).all(), (
            f"Expected all {expected}, got {values.tolist()}"
        )

    def test_composite_writes_to_path(self, three_scenes, tmp_path) -> None:
        """A ``path`` in composite mode writes the composite and returns it file-backed.

        Args:
            three_scenes: Fixture patching discovery/open (fills 10, 20, 30).
            tmp_path: pytest temp directory.

        Test scenario:
            A median composite with ``path`` writes the raster and the returned
            Dataset reads it, matching the in-memory composite.
        """
        out = tmp_path / "composite.tif"
        ds = from_earthengine(
            "COPERNICUS/S2_SR_HARMONIZED",
            bbox=_BBOX,
            start="2024-06-01",
            end="2024-06-30",
            reducer="median",
            shape=(4, 4),
            path=str(out),
        )
        assert out.exists(), "composite mode must honour 'path' and write the file"
        assert (ds.read_array() == 20).all(), "file-backed composite has wrong values"

    def test_start_end_without_reducer_raises(self) -> None:
        """A date range without a reducer is rejected with guidance.

        Test scenario:
            ``start``/``end`` but no ``reducer`` raises ``ValueError`` pointing to
            ``collection_from_earthengine``.
        """
        with pytest.raises(ValueError, match="reducer") as exc_info:
            from_earthengine(
                "COPERNICUS/S2_SR_HARMONIZED",
                bbox=_BBOX,
                start="2024-06-01",
                end="2024-06-30",
            )
        assert "collection_from_earthengine" in str(exc_info.value), (
            f"Message should suggest the collection reader: {exc_info.value}"
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"reducer": "median", "start": "2024-06-01", "end": "2024-06-30"},
            {"reducer": "median", "bbox": _BBOX, "end": "2024-06-30"},
            {"reducer": "median", "bbox": _BBOX, "start": "2024-06-01"},
        ],
    )
    def test_composite_requires_start_end_bbox(self, kwargs) -> None:
        """The composite mode demands ``start``, ``end`` and ``bbox``.

        Args:
            kwargs: A composite invocation missing one required argument.

        Test scenario:
            Any missing member of the trio raises ``ValueError``.
        """
        with pytest.raises(ValueError, match="requires 'start', 'end', and"):
            from_earthengine("COPERNICUS/S2_SR_HARMONIZED", **kwargs)

    def test_no_scenes_raises_reader_error(self, monkeypatch) -> None:
        """An empty discovery result raises ``ReaderError``.

        Test scenario:
            No scenes in the window -> ``ReaderError`` naming the asset.
        """
        monkeypatch.setattr(ee_reader, "_discover_scenes", lambda *a, **k: [])
        with pytest.raises(ReaderError, match="No Earth Engine scenes"):
            from_earthengine(
                "COPERNICUS/S2_SR_HARMONIZED",
                bbox=_BBOX,
                start="2024-06-01",
                end="2024-06-30",
                reducer="median",
            )


class TestCollectionFromEarthengine:
    """Tests for :func:`collection_from_earthengine`."""

    def test_returns_dataset_collection(self, three_scenes) -> None:
        """Discovery + aligned reads produce a per-scene ``DatasetCollection``.

        Args:
            three_scenes: Fixture patching discovery/open (three scenes).

        Test scenario:
            Three scenes become a 3-timestep collection with aligned shapes and
            the acquisition times as the time axis.
        """
        dc = collection_from_earthengine(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-30",
            bbox=_BBOX,
            shape=(4, 4),
        )
        assert isinstance(dc, DatasetCollection), (
            f"Expected a DatasetCollection, got {type(dc)}"
        )
        assert dc.time_length == 3, f"Expected 3 timesteps, got {dc.time_length}"
        assert dc.datasets[0].shape == (1, 4, 4), (
            f"Unexpected scene shape: {dc.datasets[0].shape}"
        )
        assert list(dc.time) == [
            "2024-06-01T00:00:00",
            "2024-06-02T00:00:00",
            "2024-06-03T00:00:00",
        ], f"Unexpected time axis: {list(dc.time)}"

    def test_scale_and_shape_mutually_exclusive(self) -> None:
        """Passing both ``scale`` and ``shape`` raises ``ValueError``.

        Test scenario:
            The guard rejects the ambiguous combination before any read.
        """
        with pytest.raises(ValueError, match="scale.*shape"):
            collection_from_earthengine(
                "COPERNICUS/S2_SR_HARMONIZED",
                start="2024-06-01",
                end="2024-06-30",
                bbox=_BBOX,
                scale=0.01,
                shape=(4, 4),
            )

    def test_no_scenes_raises_reader_error(self, monkeypatch) -> None:
        """An empty discovery result raises ``ReaderError``.

        Test scenario:
            No scenes in the window -> ``ReaderError``.
        """
        monkeypatch.setattr(ee_reader, "_discover_scenes", lambda *a, **k: [])
        with pytest.raises(ReaderError, match="No Earth Engine scenes"):
            collection_from_earthengine(
                "COPERNICUS/S2_SR_HARMONIZED",
                start="2024-06-01",
                end="2024-06-30",
                bbox=_BBOX,
            )

    def test_native_grid_alignment_without_scale_or_shape(self, three_scenes) -> None:
        """With neither ``scale`` nor ``shape`` all scenes share the first's grid.

        Test scenario:
            The first scene's native windowed size fixes the grid, so every
            timestep has identical dimensions.
        """
        dc = collection_from_earthengine(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-30",
            bbox=_BBOX,
        )
        shapes = {ds.shape for ds in dc.datasets}
        assert len(shapes) == 1, f"Scenes are not aligned to one grid: {shapes}"


class TestDiscoverScenes:
    """Tests for the private EEDA discovery seam :func:`_discover_scenes`."""

    def test_builds_filters_and_sorts_scenes(self, monkeypatch) -> None:
        """Discovery sets time/space filters and returns time-sorted scenes.

        Test scenario:
            Out-of-order features come back sorted by time; the attribute filter
            carries ISO datetimes and the spatial rect matches the bbox.
        """
        features = [
            _FakeFeature(
                {"gdal_dataset": "EEDAI:b", "startTime": "2024-06-02T00:00:00"}
            ),
            _FakeFeature(
                {"gdal_dataset": "EEDAI:a", "startTime": "2024-06-01T00:00:00"}
            ),
            _FakeFeature({"gdal_dataset": "", "startTime": "2024-06-03T00:00:00"}),
        ]
        layer = _FakeLayer(features)
        fake = _FakeEeda(layer)
        monkeypatch.setattr(ee_reader, "gdal", fake)

        scenes = ee_reader._discover_scenes(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-30",
            bbox_4326=(86.9, 27.9, 87.0, 28.0),
            credentials=EarthEngineCredentials.application_default(),
        )

        assert [s.connection for s in scenes] == ["EEDAI:a", "EEDAI:b"], (
            f"Scenes should be sorted by time and skip empty connections: {scenes}"
        )
        assert fake.open_calls[0] == (
            "EEDA:",
            ["COLLECTION=COPERNICUS/S2_SR_HARMONIZED"],
        ), f"Unexpected EEDA open call: {fake.open_calls}"
        assert "2024-06-01T00:00:00" in layer.attribute_filter, (
            f"Attribute filter missing start: {layer.attribute_filter}"
        )
        # Filters on acquisition time; a bare end date bounds startTime by the
        # next-day-exclusive midnight so scenes acquired that day are kept (M3/L3).
        assert "startTime < '2024-07-01T00:00:00'" in layer.attribute_filter, (
            f"Filter should bound startTime by next-day midnight: {layer.attribute_filter}"
        )
        assert "endTime" not in layer.attribute_filter, (
            f"Filter should not bound by endTime: {layer.attribute_filter}"
        )
        assert layer.spatial_rect == (86.9, 27.9, 87.0, 28.0), (
            f"Unexpected spatial filter: {layer.spatial_rect}"
        )

    def test_reader_error_when_catalog_open_fails(self, monkeypatch) -> None:
        """A ``None`` catalog open raises ``ReaderError``.

        Test scenario:
            ``OpenEx`` returning ``None`` surfaces the GDAL error as ``ReaderError``.
        """
        monkeypatch.setattr(ee_reader, "gdal", _FakeEeda(None, last_error="no access"))
        creds = EarthEngineCredentials.application_default()
        with pytest.raises(ReaderError, match="no access"):
            ee_reader._discover_scenes(
                "COPERNICUS/S2_SR_HARMONIZED",
                start="2024-06-01",
                end="2024-06-30",
                bbox_4326=(0.0, 0.0, 1.0, 1.0),
                credentials=creds,
            )

    def test_property_filter_ands_into_attribute_filter(self, monkeypatch) -> None:
        """A ``property_filter`` is ANDed onto the time selection (#62).

        Test scenario:
            A cloud-cover filter appears as a parenthesised ``AND`` clause alongside
            the ``startTime`` bounds.
        """
        layer = _FakeLayer(
            [
                _FakeFeature(
                    {"gdal_dataset": "EEDAI:a", "startTime": "2024-06-01T00:00:00"}
                )
            ]
        )
        monkeypatch.setattr(ee_reader, "gdal", _FakeEeda(layer))
        ee_reader._discover_scenes(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-30",
            bbox_4326=(0.0, 0.0, 1.0, 1.0),
            credentials=EarthEngineCredentials.application_default(),
            property_filter="CLOUDY_PIXEL_PERCENTAGE < 20",
        )
        assert "startTime >=" in layer.attribute_filter
        assert "AND (CLOUDY_PIXEL_PERCENTAGE < 20)" in layer.attribute_filter, (
            f"property_filter not ANDed on: {layer.attribute_filter}"
        )

    def test_scene_carries_cost_metadata(self, monkeypatch) -> None:
        """Scene records carry the EEDA cost fields (#61).

        Test scenario:
            ``band_count`` / ``band_max_width`` / ``sizeBytes`` fields on the feature
            surface on the returned ``_Scene``; a blank field coerces to zero.
        """
        layer = _FakeLayer(
            [
                _FakeFeature(
                    {
                        "gdal_dataset": "EEDAI:a",
                        "startTime": "2024-06-01T00:00:00",
                        "band_count": "24",
                        "band_max_width": "10980",
                        "band_max_height": "10980",
                        "band_min_pixel_size": "10",
                        "band_crs": "EPSG:32645",
                        "sizeBytes": "123456",
                    }
                )
            ]
        )
        monkeypatch.setattr(ee_reader, "gdal", _FakeEeda(layer))
        (scene,) = ee_reader._discover_scenes(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-30",
            bbox_4326=(0.0, 0.0, 1.0, 1.0),
            credentials=EarthEngineCredentials.application_default(),
        )
        assert scene.band_count == 24
        assert scene.width == 10980
        assert scene.pixel_size == 10.0
        assert scene.crs == "EPSG:32645"
        assert scene.size_bytes == 123456

    def test_estimate_cost_aggregates_scene_metadata(self, monkeypatch) -> None:
        """``estimate_earthengine_cost`` aggregates the scene records (#61).

        Test scenario:
            Two scenes of differing size aggregate into scene count, total bytes and
            the per-field maxima — with no pixel fetch (only ``_discover_scenes`` runs).
        """
        scenes = [
            ee_reader._Scene(
                "EEDAI:a", "2024-06-01T00:00:00", 24, 10980, 10980, 10.0, "", 100
            ),
            ee_reader._Scene(
                "EEDAI:b", "2024-06-02T00:00:00", 13, 5490, 5490, 20.0, "", 50
            ),
        ]
        monkeypatch.setattr(ee_reader, "_discover_scenes", lambda *a, **k: scenes)
        cost = ee_reader.estimate_earthengine_cost(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-30",
            bbox=(0.0, 0.0, 1.0, 1.0),
            credentials=EarthEngineCredentials.application_default(),
        )
        assert cost.scene_count == 2
        assert cost.total_size_bytes == 150
        assert cost.max_width == 10980
        assert cost.max_band_count == 24
        assert cost.min_pixel_size == 10.0
        assert [s.connection for s in cost.scenes] == ["EEDAI:a", "EEDAI:b"]

    def test_estimate_cost_requires_aoi(self) -> None:
        """``estimate_earthengine_cost`` needs a ``bbox`` or ``geometry`` (#61)."""
        with pytest.raises(ValueError, match="bbox.*geometry|geometry"):
            ee_reader.estimate_earthengine_cost(
                "COPERNICUS/S2_SR_HARMONIZED",
                start="2024-06-01",
                end="2024-06-30",
                credentials=EarthEngineCredentials.application_default(),
            )


class TestHelpers:
    """Tests for the private reducer / geometry helpers."""

    def test_iso_appends_time_component(self) -> None:
        """A bare date gains a midnight time; a datetime is left alone.

        Test scenario:
            ``_iso`` normalises dates for lexical catalog comparison.
        """
        assert ee_reader._iso("2024-06-01") == "2024-06-01T00:00:00", (
            "date should gain a time"
        )
        assert ee_reader._iso("2024-06-01T12:00:00") == "2024-06-01T12:00:00", (
            "datetime unchanged"
        )

    def test_end_clause_next_day_exclusive_for_bare_date(self) -> None:
        """A bare end date bounds startTime by next-day-exclusive midnight.

        Test scenario:
            ``_end_clause`` avoids the sub-millisecond lexical edge; an explicit
            datetime end stays an inclusive ``<=`` bound.
        """
        assert (
            ee_reader._end_clause("2024-06-30") == "startTime < '2024-07-01T00:00:00'"
        ), "bare end date should bound by next-day midnight (exclusive)"
        assert (
            ee_reader._end_clause("2024-06-30T12:00:00")
            == "startTime <= '2024-06-30T12:00:00'"
        ), "explicit end datetime should be an inclusive bound"

    def test_bbox_to_4326_passthrough(self) -> None:
        """A lon/lat bbox is returned unchanged.

        Test scenario:
            ``EPSG:4326`` input needs no transform.
        """
        bbox = (86.9, 27.9, 87.0, 28.0)
        assert ee_reader._bbox_to_4326(bbox, "EPSG:4326") == bbox, (
            "4326 bbox should pass through"
        )

    def test_bbox_to_4326_reprojects(self) -> None:
        """A Web-Mercator bbox is transformed to a lon/lat envelope.

        Test scenario:
            A metric ``EPSG:3857`` box near the equator maps to small lon/lat values.
        """
        out = ee_reader._bbox_to_4326((0.0, 0.0, 111319.49, 111325.14), "EPSG:3857")
        assert out[0] == pytest.approx(0.0, abs=1e-6), (
            f"min lon should be ~0, got {out[0]}"
        )
        assert out[2] == pytest.approx(1.0, abs=1e-3), (
            f"max lon should be ~1, got {out[2]}"
        )

    def test_reduce_unknown_reducer_raises(self) -> None:
        """An unknown reducer name is rejected.

        Test scenario:
            ``_reduce`` raises ``ValueError`` for an unsupported reducer.
        """
        stack = np.zeros((3, 2, 2), dtype="int16")
        with pytest.raises(ValueError, match="Unknown reducer"):
            ee_reader._reduce(stack, "bogus", None)

    def test_reduce_without_nodata(self) -> None:
        """Reduction with no nodata uses the plain NumPy path.

        Test scenario:
            The median across the scene axis is computed for every pixel.
        """
        stack = np.stack(
            [np.full((2, 2), v, dtype="int16") for v in (10, 20, 30)], axis=0
        )
        reduced = ee_reader._reduce(stack, "median", None)
        assert (reduced == 20).all(), f"Expected all 20, got {reduced.tolist()}"

    def test_reduce_masks_nodata(self) -> None:
        """Nodata pixels are masked out before reducing.

        Test scenario:
            A nodata value in one scene is ignored, so the mean skips it.
        """
        stack = np.stack(
            [
                np.array([[10, -1], [10, 10]], dtype="int16"),
                np.array([[20, 20], [20, 20]], dtype="int16"),
                np.array([[30, 30], [30, 30]], dtype="int16"),
            ],
            axis=0,
        )
        reduced = ee_reader._reduce(stack, "mean", nodata=-1)
        assert reduced[0, 1] == 25, (
            f"Masked mean of 20,30 should be 25, got {reduced[0, 1]}"
        )
        assert reduced[0, 0] == 20, (
            f"Mean of 10,20,30 should be 20, got {reduced[0, 0]}"
        )


class TestGeometryClip:
    """Tests for the polygon-cutline (`geometry`) support."""

    @staticmethod
    def _triangle():
        """Half-of-the-bbox triangle GeoDataFrame in EPSG:4326."""
        import geopandas as gpd
        from shapely.geometry import Polygon

        return gpd.GeoDataFrame(
            geometry=[Polygon([(86.9, 27.9), (87.0, 27.9), (86.9, 28.0)])],
            crs="EPSG:4326",
        )

    def test_from_earthengine_clips_to_polygon(self, patched_eedai) -> None:
        """A polygon cutline masks cells outside the polygon.

        Test scenario:
            A triangle covering ~half the bbox leaves roughly half the window
            masked with the nodata value.
        """
        ds = from_earthengine(
            "USGS/SRTMGL1_003", bbox=_BBOX, geometry=self._triangle(), shape=(10, 10)
        )
        arr = ds.read_array()
        nodata = ds.no_data_value[0]
        masked = int((arr == nodata).sum())
        assert masked > 0, "Expected some cells masked outside the polygon"
        assert masked < arr.size, "Expected some cells kept inside the polygon"

    def test_geometry_without_bbox_derives_window(self, patched_eedai) -> None:
        """A geometry with no bbox uses the geometry's envelope as the window.

        Test scenario:
            Passing only `geometry` yields a Dataset (window derived from bounds).
        """
        ds = from_earthengine(
            "USGS/SRTMGL1_003", geometry=self._triangle(), shape=(8, 8)
        )
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"

    def test_geometry_bounds_requires_total_bounds(self) -> None:
        """A geometry lacking `total_bounds` raises `ReaderError`.

        Test scenario:
            An object with no `total_bounds` cannot yield a window.
        """
        no_bounds = object()
        with pytest.raises(ReaderError, match="total_bounds"):
            ee_reader._geometry_bounds(no_bounds)

    def test_composite_clips_to_polygon(self, three_scenes) -> None:
        """The composite mode clips its output to the polygon.

        Args:
            three_scenes: Fixture patching discovery/open.

        Test scenario:
            A triangle geometry masks part of the median composite.
        """
        ds = from_earthengine(
            "COPERNICUS/S2_SR_HARMONIZED",
            bbox=_BBOX,
            geometry=self._triangle(),
            start="2024-06-01",
            end="2024-06-30",
            reducer="median",
            shape=(10, 10),
        )
        arr = ds.read_array()
        assert int((arr == ds.no_data_value[0]).sum()) > 0, "Composite not clipped"

    def test_collection_clips_each_scene(self, three_scenes) -> None:
        """Each collection scene is clipped to the polygon.

        Args:
            three_scenes: Fixture patching discovery/open.

        Test scenario:
            Every timestep has masked cells from the triangle cutline.
        """
        dc = collection_from_earthengine(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-01",
            end="2024-06-30",
            geometry=self._triangle(),
            shape=(10, 10),
        )
        for ds in dc.datasets:
            arr = ds.read_array()
            assert int((arr == ds.no_data_value[0]).sum()) > 0, "Scene not clipped"

    def test_collection_requires_bbox_or_geometry(self) -> None:
        """The collection reader requires a bbox or a geometry.

        Test scenario:
            Neither given → ValueError before any read.
        """
        with pytest.raises(ValueError, match="'bbox' or a 'geometry'"):
            collection_from_earthengine(
                "COPERNICUS/S2_SR_HARMONIZED", start="2024-06-01", end="2024-06-30"
            )


class TestSpecialReducers:
    """Tests for the ``mode`` / ``mosaic`` reducers (with and without nodata)."""

    def test_mosaic_takes_first_valid(self) -> None:
        """Mosaic returns the first non-nodata value down the scene axis.

        Test scenario:
            A masked pixel in the first scene falls through to the next scene.
        """
        stack = np.stack(
            [
                np.array([[10, -1]], dtype="int16"),
                np.array([[20, 20]], dtype="int16"),
                np.array([[30, 30]], dtype="int16"),
            ]
        )
        out = ee_reader._reduce(stack, "mosaic", nodata=-1)
        assert out.tolist() == [[10, 20]], f"Unexpected mosaic: {out.tolist()}"

    def test_mosaic_without_nodata_is_first_scene(self) -> None:
        """With no nodata, mosaic is simply the first scene.

        Test scenario:
            The plain path returns scene 0 unchanged.
        """
        stack = np.stack([np.full((2, 2), v, dtype="int16") for v in (7, 8, 9)])
        out = ee_reader._reduce(stack, "mosaic", nodata=None)
        assert (out == 7).all(), f"Expected all 7, got {out.tolist()}"

    def test_mode_ties_resolve_to_smallest(self) -> None:
        """Mode returns the most frequent value; ties pick the smallest.

        Test scenario:
            A 2-2 tie between 10 and 20 resolves to 10.
        """
        stack = np.stack([np.full((1, 1), v, dtype="int16") for v in (10, 10, 20, 20)])
        out = ee_reader._reduce(stack, "mode", nodata=None)
        assert int(out[0, 0]) == 10, f"Tie should pick smallest, got {out[0, 0]}"

    def test_mode_ignores_nodata(self) -> None:
        """Mode ignores masked values per pixel.

        Test scenario:
            A nodata sample in one scene does not become the mode.
        """
        stack = np.stack(
            [
                np.array([[5]], dtype="int16"),
                np.array([[-1]], dtype="int16"),
                np.array([[5]], dtype="int16"),
            ]
        )
        out = ee_reader._reduce(stack, "mode", nodata=-1)
        assert int(out[0, 0]) == 5, f"Mode should ignore nodata, got {out[0, 0]}"


class TestModeAllNodata:
    """Edge case: a pixel that is nodata in every scene."""

    def test_mode_all_nodata_returns_nodata(self) -> None:
        """Mode returns nodata when a pixel has no valid samples.

        Test scenario:
            Every scene is nodata at the pixel → the mode is nodata.
        """
        stack = np.stack([np.array([[-1]], dtype="int16")] * 3)
        out = ee_reader._reduce(stack, "mode", nodata=-1)
        assert int(out[0, 0]) == -1, (
            f"All-nodata pixel should stay nodata, got {out[0, 0]}"
        )


class TestCredentialLifetime:
    """Inline-JSON credentials must outlive the transient EarthEngineCredentials."""

    def test_inline_credentials_pinned_to_dataset(self, monkeypatch) -> None:
        """The returned Dataset keeps the inline key file alive (H1).

        Test scenario:
            After `from_earthengine(..., credentials={...})` returns and GC runs,
            the temp key file still exists while the Dataset is alive, and is
            cleaned up only once the Dataset is dropped.
        """
        import gc

        monkeypatch.setattr(
            ee_reader,
            "_open_eedai",
            lambda a, *, bands, credentials, **_kw: Dataset(_synthetic_srtm()),
        )
        ds = from_earthengine(
            "USGS/SRTMGL1_003", bbox=_BBOX, credentials={"type": "service_account"}
        )
        path = ds._ee_credentials.service_account_path
        gc.collect()
        assert path.is_file(), "Temp key file deleted while the Dataset is alive"
        del ds
        gc.collect()
        assert not path.exists(), (
            "Temp key file not cleaned up after the Dataset is gone"
        )


class TestReducerDtype:
    """Reducer dtype policy: no integer overflow (sum) or truncation (mean/median)."""

    def test_sum_does_not_overflow_int16(self) -> None:
        """The sum reducer widens instead of wrapping int16.

        Test scenario:
            Three int16 pixels of 20000 sum to 60000 (not the wrapped -5536).
        """
        stack = np.stack([np.full((1, 1), 20000, dtype="int16")] * 3)
        out = ee_reader._reduce(stack, "sum", None)
        assert float(out[0, 0]) == 60000.0, f"sum overflowed: {out[0, 0]}"

    def test_mean_is_not_truncated_on_ints(self) -> None:
        """The mean reducer keeps the fractional result on integer stacks.

        Test scenario:
            mean(10, 11) is 10.5, not truncated to 10.
        """
        stack = np.stack([np.full((1, 1), v, dtype="int16") for v in (10, 11)])
        out = ee_reader._reduce(stack, "mean", None)
        assert float(out[0, 0]) == 10.5, f"mean truncated: {out[0, 0]}"

    def test_value_preserving_reducers_keep_int_dtype(self) -> None:
        """min/max/mode/mosaic keep the integer stack dtype.

        Test scenario:
            An int16 stack stays int16 through a value-preserving reducer.
        """
        stack = np.stack([np.full((1, 1), v, dtype="int16") for v in (10, 20, 30)])
        assert ee_reader._reduce(stack, "max", None).dtype == np.int16, (
            "max changed dtype"
        )

    def test_lazy_wrap_installs_credential_config(self, monkeypatch, tmp_path) -> None:
        """The lazy whole-asset wrap installs the credential GDAL config (M2).

        Test scenario:
            A no-bbox read with a service-account key sets
            GOOGLE_APPLICATION_CREDENTIALS process-wide for the deferred reads.
        """
        from osgeo import gdal

        key = tmp_path / "k.json"
        key.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            ee_reader,
            "_open_eedai",
            lambda a, *, bands, credentials, **_kw: Dataset(_synthetic_srtm()),
        )
        before = gdal.GetConfigOption("GOOGLE_APPLICATION_CREDENTIALS", None)
        try:
            from_earthengine("USGS/SRTMGL1_003", credentials=str(key))
            assert gdal.GetConfigOption("GOOGLE_APPLICATION_CREDENTIALS", None) == str(
                key
            ), "Lazy wrap should install the credential config for deferred reads"
        finally:
            gdal.SetConfigOption("GOOGLE_APPLICATION_CREDENTIALS", before)


class TestGeometryCrs:
    """Tests for reconciling a geometry's CRS with the reader's `crs` (M4)."""

    def test_geometry_reprojected_to_crs(self, patched_eedai) -> None:
        """A geometry in another CRS is reprojected before deriving the window.

        Test scenario:
            A Web-Mercator triangle with default crs=EPSG:4326 yields a lon/lat
            window (small degree values), not raw metre bounds.
        """
        import geopandas as gpd
        from shapely.geometry import Polygon

        tri_4326 = Polygon([(86.9, 27.9), (87.0, 27.9), (86.9, 28.0)])
        tri_3857 = gpd.GeoDataFrame(geometry=[tri_4326], crs="EPSG:4326").to_crs(
            "EPSG:3857"
        )
        ds = from_earthengine("USGS/SRTMGL1_003", geometry=tri_3857, shape=(8, 8))
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        # Envelope came back in lon/lat degrees, not 3857 metres (~9.6e6).
        assert abs(ds.geotransform[0]) < 200, (
            f"Window not reprojected to lon/lat: origin {ds.geotransform[0]}"
        )

    def test_geometry_in_crs_passthrough_without_crs(self) -> None:
        """A geometry lacking a CRS is returned unchanged.

        Test scenario:
            An object with no `crs`/`to_crs` is assumed already in `crs`.
        """
        sentinel = object()
        assert ee_reader._geometry_in_crs(sentinel, "EPSG:4326") is sentinel, (
            "A CRS-less geometry should pass through unchanged"
        )

    def test_rejects_non_iso_dates(self) -> None:
        """A non-ISO or quote-bearing date is rejected before the filter (L1).

        Test scenario:
            An injection-style start value raises ReaderError, not a broken query.
        """
        creds = EarthEngineCredentials.application_default()
        with pytest.raises(ReaderError, match="ISO date"):
            ee_reader._discover_scenes(
                "COPERNICUS/S2_SR_HARMONIZED",
                start="2024' OR '1'='1",
                end="2024-06-30",
                bbox_4326=(0.0, 0.0, 1.0, 1.0),
                credentials=creds,
            )


def _multiband_scene(n_bands=2, fills=(10, 20), nodatas=(-1, -2)):
    """A small multi-band EPSG:4326 raster with per-band fills and nodata."""
    from osgeo import gdal, osr

    src = gdal.GetDriverByName("MEM").Create("", 20, 20, n_bands, gdal.GDT_Int16)
    src.SetGeoTransform((86.0, 0.01, 0.0, 29.0, 0.0, -0.01))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src.SetProjection(srs.ExportToWkt())
    for band in range(n_bands):
        src.GetRasterBand(band + 1).Fill(fills[band])
        src.GetRasterBand(band + 1).SetNoDataValue(nodatas[band])
    # `_composite` consumes pyramids Datasets (one per windowed scene), so hand back
    # a wrapped Dataset rather than the bare GDAL source.
    return Dataset(src)


class TestMultibandComposite:
    """Composite handling of multi-band scenes and band-count mismatches (L3)."""

    def test_per_band_nodata_preserved(self) -> None:
        """Each output band keeps its own source nodata value.

        Test scenario:
            A 2-band composite stamps the per-band nodata (-1, -2), not band-1's
            value for both.
        """
        windowed = [_multiband_scene() for _ in range(3)]
        ds = ee_reader._composite(
            windowed, "max", EarthEngineCredentials.application_default()
        )
        assert ds.shape == (2, 20, 20), f"Expected 2 bands, got {ds.shape}"
        assert tuple(ds.no_data_value) == (-1.0, -2.0), (
            f"Per-band nodata not preserved: {ds.no_data_value}"
        )

    def test_mismatched_band_counts_raise(self) -> None:
        """Scenes with different band counts raise a clear ReaderError.

        Test scenario:
            A 2-band scene and a 1-band scene cannot be composited.
        """
        windowed = [
            _multiband_scene(2),
            _multiband_scene(1, fills=(5,), nodatas=(-1,)),
        ]
        creds = EarthEngineCredentials.application_default()
        with pytest.raises(ReaderError, match="mismatched band counts"):
            ee_reader._composite(windowed, "max", creds)


class TestCredentialScope:
    """The credential config must be in effect during the windowed pixel read (M1)."""

    def test_config_active_during_windowed_read(self, monkeypatch, tmp_path) -> None:
        """The scoped config is set during `_window` and restored afterward.

        Test scenario:
            A service-account read has GOOGLE_APPLICATION_CREDENTIALS set while the
            windowing read runs, and cleared back to its prior value after.
        """
        from osgeo import gdal

        key = tmp_path / "k.json"
        key.write_text("{}", encoding="utf-8")
        seen = {}
        real_window = ee_reader._window

        def _spy(src, **kwargs):
            seen["cfg"] = gdal.GetConfigOption("GOOGLE_APPLICATION_CREDENTIALS", None)
            return real_window(src, **kwargs)

        monkeypatch.setattr(
            ee_reader,
            "_open_eedai",
            lambda a, *, bands, credentials, **_kw: Dataset(_synthetic_srtm()),
        )
        monkeypatch.setattr(ee_reader, "_window", _spy)
        before = gdal.GetConfigOption("GOOGLE_APPLICATION_CREDENTIALS", None)
        from_earthengine(
            "USGS/SRTMGL1_003", bbox=_BBOX, shape=(5, 5), credentials=str(key)
        )
        assert seen["cfg"] == str(key), "Config must be active during the windowed read"
        assert gdal.GetConfigOption("GOOGLE_APPLICATION_CREDENTIALS", None) == before, (
            "Config must be restored after the read (no global leak)"
        )


class TestRound2Coverage:
    """Extra tests closing gaps the round-2 review noted."""

    def test_multiband_mean_composite_is_float(self) -> None:
        """A multi-band mean composite is float and keeps per-band nodata.

        Test scenario:
            Two int16 bands (fills 10/20, nodata -1/-2) reduce by mean to a float
            Dataset that preserves each band's nodata.
        """
        windowed = [
            _multiband_scene(fills=(10, 20), nodatas=(-1, -2)) for _ in range(3)
        ]
        ds = ee_reader._composite(
            windowed, "mean", EarthEngineCredentials.application_default()
        )
        assert ds.shape == (2, 20, 20), f"Expected 2 bands, got {ds.shape}"
        assert ds.read_array().dtype.kind == "f", "mean composite should be floating"
        assert tuple(ds.no_data_value) == (-1.0, -2.0), (
            f"Per-band nodata not preserved: {ds.no_data_value}"
        )

    def test_geometry_in_crs_reprojects_gdf(self) -> None:
        """`_geometry_in_crs` reprojects a GeoDataFrame carrying a CRS.

        Test scenario:
            A Web-Mercator box is reprojected to EPSG:4326, so its bounds become
            lon/lat degrees.
        """
        import geopandas as gpd
        from shapely.geometry import box

        gdf_3857 = gpd.GeoDataFrame(
            geometry=[box(86.9, 27.9, 87.0, 28.0)], crs="EPSG:4326"
        ).to_crs("EPSG:3857")
        out = ee_reader._geometry_in_crs(gdf_3857, "EPSG:4326")
        assert abs(out.total_bounds[0]) < 200, (
            f"Geometry not reprojected to lon/lat: {out.total_bounds[0]}"
        )


def _synthetic_no_nodata():
    """A 200x200 EPSG:4326 raster with no nodata set on its band."""
    from osgeo import gdal, osr

    src = gdal.GetDriverByName("MEM").Create("", 200, 200, 1, gdal.GDT_Int16)
    src.SetGeoTransform((86.0, 0.01, 0.0, 29.0, 0.0, -0.01))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src.SetProjection(srs.ExportToWkt())
    src.GetRasterBand(1).Fill(7)
    return src


class TestMaterialize:
    """Tests for the block-aligned native materialisation :func:`_materialize`."""

    def test_reprojects_bbox_to_source_crs(self) -> None:
        """A bbox in another CRS is transformed to the source CRS window.

        Test scenario:
            A Web-Mercator bbox over a 4326 source still materialises the right
            constant-fill window.
        """
        from osgeo import osr

        src = _synthetic_srtm(fill=42)
        source = osr.SpatialReference()
        source.ImportFromEPSG(4326)
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target = osr.SpatialReference()
        target.ImportFromEPSG(3857)
        target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(source, target)
        (x0, y0, _), (x1, y1, _) = (
            ct.TransformPoint(86.9, 27.9),
            ct.TransformPoint(87.0, 28.0),
        )
        mem = ee_reader._materialize(Dataset(src), (x0, y0, x1, y1), "EPSG:3857")
        assert mem.band_count == 1, "Materialised copy should keep the band count"
        assert int(np.asarray(mem.read_array()).max()) == 42, (
            "Constant fill should be preserved"
        )

    def test_stitches_tiles_across_block_boundary(self) -> None:
        """A window spanning several 256-px blocks is stitched exactly.

        Test scenario:
            Over a 600x600 gradient source (so no block is constant and the window
            crosses the 256/512 boundaries), the materialised array equals the
            source's exact sub-window — the regression guard for the block-stitch
            math that the corruption fix relies on.
        """
        from osgeo import gdal, osr

        size = 600
        src = gdal.GetDriverByName("MEM").Create("", size, size, 1, gdal.GDT_Int32)
        src.SetGeoTransform((86.0, 0.001, 0.0, 29.0, 0.0, -0.001))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        src.SetProjection(srs.ExportToWkt())
        src.GetRasterBand(1).WriteArray(
            np.arange(size * size, dtype="int32").reshape(size, size)
        )
        # bbox covering native pixels ~[100, 500) on each axis -> a ~400 px window
        # that straddles the 256 and 512 block boundaries.
        bbox = (
            86.0 + 100 * 0.001,
            29.0 - 500 * 0.001,
            86.0 + 500 * 0.001,
            29.0 - 100 * 0.001,
        )
        mem = ee_reader._materialize(Dataset(src), bbox, "EPSG:4326")
        assert mem.columns > 256, (
            "window must span >1 block on x to exercise the stitch"
        )
        assert mem.rows > 256, "window must span >1 block on y to exercise the stitch"
        inverse = gdal.InvGeoTransform(src.GetGeoTransform())
        mem_gt = mem.geotransform
        origin_col, origin_row = gdal.ApplyGeoTransform(inverse, mem_gt[0], mem_gt[3])
        reference = src.GetRasterBand(1).ReadAsArray(
            round(origin_col), round(origin_row), mem.columns, mem.rows
        )
        assert np.array_equal(np.asarray(mem.read_array()), reference), (
            "Stitched block tiles differ from a direct read"
        )

    def test_preserves_rotated_geotransform_cross_terms(self) -> None:
        """A rotated source grid keeps its skew terms and stays registered.

        Test scenario:
            Over a 400x400 gradient with non-zero geotransform cross terms
            (``gt[2]``/``gt[4]``), the materialised copy carries the same skew and
            its data equals the source sub-window at the copy's origin pixel — the
            regression guard for the cross-term handling. If the skew were dropped,
            the copy's origin would map to a different source pixel and the arrays
            would diverge.
        """
        from osgeo import gdal, osr

        size = 400
        src = gdal.GetDriverByName("MEM").Create("", size, size, 1, gdal.GDT_Int32)
        gt = (100.0, 0.01, 0.002, 50.0, 0.003, -0.01)
        src.SetGeoTransform(gt)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        src.SetProjection(srs.ExportToWkt())
        src.GetRasterBand(1).WriteArray(
            np.arange(size * size, dtype="int32").reshape(size, size)
        )

        def world(col: int, row: int) -> tuple[float, float]:
            return gdal.ApplyGeoTransform(gt, col, row)

        corners = [world(100, 100), world(300, 100), world(100, 300), world(300, 300)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        mem = ee_reader._materialize(Dataset(src), bbox, "EPSG:4326")
        mem_gt = mem.geotransform
        assert (mem_gt[2], mem_gt[4]) == (gt[2], gt[4]), (
            "Rotation/skew cross terms must be preserved"
        )
        inverse = gdal.InvGeoTransform(gt)
        origin_col, origin_row = gdal.ApplyGeoTransform(inverse, mem_gt[0], mem_gt[3])
        reference = src.GetRasterBand(1).ReadAsArray(
            round(origin_col), round(origin_row), mem.columns, mem.rows
        )
        assert np.array_equal(np.asarray(mem.read_array()), reference), (
            "Rotated sub-window is mis-registered — cross-term math is wrong"
        )

    def test_raises_when_aoi_outside_asset(self) -> None:
        """An AOI that misses the asset raises ``ReaderError``.

        Test scenario:
            A bbox far from the source extent does not intersect any pixels.
        """
        ee = Dataset(_synthetic_srtm())
        with pytest.raises(ReaderError, match="does not intersect"):
            ee_reader._materialize(ee, (0.0, 0.0, 1.0, 1.0), "EPSG:4326")

    def test_band_without_nodata(self) -> None:
        """A source band with no nodata materialises without setting one.

        Test scenario:
            Covers the no-nodata branch; the fill value survives.
        """
        mem = ee_reader._materialize(
            Dataset(_synthetic_no_nodata()), _BBOX, "EPSG:4326"
        )
        assert mem.no_data_value[0] is None, "No nodata should be set"
        assert int(np.asarray(mem.read_array()).max()) == 7, (
            "Fill value should be preserved"
        )

    def test_raises_on_non_invertible_geotransform(self) -> None:
        """A non-invertible source geotransform raises ``ReaderError``.

        Test scenario:
            A degenerate (zero-scale) geotransform cannot map coordinates to pixels.
        """
        from osgeo import gdal, osr

        src = gdal.GetDriverByName("MEM").Create("", 10, 10, 1, gdal.GDT_Int16)
        src.SetGeoTransform((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        src.SetProjection(srs.ExportToWkt())
        ee = Dataset(src)
        with pytest.raises(ReaderError, match="non-invertible"):
            ee_reader._materialize(ee, _BBOX, "EPSG:4326")

    def test_raises_when_block_read_returns_none(self) -> None:
        """A failed block read (``None``) raises ``ReaderError``.

        Test scenario:
            A source band whose ``ReadAsArray`` returns ``None`` surfaces as a
            ``ReaderError``.
        """

        class _NoneReadSrc:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def GetRasterBand(self, index):  # noqa: N802
                real_band = self._real.GetRasterBand(index)

                class _Band:
                    DataType = real_band.DataType

                    def GetNoDataValue(self):  # noqa: N802
                        return real_band.GetNoDataValue()

                    def GetBlockSize(self):  # noqa: N802
                        return real_band.GetBlockSize()

                    def ReadAsArray(self, *args, **kwargs):  # noqa: N802, ARG002
                        return None

                return _Band()

        src = _NoneReadSrc(_synthetic_srtm())
        with pytest.raises(ReaderError, match="block read failed"):
            ee_reader._read_native_blocks(src, 0, 0, 200, 200)


class TestLivePixelCorrectness:
    """Live safety net: EEDAI reads must return correct, deterministic pixels."""

    @pytest.mark.live
    def test_projected_crs_reads_same_ground_as_4326(self) -> None:
        """A projected-CRS read covers the same ground as the 4326 read (#66).

        Test scenario:
            The same AOI read once as a 4326 lon/lat box and once as the equivalent
            EPSG:32645 metre box returns matching SRTM elevation statistics — the
            projected read lands on the same mountains, not a displaced/empty area.
        """
        ll = (86.92, 27.92, 86.98, 27.98)
        latlon = from_earthengine("USGS/SRTMGL1_003", bbox=ll, shape=(24, 24))
        utm = from_earthengine(
            "USGS/SRTMGL1_003", crs="EPSG:32645", bbox=_to_utm45(ll), shape=(24, 24)
        )
        assert utm.epsg == 32645, f"Expected EPSG:32645, got {utm.epsg}"
        mean_ll = float(np.mean(np.asarray(latlon.read_array())))
        mean_utm = float(np.mean(np.asarray(utm.read_array())))
        assert mean_ll > 1000, f"Expected mountainous ground, got mean {mean_ll}"
        assert abs(mean_ll - mean_utm) < 0.1 * mean_ll, (
            f"Projected read landed on different ground: {mean_ll} vs {mean_utm}"
        )

    @pytest.mark.live
    def test_nodata_tag_marks_gsw_fill(self) -> None:
        """A catalog-sourced fill is tagged on a live GSW read (#63).

        Test scenario:
            ``JRC/GSW1_4/GlobalSurfaceWater`` ``occurrence`` (Int8) uses ``-128`` for
            "never observed" but the driver reports no no-data; passing ``nodata=-128``
            tags it so downstream masking sees fill as fill, pixels unchanged.
        """
        untagged = from_earthengine(
            "JRC/GSW1_4/GlobalSurfaceWater",
            bbox=_BBOX,
            shape=(32, 32),
            bands=["occurrence"],
        )
        tagged = from_earthengine(
            "JRC/GSW1_4/GlobalSurfaceWater",
            bbox=_BBOX,
            shape=(32, 32),
            bands=["occurrence"],
            nodata=-128,
        )
        assert untagged.no_data_value[0] is None, (
            f"EEDAI unexpectedly reported a fill: {untagged.no_data_value}"
        )
        assert tagged.no_data_value[0] == -128, (
            f"Expected -128 tagged, got {tagged.no_data_value}"
        )
        assert np.array_equal(
            np.asarray(untagged.read_array()), np.asarray(tagged.read_array())
        ), "tagging nodata changed the pixels"

    @pytest.mark.live
    def test_block_size_pixels_unchanged(self) -> None:
        """A larger ``block_size`` returns identical pixels (#60).

        Test scenario:
            An SRTM window read with ``block_size=512`` is byte-identical to the
            default 256-block read — the block size is a transport knob, not a
            correctness one.
        """
        default = from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, shape=(60, 60))
        larger = from_earthengine(
            "USGS/SRTMGL1_003", bbox=_BBOX, shape=(60, 60), block_size=512
        )
        assert np.array_equal(
            np.asarray(default.read_array()), np.asarray(larger.read_array())
        ), "block_size changed the pixels"

    @pytest.mark.live
    def test_pinned_encoding_lossless_at_large_transfer(self) -> None:
        """The pinned lossless encoding keeps a large multi-band Byte read bit-exact.

        Test scenario:
            At a large (1024 px) transfer — the size the block-sizing work will use —
            the driver's ``AUTO`` default silently loses data on a multi-band Byte
            read, while the reader's pinned ``_EEDAI_PIXEL_ENCODING`` is byte-identical
            to a lossless ``NPY`` read. Guards #69 (and the future larger-block path).
        """
        from osgeo import gdal

        scene = (
            "EEDAI:projects/earthengine-public/assets/COPERNICUS/S2_SR_HARMONIZED/"
            "20240702T102601_20240702T103203_T32TLR"
        )
        creds = EarthEngineCredentials.application_default()

        def read(encoding: str) -> np.ndarray:
            with creds.activate():
                ds = gdal.OpenEx(
                    scene,
                    gdal.OF_RASTER,
                    open_options=[
                        "BLOCK_SIZE=1024",
                        "BANDS=TCI_R,TCI_G,TCI_B",
                        f"PIXEL_ENCODING={encoding}",
                    ],
                )
                return np.asarray(ds.ReadAsArray(2048, 2048, 1024, 1024))

        reference = read("NPY")
        pinned = read(ee_reader._EEDAI_PIXEL_ENCODING)
        lossy = read("AUTO")
        assert np.array_equal(pinned, reference), (
            "the pinned encoding must be byte-identical to a lossless NPY read"
        )
        assert not np.array_equal(lossy, reference), (
            "AUTO is expected to be lossy at this transfer size (proves the pin matters)"
        )

    @pytest.mark.live
    def test_srtm_values_correct_and_deterministic(self) -> None:
        """A live SRTM read has plausible elevations and repeats identically.

        Test scenario:
            No int16 garbage (every pixel a plausible elevation) and two calls
            return byte-identical arrays — the regression guard for the EEDAI
            block/overview corruption bug.
        """
        first = from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, shape=(96, 96))
        second = from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, shape=(96, 96))
        values = first.read_array()
        out_of_range = int(((values < -500) | (values > 9000)).sum())
        assert out_of_range == 0, (
            f"{out_of_range} corrupted pixels outside a plausible range"
        )
        assert np.array_equal(values, second.read_array()), (
            "Repeated read was not deterministic"
        )

    @pytest.mark.live
    def test_composite_values_correct_and_deterministic(self) -> None:
        """A live Sentinel-2 median composite has plausible reflectance and repeats.

        Test scenario:
            The composite path (discover -> per-scene EEDAI read -> reduce) returns
            non-negative, plausibly-bounded surface reflectance with no int16
            garbage, and two composites of the same window are byte-identical.
        """
        request = dict(
            bbox=_S2_BBOX,
            start="2024-06-05",
            end="2024-06-08",
            reducer="median",
            bands=["B4", "B3", "B2"],
            shape=(32, 32),
        )
        first = from_earthengine("COPERNICUS/S2_SR_HARMONIZED", **request)
        values = first.read_array()
        assert (values >= 0).all(), "Composite has negative (garbage) reflectance"
        assert int(values.max()) < 20000, (
            f"Composite reflectance implausibly large: {values.max()}"
        )
        second = from_earthengine("COPERNICUS/S2_SR_HARMONIZED", **request)
        assert np.array_equal(values, second.read_array()), (
            "Repeated composite was not deterministic"
        )

    @pytest.mark.live
    def test_collection_scene_values_correct(self) -> None:
        """Live Sentinel-2 collection scenes have plausible reflectance, no garbage.

        Test scenario:
            Every per-scene ``Dataset`` in the ``DatasetCollection`` holds
            non-negative, plausibly-bounded reflectance — the block/overview
            corruption would show as int16 extremes in one or more scenes.
        """
        collection = collection_from_earthengine(
            "COPERNICUS/S2_SR_HARMONIZED",
            start="2024-06-05",
            end="2024-06-08",
            bbox=_S2_BBOX,
            bands=["B4"],
            shape=(32, 32),
        )
        assert collection.time_length > 0, "Expected at least one scene in the window"
        for index, scene in enumerate(collection.datasets):
            values = scene.read_array()
            assert (values >= 0).all(), (
                f"Scene {index} has negative (garbage) reflectance"
            )
            assert int(values.max()) < 20000, (
                f"Scene {index} reflectance implausibly large: {values.max()}"
            )


class TestResample:
    """Tests for the ``resample`` option (:func:`_resample_alg`)."""

    def test_unknown_resample_raises(self) -> None:
        """An unknown resampling name is rejected.

        Test scenario:
            ``_resample_alg`` raises ``ValueError`` for an unsupported algorithm.
        """
        with pytest.raises(ValueError, match="Unknown resample"):
            ee_reader._resample_alg("bogus")

    def test_invalid_resample_rejected_before_network(self) -> None:
        """A bad ``resample`` name fails up front, before any EEDAI open.

        Test scenario:
            ``from_earthengine`` validates ``resample`` before touching the driver
            (no ``_open_eedai`` monkeypatch is needed — it never reaches it).
        """
        with pytest.raises(ValueError, match="Unknown resample"):
            from_earthengine("USGS/SRTMGL1_003", bbox=_BBOX, resample="neareset")

    def test_from_earthengine_honours_resample(self, patched_eedai) -> None:
        """A non-default ``resample`` is accepted end-to-end.

        Test scenario:
            ``resample="average"`` reads without error and returns a Dataset.
        """
        ds = from_earthengine(
            "USGS/SRTMGL1_003", bbox=_BBOX, shape=(5, 5), resample="average"
        )
        assert ds.shape == (1, 5, 5), f"Expected (1, 5, 5), got {ds.shape}"

    def test_resample_changes_downsampled_pixels(self) -> None:
        """The algorithm reaches the warp — nearest and average differ on downsample.

        Test scenario:
            Downsampling a 100x100 gradient to 10x10 with ``resample="nearest"``
            (pick one source pixel per cell) versus ``resample="average"`` (mean of
            the 10x10 block) yields different arrays, proving ``resample`` is applied
            in the warp rather than silently ignored.
        """
        from osgeo import gdal, osr

        size = 100
        src = gdal.GetDriverByName("MEM").Create("", size, size, 1, gdal.GDT_Int32)
        src.SetGeoTransform((86.0, 0.001, 0.0, 29.0, 0.0, -0.001))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        src.SetProjection(srs.ExportToWkt())
        src.GetRasterBand(1).WriteArray(
            np.arange(size * size, dtype="int32").reshape(size, size)
        )
        bbox = (86.0, 29.0 - size * 0.001, 86.0 + size * 0.001, 29.0)
        common = {"bbox": bbox, "crs": "EPSG:4326", "scale": None, "shape": (10, 10)}
        nearest = np.asarray(
            ee_reader._window(Dataset(src), resample="nearest", **common).read_array()
        )
        average = np.asarray(
            ee_reader._window(Dataset(src), resample="average", **common).read_array()
        )
        assert not np.array_equal(nearest, average), (
            "nearest and average must differ — resample is not reaching the warp"
        )


def _gradient_source(size: int = 400, nodata: int | None = -32768):
    """A distinct-per-pixel gradient EPSG:4326 Int32 raster over lon/lat [86,88]/[27,29].

    Args:
        size: The source is ``size`` x ``size`` pixels spanning the 2-degree box.
        nodata: Nodata value to stamp, or ``None`` to leave the band without one.

    Returns:
        An in-memory GDAL dataset whose every pixel holds a unique value, so a
        mis-placed tile is detectable in a tiled-vs-untiled comparison.
    """
    from osgeo import gdal, osr

    src = gdal.GetDriverByName("MEM").Create("", size, size, 1, gdal.GDT_Int32)
    src.SetGeoTransform((86.0, 2.0 / size, 0.0, 29.0, 0.0, -2.0 / size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src.SetProjection(srs.ExportToWkt())
    src.GetRasterBand(1).WriteArray(
        np.arange(size * size, dtype="int32").reshape(size, size)
    )
    if nodata is not None:
        src.GetRasterBand(1).SetNoDataValue(nodata)
    return src


@pytest.fixture
def patched_gradient(monkeypatch):
    """Patch the EEDAI open seam with the distinct-per-pixel gradient source.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Yields:
        None. While active, ``_open_eedai`` returns a wrapped gradient ``Dataset``.
    """
    monkeypatch.setattr(
        ee_reader,
        "_open_eedai",
        lambda a, *, bands, credentials, **_kw: Dataset(_gradient_source()),
    )


class TestTileEdges:
    """Tests for the tile-splitting helper :func:`_tile_edges`."""

    def test_splits_with_short_final_block(self) -> None:
        """A size that is not a multiple of the tile keeps a short final block.

        Test scenario:
            ``_tile_edges(10, 4)`` yields ``[(0, 4), (4, 8), (8, 10)]``.
        """
        assert ee_reader._tile_edges(10, 4) == [(0, 4), (4, 8), (8, 10)], "bad split"

    def test_single_block_when_size_within_tile(self) -> None:
        """A grid no larger than the tile size stays a single block.

        Test scenario:
            ``_tile_edges(3, 8)`` yields one block ``[(0, 3)]``.
        """
        assert ee_reader._tile_edges(3, 8) == [(0, 3)], "should be a single block"


class TestTiledRead:
    """Tests for the oversize tiling + mosaic path of :func:`from_earthengine`."""

    def test_tiled_equals_untiled(self, patched_gradient, tmp_path) -> None:
        """A tiled read reproduces the equivalent un-tiled read pixel-for-pixel.

        Test scenario:
            A 20x20 window over a distinct-per-pixel gradient, split into tiles of 7
            (a 3x3 tile grid), mosaics back to exactly the un-tiled 20x20 read.
        """
        untiled_ds = from_earthengine("X", bbox=_BBOX, shape=(20, 20))
        untiled = np.asarray(untiled_ds.read_array())
        out = tmp_path / "mosaic.tif"
        tiled_ds = from_earthengine(
            "X", bbox=_BBOX, shape=(20, 20), tile_size=7, path=str(out)
        )
        tiled = np.asarray(tiled_ds.read_array())
        assert out.exists(), "the mosaic file should be written to path"
        assert tiled.shape == untiled.shape == (20, 20), (
            f"unexpected shape {tiled.shape}"
        )
        assert tiled_ds.no_data_value[0] == untiled_ds.no_data_value[0] == -32768, (
            f"mosaic nodata must match the source: {tiled_ds.no_data_value[0]}"
        )
        assert np.array_equal(tiled, untiled), (
            "tiled mosaic differs from the un-tiled read"
        )

    def test_single_tile_when_tile_covers_grid(
        self, patched_gradient, tmp_path
    ) -> None:
        """A tile_size at least the grid size yields one tile equal to the un-tiled read.

        Test scenario:
            ``tile_size=64`` over an 8x8 grid is a single tile that still equals the
            un-tiled read.
        """
        untiled = np.asarray(
            from_earthengine("X", bbox=_BBOX, shape=(8, 8)).read_array()
        )
        out = tmp_path / "one.tif"
        tiled = np.asarray(
            from_earthengine(
                "X", bbox=_BBOX, shape=(8, 8), tile_size=64, path=str(out)
            ).read_array()
        )
        assert np.array_equal(tiled, untiled), "single-tile mosaic must equal the read"

    def test_tiled_scale_defines_the_grid(self, patched_gradient, tmp_path) -> None:
        """A scale-based tiled read produces the scale's grid over the bbox.

        Test scenario:
            ``scale=0.01`` over a 0.1-degree bbox is a 10x10 grid; tiling at 4 keeps
            that shape.
        """
        out = tmp_path / "scaled.tif"
        ds = from_earthengine("X", bbox=_BBOX, scale=0.01, tile_size=4, path=str(out))
        assert ds.shape == (1, 10, 10), f"Expected (1, 10, 10), got {ds.shape}"

    def test_tiled_scale_matches_untiled_at_rounding_boundary(
        self, patched_gradient, tmp_path
    ) -> None:
        """A scale on a ``.5`` grid boundary tiles to the same grid as the un-tiled read.

        Test scenario:
            ``scale=0.008`` over the ~0.1-degree bbox lands on a round-half-up grid
            boundary on the Y axis (13 rows via half-up, not Python's banker's 12).
            The tiled read must match the un-tiled ``scale`` read's grid and pixels.
        """
        untiled = np.asarray(
            from_earthengine("X", bbox=_BBOX, scale=0.008).read_array()
        )
        out = tmp_path / "scale_boundary.tif"
        tiled = np.asarray(
            from_earthengine(
                "X", bbox=_BBOX, scale=0.008, tile_size=5, path=str(out)
            ).read_array()
        )
        assert untiled.shape[0] == 13, (
            f"expected the half-up 13 rows, got {untiled.shape[0]}"
        )
        assert tiled.shape == untiled.shape, f"shape {tiled.shape} vs {untiled.shape}"
        assert np.array_equal(tiled, untiled), (
            "scale-boundary tiled mosaic differs from the un-tiled scale read"
        )

    def test_path_without_tile_size_writes_file(
        self, patched_gradient, tmp_path
    ) -> None:
        """A ``path`` without ``tile_size`` writes the read and returns it file-backed.

        Test scenario:
            A plain windowed read to ``path`` writes the file and returns a
            5x5 Dataset reading it.
        """
        out = tmp_path / "single.tif"
        ds = from_earthengine("X", bbox=_BBOX, shape=(5, 5), path=str(out))
        assert out.exists(), "path should be written"
        assert ds.shape == (1, 5, 5), f"Expected (1, 5, 5), got {ds.shape}"

    def test_tiled_multiband_source(self, monkeypatch, tmp_path) -> None:
        """A multi-band source tiles with each band placed correctly.

        Test scenario:
            A 3-band source with distinct per-band gradients mosaics back to the
            un-tiled read across all bands (band order + per-band placement).
        """
        from osgeo import gdal, osr

        size = 300
        src = gdal.GetDriverByName("MEM").Create("", size, size, 3, gdal.GDT_Int32)
        src.SetGeoTransform((86.0, 2.0 / size, 0.0, 29.0, 0.0, -2.0 / size))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        src.SetProjection(srs.ExportToWkt())
        base = np.arange(size * size, dtype="int32").reshape(size, size)
        for band in range(3):
            src.GetRasterBand(band + 1).WriteArray(base + band * 1_000_000)
            src.GetRasterBand(band + 1).SetNoDataValue(-32768)
        monkeypatch.setattr(
            ee_reader,
            "_open_eedai",
            lambda a, *, bands, credentials, **_kw: Dataset(src),
        )

        untiled = np.asarray(
            from_earthengine("X", bbox=_BBOX, shape=(18, 18)).read_array()
        )
        out = tmp_path / "multiband.tif"
        tiled = np.asarray(
            from_earthengine(
                "X", bbox=_BBOX, shape=(18, 18), tile_size=7, path=str(out)
            ).read_array()
        )
        assert tiled.shape == untiled.shape == (3, 18, 18), f"shape {tiled.shape}"
        assert np.array_equal(tiled, untiled), (
            "multi-band tiled mosaic differs from the un-tiled read"
        )

    def test_tiled_matches_untiled_over_footprint_edge(
        self, patched_gradient, tmp_path
    ) -> None:
        """An AOI overhanging the asset edge tiles to the same nodata-filled read.

        Test scenario:
            A bbox half outside the ``[86,88]x[27,29]`` source: a fully-outside tile
            is emitted as all-nodata (as the un-tiled warp fills the overhang), so the
            mosaic equals the un-tiled read including the nodata region.
        """
        over = (87.0, 28.0, 89.0, 30.0)
        untiled = np.asarray(
            from_earthengine("X", bbox=over, shape=(20, 20)).read_array()
        )
        out = tmp_path / "edge.tif"
        tiled = np.asarray(
            from_earthengine(
                "X", bbox=over, shape=(20, 20), tile_size=7, path=str(out)
            ).read_array()
        )
        assert (untiled == -32768).any(), "the overhang should include nodata pixels"
        assert np.array_equal(tiled, untiled), (
            "overhang tiled mosaic differs from the un-tiled nodata-filled read"
        )

    def test_tiled_fully_outside_raises_like_untiled(
        self, patched_gradient, tmp_path
    ) -> None:
        """An AOI entirely off the footprint raises, matching the un-tiled read.

        Test scenario:
            When no tile intersects the asset, the tiled read raises the same
            ``does not intersect`` ``ReaderError`` as the un-tiled read.
        """
        far = (100.0, 50.0, 101.0, 51.0)
        with pytest.raises(ReaderError, match="does not intersect"):
            from_earthengine("X", bbox=far, shape=(8, 8))
        out = tmp_path / "far.tif"
        with pytest.raises(ReaderError, match="does not intersect"):
            from_earthengine("X", bbox=far, shape=(8, 8), tile_size=4, path=str(out))

    def test_tiled_reraises_other_reader_errors(
        self, patched_gradient, monkeypatch, tmp_path
    ) -> None:
        """A tile ``ReaderError`` other than a footprint miss aborts the tiled read.

        Test scenario:
            Only ``does not intersect`` is caught per tile; any other ``ReaderError``
            (e.g. a failed read) propagates rather than being masked as nodata.
        """

        def boom(*args, **kwargs):
            raise ReaderError("boom: block read failed")

        monkeypatch.setattr(ee_reader, "_window", boom)
        out = tmp_path / "boom.tif"
        with pytest.raises(ReaderError, match="boom"):
            from_earthengine("X", bbox=_BBOX, shape=(8, 8), tile_size=4, path=str(out))

    def test_tiled_reprojects_like_untiled(self, patched_gradient, tmp_path) -> None:
        """A reprojecting (``crs`` != source) tiled read equals the un-tiled read.

        Test scenario:
            Reading a Web-Mercator bbox tiled matches the un-tiled reprojected read,
            covering the seam reasoning under reprojection.
        """
        from osgeo import osr

        source = osr.SpatialReference()
        source.ImportFromEPSG(4326)
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target = osr.SpatialReference()
        target.ImportFromEPSG(3857)
        target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ct = osr.CoordinateTransformation(source, target)
        (x0, y0, _), (x1, y1, _) = (
            ct.TransformPoint(86.92, 27.92),
            ct.TransformPoint(86.98, 27.98),
        )
        merc = (x0, y0, x1, y1)
        untiled = np.asarray(
            from_earthengine(
                "X", bbox=merc, crs="EPSG:3857", shape=(18, 18)
            ).read_array()
        )
        out = tmp_path / "merc.tif"
        tiled = np.asarray(
            from_earthengine(
                "X",
                bbox=merc,
                crs="EPSG:3857",
                shape=(18, 18),
                tile_size=7,
                path=str(out),
            ).read_array()
        )
        assert np.array_equal(tiled, untiled), (
            "reprojected tiled mosaic differs from the un-tiled reprojected read"
        )

    def test_tiled_preserves_in_window_nodata(self, monkeypatch, tmp_path) -> None:
        """Actual nodata pixels inside the window survive the tiled mosaic.

        Test scenario:
            A source with a nodata patch inside the AOI mosaics back to the un-tiled
            read, exercising merge's ``srcNodata``/``VRTNodata`` round-trip.
        """
        from osgeo import gdal, osr

        size = 300
        src = gdal.GetDriverByName("MEM").Create("", size, size, 1, gdal.GDT_Int32)
        src.SetGeoTransform((86.0, 2.0 / size, 0.0, 29.0, 0.0, -2.0 / size))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        src.SetProjection(srs.ExportToWkt())
        array = np.arange(size * size, dtype="int32").reshape(size, size)
        array[150:170, 150:170] = -32768  # a nodata patch inside the source
        src.GetRasterBand(1).WriteArray(array)
        src.GetRasterBand(1).SetNoDataValue(-32768)
        monkeypatch.setattr(
            ee_reader,
            "_open_eedai",
            lambda a, *, bands, credentials, **_kw: Dataset(src),
        )
        aoi = (86.7, 27.7, 87.3, 28.3)
        untiled = np.asarray(
            from_earthengine("X", bbox=aoi, shape=(30, 30)).read_array()
        )
        out = tmp_path / "innodata.tif"
        tiled = np.asarray(
            from_earthengine(
                "X", bbox=aoi, shape=(30, 30), tile_size=11, path=str(out)
            ).read_array()
        )
        assert (untiled == -32768).any(), "the window should include nodata pixels"
        assert np.array_equal(tiled, untiled), (
            "in-window nodata differs between tiled and un-tiled reads"
        )

    def test_tiled_cleans_up_temp_dir(self, patched_gradient, tmp_path) -> None:
        """A tiled read leaves no ``ee_tiles_*`` temp directory behind.

        Test scenario:
            The per-read temp tile directory is removed once the mosaic is written,
            so repeated oversize reads do not accumulate temp garbage.
        """
        import glob
        import os
        import tempfile

        pattern = os.path.join(tempfile.gettempdir(), "ee_tiles_*")
        before = set(glob.glob(pattern))
        out = tmp_path / "clean.tif"
        from_earthengine("X", bbox=_BBOX, shape=(16, 16), tile_size=6, path=str(out))
        leaked = set(glob.glob(pattern)) - before
        assert not leaked, f"tiled read leaked temp dir(s): {leaked}"

    def test_tiled_non_square_grid(self, patched_gradient, tmp_path) -> None:
        """A non-square window with asymmetric tile splits mosaics back exactly.

        Test scenario:
            A 12x20 output split at ``tile_size=7`` (rows -> 7+5, cols -> 7+7+6)
            reproduces the un-tiled read, guarding the row/column tile arithmetic.
        """
        untiled = np.asarray(
            from_earthengine("X", bbox=_BBOX, shape=(12, 20)).read_array()
        )
        out = tmp_path / "rect.tif"
        tiled = np.asarray(
            from_earthengine(
                "X", bbox=_BBOX, shape=(12, 20), tile_size=7, path=str(out)
            ).read_array()
        )
        assert tiled.shape == untiled.shape == (12, 20), f"shape {tiled.shape}"
        assert np.array_equal(tiled, untiled), "non-square tiled mosaic differs"

    def test_tiled_source_without_nodata(self, monkeypatch, tmp_path) -> None:
        """A source with no nodata still tiles and mosaics to the un-tiled read.

        Test scenario:
            When the source band has no nodata, the mosaic uses the ``"0"`` fill
            sentinel; a full-coverage grid still reproduces the un-tiled read.
        """
        monkeypatch.setattr(
            ee_reader,
            "_open_eedai",
            lambda a, *, bands, credentials, **_kw: Dataset(
                _gradient_source(nodata=None)
            ),
        )
        untiled_ds = from_earthengine("X", bbox=_BBOX, shape=(16, 16))
        out = tmp_path / "no_nodata.tif"
        tiled_ds = from_earthengine(
            "X", bbox=_BBOX, shape=(16, 16), tile_size=6, path=str(out)
        )
        assert np.array_equal(
            np.asarray(tiled_ds.read_array()), np.asarray(untiled_ds.read_array())
        ), "tiled mosaic differs from the un-tiled read for a no-nodata source"
        assert untiled_ds.no_data_value[0] is None, (
            "un-tiled read should have no nodata"
        )
        assert tiled_ds.no_data_value[0] is None, (
            f"mosaic must not fabricate a nodata; got {tiled_ds.no_data_value[0]}"
        )


class TestTiledValidation:
    """The oversize-tiling guards reject bad combinations before any network call."""

    def test_tile_size_requires_path(self) -> None:
        """``tile_size`` without ``path`` raises up front.

        Test scenario:
            No ``path`` to stream the mosaic to → ``ValueError``.
        """
        with pytest.raises(ValueError, match="path"):
            from_earthengine("X", bbox=_BBOX, shape=(20, 20), tile_size=7)

    def test_tile_size_requires_scale_or_shape(self) -> None:
        """``tile_size`` without ``scale``/``shape`` raises up front.

        Test scenario:
            No output grid defined → ``ValueError``.
        """
        with pytest.raises(ValueError, match="scale.*shape"):
            from_earthengine("X", bbox=_BBOX, tile_size=7, path="out.tif")

    def test_tile_size_rejects_composite(self) -> None:
        """``tile_size`` with a reducer (composite mode) raises up front.

        Test scenario:
            The oversize tiler is single-image only → ``ValueError``.
        """
        with pytest.raises(ValueError, match="composite"):
            from_earthengine(
                "X",
                bbox=_BBOX,
                shape=(8, 8),
                tile_size=4,
                path="o.tif",
                reducer="median",
            )

    def test_property_filter_rejects_single_image(self) -> None:
        """``property_filter`` without composite mode raises up front (#62).

        Test scenario:
            A single-image read has no scene set to filter → ``ValueError``.
        """
        with pytest.raises(ValueError, match="property_filter"):
            from_earthengine(
                "X",
                bbox=_BBOX,
                shape=(8, 8),
                property_filter="CLOUDY_PIXEL_PERCENTAGE < 20",
            )

    def test_tile_size_rejects_geometry(self) -> None:
        """``tile_size`` combined with a polygon ``geometry`` raises up front.

        Test scenario:
            A polygon cutline is incompatible with the tiler → ``ValueError``.
        """
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            geometry=[Polygon([(86.9, 27.9), (87.0, 27.9), (86.9, 28.0)])],
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="geometry"):
            from_earthengine("X", geometry=gdf, shape=(8, 8), tile_size=4, path="o.tif")

    def test_tile_size_must_be_positive(self) -> None:
        """A non-positive ``tile_size`` raises up front.

        Test scenario:
            ``tile_size=0`` → ``ValueError``.
        """
        with pytest.raises(ValueError, match="positive"):
            from_earthengine("X", bbox=_BBOX, shape=(8, 8), tile_size=0, path="o.tif")

    @pytest.mark.parametrize("resample", ["bilinear", "cubic", "average", "mode"])
    def test_tile_size_rejects_non_nearest_resample(self, resample) -> None:
        """A non-nearest ``resample`` with ``tile_size`` raises up front.

        Args:
            resample: A non-nearest resampling algorithm.

        Test scenario:
            Interpolating (and footprint) resamplers differ from the un-tiled read
            at tile seams, so they are rejected before any read.
        """
        with pytest.raises(ValueError, match="nearest"):
            from_earthengine(
                "X",
                bbox=_BBOX,
                shape=(8, 8),
                tile_size=4,
                path="o.tif",
                resample=resample,
            )

    def test_path_without_bbox_or_geometry(self) -> None:
        """A ``path`` with no ``bbox``/``geometry`` raises (the whole-asset read is lazy).

        Test scenario:
            Nothing to window → ``ValueError``.
        """
        with pytest.raises(ValueError, match="path"):
            from_earthengine("X", path="o.tif")


class TestTiledLive:
    """Live safety net: a tiled disk read must equal the un-tiled read."""

    @pytest.mark.live
    def test_tiled_srtm_equals_untiled(self, tmp_path) -> None:
        """A live tiled SRTM read mosaics back to exactly the un-tiled read.

        Test scenario:
            A 40x40 SRTM window read as 16-px tiles (a 3x3 grid) to disk equals the
            single un-tiled 40x40 read pixel-for-pixel.
        """
        untiled = np.asarray(
            from_earthengine(
                "USGS/SRTMGL1_003", bbox=_BBOX, shape=(40, 40)
            ).read_array()
        )
        out = tmp_path / "srtm_tiled.tif"
        tiled = np.asarray(
            from_earthengine(
                "USGS/SRTMGL1_003",
                bbox=_BBOX,
                shape=(40, 40),
                tile_size=16,
                path=str(out),
            ).read_array()
        )
        assert tiled.shape == untiled.shape, f"shape {tiled.shape} vs {untiled.shape}"
        assert np.array_equal(tiled, untiled), (
            "live tiled mosaic differs from the un-tiled read"
        )
