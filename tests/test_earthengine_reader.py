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
from pyramids_eo import collection_from_earthengine, from_earthengine
from pyramids_eo.earthengine import EarthEngineCredentials
from pyramids_eo.earthengine.reader import _Scene
from pyramids_eo.errors import ReaderError

_BBOX = (86.9, 27.9, 87.0, 28.0)


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


@pytest.fixture(scope="function")
def patched_eedai(monkeypatch):
    """Replace the EEDAI open seam with the synthetic raster.

    Args:
        monkeypatch: pytest monkeypatch fixture.

    Yields:
        None. While active, ``_open_eedai`` returns a fresh synthetic raster and
        never touches the network.
    """

    def _fake_open(asset_id, *, bands, credentials):  # noqa: ARG001
        return _synthetic_srtm()

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

        def _fake_open(asset_id, *, bands, credentials):  # noqa: ARG001
            captured["credentials"] = credentials
            return _synthetic_srtm()

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
        assert rows > 0 and cols > 0, f"Expected a non-empty window, got {rows}x{cols}"

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


@pytest.fixture(scope="function")
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
        sentinel = object()
        fake = _FakeGdal(sentinel)
        monkeypatch.setattr(ee_reader, "gdal", fake)
        creds = EarthEngineCredentials.application_default()

        result = ee_reader._open_eedai(
            "USGS/SRTMGL1_003", bands=["B4", "B3"], credentials=creds
        )

        assert result is sentinel, "Should return the driver's open result"
        conn, options = fake.calls[0]
        assert conn == "EEDAI:USGS/SRTMGL1_003", f"Unexpected connection string: {conn}"
        assert options == ["BANDS=B4,B3"], f"Unexpected open options: {options}"

    def test_no_bands_option_when_bands_none(self, monkeypatch) -> None:
        """No ``BANDS`` option is emitted when ``bands`` is ``None``.

        Test scenario:
            ``bands=None`` opens with an empty open-option list.
        """
        fake = _FakeGdal(object())
        monkeypatch.setattr(ee_reader, "gdal", fake)
        ee_reader._open_eedai(
            "USGS/SRTMGL1_003",
            bands=None,
            credentials=EarthEngineCredentials.application_default(),
        )
        _conn, options = fake.calls[0]
        assert options == [], f"Expected no open options, got {options}"

    def test_raises_reader_error_on_open_failure(self, monkeypatch) -> None:
        """A ``None`` driver result raises ``ReaderError`` with the GDAL message.

        Test scenario:
            ``OpenEx`` returning ``None`` surfaces as ``ReaderError`` including
            the last GDAL error text.
        """
        fake = _FakeGdal(None, last_error="permission denied")
        monkeypatch.setattr(ee_reader, "gdal", fake)
        with pytest.raises(ReaderError, match="permission denied") as exc_info:
            ee_reader._open_eedai(
                "USGS/SRTMGL1_003",
                bands=None,
                credentials=EarthEngineCredentials.application_default(),
            )
        assert "USGS/SRTMGL1_003" in str(exc_info.value), (
            f"Error should name the asset, got: {exc_info.value}"
        )


class TestWindow:
    """Tests for the private windowing helper :func:`_window`."""

    def test_reader_error_when_warp_returns_none(self, monkeypatch) -> None:
        """A failed warp (``None``) raises ``ReaderError``.

        Test scenario:
            ``gdal.Warp`` returning ``None`` surfaces as ``ReaderError`` naming
            the bbox and CRS.
        """

        class _WarpFails:
            def Warp(self, dest, src, **kwargs):  # noqa: N802, ARG002
                return None

            def GetLastErrorMsg(self):  # noqa: N802
                return "warp failed"

        monkeypatch.setattr(ee_reader, "gdal", _WarpFails())
        with pytest.raises(ReaderError, match="warp failed"):
            ee_reader._window(
                object(), bbox=_BBOX, crs="EPSG:4326", scale=None, shape=None
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


@pytest.fixture(scope="function")
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

    def _fake_discover(asset_id, *, start, end, bbox_4326, credentials):  # noqa: ARG001
        return scenes

    def _fake_open(connection, *, bands, credentials):  # noqa: ARG001
        return _synthetic_srtm(fill=fills[connection])

    monkeypatch.setattr(ee_reader, "_discover_scenes", _fake_discover)
    monkeypatch.setattr(ee_reader, "_open_eedai", _fake_open)
    return [10, 20, 30]


class TestFromEarthengineComposite:
    """Tests for the ImageCollection composite mode of :func:`from_earthengine`."""

    @pytest.mark.parametrize(
        "reducer, expected",
        [("median", 20), ("mean", 20), ("min", 10), ("max", 30)],
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
        assert layer.spatial_rect == (86.9, 27.9, 87.0, 28.0), (
            f"Unexpected spatial filter: {layer.spatial_rect}"
        )

    def test_reader_error_when_catalog_open_fails(self, monkeypatch) -> None:
        """A ``None`` catalog open raises ``ReaderError``.

        Test scenario:
            ``OpenEx`` returning ``None`` surfaces the GDAL error as ``ReaderError``.
        """
        monkeypatch.setattr(ee_reader, "gdal", _FakeEeda(None, last_error="no access"))
        with pytest.raises(ReaderError, match="no access"):
            ee_reader._discover_scenes(
                "COPERNICUS/S2_SR_HARMONIZED",
                start="2024-06-01",
                end="2024-06-30",
                bbox_4326=(0.0, 0.0, 1.0, 1.0),
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

    def test_build_like_2d(self) -> None:
        """A 2-D array is wrapped as a single-band georeferenced dataset.

        Test scenario:
            ``_build_like`` copies the template grid and stamps the nodata value.
        """
        template = _synthetic_srtm(fill=5)
        out = ee_reader._build_like(
            template, np.full((200, 200), 7, dtype="int16"), nodata=-32768
        )
        assert out.RasterCount == 1, f"Expected one band, got {out.RasterCount}"
        assert out.GetGeoTransform() == template.GetGeoTransform(), (
            "Geotransform not copied"
        )
        assert out.GetRasterBand(1).GetNoDataValue() == -32768, "Nodata not stamped"
        assert int(out.ReadAsArray()[0, 0]) == 7, "Array not written"

    def test_build_like_3d(self) -> None:
        """A 3-D array is wrapped band-for-band.

        Test scenario:
            ``_build_like`` creates one band per leading-axis slice.
        """
        template = _synthetic_srtm(fill=5)
        array = np.stack(
            [np.full((200, 200), b, dtype="int16") for b in (1, 2)], axis=0
        )
        out = ee_reader._build_like(template, array, nodata=None)
        assert out.RasterCount == 2, f"Expected two bands, got {out.RasterCount}"
        assert int(out.GetRasterBand(2).ReadAsArray()[0, 0]) == 2, (
            "Second band not written"
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
        with pytest.raises(ReaderError, match="total_bounds"):
            ee_reader._geometry_bounds(object())

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
