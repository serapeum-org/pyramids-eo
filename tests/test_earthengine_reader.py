"""Tests for :mod:`pyramids_eo.earthengine.reader`.

The EEDAI network open is the single seam (``_open_eedai``); the offline tests
either monkeypatch it with a synthetic in-memory raster or drive it with a faked
``gdal`` so CI needs no live Earth Engine account. A ``live`` test exercises the
real driver end-to-end and is deselected by the default ``-m "not live"`` run.
"""

from __future__ import annotations

import pytest
from pyramids.dataset import Dataset

import pyramids_eo.earthengine.reader as ee_reader
from pyramids_eo import from_earthengine
from pyramids_eo.earthengine import EarthEngineCredentials
from pyramids_eo.errors import ReaderError

_BBOX = (86.9, 27.9, 87.0, 28.0)


def _synthetic_srtm():
    """Build a 200x200 EPSG:4326 Int16 raster over lon [86, 88], lat [27, 29].

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
    band.Fill(42)
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
