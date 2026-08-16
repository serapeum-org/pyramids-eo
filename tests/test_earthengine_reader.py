"""Tests for :func:`from_earthengine`.

The EEDAI network open is the single seam (``_open_eedai``); the offline tests
monkeypatch it with a synthetic in-memory raster so CI needs no live Earth
Engine account. A ``live`` test exercises the real driver end-to-end and is
deselected by the default ``-m "not live"`` run.
"""

from __future__ import annotations

import pytest
from pyramids.dataset import Dataset

import pyramids_eo.earthengine.reader as ee_reader
from pyramids_eo import from_earthengine


def _synthetic_srtm():
    """A 200x200 EPSG:4326 Int16 raster over lon [86, 88], lat [27, 29] (0.01 deg)."""
    from osgeo import gdal, osr

    src = gdal.GetDriverByName("MEM").Create("", 200, 200, 1, gdal.GDT_Int16)
    # origin (upper-left) 86,29 ; 0.01 deg pixels, north-up
    src.SetGeoTransform((86.0, 0.01, 0.0, 29.0, 0.0, -0.01))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    src.SetProjection(srs.ExportToWkt())
    band = src.GetRasterBand(1)
    band.Fill(42)
    band.SetNoDataValue(-32768)
    return src


@pytest.fixture
def patched_eedai(monkeypatch):
    """Replace the EEDAI open seam with the synthetic raster."""

    def _fake_open(asset_id, *, bands, credentials):  # noqa: ARG001
        return _synthetic_srtm()

    monkeypatch.setattr(ee_reader, "_open_eedai", _fake_open)


def test_from_earthengine_windows_to_bbox(patched_eedai) -> None:
    ds = from_earthengine("USGS/SRTMGL1_003", bbox=(86.9, 27.9, 87.0, 28.0))
    assert isinstance(ds, Dataset)
    assert ds.epsg == 4326
    # 0.1 deg / 0.01 deg source resolution -> ~10x10 window
    _bands, rows, cols = ds.shape
    assert rows == pytest.approx(10, abs=1)
    assert cols == pytest.approx(10, abs=1)


def test_from_earthengine_honours_shape(patched_eedai) -> None:
    ds = from_earthengine(
        "USGS/SRTMGL1_003", bbox=(86.9, 27.9, 87.0, 28.0), shape=(5, 5)
    )
    assert ds.shape == (1, 5, 5)


def test_scale_and_shape_are_mutually_exclusive(patched_eedai) -> None:
    with pytest.raises(ValueError, match="scale.*shape"):
        from_earthengine(
            "USGS/SRTMGL1_003",
            bbox=(86.9, 27.9, 87.0, 28.0),
            scale=0.01,
            shape=(5, 5),
        )


def test_windowing_options_require_bbox(patched_eedai) -> None:
    from pyramids_eo.errors import ReaderError

    with pytest.raises(ReaderError, match="bbox"):
        from_earthengine("USGS/SRTMGL1_003", scale=0.01)


def test_no_bbox_wraps_full_asset(patched_eedai) -> None:
    ds = from_earthengine("USGS/SRTMGL1_003")
    assert isinstance(ds, Dataset)
    assert ds.shape == (1, 200, 200)


@pytest.mark.live
def test_from_earthengine_live_srtm() -> None:
    """End-to-end read of a public EE asset (needs ADC / service-account creds)."""
    ds = from_earthengine("USGS/SRTMGL1_003", bbox=(86.9, 27.9, 87.0, 28.0))
    assert isinstance(ds, Dataset)
    assert ds.epsg == 4326
    _bands, rows, cols = ds.shape
    assert rows > 0 and cols > 0
