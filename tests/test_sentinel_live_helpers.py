"""Offline unit tests for the pure helpers in `test_sentinel_live`.

The live tests need network and CDSE credentials, but their path- and
window-deriving helpers are pure functions whose logic should be guarded in the
default (offline) suite too. These import those helpers and exercise them
against fake STAC items — no network, and no `live` marker, so they run in the
normal suite.
"""

from __future__ import annotations

import pytest

from tests.test_sentinel_live import (
    _centre_window,
    _configure_cdse_s3,
    _safe_mtd_path,
)

_SAFE = "S2A_MSIL2A_20240820T105621_N0511_R094_T30TVK_20240820T152852.SAFE"


def _fake_item() -> dict:
    """A minimal STAC item with a non-s3 asset, an s3 band href, and a proj bbox."""
    return {
        "assets": {
            "thumbnail": {"href": "https://download.example/preview.jpg"},
            "B04_10m": {
                "href": (
                    f"s3://eodata/Sentinel-2/MSI/L2A/2024/08/20/{_SAFE}/"
                    "GRANULE/L2A_T30TVK/IMG_DATA/R10m/T30TVK_B04_10m.jp2"
                ),
                "proj:bbox": [399960, 4390200, 509760, 4500000],
            },
        }
    }


class TestSafeMtdPath:
    """Tests for `_safe_mtd_path`."""

    def test_derives_vsis3_metadata_path_from_s3_asset(self):
        """The `.SAFE` prefix of an s3 band href becomes a `/vsis3` MTD path.

        Test scenario:
            A band href under a `.SAFE` product resolves to that product's
            `MTD_MSIL2A.xml` rewritten as a `/vsis3/eodata/...` path.
        """
        path = _safe_mtd_path(_fake_item())
        expected = f"/vsis3/eodata/Sentinel-2/MSI/L2A/2024/08/20/{_SAFE}/MTD_MSIL2A.xml"
        assert path == expected, f"Expected {expected}, got {path}"

    def test_skips_non_s3_assets(self):
        """A leading non-s3 (https) asset is skipped in favour of the s3 one.

        Test scenario:
            The first asset is an https thumbnail; the resolver must ignore it
            and derive the path from the s3 band asset.
        """
        path = _safe_mtd_path(_fake_item())
        assert path.startswith("/vsis3/eodata/"), f"Not a /vsis3 path: {path}"
        assert path.endswith("/MTD_MSIL2A.xml"), f"Not an MTD path: {path}"


class TestCentreWindow:
    """Tests for `_centre_window`."""

    def test_centres_a_default_window_on_the_tile(self):
        """The window is centred on the proj bbox with the default 1500 m size.

        Test scenario:
            A 1500 m window is centred on the mid-point of the tile's projected
            bbox, so both edges span 1500 m and the centre matches the bbox mid.
        """
        minx, miny, maxx, maxy = _centre_window(_fake_item())
        assert (maxx - minx, maxy - miny) == (1500.0, 1500.0), "window not 1500 m"
        assert (minx + maxx) / 2 == pytest.approx((399960 + 509760) / 2), "x off-centre"
        assert (miny + maxy) / 2 == pytest.approx((4390200 + 4500000) / 2), (
            "y off-centre"
        )

    def test_respects_a_custom_window_size(self):
        """A custom `metres` sets the window edge length.

        Test scenario:
            Passing metres=800 yields an 800 m square window.
        """
        minx, _, maxx, _ = _centre_window(_fake_item(), metres=800)
        assert maxx - minx == pytest.approx(800.0), f"edge not 800 m: {maxx - minx}"


class TestConfigureCdseS3:
    """Tests for `_configure_cdse_s3`."""

    def test_sets_gdal_s3_config_from_env(self, monkeypatch):
        """The CDSE keys from the environment land in GDAL's S3 config options.

        Test scenario:
            With `CDSE_S3_ACCESS_KEY` / `CDSE_S3_SECRET_KEY` set, the helper
            writes them plus the fixed CDSE endpoint into GDAL's config; the
            options are restored afterwards so no state leaks to other tests.
        """
        import pyramids  # noqa: F401  (activates the bundled GDAL)
        from osgeo import gdal

        touched = (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT",
            "AWS_VIRTUAL_HOSTING",
            "AWS_HTTPS",
            "GDAL_DISABLE_READDIR_ON_OPEN",
        )
        saved = {k: gdal.GetConfigOption(k) for k in touched}
        monkeypatch.setenv("CDSE_S3_ACCESS_KEY", "ACCESS_placeholder")
        monkeypatch.setenv("CDSE_S3_SECRET_KEY", "SECRET_placeholder")
        try:
            _configure_cdse_s3()
            assert gdal.GetConfigOption("AWS_ACCESS_KEY_ID") == "ACCESS_placeholder"
            assert gdal.GetConfigOption("AWS_SECRET_ACCESS_KEY") == "SECRET_placeholder"
            assert (
                gdal.GetConfigOption("AWS_S3_ENDPOINT")
                == "eodata.dataspace.copernicus.eu"
            )
        finally:
            for key, value in saved.items():
                gdal.SetConfigOption(key, value)
