"""Live Sentinel-2 tests against a real CDSE granule read in place over ``/vsis3``.

These prove the ``pyramids_eo.sentinel`` pipeline on *real* pixels — something the
synthetic zero-valued fixtures in ``test_sentinel_s2.py`` cannot do. They open a
real Level-2A ``.SAFE`` straight off the Copernicus Data Space Ecosystem (CDSE)
``eodata`` S3 store, so only the requested band windows are fetched (no multi-GB
download).

Running them:
    - They are marked ``live`` and are deselected by the default suite
      (``-m "not plot and not live"``). Run them with ``pytest -m live``.
    - Reading the S3 bytes needs CDSE S3 credentials in the environment:
      ``CDSE_S3_ACCESS_KEY`` and ``CDSE_S3_SECRET_KEY`` (generate them at
      https://eodata-s3keysmanager.dataspace.copernicus.eu). The STAC *search*
      that locates a scene is anonymous; only the pixel reads are authenticated.
      Per the project's testing policy the ``live`` marker — not an environment
      check — decides whether these run; the credentials only supply the endpoint.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np
import pytest

pytestmark = pytest.mark.live

#: CDSE STAC search endpoint (anonymous) and the L2A collection id.
_STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
_COLLECTION = "sentinel-2-l2a"
#: A small, usually-clear land AOI (Madrid) to search over, in EPSG:4326.
_AOI = [-3.75, 40.38, -3.66, 40.46]
#: CDSE S3 (eodata) endpoint the signed ``/vsis3`` reads go through.
_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"


def _configure_cdse_s3() -> None:
    """Point GDAL's ``/vsis3`` reader at CDSE ``eodata`` using env credentials.

    Sets the S3 endpoint, the ``CDSE_S3_*`` access/secret keys, and the
    path-style / HTTPS options CDSE requires. Reads the keys from the
    environment (the endpoint credentials the ``live`` marker's tests need).
    """
    from osgeo import gdal

    config = {
        "AWS_S3_ENDPOINT": _S3_ENDPOINT,
        "AWS_ACCESS_KEY_ID": os.environ.get("CDSE_S3_ACCESS_KEY", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("CDSE_S3_SECRET_KEY", ""),
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_HTTPS": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    }
    for key, value in config.items():
        gdal.SetConfigOption(key, value)


def _search_clearest_l2a(months: int = 3, limit: int = 30) -> dict:
    """Return the least-cloudy L2A STAC item over the AOI in the recent window.

    Args:
        months: How many months back to search (kept small so CDSE serves the
            product from hot storage).
        limit: Maximum items to score by cloud cover.

    Returns:
        The chosen STAC item (a GeoJSON feature dict).
    """
    import datetime as dt

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=30 * months)
    body = {
        "collections": [_COLLECTION],
        "bbox": _AOI,
        "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
        "limit": limit,
    }
    request = urllib.request.Request(
        _STAC_SEARCH,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            features = json.load(response).get("features", [])
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        pytest.fail(f"CDSE STAC search failed: {exc}")
    if not features:  # pragma: no cover - depends on CDSE catalog state
        pytest.fail("CDSE STAC returned no Sentinel-2 L2A scenes for the AOI/window")
    features.sort(key=lambda f: f["properties"].get("eo:cloud_cover", 999.0))
    return features[0]


def _safe_mtd_path(item: dict) -> str:
    """Derive the ``/vsis3`` product-metadata path from a STAC item's assets.

    The item's band assets point at ``s3://eodata/<…>.SAFE/GRANULE/…``; the
    ``.SAFE`` prefix plus ``MTD_MSIL2A.xml`` is the container GDAL opens.
    """
    href = next(
        a["href"]
        for a in item["assets"].values()
        if a.get("href", "").startswith("s3://")
    )
    safe_root = href.split(".SAFE/", 1)[0] + ".SAFE"
    key = safe_root[len("s3://") :]
    return f"/vsis3/{key}/MTD_MSIL2A.xml"


def _centre_window(item: dict, metres: int = 1500) -> tuple[float, float, float, float]:
    """A small bbox at the tile centre, in the product's native UTM CRS.

    Reading a small central window keeps the S3 fetch light and lands on valid
    (non-fill) pixels.
    """
    proj_bbox = next(
        a["proj:bbox"] for a in item["assets"].values() if "proj:bbox" in a
    )
    minx, miny, maxx, maxy = proj_bbox
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half = metres / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


@pytest.fixture(scope="module")
def scene() -> dict:
    """Resolve one clear CDSE L2A scene and configure S3 access once per module."""
    _configure_cdse_s3()
    item = _search_clearest_l2a()
    return {
        "path": _safe_mtd_path(item),
        "bbox": _centre_window(item),
        "id": item["id"],
        "cloud_cover": item["properties"].get("eo:cloud_cover"),
    }


def test_open_product_reads_real_l2a_over_s3(scene):
    """A real CDSE L2A ``.SAFE`` opens over ``/vsis3`` with the expected model."""
    from pyramids_eo.sentinel import open_product
    from pyramids_eo.sentinel.s2 import S2Level

    product = open_product(scene["path"])
    assert product.level is S2Level.L2A
    assert product.quantification == 10000.0
    assert product.resolutions == [10, 20, 60]
    assert len(product.epsg_codes) == 1
    assert "SCL" in product.available_bands


def test_reflectance_is_physical_with_baseline_offset(scene):
    """A scaled read yields physical reflectance; the baseline offset is applied."""
    from pyramids_eo.sentinel import from_sentinel2

    ds = from_sentinel2(scene["path"], bands=["B04", "B03", "B02"], bbox=scene["bbox"])
    assert ds.band_count == 3
    assert ds.cell_size == 10.0

    reflectance = np.asarray(ds.read_array(scaled=True), dtype="float64")
    finite = reflectance[np.isfinite(reflectance)]
    # Surface reflectance sits in [0, ~1.6]; a clear land scene is well inside it.
    assert finite.min() >= -0.2
    assert np.percentile(finite, 99) <= 2.0
    assert 0.01 < float(np.nanmean(finite)) < 0.9
    # Post-baseline-04.00 products carry scale 1/quant and a non-zero offset.
    assert ds.scale[0] == pytest.approx(1.0 / 10000.0)
    assert ds.offset[0] <= 0.0


def test_cross_resolution_harmonise_onto_finest_grid(scene):
    """A 10 m and a 20 m band come back harmonised onto the 10 m grid."""
    from pyramids_eo.sentinel import from_sentinel2

    ds = from_sentinel2(scene["path"], bands=["B04", "B11"], bbox=scene["bbox"])
    assert ds.band_count == 2
    assert ds.cell_size == 10.0
    assert ds.shape[0] == 2


def test_scl_masking_marks_pixels_nodata(scene):
    """Masking an SCL class returns the same grid with a no-data value set."""
    from pyramids_eo.sentinel import from_sentinel2
    from pyramids_eo.sentinel.s2.masks import SclClass

    unmasked = from_sentinel2(scene["path"], bands=["B04"], bbox=scene["bbox"])
    masked = from_sentinel2(
        scene["path"],
        bands=["B04"],
        bbox=scene["bbox"],
        mask_scl=[SclClass.CLOUD_HIGH_PROBA, SclClass.CLOUD_MEDIUM_PROBA],
    )
    assert masked.shape == unmasked.shape
    assert any(v is not None for v in masked.no_data_value)
