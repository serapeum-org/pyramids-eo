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
import time
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
    features: list[dict] = []
    for attempt in range(3):  # tolerate a transient CDSE STAC hiccup
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                features = json.load(response).get("features", [])
            break
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ):  # pragma: no cover - network dependent
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if not features:  # pragma: no cover - depends on CDSE catalog state
        pytest.fail("CDSE STAC returned no Sentinel-2 L2A scenes for the AOI/window")
    features.sort(key=lambda f: _cloud_cover_key(f))
    return features[0]


def _cloud_cover_key(feature: dict) -> float:
    """Cloud-cover sort key; a missing or null value sorts last (not as 0)."""
    value = (feature.get("properties") or {}).get("eo:cloud_cover")
    return 999.0 if value is None else float(value)


def _safe_mtd_path(item: dict) -> str:
    """Derive the ``/vsis3`` product-metadata path from a STAC item's assets.

    The item's band assets point at ``s3://eodata/<…>.SAFE/GRANULE/…``; the
    ``.SAFE`` prefix plus ``MTD_MSIL2A.xml`` is the container GDAL opens.
    """
    href = next(
        a["href"]
        for a in item["assets"].values()
        if a.get("href", "").startswith("s3://") and ".SAFE/" in a["href"]
    )
    safe_root = href.split(".SAFE/", 1)[0] + ".SAFE"
    key = safe_root[len("s3://") :]
    return f"/vsis3/{key}/MTD_MSIL2A.xml"


def _centre_window(item: dict, metres: int = 1500) -> tuple[float, float, float, float]:
    """A small bbox at the tile centre, in the product's native UTM CRS.

    This is the centre of the *whole tile*, not the small lon/lat search AOI used
    to pick the scene — the search only chooses which (clearest) tile to read;
    the pixels come from its centre. Reading a small central window keeps the S3
    fetch light and lands on valid (non-fill) pixels.
    """
    proj_bbox = next(
        a["proj:bbox"] for a in item["assets"].values() if "proj:bbox" in a
    )
    minx, miny, maxx, maxy = proj_bbox
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half = metres / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


_S3_CONFIG_KEYS = (
    "AWS_S3_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_VIRTUAL_HOSTING",
    "AWS_HTTPS",
    "GDAL_DISABLE_READDIR_ON_OPEN",
)


@pytest.fixture(scope="module")
def scene():
    """Resolve one clear CDSE L2A scene, restoring the global GDAL config after.

    ``_configure_cdse_s3`` mutates process-global GDAL options (including the
    credentials); this snapshots and restores them on teardown so no state — and
    no secret — leaks to any later GDAL operation in the same pytest session.
    """
    from osgeo import gdal

    saved = {key: gdal.GetConfigOption(key) for key in _S3_CONFIG_KEYS}
    _configure_cdse_s3()
    item = _search_clearest_l2a()
    try:
        yield {
            "path": _safe_mtd_path(item),
            "bbox": _centre_window(item),
        }
    finally:
        for key, value in saved.items():
            gdal.SetConfigOption(key, value)


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
    # Post-baseline-04.00 products carry scale 1/quant and a strictly negative
    # offset; a recent scene is always baseline >= 04.00, so 0.0 would mean the
    # offset was silently dropped.
    assert ds.scale[0] == pytest.approx(1.0 / 10000.0)
    assert ds.offset[0] < 0.0


def test_cross_resolution_harmonise_onto_finest_grid(scene):
    """A 10 m and a 20 m band come back harmonised onto the 10 m grid."""
    from pyramids_eo.sentinel import from_sentinel2

    ds = from_sentinel2(scene["path"], bands=["B04", "B11"], bbox=scene["bbox"])
    assert ds.band_count == 2
    assert ds.cell_size == 10.0
    # The resampled 20 m band (B11) must carry real values, not an empty grid.
    b11 = np.asarray(ds.read_array())[1]
    valid = b11[b11 != ds.no_data_value[1]]
    assert valid.size > 0, "harmonised B11 is entirely no-data"
    assert np.any(valid > 0), "harmonised B11 has no positive values"


def test_scl_masking_is_class_sensitive(scene):
    """Masking a class present in the window drops more pixels than an absent one.

    Reads the SCL band to learn which classes are actually present, then masks
    B04 twice — once with the most-common present class, once with a class that
    does not occur in the window — and asserts the present-class mask sets
    strictly more pixels to no-data. This fails if ``mask_scl=`` were ignored or
    ``scl_mask`` were a no-op (both counts would be equal), without depending on
    the reader's masked-vs-unmasked grid staying identical.
    """
    from pyramids_eo.sentinel import from_sentinel2
    from pyramids_eo.sentinel.s2.masks import SclClass

    scl = np.asarray(
        from_sentinel2(scene["path"], bands=["SCL"], bbox=scene["bbox"]).read_array()
    )
    present = {int(c) for c in np.unique(scl)} - {int(SclClass.NODATA)}
    assert present, "window has no maskable SCL class"
    target = SclClass(max(present, key=lambda c: int(np.count_nonzero(scl == c))))
    absent_code = next((c for c in range(1, 12) if c not in present), None)
    if absent_code is None:  # pragma: no cover - a 1500 m window never holds all 11
        pytest.skip("window contains every SCL class; no absent class to compare")
    absent = SclClass(absent_code)

    masked_present = from_sentinel2(
        scene["path"], bands=["B04"], bbox=scene["bbox"], mask_scl=[target]
    )
    masked_absent = from_sentinel2(
        scene["path"], bands=["B04"], bbox=scene["bbox"], mask_scl=[absent]
    )
    present_nodata = int(
        np.count_nonzero(
            np.asarray(masked_present.read_array()) == masked_present.no_data_value[0]
        )
    )
    absent_nodata = int(
        np.count_nonzero(
            np.asarray(masked_absent.read_array()) == masked_absent.no_data_value[0]
        )
    )
    assert present_nodata > absent_nodata, (
        "masking a present class removed no more pixels than masking an absent one"
    )


def test_grid_is_stable_across_band_selection(scene):
    """The output grid is identical for one band, many bands, and a masked read.

    A bbox read must return the same rows x cols regardless of how many bands are
    requested or whether SCL masking is applied — the invariant #81 broke, where
    a single-band or masked read shrank to the valid-data extent while a
    multi-band read kept the full window.
    """
    from pyramids_eo.sentinel import from_sentinel2
    from pyramids_eo.sentinel.s2.masks import SclClass

    one = from_sentinel2(scene["path"], bands=["B04"], bbox=scene["bbox"])
    many = from_sentinel2(scene["path"], bands=["B04", "SCL"], bbox=scene["bbox"])
    masked = from_sentinel2(
        scene["path"],
        bands=["B04"],
        bbox=scene["bbox"],
        mask_scl=[SclClass.VEGETATION],
    )
    grid = one.shape[1:]
    # Absolute check (not just one == many): the 1500 m window at 10 m is the
    # full ~150x150 grid, so a systematic shrink (e.g. the #81 trim to the valid
    # extent) fails here even if it hit every band count equally.
    assert abs(grid[0] - 150) <= 1, f"grid rows {grid[0]} not the full ~150 window"
    assert abs(grid[1] - 150) <= 1, f"grid cols {grid[1]} not the full ~150 window"
    assert many.shape[1:] == grid, f"multi-band grid {many.shape[1:]} != {grid}"
    assert masked.shape[1:] == grid, f"masked grid {masked.shape[1:]} != {grid}"
