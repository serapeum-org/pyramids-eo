"""Sentinel-2 reader tests against GDAL's synthetic L1C / L2A fixtures.

The fixtures are structural stubs (zero-valued JP2s), so tests assert structure,
metadata, CRS, band selection, reflectance tags, and masking — not pixel values.
A real-granule check belongs behind the ``live`` marker.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids_eo.errors import ProductError
from pyramids_eo.sentinel import from_sentinel2, open_product, scl_mask
from pyramids_eo.sentinel.s2 import S2Level, S2Product
from pyramids_eo.sentinel.s2.masks import SclClass

_DATA = Path(__file__).parent / "data" / "sentinel2"
_L2A = _DATA / "fake_l2a" / "S2A_USER_PRD_MSIL2A.SAFE" / "S2A_USER_MTD_SAFL2A.xml"
_L2A_SAFE = _DATA / "fake_l2a" / "S2A_USER_PRD_MSIL2A.SAFE"
_L1C = _DATA / "fake_l1c" / "S2A_OPER_PRD_MSIL1C.SAFE" / "S2A_OPER_MTD_SAFL1C.xml"


@pytest.fixture(scope="session")
def l2a_zip(tmp_path_factory) -> Path:
    """Zip the L2A ``.SAFE`` fixture so the ``/vsizip/`` read path can be tested.

    Built at test time rather than committed as a binary.
    """
    import shutil

    out = tmp_path_factory.mktemp("s2zip") / "fake_l2a"
    archive = shutil.make_archive(str(out), "zip", root_dir=str(_DATA / "fake_l2a"))
    return Path(archive)


# -- product model ---------------------------------------------------------


def test_open_product_l2a_model():
    product = open_product(_L2A)
    assert isinstance(product, S2Product)
    assert product.level is S2Level.L2A
    assert product.mission == "sentinel-2"
    assert product.product_type.startswith("S2MSI2A")
    assert product.baseline == "01.03"
    assert product.quantification == 1000.0
    assert product.cloud_cover == 0.0
    assert product.resolutions == [60]
    assert product.epsg_codes == [32632]
    assert "SCL" in product.available_bands


def test_open_product_opens_safe_directory():
    # GDAL opens the .SAFE directory as well as the metadata XML.
    product = open_product(_L2A_SAFE)
    assert isinstance(product, S2Product)
    assert product.level is S2Level.L2A


def test_open_product_reads_zip_in_place(l2a_zip):
    # A .zip is read via /vsizip/ without extraction; the reader works through it.
    product = open_product(l2a_zip)
    assert isinstance(product, S2Product)
    assert product.level is S2Level.L2A
    ds = from_sentinel2(l2a_zip, bands=["B04"])
    assert ds.band_count == 1


def test_open_product_l1c_level():
    product = open_product(_L1C)
    assert product.level is S2Level.L1C
    assert any(b.upper().startswith("B") for b in product.available_bands)


def test_band_name_alias_b4_equals_b04():
    product = open_product(_L2A)
    assert product.resolution_of("B4") == product.resolution_of("B04") == 60


def test_missing_band_raises():
    product = open_product(_L2A)
    with pytest.raises(ProductError):
        product.resolution_of("B99")


def test_subdataset_for_ambiguity_and_miss():
    product = open_product(_L2A)
    assert product.subdataset_for(60).resolution_m == 60
    with pytest.raises(ProductError):
        product.subdataset_for(10)  # not in this 60m-only fixture


# -- reader ----------------------------------------------------------------


def test_from_sentinel2_selected_bands_and_order():
    ds = from_sentinel2(_L2A, bands=["B8A", "B04"])
    assert ds.band_count == 2
    # Order follows the request (B8A first).
    assert ds.band_names[0].upper().startswith("B8A")
    assert ds.epsg == 32632
    assert ds.cell_size == 60.0


def test_default_bands_are_all_spectral():
    ds = from_sentinel2(_L2A)
    # B1..B12 minus B8 (10 m only, absent here) plus B8A = 12.
    assert ds.band_count == 12


def test_reflectance_tags_and_scaled_read():
    product = open_product(_L2A)
    ds = from_sentinel2(product, bands=["B04", "B8A"])
    quant = product.quantification
    assert ds.scale == pytest.approx([1.0 / quant, 1.0 / quant])
    assert ds.offset == pytest.approx([0.0, 0.0])
    raw = ds.read_array()
    scaled = ds.read_array(scaled=True)
    assert np.allclose(scaled, raw / quant)


def test_reflectance_false_leaves_identity_scale():
    ds = from_sentinel2(_L2A, bands=["B04"], reflectance=False)
    assert ds.scale == pytest.approx([1.0])
    assert ds.offset == pytest.approx([0.0])


def test_reproject_to_wgs84():
    ds = from_sentinel2(_L2A, bands=["B04"], crs=4326)
    assert ds.epsg == 4326


def test_requesting_absent_band_raises():
    with pytest.raises(ProductError):
        from_sentinel2(_L2A, bands=["B08"])  # 10 m band, absent from 60m fixture


# -- SCL masking -----------------------------------------------------------


def test_scl_mask_via_reader_sets_nodata_shape_preserved():
    ds = from_sentinel2(
        _L2A, bands=["B04", "B8A"], mask_scl=[SclClass.CLOUD_HIGH_PROBA, "CLOUD_SHADOW"]
    )
    assert ds.band_count == 2
    assert ds.shape == (2, 1830, 1830)
    assert all(v is not None for v in ds.no_data_value)


def test_scl_mask_standalone_finds_embedded_scl():
    # Read a subdataset that includes SCL, then mask via the embedded band.
    product = open_product(_L2A)
    full = product.subdataset_for(60).open()
    masked = scl_mask(full, [SclClass.WATER])
    assert masked.band_count == full.band_count


def test_scl_mask_unknown_class_name_raises():
    product = open_product(_L2A)
    full = product.subdataset_for(60).open()
    with pytest.raises(ProductError):
        scl_mask(full, ["NOT_A_CLASS"])
