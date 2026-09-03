"""The composite/enhance wrappers keep a non-EPSG (geostationary) CRS."""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from pyramids_eo.composites._common import _wrap_like
from pyramids_eo.enhance import _wrap

# A minimal geostationary WKT: it carries no EPSG authority, so Dataset.epsg is
# None while Dataset.crs holds the projection — the SEVIRI / FCI L1C shape.
_GEOS_WKT = (
    'PROJCS["geos",GEOGCS["sphere",DATUM["D",SPHEROID["S",6378169,295.488065897]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Geostationary_Satellite"],PARAMETER["central_meridian",0],'
    'PARAMETER["satellite_height",35785831],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)


@pytest.fixture
def geostationary():
    """A 2x2 dataset whose CRS has no EPSG code."""
    ds = Dataset.from_array(
        np.ones((2, 2)),
        geo_ref=GeoReference(geo=(0.0, 1000.0, 0.0, 2000.0, 0.0, -1000.0), epsg=None),
        no_data_value=np.nan,
    )
    ds.crs = _GEOS_WKT
    return ds


def test_geostationary_template_has_no_epsg_but_has_crs(geostationary):
    """The fixture reproduces the condition the fix targets."""
    assert geostationary.epsg is None
    assert "Geostationary" in geostationary.crs


def test_wrap_like_carries_the_non_epsg_crs(geostationary):
    """_wrap_like keeps the geostationary projection rather than dropping it."""
    out = _wrap_like(np.zeros((2, 2)), geostationary)
    assert out.crs, "_wrap_like dropped the CRS of a geostationary template"
    assert "Geostationary" in out.crs


def test_wrap_carries_the_non_epsg_crs(geostationary):
    """_wrap keeps the geostationary projection rather than dropping it."""
    out = _wrap(np.zeros((2, 2)), geostationary, "float32")
    assert out.crs, "_wrap dropped the CRS of a geostationary template"
    assert "Geostationary" in out.crs


def test_wrap_like_still_carries_an_epsg_template():
    """An EPSG-coded template is unaffected by the non-EPSG branch."""
    src = Dataset.from_array(
        np.ones((2, 2)),
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
    )
    out = _wrap_like(np.zeros((2, 2)), src)
    assert out.epsg == 4326
