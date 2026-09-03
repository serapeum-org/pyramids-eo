"""A CRS with no EPSG code survives the composite / enhance / reader wrappers."""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from pyramids_eo.composites._common import _wrap_like
from pyramids_eo.enhance import _wrap, stretch

# A minimal geostationary WKT: it carries no EPSG authority, so Dataset.epsg is
# None while Dataset.crs holds the projection — the SEVIRI / FCI L1C shape.
_GEOS_WKT = (
    'PROJCS["geos",GEOGCS["sphere",DATUM["D",SPHEROID["S",6378169,295.488065897]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Geostationary_Satellite"],PARAMETER["central_meridian",0],'
    'PARAMETER["satellite_height",35785831],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)

_GEO = (0.0, 1000.0, 0.0, 2000.0, 0.0, -1000.0)


def _geostationary(values=None):
    """A 2x2 dataset whose CRS has no EPSG code."""
    arr = np.ones((2, 2)) if values is None else np.asarray(values, dtype=float)
    ds = Dataset.from_array(
        arr, geo_ref=GeoReference(geo=_GEO, epsg=None), no_data_value=np.nan
    )
    ds.crs = _GEOS_WKT
    return ds


@pytest.fixture
def geostationary():
    """A geostationary-georeferenced dataset with no EPSG code."""
    return _geostationary()


def test_geostationary_template_has_no_epsg_but_has_crs(geostationary):
    """The fixture reproduces the precondition the fix targets."""
    assert geostationary.epsg is None, "fixture should have no EPSG code"
    assert "Geostationary" in geostationary.crs, "fixture should carry a geos WKT"


@pytest.mark.parametrize("wrapper", ["_wrap_like", "_wrap"])
def test_wrappers_carry_the_non_epsg_crs(geostationary, wrapper):
    """Both result wrappers keep the geostationary projection."""
    out = (
        _wrap_like(np.zeros((2, 2)), geostationary)
        if wrapper == "_wrap_like"
        else _wrap(np.zeros((2, 2)), geostationary, "float32")
    )
    assert out.crs, f"{wrapper} dropped the CRS of a geostationary template"
    assert "Geostationary" in out.crs, f"{wrapper} did not preserve the geos WKT"


@pytest.mark.parametrize("wrapper", ["_wrap_like", "_wrap"])
def test_wrappers_preserve_the_geotransform(geostationary, wrapper):
    """Carrying the CRS must not disturb the georeference."""
    out = (
        _wrap_like(np.zeros((2, 2)), geostationary)
        if wrapper == "_wrap_like"
        else _wrap(np.zeros((2, 2)), geostationary, "float32")
    )
    assert tuple(out.geotransform) == _GEO, f"{wrapper} altered the geotransform"


def test_wrap_like_tolerates_a_template_without_crs():
    """The template predicate requires read_array + geotransform only."""

    class Duck:
        geotransform = _GEO
        epsg = None

        def read_array(self, *args, **kwargs):
            return np.zeros((2, 2))

    out = _wrap_like(np.zeros((2, 2)), Duck())
    assert out is not None, "a crs-less duck template should not raise"


def test_wrap_like_keeps_an_epsg_template_epsg_coded():
    """An EPSG-coded template keeps its code and gains a real CRS."""
    src = Dataset.from_array(
        np.ones((2, 2)),
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
    )
    out = _wrap_like(np.zeros((2, 2)), src)
    assert out.epsg == 4326, f"expected EPSG 4326, got {out.epsg}"
    assert out.crs, "an EPSG-coded template should still produce a CRS"


def test_public_enhance_entry_point_preserves_the_crs():
    """stretch — a function users actually call — keeps the projection."""
    out = stretch(_geostationary([[0.1, 0.2], [0.3, 0.4]]))
    assert out.crs, "stretch dropped the geostationary CRS"
    assert "Geostationary" in out.crs, "stretch lost the geos WKT"
