"""Integration: `Dataset.read_file` routes a claimed source into a pyramids-eo driver.

Exercised through the SEVIRI driver, which is discovered unconditionally (no backend extra). The
Earth Engine driver shares the same dispatch but is gated behind the `ee` extra, so it is covered
at the unit level in `test_drivers_earthengine.py` instead.
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference


def _parser(path, channel):
    """A `.nat` decoder stub returning a constant synthetic radiance Dataset."""
    return Dataset.from_array(
        np.full((2, 2), 7.0),
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    )


def test_explicit_driver_routes_into_seviri():
    """`read_file(driver="seviri", ...)` opens a .nat source through the pyramids-eo driver."""
    out = Dataset.read_file(
        "scene.nat",
        driver="seviri",
        open_options={"channel": "IR_108", "calibrate": False, "parse": _parser},
    )
    assert isinstance(out, Dataset)
    assert np.allclose(out.read_array(), 7.0), (
        "driver output should carry the decoded radiance"
    )


def test_can_open_fallback_routes_into_seviri():
    """A bare `read_file(".nat")` GDAL cannot open falls through to the claiming driver."""
    out = Dataset.read_file(
        "no_such_scene.nat",
        open_options={"channel": "IR_108", "calibrate": False, "parse": _parser},
    )
    assert isinstance(out, Dataset)
    assert np.allclose(out.read_array(), 7.0)


def test_dispatch_matches_the_facade():
    """The dispatch path returns the same values as calling read_seviri directly."""
    from pyramids_eo.sensors.readers import read_seviri

    via_dispatch = Dataset.read_file(
        "scene.nat",
        driver="seviri",
        open_options={"channel": "IR_108", "calibrate": False, "parse": _parser},
    )
    via_facade = read_seviri("scene.nat", "IR_108", calibrate=False, parse=_parser)
    assert np.allclose(via_dispatch.read_array(), via_facade.read_array())


def test_unknown_driver_name_raises():
    """An explicit driver= that names no registered driver raises."""
    with pytest.raises(Exception):  # DriverNotExistError from pyramids
        Dataset.read_file("scene.nat", driver="not_a_driver")
