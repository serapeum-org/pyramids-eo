"""Integration: `Dataset.read_file` routes a claimed source into a pyramids-eo driver.

Both pyramids-eo drivers are discovered unconditionally (neither gates on an optional backend),
so both are exercised through the real dispatch here: SEVIRI with an injected `.nat` decoder, and
Earth Engine with `from_earthengine` monkeypatched (read_file forwards only `open_options`, which
is the path this proves).
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

import pyramids_eo


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


def test_ee_source_routes_into_earthengine_via_open_options(monkeypatch):
    """A bare `read_file("ee://…")` routes into EarthEngineDriver, carrying open_options through.

    read_file forwards only open_options to the driver, so the window/bands/credentials the EE
    read needs are supplied there; this proves that whole path end to end.
    """
    from pyramids.dataset._drivers import clear_driver_cache

    captured = {}
    monkeypatch.setattr(
        pyramids_eo, "from_earthengine",
        lambda asset_id, *, window, bands, credentials, **o: captured.update(
            asset_id=asset_id, bbox=window.bbox, bands=bands, credentials=credentials
        )
        or Dataset.from_array(
            np.zeros((1, 1)),
            geo_ref=GeoReference(top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326),
        ),
    )
    clear_driver_cache()
    out = Dataset.read_file(
        "ee://USGS/SRTMGL1_003",
        open_options={"bbox": (0.0, 0.0, 1.0, 1.0), "bands": ["elevation"], "credentials": "C"},
    )
    clear_driver_cache()
    assert isinstance(out, Dataset)
    assert captured["asset_id"] == "USGS/SRTMGL1_003", "scheme stripped and routed to EE driver"
    assert captured["bbox"] == (0.0, 0.0, 1.0, 1.0)
    assert captured["bands"] == ["elevation"]
    assert captured["credentials"] == "C"


def test_unknown_driver_name_raises():
    """An explicit driver= that names no registered driver raises."""
    with pytest.raises(Exception):  # DriverNotExistError from pyramids
        Dataset.read_file("scene.nat", driver="not_a_driver")
