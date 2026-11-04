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
        pyramids_eo,
        "from_earthengine",
        lambda asset_id, *, window, bands, credentials, **o: (
            captured.update(
                asset_id=asset_id,
                bbox=window.bbox,
                bands=bands,
                credentials=credentials,
            )
            or Dataset.from_array(
                np.zeros((1, 1)),
                geo_ref=GeoReference(
                    top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326
                ),
            )
        ),
    )
    clear_driver_cache()
    out = Dataset.read_file(
        "ee://USGS/SRTMGL1_003",
        open_options={
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "bands": ["elevation"],
            "credentials": "C",
        },
    )
    clear_driver_cache()
    assert isinstance(out, Dataset)
    assert captured["asset_id"] == "USGS/SRTMGL1_003", (
        "scheme stripped and routed to EE driver"
    )
    assert captured["bbox"] == (0.0, 0.0, 1.0, 1.0)
    assert captured["bands"] == ["elevation"]
    assert captured["credentials"] == "C"


def test_s2_explicit_driver_routes_into_sentinel2(monkeypatch):
    """read_file(path, driver="s2") routes into Sentinel2Driver (GDAL bypassed)."""
    captured = {}
    monkeypatch.setattr(
        pyramids_eo,
        "from_sentinel2",
        lambda path, *, bands, bbox, crs, **o: (
            captured.update(path=path, bands=bands)
            or Dataset.from_array(
                np.zeros((1, 1)),
                geo_ref=GeoReference(
                    top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326
                ),
            )
        ),
    )
    out = Dataset.read_file("X.SAFE", driver="s2", open_options={"bands": ["B04"]})
    assert isinstance(out, Dataset)
    assert captured["path"] == "X.SAFE"
    assert captured["bands"] == ["B04"]


def test_fci_bare_directory_routes_into_fci(tmp_path, monkeypatch):
    """A bare read_file(granule_dir) GDAL cannot open falls through to FciDriver."""
    from pyramids.dataset._drivers import clear_driver_cache

    import pyramids_eo.sensors.readers as readers

    (tmp_path / "W_XX_FCI-1C-RRAD-FDHSI-FD--CHK-BODY--x.nc").write_bytes(b"")
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_fci_l1c",
        lambda paths, *, channel, channels, **o: (
            captured.update(paths=paths, channel=channel)
            or Dataset.from_array(
                np.zeros((1, 1)),
                geo_ref=GeoReference(
                    top_left_corner=(0.0, 1.0), cell_size=1.0, epsg=4326
                ),
            )
        ),
    )
    clear_driver_cache()
    out = Dataset.read_file(str(tmp_path), open_options={"channel": "ir_105"})
    clear_driver_cache()
    assert isinstance(out, Dataset)
    assert captured["channel"] == "ir_105"
    assert captured["paths"] == [
        str(tmp_path / "W_XX_FCI-1C-RRAD-FDHSI-FD--CHK-BODY--x.nc")
    ]


def test_unknown_driver_name_raises():
    """An explicit driver= that names no registered driver raises DriverNotExistError."""
    from pyramids.base._errors import DriverNotExistError

    with pytest.raises(DriverNotExistError):
        Dataset.read_file("scene.nat", driver="not_a_driver")


def test_sentinel2_preempts_the_safe_suffix():
    """`select_preempting_driver` picks the S2 driver for a bare `.SAFE` path.

    This is the opt-in itself: pre-emption is consulted only for a driver with a
    non-empty `preempts`, so before the declaration this returns None and a bare
    `.SAFE` reaches GDAL's container first.
    """
    from pyramids.dataset._drivers import clear_driver_cache, select_preempting_driver

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    clear_driver_cache()
    try:
        picked = select_preempting_driver("/data/S2A_MSIL2A_20240101.SAFE")
    finally:
        clear_driver_cache()
    assert picked is Sentinel2Driver, (
        f"a bare .SAFE should be pre-empted by the s2 driver, got {picked!r}"
    )


def test_sentinel2_preempts_a_directory_with_a_trailing_slash():
    """A `.SAFE/` path is pre-empted too — core tokenises the suffix slash-tolerantly."""
    from pyramids.dataset._drivers import clear_driver_cache, select_preempting_driver

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    clear_driver_cache()
    try:
        picked = select_preempting_driver("/data/S2A_MSIL2A_20240101.SAFE/")
    finally:
        clear_driver_cache()
    assert picked is Sentinel2Driver, (
        f"a trailing slash should not defeat pre-emption, got {picked!r}"
    )


def test_sentinel2_does_not_preempt_a_zipped_product():
    """A `.SAFE.zip` is NOT pre-empted, which is why `.safe.zip` is not declared.

    Core tokenises only the final suffix, so a zipped product presents as `.zip`:
    declaring `.safe.zip` in `preempts` could never match it, and declaring `.zip`
    would claim every zip archive. The zipped form therefore keeps the GDAL-first
    path and the explicit `driver="s2"` escape.
    """
    from pyramids.dataset._drivers import clear_driver_cache, select_preempting_driver

    clear_driver_cache()
    try:
        picked = select_preempting_driver("/data/S2A_MSIL2A_20240101.SAFE.zip")
    finally:
        clear_driver_cache()
    assert picked is None, (
        f"a zipped product should keep the GDAL-first path, got {picked!r}"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("ee://USGS/SRTMGL1_003", id="earthengine"),
        pytest.param("/data/scene.nat", id="seviri"),
        pytest.param("/data/granule", id="fci-directory"),
        pytest.param("/data/S3A_OL_1_EFR.SEN3", id="sentinel3"),
    ],
)
def test_no_other_source_is_preempted(source):
    """Only Sentinel-2 opts in; every other source keeps the GDAL-first order.

    The other five drivers read formats GDAL cannot open at all, so they already
    auto-route through the post-GDAL `can_open` scan and gain nothing from
    pre-emption. Pinned so the opt-in stays scoped to the one format that needs it.
    """
    from pyramids.dataset._drivers import clear_driver_cache, select_preempting_driver

    clear_driver_cache()
    try:
        picked = select_preempting_driver(source)
    finally:
        clear_driver_cache()
    assert picked is None, f"{source!r} should not be pre-empted, got {picked!r}"


def test_s2_preempts_declaration_is_well_formed():
    """`preempts` is lowercase and dot-prefixed, so core's token match can hit it.

    The public conformance kit does not validate `preempts`, so this is pinned
    here: a stray capital or missing dot would silently never match and the
    pre-emption would quietly not happen.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.preempts, "preempts must be non-empty to opt in"
    for entry in Sentinel2Driver.preempts:
        assert entry == entry.lower(), f"{entry!r} must be lowercase"
        assert entry.startswith("."), f"{entry!r} must be dot-prefixed"
        assert entry in Sentinel2Driver.suffixes, (
            f"{entry!r} should also be a declared suffix"
        )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("S2A_MSIL2A.SAFE/manifest.safe", id="posix-separator"),
        pytest.param(r"C:\data\S2A_MSIL2A.SAFE\manifest.safe", id="windows-separator"),
        pytest.param("manifest.safe", id="bare"),
        pytest.param("MANIFEST.SAFE", id="uppercase"),
    ],
)
def test_a_manifest_is_not_preempted(source):
    """A `manifest.safe` is not claimed, so pre-emption cannot recurse through it.

    Regression: `manifest.safe` is the metadata file *inside* a SAFE product — and
    Sentinel-1's entry point — yet it shares the `.safe` suffix. While the driver
    was only consulted after GDAL failed, a loose `can_open` was harmless. Once
    `preempts` makes the claim load-bearing, claiming the manifest sends
    `from_sentinel2` -> `open_product` -> `read_file` straight back into the same
    driver, which recursed until `RecursionError`.
    """
    from pyramids.dataset._drivers import clear_driver_cache, select_preempting_driver

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.can_open(source) is False, (
        f"{source!r} is a manifest, not a product, and must not be claimed"
    )
    clear_driver_cache()
    try:
        picked = select_preempting_driver(source)
    finally:
        clear_driver_cache()
    assert picked is None, f"{source!r} must not pre-empt GDAL, got {picked!r}"
