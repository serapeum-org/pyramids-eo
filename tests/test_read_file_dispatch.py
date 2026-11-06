"""Integration: `Dataset.read_file` routes a claimed source into a pyramids-eo driver.

Both pyramids-eo drivers are discovered unconditionally (neither gates on an optional backend),
so both are exercised through the real dispatch here: SEVIRI with an injected `.nat` decoder, and
Earth Engine with `from_earthengine` monkeypatched (read_file forwards only `open_options`, which
is the path this proves).
"""

import pathlib

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


def test_sentinel2_preempts_a_zipped_product(tmp_path):
    """`select_preempting_driver` picks the s2 driver for a real `.SAFE.zip`.

    The zipped form is the one GDAL opens and hands back an empty container for,
    so it is the form worth pre-empting. A real archive is required because the
    claim is decided by what is inside it, not by the name.
    """
    from pyramids.dataset._drivers import clear_driver_cache, select_preempting_driver

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    archive = _zip_product(_S2_PRODUCT, tmp_path, "S2A_OPER_PRD_MSIL1C.SAFE")
    clear_driver_cache()
    try:
        picked = select_preempting_driver(archive)
    finally:
        clear_driver_cache()
    assert picked is Sentinel2Driver, (
        f"a zipped product should be pre-empted by the s2 driver, got {picked!r}"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("/data/archive.zip", id="plain-archive"),
        pytest.param("/data/backup.ZIP", id="uppercase-archive"),
    ],
)
def test_an_ordinary_archive_is_not_preempted(source):
    """Declaring the bare `.zip` does not hand every archive to this driver.

    Test scenario:
        `preempts` is a coarse suffix declaration; `can_open` gates every claim,
        so a zip that is not a `.SAFE.zip` is left to GDAL. Pinned because the
        declaration alone looks far broader than the behaviour.
    """
    from pyramids.dataset._drivers import clear_driver_cache, select_preempting_driver

    clear_driver_cache()
    try:
        picked = select_preempting_driver(source)
    finally:
        clear_driver_cache()
    assert picked is None, (
        f"{source!r} is not a product and must not be claimed, got {picked!r}"
    )


def test_s2_preempts_declaration_is_well_formed():
    """`preempts` is lowercase and dot-prefixed, so core's token match can hit it.

    The conformance kit only partly covers this: it rejects an uppercase entry
    (`(".ZIP",)` fails `assert_raster_driver_conforms`) but accepts a missing dot
    (`("zip",)`) and a scheme-shaped one (`(".zip://",)`), either of which would
    silently never match and quietly disable the pre-emption. Note the entry is
    deliberately *not* required to appear in `suffixes` — `.zip` is a container
    suffix, not a product one.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.preempts, "preempts must be non-empty to opt in"
    for entry in Sentinel2Driver.preempts:
        assert entry == entry.lower(), f"{entry!r} must be lowercase"
        assert entry.startswith("."), f"{entry!r} must be dot-prefixed"
        assert "://" not in entry, f"{entry!r} is a suffix, not a scheme"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("S2A_MSIL2A.SAFE/manifest.safe", id="posix-separator"),
        pytest.param(r"C:\data\S2A_MSIL2A.SAFE\manifest.safe", id="windows-separator"),
        pytest.param("manifest.safe", id="bare"),
        pytest.param("MANIFEST.SAFE", id="uppercase"),
    ],
)
def test_a_manifest_is_not_claimed(source):
    """`can_open` rejects a `manifest.safe`: the metadata file, not the product.

    Test scenario:
        A manifest shares the `.safe` suffix but is the file *inside* a product,
        and is Sentinel-1's entry point. Claiming it sends `from_sentinel2` ->
        `open_product` -> `read_file` back into this driver.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.can_open(source) is False, (
        f"{source!r} is a manifest, not a product, and must not be claimed"
    )


def test_a_bare_zipped_product_reads_through_preemption(tmp_path):
    """End to end: a bare `read_file` of a `.SAFE.zip` reads the product.

    Test scenario:
        This is the behaviour the pre-emption exists for, exercised through the
        real `Dataset.read_file` rather than through `select_preempting_driver`.
        Without the opt-in GDAL opens the archive first and returns an empty
        container (0 bands) while silently ignoring `open_options`; with it the
        pyramids-eo reader runs and returns the requested band.
    """
    import shutil

    from pyramids.dataset._drivers import clear_driver_cache

    product = (
        pathlib.Path(__file__).parent
        / "data"
        / "sentinel2"
        / "fake_l1c"
        / "S2A_OPER_PRD_MSIL1C.SAFE"
    )
    archive = tmp_path / "S2A_OPER_PRD_MSIL1C.SAFE"
    shutil.make_archive(
        str(archive), "zip", root_dir=str(product.parent), base_dir=product.name
    )

    clear_driver_cache()
    try:
        out = Dataset.read_file(
            str(archive) + ".zip", open_options={"bands": ["B1"], "resolution": 60}
        )
    finally:
        clear_driver_cache()
    assert isinstance(out, Dataset)
    assert out.band_count == 1, (
        f"the pyramids-eo reader should return the requested band, got "
        f"{out.band_count} (0 means GDAL's container won the race)"
    )


def _zip_product(product, tmp_path, stem):
    """Zip a `.SAFE` fixture directory and return the archive path as a string."""
    import shutil

    archive = tmp_path / stem
    shutil.make_archive(
        str(archive), "zip", root_dir=str(product.parent), base_dir=product.name
    )
    return str(archive) + ".zip"


_DATA = pathlib.Path(__file__).parent / "data"
_S2_PRODUCT = _DATA / "sentinel2" / "fake_l1c" / "S2A_OPER_PRD_MSIL1C.SAFE"
_S1_PRODUCT = _DATA / "sentinel1" / "test.SAFE"


def test_a_sentinel1_product_directory_is_not_claimed():
    """`can_open` rejects a Sentinel-1 `.SAFE`, which GDAL reads itself.

    Test scenario:
        ESA shares the `.SAFE` suffix between missions, so claiming by suffix
        would take Sentinel-1 products into the Sentinel-2 reader and fail on
        them. The driver distinguishes them by looking for Sentinel-2 metadata.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.can_open(str(_S2_PRODUCT)) is True, (
        "a real Sentinel-2 product must still be claimed"
    )
    assert Sentinel2Driver.can_open(str(_S1_PRODUCT)) is False, (
        "a Sentinel-1 product shares the suffix but must be left to GDAL"
    )


def test_a_zipped_sentinel1_product_is_not_preempted(tmp_path):
    """A zipped Sentinel-1 product keeps GDAL's read.

    Test scenario:
        `.SAFE.zip` is shared across missions exactly as `.SAFE` is, so
        pre-empting `.zip` would break zipped Sentinel-1 reads unless the claim
        is mission-aware. Exercised through `Dataset.read_file`, since this is
        the regression a declaration-only test cannot see.
    """
    from pyramids.dataset._drivers import clear_driver_cache, select_preempting_driver

    archive = _zip_product(_S1_PRODUCT, tmp_path, "S1A_IW_GRDH.SAFE")
    clear_driver_cache()
    try:
        picked = select_preempting_driver(archive)
        out = Dataset.read_file(archive)
    finally:
        clear_driver_cache()
    assert picked is None, (
        f"a zipped Sentinel-1 product must not be pre-empted, got {picked!r}"
    )
    assert out.band_count == 2, (
        f"GDAL should still serve the Sentinel-1 product, got {out.band_count} bands"
    )


def test_an_uppercase_zip_extension_is_not_claimed(tmp_path):
    """A `.SAFE.ZIP` is left to GDAL, because the archive reader cannot take it.

    Test scenario:
        The downstream archive reader infers the kind from the extension and
        rejects `.ZIP` even under an explicit `driver="s2"`. Claiming one would
        route a read the stack cannot serve, replacing GDAL's result with an
        error, so the uppercase form is deliberately not claimed.
    """
    import shutil

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    lower = _zip_product(_S2_PRODUCT, tmp_path, "S2A_OPER_PRD_MSIL1C.SAFE")
    # A case-insensitive filesystem treats `.zip` and `.ZIP` as one name, so the
    # uppercase copy needs its own directory.
    upper_dir = tmp_path / "upper"
    upper_dir.mkdir()
    upper = upper_dir / "S2A_OPER_PRD_MSIL1C.SAFE.ZIP"
    shutil.copyfile(lower, upper)

    assert Sentinel2Driver.can_open(lower) is True, "the lowercase archive is claimed"
    assert Sentinel2Driver.can_open(str(upper)) is False, (
        "the uppercase archive must be left to GDAL, which can still open it"
    )


def test_an_archive_without_sentinel2_metadata_is_not_claimed(tmp_path):
    """A `.SAFE.zip` holding something else is not claimed.

    Test scenario:
        The suffix is only a hint; the claim is decided by what is inside. An
        archive named like a product but holding no Sentinel-2 metadata belongs
        to GDAL.
    """
    import zipfile

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    impostor = tmp_path / "NOT_A_PRODUCT.SAFE.zip"
    with zipfile.ZipFile(impostor, "w") as archive:
        archive.writestr("NOT_A_PRODUCT.SAFE/measurement/band.tif", b"not really a tif")
        archive.writestr("NOT_A_PRODUCT.SAFE/manifest.safe", b"<xml/>")
    assert Sentinel2Driver.can_open(str(impostor)) is False, (
        "an archive with no Sentinel-2 metadata must not be claimed"
    )


def test_a_corrupt_archive_is_not_claimed(tmp_path):
    """A `.SAFE.zip` that is not a readable archive is left to GDAL.

    Test scenario:
        The sniff has to open the file, so it must survive a truncated or
        non-archive payload rather than propagating. Covers the zip branch's
        failure path.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    corrupt = tmp_path / "S2A_BROKEN.SAFE.zip"
    corrupt.write_bytes(b"this is definitely not a zip archive")
    assert Sentinel2Driver.can_open(str(corrupt)) is False, (
        "an unreadable archive must not be claimed"
    )


def test_an_unreadable_directory_is_not_claimed(tmp_path, monkeypatch):
    """A `.SAFE` directory that cannot be listed is left to GDAL.

    Test scenario:
        The sniff reads a directory listing, which can fail on a permission or
        I/O error. That must not propagate out of `can_open`, since pyramids
        treats a raising `can_open` as a broken driver and warns. Covers the
        directory branch's failure path.
    """
    import os

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    product = tmp_path / "S2A_UNREADABLE.SAFE"
    product.mkdir()

    def _deny(_path):
        raise OSError("permission denied")

    monkeypatch.setattr(os, "listdir", _deny)
    assert Sentinel2Driver.can_open(str(product)) is False, (
        "an unlistable directory must not be claimed, and must not raise"
    )
