"""Integration: `Dataset.read_file` routes a claimed source into a pyramids-eo driver.

All pyramids-eo drivers are discovered unconditionally (none gates on an optional backend),
so they are exercised through the real dispatch here: SEVIRI with an injected `.nat` decoder, and
Earth Engine with `from_earthengine` monkeypatched (read_file forwards only `open_options`, which
is the path this proves). The Sentinel-2 cases additionally cover which sources `can_open` claims,
since ESA shares the `.SAFE` suffix across missions.
"""

import os
import pathlib
import shutil
import zipfile

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


_DATA = pathlib.Path(__file__).parent / "data"
_S2_PRODUCT = _DATA / "sentinel2" / "fake_l1c" / "S2A_OPER_PRD_MSIL1C.SAFE"
_S1_PRODUCT = _DATA / "sentinel1" / "test.SAFE"
#: A compact-naming product (`MTD_MSIL1C.xml`); `_S2_PRODUCT` uses the legacy
#: `*_MTD_SAF*.xml` spelling, so both marker globs need their own fixture.
_S2_COMPACT = (
    _DATA
    / "sentinel2"
    / "fake_l1c_baseline_5_09"
    / "S2B_MSIL1C_20230823T095559_N0509_R122_T34UCF_20230823T120234.SAFE"
)


def _zip_product(product, tmp_path, stem):
    """Zip a `.SAFE` fixture directory and return the archive path as a string."""
    archive = tmp_path / stem
    shutil.make_archive(
        str(archive), "zip", root_dir=str(product.parent), base_dir=product.name
    )
    return str(archive) + ".zip"


def test_a_sentinel2_product_directory_is_claimed():
    """`can_open` claims a real Sentinel-2 product directory.

    Test scenario:
        The positive case for the content check — a product carrying Sentinel-2
        metadata is still claimed, so the fallback keeps working.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.can_open(str(_S2_PRODUCT)) is True, (
        "a real Sentinel-2 product must be claimed"
    )


def test_a_sentinel1_product_directory_is_claimed_but_never_reached():
    """A Sentinel-1 `.SAFE` directory is claimed, and that is harmless.

    Test scenario:
        The directory branch is deliberately not gated on the metadata markers,
        exactly as on `main` — so it claims a Sentinel-1 directory too. That
        costs nothing, because `.safe` is not pre-empted: GDAL opens such a
        product at step 3 and the fallback never runs. The guarantee that
        matters is the end-to-end one next door, which pins that GDAL still
        serves it. Where the claim *must* discriminate is the zipped form, which
        is pre-empted ahead of GDAL.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.can_open(str(_S1_PRODUCT)) is True, (
        "the directory branch claims any .SAFE, matching main"
    )


def test_a_sentinel1_product_still_reads_through_gdal():
    """End to end: a Sentinel-1 `.SAFE` keeps GDAL's read.

    Test scenario:
        The regression guard. GDAL's `SAFE` driver serves this product, and
        nothing in pyramids-eo may take it away — exercised through the real
        `Dataset.read_file`, since a `can_open` assertion alone would not notice
        the dispatch changing.
    """
    from pyramids.dataset._drivers import clear_driver_cache

    clear_driver_cache()
    try:
        out = Dataset.read_file(str(_S1_PRODUCT))
    finally:
        clear_driver_cache()
    assert out.band_count == 2, (
        f"GDAL should still serve the Sentinel-1 product, got {out.band_count} bands"
    )


def test_a_zipped_sentinel1_product_is_not_claimed(tmp_path):
    """A zipped Sentinel-1 product is not claimed either.

    Test scenario:
        `.SAFE.zip` is shared across missions exactly as `.SAFE` is, so the
        content check has to see inside the archive, not just read its name.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    archive = _zip_product(_S1_PRODUCT, tmp_path, "S1A_IW_GRDH.SAFE")
    assert Sentinel2Driver.can_open(archive) is False, (
        "a zipped Sentinel-1 product must be left to GDAL"
    )


def test_a_zipped_sentinel2_product_is_claimed(tmp_path):
    """A zipped Sentinel-2 product is claimed, so `driver="s2"` can read it.

    Test scenario:
        GDAL would open the archive on the bare path, which is why this form is
        pre-empted; the claim has to succeed for that to work, and for an
        explicit `driver="s2"` too.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    archive = _zip_product(_S2_PRODUCT, tmp_path, "S2A_OPER_PRD_MSIL1C.SAFE")
    assert Sentinel2Driver.can_open(archive) is True, (
        "a zipped Sentinel-2 product must be claimable"
    )


def test_an_uppercase_zip_extension_is_not_claimed(tmp_path):
    """A `.SAFE.ZIP` is left to GDAL, because the archive reader cannot take it.

    Test scenario:
        The downstream archive reader infers the kind from the extension and
        rejects `.ZIP`, even under an explicit `driver="s2"`. Claiming one would
        replace GDAL's result with an error, so the uppercase form is not
        claimed. The copy needs its own directory, since a case-insensitive
        filesystem treats `.zip` and `.ZIP` as one name.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    lower = _zip_product(_S2_PRODUCT, tmp_path, "S2A_OPER_PRD_MSIL1C.SAFE")
    upper_dir = tmp_path / "upper"
    upper_dir.mkdir()
    upper = upper_dir / "S2A_OPER_PRD_MSIL1C.SAFE.ZIP"
    shutil.copyfile(lower, upper)
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
    impostor = tmp_path / "NOT_A_PRODUCT.SAFE.zip"
    with zipfile.ZipFile(impostor, "w") as archive:
        archive.writestr("NOT_A_PRODUCT.SAFE/measurement/band.tif", b"not really a tif")
        archive.writestr("NOT_A_PRODUCT.SAFE/manifest.safe", b"<xml/>")

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.can_open(str(impostor)) is False, (
        "an archive with no Sentinel-2 metadata must not be claimed"
    )


def test_a_corrupt_archive_is_not_claimed(tmp_path):
    """A `.SAFE.zip` that is not a readable archive is left to GDAL.

    Test scenario:
        The check has to open the file, so it must survive a truncated or
        non-archive payload rather than propagating — pyramids treats a raising
        `can_open` as a broken driver.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    corrupt = tmp_path / "S2A_BROKEN.SAFE.zip"
    corrupt.write_bytes(b"this is definitely not a zip archive")
    assert Sentinel2Driver.can_open(str(corrupt)) is False, (
        "an unreadable archive must not be claimed"
    )


def test_a_directory_stat_failure_is_not_claimed(tmp_path, monkeypatch):
    """A `.SAFE` path whose type cannot be determined is left to GDAL.

    Test scenario:
        The directory branch still touches the filesystem to tell a directory
        from a file and from an absent path. A permission or I/O error there must
        not propagate, because pyramids treats a raising `can_open` as a broken
        driver and warns.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    product = tmp_path / "S2A_UNSTATABLE.SAFE"
    product.mkdir()

    def _deny(_path):
        raise OSError("permission denied")

    monkeypatch.setattr(os.path, "isdir", _deny)
    assert Sentinel2Driver.can_open(str(product)) is False, (
        "a stat failure must not be claimed, and must not raise"
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


def test_a_zipped_product_is_preempted(tmp_path):
    """`select_preempting_driver` picks the s2 driver for a real `.SAFE.zip`.

    Test scenario:
        The opt-in itself. A real archive is required, because the claim is
        decided by what is inside it rather than by the name.
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
        f"a zipped Sentinel-2 product should be pre-empted, got {picked!r}"
    )


def test_a_bare_zipped_product_reads_through_preemption(tmp_path):
    """End to end: a bare `read_file` of a `.SAFE.zip` reads the product.

    Test scenario:
        The behaviour #143 asked for, exercised through the real
        `Dataset.read_file` rather than through `select_preempting_driver`.
        Without the opt-in GDAL opens the archive first and returns an empty
        container (0 bands) while silently ignoring `open_options`; with it the
        pyramids-eo reader runs and returns the requested band. The read is
        narrowed deliberately — a whole product is ~10 GiB.
    """
    from pyramids.dataset._drivers import clear_driver_cache

    archive = _zip_product(_S2_PRODUCT, tmp_path, "S2A_OPER_PRD_MSIL1C.SAFE")
    clear_driver_cache()
    try:
        out = Dataset.read_file(
            archive, open_options={"bands": ["B1"], "resolution": 60}
        )
    finally:
        clear_driver_cache()
    assert isinstance(out, Dataset)
    assert out.band_count == 1, (
        f"the pyramids-eo reader should return the requested band, got "
        f"{out.band_count} (0 means GDAL's container won the race)"
    )


def test_an_explicit_driver_still_wins_over_preemption(tmp_path):
    """`driver="s2"` short-circuits ahead of the pre-empt scan.

    Test scenario:
        An explicit driver is resolved before pre-emption is considered, so
        naming the driver keeps working and stays the way to be deliberate.
    """
    from pyramids.dataset._drivers import clear_driver_cache

    archive = _zip_product(_S2_PRODUCT, tmp_path, "S2A_OPER_PRD_MSIL1C.SAFE")
    clear_driver_cache()
    try:
        out = Dataset.read_file(
            archive, driver="s2", open_options={"bands": ["B1"], "resolution": 60}
        )
    finally:
        clear_driver_cache()
    assert out.band_count == 1, (
        f"an explicit driver=s2 must still read the product, got {out.band_count}"
    )


def test_vsi_skips_the_preempt_scan(tmp_path):
    """An explicit `vsi=` outranks pre-emption, as pyramids documents.

    Test scenario:
        `vsi=` is a deliberate "open member N of this archive" intent, so core
        skips the pre-empt scan for it. Note this is not a way to force GDAL:
        the post-GDAL `can_open` fallback can still claim the source, so the
        read may well arrive here anyway — what is pinned is that passing
        `vsi=` does not break the read.
    """
    from pyramids.dataset._drivers import clear_driver_cache

    archive = _zip_product(_S2_PRODUCT, tmp_path, "S2A_OPER_PRD_MSIL1C.SAFE")
    clear_driver_cache()
    try:
        out = Dataset.read_file(
            archive, vsi="zip", open_options={"bands": ["B1"], "resolution": 60}
        )
    finally:
        clear_driver_cache()
    assert isinstance(out, Dataset), "vsi= must not break a zipped product read"


def test_preempts_declaration_is_well_formed():
    """`preempts` is exactly the lowercase, dot-prefixed `.zip`.

    Test scenario:
        The conformance kit only partly covers this: it rejects an uppercase
        entry (`(".ZIP",)` fails `assert_raster_driver_conforms`) but accepts a
        missing dot (`("zip",)`) and a scheme-shaped one (`(".zip://",)`),
        either of which would silently never match. The exact tuple is pinned as
        a deliberate-change tripwire rather than a safety net: the content check
        is what actually keeps Sentinel-1 and ordinary archives out, so adding
        `.safe` here breaks this assertion and nothing else. Widening the tuple
        should be a decision, not a side effect.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.preempts == (".zip",), (
        f"preempts should be exactly ('.zip',), got {Sentinel2Driver.preempts!r}"
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("/vsimem/scene.SAFE.zip", id="vsimem"),
        pytest.param("/vsicurl/https://host/x.SAFE.zip", id="vsicurl"),
        pytest.param("/vsis3/bucket/x.SAFE.zip", id="vsis3"),
        pytest.param("/vsizip/x.SAFE.zip", id="vsizip"),
        pytest.param("s3://bucket/x.SAFE.zip", id="s3-scheme"),
        pytest.param("https://host/x.SAFE.zip", id="https-scheme"),
        pytest.param("//server/share/x.SAFE.zip", id="unc"),
    ],
)
def test_a_non_local_archive_is_never_claimed(source):
    """A URL, `/vsi*` path or UNC share is left to GDAL.

    Test scenario:
        `can_open` cannot sniff these with `os`/`zipfile`, and GDAL often reads
        them. Claiming one on the strength of its suffix would pre-empt GDAL out
        of a read it can serve — which is exactly what happened before this
        guard: a GDAL-readable `/vsimem/scene.SAFE.zip` went from a working read
        to `FileNotFoundError`, because `os.path.exists` is False for every
        virtual path and the cannot-sniff shortcut then claimed it.
    """
    from pyramids.dataset._drivers import clear_driver_cache, select_preempting_driver

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.can_open(source) is False, (
        f"{source!r} cannot be inspected and must not be claimed"
    )
    clear_driver_cache()
    try:
        picked = select_preempting_driver(source)
    finally:
        clear_driver_cache()
    assert picked is None, f"{source!r} must not pre-empt GDAL, got {picked!r}"


def test_a_gdal_readable_vsimem_archive_still_reads(tmp_path):
    """End to end: a GDAL-readable `/vsimem` source named `.SAFE.zip` still reads.

    Test scenario:
        The regression this guard exists for, through the real `Dataset.read_file`.
        The name matches the pre-empt suffix but the content is an ordinary
        raster GDAL serves, so GDAL must keep the read.
    """
    import numpy as np
    from osgeo import gdal
    from pyramids.dataset._drivers import clear_driver_cache

    path = "/vsimem/test_s2_guard.SAFE.zip"
    driver = gdal.GetDriverByName("GTiff")
    handle = driver.Create(path, 4, 4, 1)
    handle.GetRasterBand(1).WriteArray(np.ones((4, 4)))
    handle = None
    clear_driver_cache()
    try:
        out = Dataset.read_file(path)
    finally:
        clear_driver_cache()
        gdal.Unlink(path)
    assert out.band_count == 1, (
        f"GDAL should still serve this source, got {out.band_count} bands"
    )


@pytest.mark.parametrize(
    "product, marker",
    [
        pytest.param(_S2_PRODUCT, "*_MTD_SAF*.xml", id="legacy-naming"),
        pytest.param(_S2_COMPACT, "MTD_MSIL*.xml", id="compact-naming"),
    ],
)
def test_both_metadata_namings_are_recognised(product, marker, tmp_path):
    """Each `_S2_MARKERS` glob is covered by a product that only matches it.

    Args:
        product: A committed `.SAFE` fixture.
        marker: The glob that fixture's metadata file matches.
        tmp_path: Where the zipped copy is built.

    Test scenario:
        ESA renamed the metadata file in 2016, so the driver carries two globs.
        Testing only one fixture leaves the other glob dead — a mutation that
        deleted `MTD_MSIL*.xml` passed the whole suite, because every other test
        here uses the legacy-named product. Both the directory and the zipped
        form are checked for each naming.
    """
    import fnmatch

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    names = [entry.name for entry in product.iterdir()]
    assert any(fnmatch.fnmatch(n, marker) for n in names), (
        f"fixture {product.name} should carry a {marker} file, found {names}"
    )
    assert Sentinel2Driver.can_open(str(product)) is True, (
        f"the {marker} naming must be recognised in a directory"
    )
    archive = _zip_product(product, tmp_path, product.name)
    assert Sentinel2Driver.can_open(archive) is True, (
        f"the {marker} naming must be recognised inside an archive"
    )


def test_a_zip_directory_entry_is_not_metadata(tmp_path):
    """A zip entry that is a *folder* named like the metadata file is not metadata.

    Test scenario:
        A zip index lists folders as entries ending in `/`. Matching those against
        the metadata globs would claim an archive that contains no metadata file
        at all — only a directory that happens to share the name.
    """
    import zipfile

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    archive = tmp_path / "DIRENTRY.SAFE.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("X.SAFE/MTD_MSIL1C.xml/", b"")
        handle.writestr("X.SAFE/measurement/band.tif", b"not a product")
    assert Sentinel2Driver.can_open(str(archive)) is False, (
        "a directory entry is not a metadata file and must not win the claim"
    )


def test_a_deep_marker_does_not_identify_a_product(tmp_path):
    """Metadata buried inside an archive does not make it a Sentinel-2 product.

    Test scenario:
        A zipped Sentinel-1 product that happens to carry a stray
        `notes/backup/MTD_MSIL1C.xml` is still a Sentinel-1 product. Matching the
        markers at any depth claimed it away from GDAL, which reads it — the same
        failure as taking it on its name. Only depth 0 or 1 can identify a
        product, because an archive wraps it in at most one folder.
    """
    import zipfile

    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    archive = _zip_product(_S1_PRODUCT, tmp_path, "S1A_IW_GRDH.SAFE")
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("notes/backup/MTD_MSIL1C.xml", b"<xml/>")
    assert Sentinel2Driver.can_open(archive) is False, (
        "a marker nested inside the archive must not win the claim"
    )


def test_a_marker_under_a_single_root_folder_is_recognised(tmp_path):
    """The usual archive shape — one wrapper folder — is still recognised.

    Test scenario:
        `make_archive` wraps the product in its own folder, so the metadata sits
        at depth 1 (`X.SAFE/MTD_MSIL1C.xml`). The depth limit must admit that,
        or the feature would never fire on a real archive.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    archive = _zip_product(_S2_COMPACT, tmp_path, _S2_COMPACT.name)
    assert Sentinel2Driver.can_open(archive) is True, (
        "a product wrapped in one folder must still be claimed"
    )


def test_a_product_directory_is_claimed_without_a_marker(tmp_path):
    """Any `.SAFE` directory is claimed: this branch runs only after GDAL failed.

    Test scenario:
        `.safe` is not pre-empted, so a directory reaches `can_open` only through
        the post-GDAL fallback — and GDAL reads a Sentinel-1 `.SAFE` directory
        itself, so it has already declined anything that gets here. Gating the
        branch on the metadata markers bought no discrimination and turned away a
        product whose root metadata is a bare `manifest.safe`, which
        `from_sentinel2` reads.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    product = tmp_path / "S2A_MANIFEST_ONLY.SAFE"
    product.mkdir()
    (product / "manifest.safe").write_text("<xml/>", encoding="utf-8")
    assert Sentinel2Driver.can_open(str(product)) is True, (
        "a .SAFE directory must be claimed on the fallback, marker or not"
    )


def test_a_folder_entry_named_like_metadata_is_skipped(tmp_path):
    """A zip folder entry is skipped explicitly, not incidentally.

    Test scenario:
        Pins the `endswith("/")` skip on its own. Without it the base name would
        still come out empty for `X.SAFE/MTD_MSIL1C.xml/` and fail to match, so
        the guard is belt-and-braces — but a later refactor that reintroduces
        `rstrip("/")` on the base name would silently start claiming folders.
    """
    import zipfile

    from pyramids_eo.drivers.sentinel2 import _holds_s2_metadata

    assert _holds_s2_metadata(["X.SAFE/MTD_MSIL1C.xml/"]) is False, (
        "a folder entry is not a metadata file"
    )
    assert _holds_s2_metadata(["X.SAFE/MTD_MSIL1C.xml"]) is True, (
        "the same name as a file is a metadata file"
    )
    archive = tmp_path / "FOLDERONLY.SAFE.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("X.SAFE/MTD_MSIL1C.xml/", b"")
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    assert Sentinel2Driver.can_open(str(archive)) is False


def test_backslash_separated_zip_entries_are_handled():
    """A zip written with backslash separators still resolves to its base name.

    Test scenario:
        Some writers emit `X.SAFE\\MTD_MSIL1C.xml`. The separator is normalised
        before the depth count and the base-name split, so such an archive is
        recognised rather than silently unclaimed.
    """
    from pyramids_eo.drivers.sentinel2 import _holds_s2_metadata

    assert _holds_s2_metadata(["X.SAFE\\MTD_MSIL1C.xml"]) is True, (
        "a backslash-separated entry must resolve to its base name"
    )
    assert _holds_s2_metadata(["deep\\deeper\\MTD_MSIL1C.xml"]) is False, (
        "the depth limit must apply to backslash separators too"
    )


def test_the_marker_match_is_case_insensitive_but_not_glob_folded():
    """Marker matching folds case on the name, not on the pattern.

    Test scenario:
        `fnmatchcase` on pre-upper-cased input makes the result independent of the
        host filesystem's case rules, which `fnmatch` would otherwise borrow. Both
        spellings a product uses in practice are accepted.
    """
    from pyramids_eo.drivers.sentinel2 import _holds_s2_metadata

    assert _holds_s2_metadata(["MTD_MSIL1C.xml"]) is True
    assert _holds_s2_metadata(["mtd_msil1c.XML"]) is True
    assert _holds_s2_metadata(["MTD_MSIL1C.txt"]) is False


def test_an_existing_non_directory_safe_is_not_claimed(tmp_path):
    """A plain *file* named `*.SAFE` is not a product.

    Test scenario:
        Pins the existing-non-directory guard: the not-on-disk shortcut must not
        swallow a path that does exist but is a file, which is not a product and
        which GDAL may be able to open.
    """
    from pyramids_eo.drivers.sentinel2 import Sentinel2Driver

    impostor = tmp_path / "NOT_A_PRODUCT.SAFE"
    impostor.write_text("not a directory", encoding="utf-8")
    assert Sentinel2Driver.can_open(str(impostor)) is False, (
        "an existing file is not a product directory"
    )
