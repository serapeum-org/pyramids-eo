"""Unit tests for `pyramids_eo.drivers.sentinel2.Sentinel2Driver`.

The driver delegates to `from_sentinel2`; the facade is monkeypatched to capture the call, so
these tests pin the mapping (path, bbox/bands/crs, option pass-through, reserved-key
non-collision) without a real `.SAFE` product.
"""

import pytest

import pyramids_eo
from pyramids_eo.drivers.sentinel2 import Sentinel2Driver


@pytest.mark.parametrize(
    "source, expected",
    [
        ("S2A_MSIL2A.SAFE", True),
        ("S2A_MSIL2A.safe", True),
        ("S2A_MSIL2A.SAFE/", True),
        ("scene.tif", False),
        ("ee://x", False),
        (42, False),
    ],
)
def test_can_open(source, expected):
    """`can_open` claims only `.SAFE` string sources, case-insensitively, trailing slash OK."""
    assert Sentinel2Driver.can_open(source) is expected


def test_conforms_to_the_contract():
    """The driver satisfies pyramids' raster-driver conformance kit."""
    from pyramids.dataset.driver_conformance import assert_raster_driver_conforms

    assert_raster_driver_conforms(Sentinel2Driver)


def test_claims():
    """The class-level claims are the Sentinel-2 ones (no backend extra)."""
    assert Sentinel2Driver.name == "s2"
    assert Sentinel2Driver.suffixes == (".safe",)
    assert Sentinel2Driver.schemes == ()
    assert Sentinel2Driver.extra is None
    assert Sentinel2Driver.capabilities == frozenset({"read"})


def test_read_maps_path_bbox_bands_crs(monkeypatch):
    """`read` forwards the .SAFE path and maps bbox/bands/crs plus extra options."""
    captured = {}

    def _fake(path, *, bands, bbox, crs, **options):
        captured.update(path=path, bands=bands, bbox=bbox, crs=crs, options=options)
        return "DATASET"

    monkeypatch.setattr(pyramids_eo, "from_sentinel2", _fake)
    out = Sentinel2Driver().read(
        "X.SAFE",
        bbox=(0.0, 0.0, 1.0, 1.0),
        bands=["B04"],
        crs=4326,
        options={"resolution": 10, "reflectance": True},
    )
    assert out == "DATASET"
    assert captured["path"] == "X.SAFE"
    assert captured["bbox"] == (0.0, 0.0, 1.0, 1.0)
    assert captured["bands"] == ["B04"]
    assert captured["crs"] == 4326
    assert captured["options"] == {"resolution": 10, "reflectance": True}


def test_read_takes_bbox_bands_crs_from_options(monkeypatch):
    """read_file forwards only options, so bbox/bands/crs may arrive there without colliding."""
    captured = {}
    monkeypatch.setattr(
        pyramids_eo,
        "from_sentinel2",
        lambda path, *, bands, bbox, crs, **o: captured.update(
            bands=bands, bbox=bbox, crs=crs
        ),
    )
    Sentinel2Driver().read(
        "X.SAFE", options={"bbox": (0.0, 0.0, 2.0, 2.0), "bands": ["B08"], "crs": 32631}
    )
    assert captured["bbox"] == (0.0, 0.0, 2.0, 2.0)
    assert captured["bands"] == ["B08"]
    assert captured["crs"] == 32631


def test_explicit_keyword_wins_over_option(monkeypatch):
    """A non-None explicit keyword overrides the same key in options."""
    captured = {}
    monkeypatch.setattr(
        pyramids_eo,
        "from_sentinel2",
        lambda path, *, bands, bbox, crs, **o: captured.update(bands=bands),
    )
    Sentinel2Driver().read("X.SAFE", bands=["EXPLICIT"], options={"bands": ["OPT"]})
    assert captured["bands"] == ["EXPLICIT"]
