"""Unit tests for `pyramids_eo.drivers.earthengine.EarthEngineDriver`.

The driver delegates to `from_earthengine` / `collection_from_earthengine`; the facades are
monkeypatched to capture the call, so these tests pin the *mapping* (scheme stripping, window
construction, auth -> credentials, option pass-through) without the Earth Engine backend.
"""

import pytest

import pyramids_eo
from pyramids_eo.drivers.earthengine import EarthEngineDriver


@pytest.mark.parametrize(
    "source, expected",
    [
        ("ee://COPERNICUS/S2_SR_HARMONIZED", True),
        ("EE://MODIS/006", True),
        ("COPERNICUS/S2_SR_HARMONIZED", False),
        ("scene.tif", False),
        (42, False),
    ],
)
def test_can_open(source, expected):
    """`can_open` claims only ``ee://`` string sources, case-insensitively."""
    assert EarthEngineDriver.can_open(source) is expected


def test_conforms_to_the_contract():
    """The driver satisfies pyramids' raster-driver conformance kit."""
    from pyramids.dataset.driver_conformance import assert_raster_driver_conforms

    assert_raster_driver_conforms(EarthEngineDriver)


def test_claims():
    """The class-level claims are the Earth Engine ones."""
    assert EarthEngineDriver.name == "earthengine"
    assert EarthEngineDriver.suffixes == ()
    assert EarthEngineDriver.schemes == ("ee",)
    assert EarthEngineDriver.extra == "ee"
    assert {"read", "read_collection", "needs_auth"} <= EarthEngineDriver.capabilities


def test_read_maps_scheme_window_bands_and_auth(monkeypatch):
    """`read` strips ``ee://``, builds a Window from bbox/crs, and routes auth to credentials."""
    captured = {}

    def _fake(asset_id, *, window, bands, credentials, **options):
        captured.update(
            asset_id=asset_id,
            window=window,
            bands=bands,
            credentials=credentials,
            options=options,
        )
        return "DATASET"

    monkeypatch.setattr(pyramids_eo, "from_earthengine", _fake)
    out = EarthEngineDriver().read(
        "ee://COPERNICUS/S2_SR_HARMONIZED",
        bbox=(0.0, 0.0, 1.0, 1.0),
        crs=4326,
        bands=["B4"],
        options={"reducer": "median"},
        auth="CREDS",
    )
    assert out == "DATASET"
    assert captured["asset_id"] == "COPERNICUS/S2_SR_HARMONIZED", "scheme not stripped"
    assert captured["window"].bbox == (0.0, 0.0, 1.0, 1.0)
    assert captured["window"].crs == 4326
    assert captured["bands"] == ["B4"]
    assert captured["credentials"] == "CREDS", "auth should map to credentials"
    assert captured["options"] == {"reducer": "median"}


def test_read_without_bbox_uses_a_default_window(monkeypatch):
    """With no bbox/crs the driver passes a default (whole-asset) Window."""
    captured = {}
    monkeypatch.setattr(
        pyramids_eo,
        "from_earthengine",
        lambda asset_id, *, window, bands, credentials, **o: captured.update(
            window=window
        ),
    )
    EarthEngineDriver().read("ee://X", auth=None)
    assert captured["window"].bbox is None, "default window should have no bbox"


def test_read_collection_maps_time_tuple(monkeypatch):
    """`read_collection` accepts a ``time=(start, end)`` pair and strips the scheme."""
    captured = {}

    def _fake(asset_id, **kwargs):
        captured.update(asset_id=asset_id, **kwargs)
        return "COLLECTION"

    monkeypatch.setattr(pyramids_eo, "collection_from_earthengine", _fake)
    out = EarthEngineDriver().read_collection(
        "ee://COPERNICUS/S2_SR_HARMONIZED", time=("2024-01-01", "2024-02-01")
    )
    assert out == "COLLECTION"
    assert captured["asset_id"] == "COPERNICUS/S2_SR_HARMONIZED"
    assert captured["start"] == "2024-01-01" and captured["end"] == "2024-02-01"


def test_read_collection_without_dates_raises(monkeypatch):
    """A collection read needs a date range, by `time` or by `start`/`end`."""
    monkeypatch.setattr(pyramids_eo, "collection_from_earthengine", lambda *a, **k: "X")
    with pytest.raises(ValueError, match="date range"):
        EarthEngineDriver().read_collection("ee://X")


def test_read_collection_maps_auth_to_credentials(monkeypatch):
    """`read_collection` remaps an `auth` kwarg to `credentials` for the facade."""
    captured = {}
    monkeypatch.setattr(
        pyramids_eo, "collection_from_earthengine",
        lambda asset_id, **kwargs: captured.update(asset_id=asset_id, **kwargs),
    )
    EarthEngineDriver().read_collection(
        "ee://X", start="2024-01-01", end="2024-02-01", auth="CREDS"
    )
    assert captured["credentials"] == "CREDS", "auth should map to credentials"
    assert "auth" not in captured, "the auth key should be consumed, not forwarded"
