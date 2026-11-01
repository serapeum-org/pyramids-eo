"""Unit tests for `pyramids_eo.drivers.seviri.SeviriDriver`.

The driver delegates to `read_seviri`, which is exercised for real here via an injected `.nat`
decoder (`parse=`) that returns a synthetic radiance Dataset — no heavy backend needed.
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from pyramids_eo.drivers.seviri import SeviriDriver


def _radiance_ds(value: float = 7.0) -> Dataset:
    """A 2x2 WGS84 Dataset standing in for a decoded SEVIRI granule."""
    return Dataset.from_array(
        np.full((2, 2), value),
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    )


def _parser(path, channel):
    """A `.nat` decoder stub: return a constant synthetic radiance Dataset."""
    return _radiance_ds(7.0)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("MSG_SEVIRI.nat", True),
        ("scene.NAT", True),
        ("scene.tif", False),
        ("ee://x", False),
        (42, False),
    ],
)
def test_can_open(source, expected):
    """`can_open` claims only ``.nat`` string sources, case-insensitively."""
    assert SeviriDriver.can_open(source) is expected


def test_conforms_to_the_contract():
    """The driver satisfies pyramids' raster-driver conformance kit."""
    from pyramids.dataset.driver_conformance import assert_raster_driver_conforms

    assert_raster_driver_conforms(SeviriDriver)


def test_claims():
    """The class-level claims are the SEVIRI ones (no backend extra)."""
    assert SeviriDriver.name == "seviri"
    assert SeviriDriver.suffixes == (".nat",)
    assert SeviriDriver.schemes == ()
    assert SeviriDriver.extra is None
    assert SeviriDriver.capabilities == frozenset({"read"})


def test_read_channel_from_options_delegates(monkeypatch):
    """`read` takes the channel from options and delegates the decode to `read_seviri`."""
    out = SeviriDriver().read(
        "scene.nat", options={"channel": "IR_108", "calibrate": False, "parse": _parser}
    )
    assert np.allclose(out.read_array(), 7.0), (
        "the injected decoder's radiance should flow out"
    )


def test_read_channel_from_bands():
    """The channel may be given as a one-element `bands` list instead of in options."""
    out = SeviriDriver().read(
        "scene.nat", bands=["IR_108"], options={"calibrate": False, "parse": _parser}
    )
    assert np.allclose(out.read_array(), 7.0)


def test_read_without_channel_raises():
    """A SEVIRI read needs a channel, via bands or options."""
    driver = SeviriDriver()
    with pytest.raises(ValueError, match="channel"):
        driver.read("scene.nat", options={"parse": _parser})


def test_read_rejects_multiple_bands():
    """More than one band is rejected, not silently truncated to the first channel."""
    driver = SeviriDriver()
    with pytest.raises(ValueError, match="one channel"):
        driver.read("scene.nat", bands=["IR_108", "IR_120"], options={"parse": _parser})


@pytest.mark.parametrize("bare", ["I", "IR_108", b"IR_108"])
def test_read_rejects_bare_string_bands(bare):
    """A bare str/bytes bands is rejected, so a 1-char string cannot smuggle a wrong channel."""
    driver = SeviriDriver()
    with pytest.raises(ValueError, match="list of channel names"):
        driver.read("scene.nat", bands=bare, options={"parse": _parser})


def test_read_accepts_a_single_band_generator():
    """A generator yielding exactly one channel is materialised and read (no TypeError)."""
    out = SeviriDriver().read(
        "scene.nat",
        bands=(b for b in ["IR_108"]),
        options={"calibrate": False, "parse": _parser},
    )
    assert np.allclose(out.read_array(), 7.0)


def test_read_rejects_multi_band_generator():
    """A generator yielding more than one channel is a clean ValueError, not a TypeError."""
    driver = SeviriDriver()
    with pytest.raises(ValueError, match="one channel"):
        driver.read(
            "scene.nat",
            bands=(b for b in ["IR_108", "IR_120"]),
            options={"parse": _parser},
        )


def test_read_coerces_string_boolean_and_numeric_open_options(monkeypatch):
    """`calibrate=false` disables calibration instead of enabling it.

    SEVIRI had the same defect as the other channel-at-a-time drivers and kept it
    after they were fixed: `read_file` normalises open_options to KEY=VALUE
    strings, and the truthy string "false" switched calibration ON, so a caller
    who asked for raw radiance silently got calibrated values. The two numeric
    keywords arrived as strings for the same reason.
    """
    import pyramids_eo.sensors.readers as readers

    captured: dict = {}
    monkeypatch.setattr(
        readers, "read_seviri", lambda *a, **k: captured.update(k) or "dataset"
    )
    SeviriDriver().read(
        "scene.nat",
        options=[
            "channel=IR_108",
            "calibrate=false",
            "sun_earth_distance=0.983",
            "subsatellite_longitude=9.5",
        ],
    )
    assert captured["calibrate"] is False, captured
    assert captured["sun_earth_distance"] == 0.983, captured
    assert captured["subsatellite_longitude"] == 9.5, captured


@pytest.mark.parametrize(
    "option, message",
    [("calibrate=maybe", "must be a boolean"), ("sun_earth_distance=nan", "finite")],
)
def test_read_rejects_unreadable_options(option, message):
    """An unreadable boolean or non-finite numeric option is rejected by name."""
    driver = SeviriDriver()
    with pytest.raises(ValueError, match=message):
        driver.read("scene.nat", options=["channel=IR_108", option])
