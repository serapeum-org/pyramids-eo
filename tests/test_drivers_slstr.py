"""Unit tests for `pyramids_eo.drivers.slstr.SlstrDriver`.

The driver delegates to `read_slstr`; the facade is monkeypatched to capture the call, so
these tests pin the claim set, the `can_open` content sniff (which must distinguish an
SLSTR granule from an OLCI one sharing the `.SEN3` suffix) and the channel mapping,
without decoding real NetCDF files.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyramids_eo.sensors.readers as readers
from pyramids_eo.drivers.slstr import SlstrDriver
from tests._sentinel3_granule import packed_counts, slstr_granule


def _granule(tmp_path, names, name="S3A_SL_1_RBT.SEN3"):
    """Create a granule directory holding empty files `names`; return its path."""
    granule = tmp_path / name
    granule.mkdir(parents=True, exist_ok=True)
    for entry in names:
        (granule / entry).write_bytes(b"")
    return granule


def test_can_open_slstr_scheme():
    """A slstr:// source is always claimed (cheap string check, no file access)."""
    assert SlstrDriver.can_open("slstr://anywhere") is True
    assert SlstrDriver.can_open("SLSTR://ANYWHERE") is True


@pytest.mark.parametrize(
    "measurement",
    ["S5_radiance_an.nc", "S8_BT_in.nc", "F1_BT_fn.nc", "S5_radiance_ao.nc"],
)
def test_can_open_granule_with_a_measurement_file(tmp_path, measurement):
    """A .SEN3 directory holding any SLSTR measurement file is claimed."""
    granule = _granule(tmp_path, [measurement])
    assert SlstrDriver.can_open(str(granule)) is True


def test_can_open_rejects_an_olci_granule(tmp_path):
    """An OLCI granule shares the .SEN3 suffix but is not claimed by the SLSTR driver."""
    granule = _granule(
        tmp_path, ["Oa08_radiance.nc", "geo_coordinates.nc"], name="S3A_OL_1_EFR.SEN3"
    )
    assert SlstrDriver.can_open(str(granule)) is False


def test_can_open_rejects_an_unrelated_netcdf_directory(tmp_path):
    """A directory of unrelated .nc files is not claimed."""
    granule = _granule(tmp_path, ["random.nc", "other.nc"])
    assert SlstrDriver.can_open(str(granule)) is False


@pytest.mark.parametrize("source", ["scene.tif", 42, "ee://x", "", None])
def test_can_open_rejects_non_granule(source):
    """A plain file, a non-string, another scheme, or an empty source is not claimed."""
    assert SlstrDriver.can_open(source) is False


def test_can_open_glob_matching_a_granule(tmp_path):
    """A glob whose matches include an SLSTR granule directory is claimed."""
    _granule(tmp_path, ["S5_radiance_an.nc"])
    assert SlstrDriver.can_open(str(tmp_path / "*.SEN3")) is True


def test_can_open_glob_without_a_granule(tmp_path):
    """A glob matching only non-SLSTR directories is not claimed."""
    _granule(tmp_path, ["random.nc"])
    assert SlstrDriver.can_open(str(tmp_path / "*.SEN3")) is False


def test_can_open_granule_name_with_glob_metachar(tmp_path):
    """A real granule whose name contains a glob metachar is still claimed."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"], name="granule[1].SEN3")
    assert SlstrDriver.can_open(str(granule)) is True


def test_can_open_existing_single_file_is_false(tmp_path):
    """An existing single file (not a granule directory) is not claimed."""
    band = tmp_path / "S5_radiance_an.nc"
    band.write_bytes(b"")
    assert SlstrDriver.can_open(str(band)) is False


def test_can_open_never_raises_on_a_filesystem_error(tmp_path, monkeypatch):
    """A directory whose listing raises OSError yields False, never an exception (contract).

    `_holds_measurement` enumerates with `iterdir` (one pass instead of a glob per
    pattern), so this patches that -- patching `glob` no longer reaches the probe.
    """
    from pathlib import Path

    def _boom(self):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "iterdir", _boom)
    assert SlstrDriver.can_open(str(tmp_path)) is False


def test_conforms_to_the_contract():
    """The driver satisfies pyramids' raster-driver conformance kit."""
    from pyramids.dataset.driver_conformance import assert_raster_driver_conforms

    assert_raster_driver_conforms(SlstrDriver)


def test_claims():
    """The class-level claims are the SLSTR ones (.SEN3 suffix, slstr:// scheme)."""
    assert SlstrDriver.name == "slstr"
    assert SlstrDriver.suffixes == (".sen3",)
    assert SlstrDriver.schemes == ("slstr",)
    assert SlstrDriver.extra is None
    assert SlstrDriver.capabilities == frozenset({"read"})


def test_read_passes_the_channel_view_and_options(tmp_path, monkeypatch):
    """`read` forwards the granule, channel and the SLSTR-specific view/grid options."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_slstr",
        lambda s, *, channel, channels, **o: captured.update(
            source=s, channel=channel, channels=channels, options=o
        ),
    )
    SlstrDriver().read(
        str(granule), options={"channel": "S5", "view": "oblique", "grid": "a"}
    )
    assert captured["source"] == str(granule)
    assert captured["channel"] == "S5"
    assert captured["options"] == {"view": "oblique", "grid": "a"}


def test_read_strips_the_slstr_scheme(tmp_path, monkeypatch):
    """A slstr:// prefix is stripped before the reader is called."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    captured = {}
    monkeypatch.setattr(readers, "read_slstr", lambda s, **k: captured.update(source=s))
    SlstrDriver().read("slstr://" + str(granule), options={"channel": "S5"})
    assert captured["source"] == str(granule)


def test_read_channel_from_single_band(tmp_path, monkeypatch):
    """A one-element bands list is read as a single channel."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_slstr",
        lambda s, *, channel, channels, **o: captured.update(
            channel=channel, channels=channels
        ),
    )
    SlstrDriver().read(str(granule), bands=["S5"])
    assert captured["channel"] == "S5"
    assert captured["channels"] is None


def test_read_channels_from_multiple_bands(tmp_path, monkeypatch):
    """A multi-element bands list becomes the `channels` set."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_slstr",
        lambda s, *, channel, channels, **o: captured.update(
            channel=channel, channels=channels
        ),
    )
    SlstrDriver().read(str(granule), bands=["S5", "S8"])
    assert captured["channel"] is None
    assert captured["channels"] == ["S5", "S8"]


def test_read_bands_as_numpy_array_maps_to_channels(tmp_path, monkeypatch):
    """A numpy-array bands maps to the channels set, not an 'ambiguous truth value' error."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_slstr",
        lambda s, *, channel, channels, **o: captured.update(channels=channels),
    )
    SlstrDriver().read(str(granule), bands=np.array(["S5", "S8"]))
    assert list(captured["channels"]) == ["S5", "S8"]


def test_read_without_a_channel_raises(tmp_path):
    """An SLSTR read needs a channel, via bands or options."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    driver = SlstrDriver()
    source = str(granule)
    with pytest.raises(ValueError, match="needs a channel"):
        driver.read(source)


@pytest.mark.parametrize("bands", [[], None])
def test_read_empty_or_absent_bands_needs_a_channel(tmp_path, bands):
    """An empty or absent bands list names no channel, so the clear error is raised."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    driver = SlstrDriver()
    source = str(granule)
    with pytest.raises(ValueError, match="needs a channel"):
        driver.read(source, bands=bands)


def test_read_empty_channels_list_needs_a_channel(tmp_path):
    """An empty channels list in options is a clear error, symmetric with bands=[]."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    driver = SlstrDriver()
    source = str(granule)
    with pytest.raises(ValueError, match="needs a channel"):
        driver.read(source, options={"channels": []})


@pytest.mark.parametrize("bands", ["S5", 42, {"S5": 1}])
def test_read_rejects_non_sequence_bands(tmp_path, bands):
    """A bare string, int or dict bands gives the clear error, not a TypeError."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    driver = SlstrDriver()
    source = str(granule)
    with pytest.raises(ValueError, match="list of channel names"):
        driver.read(source, bands=bands)


def test_read_rejects_both_channel_and_channels(tmp_path):
    """Giving both channel and channels in options is rejected at the driver boundary."""
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    driver = SlstrDriver()
    source = str(granule)
    with pytest.raises(ValueError, match="not both"):
        driver.read(source, options={"channel": "S5", "channels": ["S8"]})


def test_holds_measurement_rejects_a_file(tmp_path):
    """A plain file is never treated as a granule directory."""
    from pyramids_eo.drivers.slstr import _holds_measurement

    band = tmp_path / "S5_radiance_an.nc"
    band.write_bytes(b"")
    assert _holds_measurement(band) is False


def test_can_open_glob_matching_a_measurement_directory(tmp_path):
    """A glob whose matches include a measurement-bearing directory is claimed."""
    granule = _granule(tmp_path, ["S8_BT_in.nc"])
    assert SlstrDriver.can_open(str(granule.parent / "*.SEN3")) is True


def test_read_coerces_a_string_boolean_open_option(tmp_path):
    """`geolocation=false` as a KEY=VALUE string disables geolocation, not enables it.

    `Dataset.read_file` normalises open_options to the GDAL KEY=VALUE list form, so a
    boolean reader keyword arrives as the string "false" -- which is truthy, and would
    therefore silently turn geolocation ON. The OLCI driver has this test; SLSTR shares
    the helper but had nothing pinning it, so a driver-specific regression (a missing
    `coerce_booleans` call, or the wrong key tuple) would have passed here.
    """
    granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
    dataset = SlstrDriver().read(
        str(granule), options=["channel=S5", "geolocation=false"]
    )
    assert not dataset.raster.GetMetadata("GEOLOCATION"), (
        "geolocation=false must disable the domain"
    )
    assert dataset.raster.GetMetadata()["geolocation"] == (
        "none (placeholder pixel grid)"
    ), dataset.raster.GetMetadata()


@pytest.mark.parametrize("spelling", ["true", "TRUE", "yes", "on", "1"])
def test_read_accepts_the_truthy_boolean_spellings(tmp_path, spelling):
    """Every accepted true spelling enables geolocation.

    Args:
        spelling: One accepted string spelling of boolean true.
    """
    granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
    dataset = SlstrDriver().read(
        str(granule), options=["channel=S5", f"geolocation={spelling}"]
    )
    assert dataset.raster.GetMetadata("GEOLOCATION"), f"{spelling!r} should enable it"


@pytest.mark.parametrize("spelling", ["false", "FALSE", "no", "off", "0"])
def test_read_accepts_the_falsy_boolean_spellings(tmp_path, spelling):
    """Every accepted false spelling disables geolocation.

    Args:
        spelling: One accepted string spelling of boolean false.
    """
    granule = slstr_granule(tmp_path, channels={"S5": packed_counts()}, geodetic=False)
    dataset = SlstrDriver().read(
        str(granule), options=["channel=S5", f"geolocation={spelling}"]
    )
    assert not dataset.raster.GetMetadata("GEOLOCATION"), (
        f"{spelling!r} should disable the domain"
    )


def test_read_rejects_a_non_boolean_spelling(tmp_path):
    """An unrecognised boolean spelling is rejected rather than guessed at."""
    granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
    driver = SlstrDriver()
    source = str(granule)
    with pytest.raises(ValueError, match="must be a boolean"):
        driver.read(source, options=["channel=S5", "geolocation=maybe"])


def test_the_channel_error_does_not_advertise_the_channels_option(tmp_path):
    """The error must not teach `options={'channels': [...]}`, which read_file rejects.

    pyramids' plugin layer requires a driver to return one Dataset, so a channel set
    through read_file raises `TypeError: raster driver 'slstr' returned dict`.
    """
    granule = _granule(tmp_path, ["S5_radiance_an.nc"])
    source = str(granule)
    with pytest.raises(ValueError, match="needs a channel") as error:
        SlstrDriver().read(source)
    assert "channels" not in str(error.value), (
        f"the error must not advertise a channel set: {error.value}"
    )


def test_read_file_routes_a_single_channel_end_to_end(tmp_path):
    """`Dataset.read_file` reaches `read_slstr` through the real plugin path.

    Every other driver test monkeypatches the reader or uses empty placeholder files,
    so nothing exercised the genuine read_file -> entry point -> can_open -> read ->
    reader chain for SLSTR against a decodable granule.
    """
    from pyramids.dataset import Dataset

    granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
    dataset = Dataset.read_file(str(granule), open_options={"channel": "S5"})
    assert dataset.read_array().shape == (2, 3), "read_file should decode the channel"
    assert dataset.raster.GetMetadata()["sensor"] == "slstr", (
        dataset.raster.GetMetadata()
    )
    assert dataset.raster.GetMetadata("GEOLOCATION").get("X_DATASET"), (
        "the swath domain must survive the trip through read_file"
    )


def test_read_file_coerces_a_string_boolean(tmp_path):
    """Through `read_file`, `geolocation` given as a string still disables the domain.

    Test scenario:
        This is the path a caller actually takes, and the one the string-boolean
        defect lived on: read_file hands the driver the option uninterpreted, so the
        coercion has to happen inside the driver rather than being assumed of pyramids.
    """
    from pyramids.dataset import Dataset

    granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
    dataset = Dataset.read_file(
        str(granule), open_options={"channel": "S5", "geolocation": "false"}
    )
    assert not dataset.raster.GetMetadata("GEOLOCATION"), (
        "geolocation='false' through read_file must disable the domain"
    )


def test_read_file_rejects_a_non_boolean_through_the_plugin_path(tmp_path):
    """A nonsense boolean surfaces as the driver's ValueError, not a silent true."""
    from pyramids.dataset import Dataset

    granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
    source = str(granule)
    with pytest.raises(ValueError, match="must be a boolean"):
        Dataset.read_file(
            source, open_options={"channel": "S5", "geolocation": "maybe"}
        )


def test_read_file_passes_the_view_and_grid_through(tmp_path):
    """`view` and `grid` reach the reader intact through `read_file`.

    Test scenario:
        These are the two arguments SLSTR adds over OLCI, and both are forwarded as
        leftover `**settings`. A driver that popped or renamed one would silently read
        the nadir `a` grid instead, which is a different footprint at a different
        resolution -- and still a valid-looking Dataset.
    """
    from pyramids.dataset import Dataset

    granule = slstr_granule(tmp_path, channels={"S5": packed_counts()}, view="o")
    dataset = Dataset.read_file(
        str(granule),
        open_options={"channel": "S5", "view": "oblique", "grid": "a"},
    )
    meta = dataset.raster.GetMetadata()
    assert (meta["view"], meta["grid"]) == ("oblique", "ao"), meta


def test_read_file_accepts_the_slstr_scheme(tmp_path):
    """A `slstr://` source routes to this driver and is stripped before the reader."""
    from pyramids.dataset import Dataset

    granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
    dataset = Dataset.read_file(f"slstr://{granule}", open_options={"channel": "S5"})
    assert dataset.read_array().shape == (2, 3), "the scheme should route and strip"
