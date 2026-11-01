"""Unit tests for `pyramids_eo.drivers.olci.OlciDriver`.

The driver delegates to `read_olci`; the facade is monkeypatched to capture the call, so
these tests pin the claim set, the `can_open` content sniff (which must distinguish an OLCI
granule from an SLSTR one sharing the `.SEN3` suffix) and the band mapping, without
decoding real NetCDF files.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyramids_eo.sensors.readers as readers
from pyramids_eo.drivers.olci import OlciDriver
from tests._sentinel3_granule import olci_granule


def _granule(tmp_path, names, name="S3A_OL_1_EFR.SEN3"):
    """Create a granule directory holding empty files `names`; return its path."""
    granule = tmp_path / name
    granule.mkdir(parents=True, exist_ok=True)
    for entry in names:
        (granule / entry).write_bytes(b"")
    return granule


def test_can_open_olci_scheme():
    """An olci:// source is always claimed (cheap string check, no file access)."""
    assert OlciDriver.can_open("olci://anywhere") is True
    assert OlciDriver.can_open("OLCI://ANYWHERE") is True


def test_can_open_granule_with_radiance_bands(tmp_path):
    """A .SEN3 directory holding at least one Oa*_radiance.nc is claimed."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc", "geo_coordinates.nc"])
    assert OlciDriver.can_open(str(granule)) is True


def test_can_open_rejects_an_slstr_granule(tmp_path):
    """An SLSTR granule shares the .SEN3 suffix but is not claimed by the OLCI driver."""
    granule = _granule(
        tmp_path, ["S5_radiance_an.nc", "geodetic_an.nc"], name="S3A_SL_1_RBT.SEN3"
    )
    assert OlciDriver.can_open(str(granule)) is False


def test_can_open_rejects_an_unrelated_netcdf_directory(tmp_path):
    """A directory of unrelated .nc files is not claimed."""
    granule = _granule(tmp_path, ["random.nc", "other.nc"])
    assert OlciDriver.can_open(str(granule)) is False


@pytest.mark.parametrize("source", ["scene.tif", 42, "ee://x", "", None])
def test_can_open_rejects_non_granule(source):
    """A plain file, a non-string, another scheme, or an empty source is not claimed."""
    assert OlciDriver.can_open(source) is False


def test_can_open_glob_matching_a_granule(tmp_path):
    """A glob whose matches include an OLCI granule directory is claimed."""
    _granule(tmp_path, ["Oa08_radiance.nc"])
    assert OlciDriver.can_open(str(tmp_path / "*.SEN3")) is True


def test_can_open_glob_without_a_granule(tmp_path):
    """A glob matching only non-OLCI directories is not claimed."""
    _granule(tmp_path, ["random.nc"])
    assert OlciDriver.can_open(str(tmp_path / "*.SEN3")) is False


def test_can_open_granule_name_with_glob_metachar(tmp_path):
    """A real granule whose name contains a glob metachar is still claimed."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"], name="granule[1].SEN3")
    assert OlciDriver.can_open(str(granule)) is True


def test_can_open_existing_single_file_is_false(tmp_path):
    """An existing single file (not a granule directory) is not claimed."""
    band = tmp_path / "Oa08_radiance.nc"
    band.write_bytes(b"")
    assert OlciDriver.can_open(str(band)) is False


def test_can_open_never_raises_on_a_filesystem_error(tmp_path, monkeypatch):
    """A directory whose glob raises OSError yields False, never an exception (contract)."""
    from pathlib import Path

    def _boom(self, pattern):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "glob", _boom)
    assert OlciDriver.can_open(str(tmp_path)) is False


def test_conforms_to_the_contract():
    """The driver satisfies pyramids' raster-driver conformance kit."""
    from pyramids.dataset.driver_conformance import assert_raster_driver_conforms

    assert_raster_driver_conforms(OlciDriver)


def test_claims():
    """The class-level claims are the OLCI ones (.SEN3 suffix, olci:// scheme)."""
    assert OlciDriver.name == "olci"
    assert OlciDriver.suffixes == (".sen3",)
    assert OlciDriver.schemes == ("olci",)
    assert OlciDriver.extra is None
    assert OlciDriver.capabilities == frozenset({"read"})


def test_read_passes_the_channel_and_options(tmp_path, monkeypatch):
    """`read` forwards the granule, channel and remaining options to read_olci."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_olci",
        lambda s, *, channel, channels, **o: captured.update(
            source=s, channel=channel, channels=channels, options=o
        ),
    )
    OlciDriver().read(str(granule), options={"channel": "Oa08", "geolocation": False})
    assert captured["source"] == str(granule)
    assert captured["channel"] == "Oa08"
    assert captured["channels"] is None
    assert captured["options"] == {"geolocation": False}


def test_read_strips_the_olci_scheme(tmp_path, monkeypatch):
    """An olci:// prefix is stripped before the reader is called."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    captured = {}
    monkeypatch.setattr(readers, "read_olci", lambda s, **k: captured.update(source=s))
    OlciDriver().read("olci://" + str(granule), options={"channel": "Oa08"})
    assert captured["source"] == str(granule)


def test_read_channel_from_single_band(tmp_path, monkeypatch):
    """A one-element bands list is read as a single channel."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_olci",
        lambda s, *, channel, channels, **o: captured.update(
            channel=channel, channels=channels
        ),
    )
    OlciDriver().read(str(granule), bands=["Oa08"])
    assert captured["channel"] == "Oa08"
    assert captured["channels"] is None


def test_read_channels_from_multiple_bands(tmp_path, monkeypatch):
    """A multi-element bands list becomes the `channels` set, not a dropped-to-first band."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_olci",
        lambda s, *, channel, channels, **o: captured.update(
            channel=channel, channels=channels
        ),
    )
    OlciDriver().read(str(granule), bands=["Oa08", "Oa17"])
    assert captured["channel"] is None
    assert captured["channels"] == ["Oa08", "Oa17"]


def test_read_bands_as_numpy_array_maps_to_channels(tmp_path, monkeypatch):
    """A numpy-array bands maps to the channels set, not an 'ambiguous truth value' error."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_olci",
        lambda s, *, channel, channels, **o: captured.update(channels=channels),
    )
    OlciDriver().read(str(granule), bands=np.array(["Oa08", "Oa17"]))
    assert list(captured["channels"]) == ["Oa08", "Oa17"]


def test_read_without_a_band_raises(tmp_path):
    """An OLCI read needs a band, via bands or options."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    with pytest.raises(ValueError, match="needs a band"):
        OlciDriver().read(str(granule))


@pytest.mark.parametrize("bands", [[], None])
def test_read_empty_or_absent_bands_needs_a_band(tmp_path, bands):
    """An empty or absent bands list names no band, so the clear band error is raised."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    with pytest.raises(ValueError, match="needs a band"):
        OlciDriver().read(str(granule), bands=bands)


def test_read_empty_channels_list_needs_a_band(tmp_path):
    """An empty channels list in options is a clear band error, symmetric with bands=[]."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    with pytest.raises(ValueError, match="needs a band"):
        OlciDriver().read(str(granule), options={"channels": []})


@pytest.mark.parametrize("bands", ["Oa08", 42, {"Oa08": 1}])
def test_read_rejects_non_sequence_bands(tmp_path, bands):
    """A bare string, int or dict bands gives the clear error, not a TypeError."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    with pytest.raises(ValueError, match="list of channel names"):
        OlciDriver().read(str(granule), bands=bands)


def test_read_rejects_both_channel_and_channels(tmp_path):
    """Giving both channel and channels in options is rejected at the driver boundary."""
    granule = _granule(tmp_path, ["Oa08_radiance.nc"])
    with pytest.raises(ValueError, match="not both"):
        OlciDriver().read(
            str(granule), options={"channel": "Oa08", "channels": ["Oa17"]}
        )


def test_read_coerces_a_string_boolean_open_option(tmp_path):
    """`geolocation=false` as a KEY=VALUE string disables geolocation, not enables it.

    `Dataset.read_file` normalises open_options to the GDAL KEY=VALUE list form, so a
    boolean reader keyword arrives as the string "false" -- which is truthy, and used to
    silently turn geolocation ON. That is the common path through read_file, not an
    exotic one.
    """
    granule = olci_granule(tmp_path)
    dataset = OlciDriver().read(
        str(granule), options=["channel=Oa08", "geolocation=false"]
    )
    assert not dataset.raster.GetMetadata("GEOLOCATION"), (
        "geolocation=false must disable the domain"
    )


@pytest.mark.parametrize("spelling", ["true", "TRUE", "yes", "on", "1"])
def test_read_accepts_the_truthy_boolean_spellings(tmp_path, spelling):
    """Every accepted true spelling enables geolocation."""
    granule = olci_granule(tmp_path)
    dataset = OlciDriver().read(
        str(granule), options=["channel=Oa08", f"geolocation={spelling}"]
    )
    assert dataset.raster.GetMetadata("GEOLOCATION"), f"{spelling!r} should enable it"


def test_read_rejects_a_non_boolean_spelling(tmp_path):
    """An unrecognised boolean spelling is rejected rather than guessed at."""
    granule = olci_granule(tmp_path)
    with pytest.raises(ValueError, match="must be a boolean"):
        OlciDriver().read(str(granule), options=["channel=Oa08", "geolocation=maybe"])


def test_the_band_error_does_not_advertise_the_channels_option(tmp_path):
    """The error must not teach `options={'channels': [...]}`, which read_file rejects.

    pyramids' plugin layer requires a driver to return one Dataset, so a channel set
    through read_file raises `TypeError: raster driver 'olci' returned dict`. The
    message a user is most likely to hit used to recommend exactly that.
    """
    granule = olci_granule(tmp_path)
    with pytest.raises(ValueError, match="needs a band") as error:
        OlciDriver().read(str(granule))
    assert "channels" not in str(error.value), (
        f"the error must not advertise a channel set: {error.value}"
    )


def test_read_file_routes_a_single_channel_end_to_end(tmp_path):
    """`Dataset.read_file` reaches the reader through the real plugin path.

    The other driver tests monkeypatch the reader, so nothing exercised the genuine
    read_file -> entry point -> can_open -> read -> reader chain, which is how the
    string-boolean and channel-set defects both survived.
    """
    from pyramids.dataset import Dataset

    granule = olci_granule(tmp_path)
    dataset = Dataset.read_file(str(granule), open_options={"channel": "Oa08"})
    assert dataset.read_array().shape == (2, 3), "read_file should decode the band"


@pytest.mark.parametrize("value", [1, 0, None, ["false"]])
def test_coerce_booleans_leaves_non_string_values_alone(value):
    """A non-string, non-bool option is passed through for the reader to reject.

    The driver only translates the KEY=VALUE string spellings; it does not try to
    second-guess a caller who passed some other type in a dict.
    """
    from pyramids_eo.drivers._base import coerce_booleans

    settings = {"geolocation": value}
    coerce_booleans(settings, ("geolocation",), driver="olci")
    assert settings["geolocation"] is value, settings
