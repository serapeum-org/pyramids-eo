"""Unit tests for `pyramids_eo.drivers.fci.FciDriver`.

The driver delegates to `read_fci_l1c`; the facade is monkeypatched to capture the call, so these
tests pin the source expansion (directory/glob/fci:// → sorted chunk list) and the channel mapping
without decoding real NetCDF chunks.
"""

import numpy as np
import pytest

import pyramids_eo.sensors.readers as readers
from pyramids_eo.drivers.fci import FciDriver


def _make_granule(tmp_path, names):
    """Create empty files named `names` under tmp_path; return (dir, [paths])."""
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"")
        paths.append(str(p))
    return tmp_path, paths


def test_can_open_fci_scheme():
    """An fci:// source is always claimed (cheap string check, no file access)."""
    assert FciDriver.can_open("fci://anywhere") is True
    assert FciDriver.can_open("FCI://ANYWHERE") is True


def test_can_open_directory_with_fdhsi_chunks(tmp_path):
    """A directory holding at least one *FDHSI*.nc chunk is claimed."""
    _make_granule(tmp_path, ["W_XX_FCI-1C-RRAD-FDHSI-FD--CHK-BODY--x.nc", "note.txt"])
    assert FciDriver.can_open(str(tmp_path)) is True


def test_can_open_rejects_a_plain_netcdf_directory(tmp_path):
    """A directory of unrelated .nc files (no FDHSI token) is not claimed."""
    _make_granule(tmp_path, ["random.nc", "other.nc"])
    assert FciDriver.can_open(str(tmp_path)) is False


@pytest.mark.parametrize("source", ["scene.tif", 42, "ee://x"])
def test_can_open_rejects_non_granule(source):
    """A plain file, a non-string, or another scheme is not claimed."""
    assert FciDriver.can_open(source) is False


def test_can_open_glob_with_fdhsi_match(tmp_path):
    """A glob whose matches include an FDHSI chunk is claimed."""
    (tmp_path / "W_XX_FCI-1C-FDHSI_x.nc").write_bytes(b"")
    assert FciDriver.can_open(str(tmp_path / "*.nc")) is True


def test_can_open_glob_without_fdhsi_match(tmp_path):
    """A glob matching only non-FDHSI files is not claimed."""
    (tmp_path / "random.nc").write_bytes(b"")
    assert FciDriver.can_open(str(tmp_path / "*.nc")) is False


def test_can_open_directory_name_with_glob_metachar(tmp_path):
    """A real granule directory whose name contains a glob metachar is still claimed."""
    granule = tmp_path / "granule[1]"
    granule.mkdir()
    (granule / "W_XX_FCI-1C-FDHSI_x.nc").write_bytes(b"")
    assert FciDriver.can_open(str(granule)) is True


def test_can_open_existing_single_file_is_false(tmp_path):
    """An existing single file (not a granule directory) is not claimed."""
    chunk = tmp_path / "W_XX_FCI-1C-FDHSI_x.nc"
    chunk.write_bytes(b"")
    assert FciDriver.can_open(str(chunk)) is False


def test_conforms_to_the_contract():
    """The driver satisfies pyramids' raster-driver conformance kit."""
    from pyramids.dataset.driver_conformance import assert_raster_driver_conforms

    assert_raster_driver_conforms(FciDriver)


def test_claims():
    """The class-level claims are the FCI ones (fci:// scheme, no backend extra)."""
    assert FciDriver.name == "fci"
    assert FciDriver.suffixes == ()
    assert FciDriver.schemes == ("fci",)
    assert FciDriver.extra is None
    assert FciDriver.capabilities == frozenset({"read"})


def test_read_expands_a_directory_to_the_sorted_chunk_list(tmp_path, monkeypatch):
    """`read` globs a directory's *.nc chunks into a sorted list for read_fci_l1c."""
    granule, paths = _make_granule(tmp_path, ["c1.nc", "c0.nc", "ignore.txt"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_fci_l1c",
        lambda p, *, channel, channels, **o: captured.update(
            paths=p, channel=channel, channels=channels, options=o
        ),
    )
    FciDriver().read(str(granule), options={"channel": "ir_105", "calibrate": False})
    assert captured["paths"] == sorted(
        [str(tmp_path / "c0.nc"), str(tmp_path / "c1.nc")]
    )
    assert captured["channel"] == "ir_105"
    assert captured["channels"] is None
    assert captured["options"] == {"calibrate": False}


def test_read_strips_the_fci_scheme(tmp_path, monkeypatch):
    """An fci:// prefix is stripped before the directory is expanded."""
    granule, _ = _make_granule(tmp_path, ["c0.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_fci_l1c",
        lambda p, *, channel, channels, **o: captured.update(paths=p),
    )
    FciDriver().read("fci://" + str(granule), options={"channel": "ir_105"})
    assert captured["paths"] == [str(tmp_path / "c0.nc")]


def test_read_channel_from_single_band(tmp_path, monkeypatch):
    """A one-element bands list is read as a single channel."""
    granule, _ = _make_granule(tmp_path, ["c0.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_fci_l1c",
        lambda p, *, channel, channels, **o: captured.update(
            channel=channel, channels=channels
        ),
    )
    FciDriver().read(str(granule), bands=["ir_105"])
    assert captured["channel"] == "ir_105"
    assert captured["channels"] is None


def test_read_channels_from_multiple_bands(tmp_path, monkeypatch):
    """A multi-element bands list becomes the `channels` set, not a dropped-to-first channel."""
    granule, _ = _make_granule(tmp_path, ["c0.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_fci_l1c",
        lambda p, *, channel, channels, **o: captured.update(
            channel=channel, channels=channels
        ),
    )
    FciDriver().read(str(granule), bands=["ir_105", "vis_06"])
    assert captured["channel"] is None
    assert captured["channels"] == ["ir_105", "vis_06"]


def test_read_without_channel_raises(tmp_path):
    """An FCI read needs a channel, via bands or options."""
    granule, _ = _make_granule(tmp_path, ["c0.nc"])
    driver = FciDriver()
    with pytest.raises(ValueError, match="needs a channel"):
        driver.read(str(granule))


def test_read_empty_bands_list_needs_a_channel(tmp_path):
    """An empty bands list names no channel, so the clear channel error is raised."""
    granule, _ = _make_granule(tmp_path, ["c0.nc"])
    driver = FciDriver()
    with pytest.raises(ValueError, match="needs a channel"):
        driver.read(str(granule), bands=[])


def test_read_rejects_bare_string_bands(tmp_path):
    """A bare-string bands is rejected, not iterated into per-character channels."""
    granule, _ = _make_granule(tmp_path, ["c0.nc"])
    driver = FciDriver()
    with pytest.raises(ValueError, match="list of channel names"):
        driver.read(str(granule), bands="ir_105")


def test_read_bands_as_numpy_array_maps_to_channels(tmp_path, monkeypatch):
    """A numpy-array bands maps to the channels set, not an 'ambiguous truth value' error."""
    granule, _ = _make_granule(tmp_path, ["c0.nc"])
    captured = {}
    monkeypatch.setattr(
        readers,
        "read_fci_l1c",
        lambda p, *, channel, channels, **o: captured.update(
            channel=channel, channels=channels
        ),
    )
    FciDriver().read(str(granule), bands=np.array(["ir_105", "vis_06"]))
    assert captured["channel"] is None
    assert list(captured["channels"]) == ["ir_105", "vis_06"]


def test_read_empty_source_raises_a_clear_error(tmp_path):
    """A source that expands to no chunk files is a clear error, not a channel-coverage one."""
    empty = tmp_path / "empty"
    empty.mkdir()
    driver = FciDriver()
    with pytest.raises(ValueError, match="no chunk files"):
        driver.read(str(empty), options={"channel": "ir_105"})


def test_read_rejects_both_channel_and_channels(tmp_path):
    """Giving both channel and channels in options is rejected at the driver boundary."""
    granule, _ = _make_granule(tmp_path, ["c0.nc"])
    driver = FciDriver()
    with pytest.raises(ValueError, match="not both"):
        driver.read(str(granule), options={"channel": "ir_105", "channels": ["vis_06"]})
