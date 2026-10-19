"""Unit tests for the shared raster-driver helpers in `pyramids_eo.drivers._base`."""

import pytest

from pyramids_eo.drivers._base import (
    RASTER_DRIVER_API_VERSION,
    expand_sources,
    options_as_dict,
)


def test_api_version_is_one():
    """The declared contract version matches pyramids' RASTER_DRIVER_API_VERSION (1)."""
    assert RASTER_DRIVER_API_VERSION == 1


def test_options_none_is_empty_dict():
    """`None` options normalise to an empty dict."""
    assert options_as_dict(None) == {}


def test_options_dict_is_copied():
    """A dict is returned as a shallow copy, leaving the caller's object untouched."""
    src = {"channel": "IR_108"}
    out = options_as_dict(src)
    assert out == src
    assert out is not src


def test_options_key_value_list_parsed():
    """A GDAL-style ``KEY=VALUE`` sequence parses to a dict of strings."""
    assert options_as_dict(["reducer=median", "tile_size=256"]) == {
        "reducer": "median",
        "tile_size": "256",
    }


def test_options_value_with_equals_keeps_the_rest():
    """Only the first ``=`` splits a ``KEY=VALUE`` item; later ``=`` stay in the value."""
    assert options_as_dict(["expr=a=b"]) == {"expr": "a=b"}


def test_options_item_without_equals_raises():
    """A sequence item that is not ``KEY=VALUE`` is a ValueError naming the item."""
    with pytest.raises(ValueError, match="channel"):
        options_as_dict(["channel"])


@pytest.mark.parametrize("bare", ["channel=IR_108", b"channel=IR_108"])
def test_options_bare_string_rejected(bare):
    """A bare str/bytes is rejected outright, not iterated character by character."""
    with pytest.raises(ValueError, match="bare"):
        options_as_dict(bare)


def test_options_accepts_a_non_dict_mapping():
    """Any Mapping, not only an exact dict, is coerced to a plain dict (not iterated as keys)."""
    from types import MappingProxyType

    out = options_as_dict(MappingProxyType({"channel": "IR_108"}))
    assert out == {"channel": "IR_108"}
    assert type(out) is dict


def test_expand_sources_single_file_is_a_one_element_list():
    """A plain path that is not a directory or glob is returned as a single-element list."""
    assert expand_sources("granule/chunk_0.nc") == ["granule/chunk_0.nc"]


def test_expand_sources_directory_globs_the_pattern(tmp_path):
    """A directory source expands to its sorted entries matching the pattern."""
    for name in ["c1.nc", "c0.nc", "note.txt"]:
        (tmp_path / name).write_bytes(b"")
    out = expand_sources(str(tmp_path), pattern="*.nc")
    assert out == sorted([str(tmp_path / "c0.nc"), str(tmp_path / "c1.nc")])


def test_expand_sources_glob_string_is_expanded(tmp_path):
    """A glob string expands to its sorted matches, ignoring the directory pattern."""
    for name in ["b.nc", "a.nc"]:
        (tmp_path / name).write_bytes(b"")
    out = expand_sources(str(tmp_path / "*.nc"))
    assert out == sorted([str(tmp_path / "a.nc"), str(tmp_path / "b.nc")])


def test_expand_sources_empty_directory_is_empty(tmp_path):
    """A directory with no match expands to an empty list."""
    assert expand_sources(str(tmp_path), pattern="*.nc") == []


def test_expand_sources_directory_name_with_glob_metachar(tmp_path):
    """A real directory whose name contains a glob metachar is treated literally, not as a pattern."""
    granule = tmp_path / "granule[1]"
    granule.mkdir()
    (granule / "c0.nc").write_bytes(b"")
    assert expand_sources(str(granule), pattern="*.nc") == [str(granule / "c0.nc")]


def test_expand_sources_existing_file_with_glob_metachar(tmp_path):
    """A real file whose name contains a glob metachar is returned literally, not expanded."""
    chunk = tmp_path / "a[1].nc"
    chunk.write_bytes(b"")
    assert expand_sources(str(chunk)) == [str(chunk)]
