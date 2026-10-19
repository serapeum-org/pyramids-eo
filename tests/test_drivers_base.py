"""Unit tests for the shared raster-driver helpers in `pyramids_eo.drivers._base`."""

import pytest

from pyramids_eo.drivers._base import RASTER_DRIVER_API_VERSION, options_as_dict


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
    assert out == src and out is not src


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
    assert out == {"channel": "IR_108"} and type(out) is dict
