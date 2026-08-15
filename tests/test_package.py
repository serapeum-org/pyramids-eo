"""Smoke tests: the package imports and exposes its public surface."""

from __future__ import annotations

import pytest

import pyramids_eo
from pyramids_eo.errors import EOError, ReaderError


def test_version_is_exposed():
    assert isinstance(pyramids_eo.__version__, str)
    assert pyramids_eo.__version__


def test_error_hierarchy():
    assert issubclass(ReaderError, EOError)
    with pytest.raises(EOError):
        raise ReaderError("boom")


def test_subpackages_import():
    import pyramids_eo.readers  # noqa: F401
    import pyramids_eo.registry  # noqa: F401
