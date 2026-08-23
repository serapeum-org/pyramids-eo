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
    import pyramids_eo.sensors.readers  # noqa: F401
    import pyramids_eo.sensors.registry  # noqa: F401


def test_sensors_facade_reexports():
    """The `sensors` facade exposes every name in its `__all__`."""
    import pyramids_eo.sensors as sensors

    missing = [name for name in sensors.__all__ if not hasattr(sensors, name)]
    assert not missing, f"sensors facade is missing {missing} declared in __all__"


def test_sensors_facade_all_is_union_of_subpackages():
    """The facade `__all__` is exactly the readers + registry public surface."""
    from pyramids_eo.sensors import __all__ as facade_all
    from pyramids_eo.sensors.readers import __all__ as readers_all
    from pyramids_eo.sensors.registry import __all__ as registry_all

    expected = set(readers_all) | set(registry_all)
    assert set(facade_all) == expected, (
        f"facade __all__ {sorted(facade_all)} != readers|registry {sorted(expected)}"
    )
