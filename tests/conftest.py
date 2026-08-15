"""Shared test configuration.

``pyramids`` activates its (optionally vendored) osgeo on import, so import it
before anything reaches ``from osgeo import ...``. pyramids-eo builds on that
engine, so the same ordering guard applies here.
"""

from __future__ import annotations

# isort: off
import pyramids as _pyramids_bootstrap  # noqa: F401
# isort: on

import pytest

from tests._marks import EXTRA_MARKERS


def pytest_collection_modifyitems(config, items):
    """Auto-skip extras-gated tests when the extra's headline dep is missing.

    A test tagged ``@pytest.mark.<extra>`` (e.g. ``plot``) is skipped when the
    extra is not installed, mirroring pyramids' collection hook.
    """
    for item in items:
        for marker_name, (available, reason) in EXTRA_MARKERS.items():
            if marker_name in item.keywords and not available:
                item.add_marker(pytest.mark.skip(reason=reason))
