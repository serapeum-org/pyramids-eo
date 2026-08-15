"""Optional-dependency probes for extras-gated tests.

One source of truth for every ``[project.optional-dependencies]`` group. The
``pytest_collection_modifyitems`` hook in :mod:`tests.conftest` reads
:data:`EXTRA_MARKERS` to auto-skip a test tagged ``@pytest.mark.<extra>`` when
the extra is not installed.
"""

from __future__ import annotations

import importlib.util


def _has(*module_names: str) -> bool:
    """Return True iff every named top-level module is importable."""
    for name in module_names:
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            return False
    return True


HAS_CLEOPATRA = _has("cleopatra")

# marker name -> (is-available, skip-reason). Marker names use underscores
# (valid ``pytest.mark.<name>`` identifiers); PyPI extra names use hyphens.
EXTRA_MARKERS: dict[str, tuple[bool, str]] = {
    "plot": (HAS_CLEOPATRA, "requires the [viz] extra (cleopatra)"),
}
