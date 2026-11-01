"""Optional-dependency probes for extras-gated tests.

One source of truth for every ``[project.optional-dependencies]`` group (``viz`` →
cleopatra, ``lazy`` → dask). The ``pytest_collection_modifyitems`` hook in
:mod:`tests.conftest` reads :data:`EXTRA_MARKERS` to auto-skip a test tagged
``@pytest.mark.<extra>`` when the extra is not installed.

Each probe names the module the *guarded code* imports, not the distribution — so a
partial install of an extra's dependency skips rather than fails.
"""

from __future__ import annotations

import importlib.util


def _has(*module_names: str) -> bool:
    """Return True iff every named module is importable.

    A dotted name (``"dask.array"``) probes that submodule, which is what a guard
    importing it needs: a modular install can ship the parent without it. Probing a
    submodule imports its parent as a side effect, and that parent's import can
    raise any ``ImportError``, not only ``ModuleNotFoundError`` — so both are
    treated as "absent".
    """
    for name in module_names:
        try:
            spec = importlib.util.find_spec(name)
        except ImportError:
            spec = None
        if spec is None:
            return False
    return True


HAS_CLEOPATRA = _has("cleopatra")
# `dask.array`, not `dask`: DatasetCollection.data does `import dask.array as da`, so
# a dask core installed without the array stack must count as absent or the lazy
# tests fail hard instead of skipping.
HAS_DASK = _has("dask.array")

EXTRA_MARKERS: dict[str, tuple[bool, str]] = {
    "plot": (HAS_CLEOPATRA, "requires the [viz] extra (cleopatra)"),
    "lazy": (HAS_DASK, "requires pyramids-core[lazy] (dask)"),
}
