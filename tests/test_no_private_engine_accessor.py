"""Guard: no pyramids ``Dataset`` engine-accessor usage in the package source.

pyramids-core is making its ``Dataset`` engine collaborators (``Bands``, ``IO``,
``COG``, …) internal (serapeum-org/pyramids-core#1623) with no deprecation shim, so
``ds.bands.<method>`` stops resolving once that lands. Production code must reach the
public ``Dataset`` facade (e.g. ``ds.select_bands``) instead of the private engine
accessor (``ds.bands.select``). This test fails if any accessor hop creeps back into
``src/pyramids_eo``.

Scoped to ``src/`` on purpose: ``scipy.io`` in the test suite collides with the ``.io.``
pattern, so a blanket scan of ``tests/`` would false-positive on an unrelated module.
"""

from __future__ import annotations

import re
from pathlib import Path

# The private pyramids Dataset engine accessors this package must not reach through.
_ENGINE_ACCESSOR = re.compile(r"\.(bands|cog|io|vectorize)\.")

_SRC = Path(__file__).resolve().parent.parent / "src" / "pyramids_eo"


def test_no_dataset_engine_accessor_in_source():
    """No ``.bands.`` / ``.cog.`` / ``.io.`` engine hop remains in the package source."""
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _ENGINE_ACCESSOR.search(line):
                rel = path.relative_to(_SRC.parent.parent)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "pyramids Dataset engine accessors must use the public facade "
        "(e.g. ds.select_bands), not ds.bands.select (pyramids-core#1623):\n  "
        + "\n  ".join(offenders)
    )
