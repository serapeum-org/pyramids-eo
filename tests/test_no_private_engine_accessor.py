"""Guard: no pyramids ``Dataset`` engine-accessor usage in this package.

pyramids-core is making its ``Dataset`` engine collaborators internal
(serapeum-org/pyramids-core#1623) with no deprecation shim, so ``ds.<engine>.<method>``
stops resolving once that lands. Code must reach the public ``Dataset`` facade
(e.g. ``ds.select_bands``) instead of the private engine accessor (``ds.bands.select``).
This test fails if any accessor hop creeps back into ``src/`` or ``tests/``.

``Dataset.__init__`` wires eight collaborators — ``io``, ``spatial``, ``bands``,
``analysis``, ``cell``, ``vectorize``, ``cog`` and ``georef`` — and all eight are
checked. The check is lexical, not semantic: it scans source text, so an accessor reached
indirectly (``getattr(ds, "bands")``, ``ds.bands["x"]``, an alias) is out of scope by
design — it catches the direct ``X.<engine>.<method>`` form the migration removes.
Being lexical it is deliberately conservative: it flags ``.<engine>.`` on *any*
object, not only a ``Dataset`` (``other.cell.value`` would match). That is the safe
direction for a guard; a legitimate non-Dataset use is whitelisted by adding its
dotted prefix to ``_ALLOWED`` below, which is stripped from each line before the
match. Two allowances already apply — ``scipy.io`` (an unrelated standard module
colliding with the ``io`` accessor) and this guard file itself (it names the banned
forms as documentation).
"""

from __future__ import annotations

import re
from pathlib import Path

# The eight pyramids Dataset engine collaborators (pyramids-core#1623).
_ENGINE_ACCESSOR = re.compile(
    r"\.(io|spatial|bands|analysis|cell|vectorize|cog|georef)\."
)

# An unrelated standard module whose dotted name collides with the ``io`` accessor.
_ALLOWED = ("scipy.io",)

_THIS_FILE = Path(__file__).resolve()
_ROOT = _THIS_FILE.parent.parent
_SCAN_ROOTS = (_ROOT / "src" / "pyramids_eo", _ROOT / "tests")


def test_no_dataset_engine_accessor_in_source():
    """No direct ``.<engine>.`` hop on a Dataset remains in ``src/`` or ``tests/``."""
    offenders: list[str] = []
    for root in _SCAN_ROOTS:
        for path in root.rglob("*.py"):
            if path.resolve() == _THIS_FILE:
                continue
            text = path.read_text(encoding="utf-8", errors="surrogateescape")
            for lineno, line in enumerate(text.splitlines(), 1):
                probe = line
                for allowed in _ALLOWED:
                    probe = probe.replace(allowed, "")
                if _ENGINE_ACCESSOR.search(probe):
                    offenders.append(f"{path.relative_to(_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "pyramids Dataset engine accessors must use the public facade "
        "(e.g. ds.select_bands), not ds.bands.select (pyramids-core#1623):\n  "
        + "\n  ".join(offenders)
    )
