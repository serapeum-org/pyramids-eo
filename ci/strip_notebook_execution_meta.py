"""Strip per-cell ``metadata.execution`` timestamps from the docs notebooks.

`nbclient` stamps every executed cell with `iopub.execute_input`,
`iopub.status.busy`, `iopub.status.idle` and `shell.execute_reply` wall-clock
times. They change on every run, so a docs rebuild rewrites each notebook even
when nothing about the content moved, burying the real diff.

Baked **outputs** are deliberately kept: `mkdocs-jupyter` renders the notebooks
with `execute: false`, so the saved outputs are what the docs site shows. This
strips only the timing metadata, which nothing renders.

The notebooks are discovered by globbing `NOTEBOOK_ROOT` rather than taken from
the command line, so the paths written are ones this script constructed from a
literal root. The pre-commit hook therefore runs with `pass_filenames: false`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: Where the example notebooks live, relative to the repository root.
NOTEBOOK_ROOT = Path("docs/examples")


def strip(path: Path) -> bool:
    """Remove `metadata.execution` from every cell; return True if changed.

    Args:
        path: The notebook to rewrite.

    Returns:
        True when the notebook carried timing metadata and was rewritten.
    """
    notebook = json.loads(path.read_bytes().decode("utf-8"))
    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("metadata", {}).pop("execution", None) is not None:
            changed = True
    if not changed:
        return False
    text = json.dumps(notebook, indent=4, ensure_ascii=True) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return True


def main() -> int:
    """Strip every notebook under `NOTEBOOK_ROOT`; non-zero if any changed."""
    rewritten = [path for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")) if strip(path)]
    for path in rewritten:
        print(f"stripped execution metadata: {path.as_posix()}")
    return 1 if rewritten else 0


if __name__ == "__main__":
    sys.exit(main())
