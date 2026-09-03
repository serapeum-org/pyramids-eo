"""Strip per-cell ``metadata.execution`` timestamps from notebooks.

`nbclient` stamps every executed cell with `iopub.execute_input`,
`iopub.status.busy`, `iopub.status.idle` and `shell.execute_reply` wall-clock
times. They change on every run, so a docs rebuild rewrites each notebook even
when nothing about the content moved, burying the real diff.

Baked **outputs** are deliberately kept: `mkdocs-jupyter` renders the notebooks
with `execute: false`, so the saved outputs are what the docs site shows. This
strips only the timing metadata, which nothing renders.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def strip(path: Path) -> bool:
    """Remove `metadata.execution` from every cell; return True if changed."""
    original = path.read_bytes()
    notebook = json.loads(original.decode("utf-8"))
    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("metadata", {}).pop("execution", None) is not None:
            changed = True
    if not changed:
        return False
    text = json.dumps(notebook, indent=4, ensure_ascii=True) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return True


def main(paths: list[str]) -> int:
    """Strip each notebook named on the command line."""
    rewritten = [name for name in paths if strip(Path(name))]
    for name in rewritten:
        print(f"stripped execution metadata: {name}")
    return 1 if rewritten else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
