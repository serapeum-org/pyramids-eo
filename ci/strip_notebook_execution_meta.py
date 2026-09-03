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


def _resolve(name: str, root: Path) -> Path:
    """Resolve `name` to a notebook inside `root`, or reject it.

    pre-commit passes the staged file list, but the paths still arrive as plain
    argv strings. Confining them to `root` and to a `.ipynb` suffix keeps the
    rewrite below from ever touching a file outside the working tree.

    Args:
        name: A path as given on the command line.
        root: The directory the path must live under.

    Returns:
        The resolved path.

    Raises:
        ValueError: The path escapes `root` or is not a notebook.
    """
    path = Path(name).resolve()
    if path.suffix != ".ipynb":
        raise ValueError(f"not a notebook: {name}")
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes {root}: {name}")
    return path


def strip(path: Path) -> bool:
    """Remove `metadata.execution` from every cell; return True if changed."""
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


def main(names: list[str]) -> int:
    """Strip each notebook named on the command line."""
    root = Path.cwd().resolve()
    rewritten = [name for name in names if strip(_resolve(name, root))]
    for name in rewritten:
        print(f"stripped execution metadata: {name}")
    return 1 if rewritten else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
