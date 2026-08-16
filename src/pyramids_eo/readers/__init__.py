"""Instrument readers — the package entry points.

Each reader takes a local file / path / bytes and returns a pyramids `Dataset`
/ `DatasetCollection` (or curvilinear equivalent).

* `read_fci` — MTG-FCI L1C (FDHSI): stitch a channel across its chunk set and
  calibrate to reflectance / brightness temperature.
"""

from __future__ import annotations

from pyramids_eo.readers.fci import read_fci

__all__ = [
    "read_fci",
]
