"""Instrument readers — the package entry points.

Each reader takes a file path / chunk set / `Dataset` and returns a pyramids
`Dataset` / `DatasetCollection` (or curvilinear equivalent).

* `read_fci` — MTG-FCI L1C (FDHSI): stitch a channel across its chunk set and
  calibrate to reflectance / brightness temperature.
* `read_seviri` — MSG-SEVIRI native (`.nat`): calibrate + geolocate a channel
  (the raw `.nat` byte decode is injectable — see its module warning).
* `harmonise` — align multi-resolution bands onto one common target grid.
"""

from __future__ import annotations

from pyramids_eo.sensors.readers.fci import read_fci
from pyramids_eo.sensors.readers.harmonise import harmonise
from pyramids_eo.sensors.readers.seviri import read_seviri

__all__ = [
    "harmonise",
    "read_fci",
    "read_seviri",
]
