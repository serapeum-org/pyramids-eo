"""Instrument readers — the package entry points.

Each reader takes a file path / chunk set / `Dataset` and returns a pyramids
`Dataset` / `DatasetCollection` (or curvilinear equivalent).

* `read_fci` — MTG-FCI L1C (FDHSI): stitch a channel (or several — `channels=[...]`
  returns a `dict`) across its chunk set and calibrate to reflectance / brightness
  temperature.
* `read_fci_l1c` — decode + stitch + calibrate one channel or several
  (`channels=[...]` returns a `dict`, opening each chunk once) across real MTG-FCI
  L1C FDHSI chunk files (packed radiance in nested groups, per-granule
  coefficients, geostationary grid).
* `available_channels` — list the VIS/IR channels present in FCI L1C FDHSI chunk(s).
* `open_fci_l1c_chunk` — lower-level `open_chunk` reading the FCI L1C FDHSI nested
  `data/<channel>/measured/effective_radiance` radiance for use with `read_fci`.
* `read_seviri` — MSG-SEVIRI Level-1.5 native (`.nat`): decode + calibrate +
  geolocate a VIS/IR channel (via `parse_seviri_native`, or an injected parser).
* `parse_seviri_native` — decode one VIS/IR channel of an MSG Level-1.5 native
  `.nat` granule to a geolocated radiance `Dataset` (the default `read_seviri`
  parser).
* `harmonise` — align multi-resolution bands onto one common target grid.
"""

from __future__ import annotations

from pyramids_eo.sensors.readers.fci import open_fci_l1c_chunk, read_fci
from pyramids_eo.sensors.readers.fci_l1c import available_channels, read_fci_l1c
from pyramids_eo.sensors.readers.harmonise import harmonise
from pyramids_eo.sensors.readers.seviri import parse_seviri_native, read_seviri

__all__ = [
    "available_channels",
    "harmonise",
    "open_fci_l1c_chunk",
    "parse_seviri_native",
    "read_fci",
    "read_fci_l1c",
    "read_seviri",
]
