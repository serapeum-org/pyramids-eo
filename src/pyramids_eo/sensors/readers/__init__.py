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
* `read_olci` — Sentinel-3 OLCI L1B (EFR / ERR): decode one radiance band (or
  several — `channels=[...]` returns a `dict`) from a `.SEN3` granule onto its
  native swath grid, geolocated by a GDAL `GEOLOCATION` domain.
* `available_olci_channels` — list the OLCI radiance bands a `.SEN3` granule carries.
* `read_slstr` — Sentinel-3 SLSTR L1 RBT: decode one channel (or several) for a
  chosen `view` (nadir / oblique) and grid, likewise swath-geolocated.
* `harmonise` — align multi-resolution bands onto one common target grid.

The Sentinel-3 readers decode to arrays plus CF metadata only — radiometric
calibration choices stay with the caller (issue #136). Being swath instruments,
their results carry a GDAL `GEOLOCATION` domain over the granule's per-pixel
longitude / latitude arrays rather than an affine geotransform, and they resample
nothing.
"""

from __future__ import annotations

from pyramids_eo.sensors.readers.fci import open_fci_l1c_chunk, read_fci
from pyramids_eo.sensors.readers.fci_l1c import available_channels, read_fci_l1c
from pyramids_eo.sensors.readers.harmonise import harmonise
from pyramids_eo.sensors.readers.olci import available_olci_channels, read_olci
from pyramids_eo.sensors.readers.seviri import parse_seviri_native, read_seviri
from pyramids_eo.sensors.readers.slstr import read_slstr

__all__ = [
    "available_channels",
    "available_olci_channels",
    "harmonise",
    "open_fci_l1c_chunk",
    "parse_seviri_native",
    "read_fci",
    "read_fci_l1c",
    "read_olci",
    "read_seviri",
    "read_slstr",
]
