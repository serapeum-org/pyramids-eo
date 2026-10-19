"""Raster-reader drivers for the ``pyramids.raster_drivers`` plugin hook.

pyramids-core 0.70 added a ``pyramids.raster_drivers`` entry-point group: a third-party package
registers a driver class there, and ``Dataset.read_file(<source>)`` routes a source GDAL cannot
open into that driver with no import coupling. This subpackage registers pyramids-eo's readers:

- :class:`~pyramids_eo.drivers.earthengine.EarthEngineDriver` — ``ee://<asset-id>`` sources.
- :class:`~pyramids_eo.drivers.seviri.SeviriDriver` — SEVIRI native ``.nat`` granules.
- :class:`~pyramids_eo.drivers.sentinel2.Sentinel2Driver` — Sentinel-2 ``.SAFE`` products.
- :class:`~pyramids_eo.drivers.fci.FciDriver` — MTG-FCI L1C FDHSI granules (a chunk set named by a
  directory or glob).

The entry points in ``pyproject.toml`` are the whole integration; installing the package is
enough for ``read_file`` to find these. Each driver is a thin wrapper over an existing reader
(``from_earthengine`` / ``read_seviri`` / ``from_sentinel2`` / ``read_fci_l1c``), which stays the
ergonomic public API.

A single ``read_file`` source string can name a *set* of files — a directory or a glob — which is
how a multi-chunk FCI granule is addressed (see
:func:`~pyramids_eo.drivers._base.expand_sources`). The one reader reachable only via an explicit
``driver="s2"`` is Sentinel-2: GDAL ships its own ``SENTINEL2`` driver and opens a ``.SAFE`` on the
plain ``read_file`` path before any plugin, so a bare-path auto-route waits on
serapeum-org/pyramids#1344. See ``planning/scope/raster-driver-plugin-plan.md``.
"""

from __future__ import annotations

from pyramids_eo.drivers.earthengine import EarthEngineDriver
from pyramids_eo.drivers.fci import FciDriver
from pyramids_eo.drivers.sentinel2 import Sentinel2Driver
from pyramids_eo.drivers.seviri import SeviriDriver

__all__ = ["EarthEngineDriver", "FciDriver", "Sentinel2Driver", "SeviriDriver"]
