"""Raster-reader drivers for the ``pyramids.raster_drivers`` plugin hook.

pyramids-core 0.70 added a ``pyramids.raster_drivers`` entry-point group: a third-party package
registers a driver class there, and ``Dataset.read_file(<source>)`` routes a source GDAL cannot
open into that driver with no import coupling. This subpackage registers pyramids-eo's readers:

- :class:`~pyramids_eo.drivers.earthengine.EarthEngineDriver` — ``ee://<asset-id>`` sources.
- :class:`~pyramids_eo.drivers.seviri.SeviriDriver` — SEVIRI native ``.nat`` granules.
- :class:`~pyramids_eo.drivers.sentinel2.Sentinel2Driver` — Sentinel-2 ``.SAFE`` products.
- :class:`~pyramids_eo.drivers.fci.FciDriver` — MTG-FCI L1C FDHSI granules (a chunk set named by a
  directory or glob).
- :class:`~pyramids_eo.drivers.olci.OlciDriver` — Sentinel-3 OLCI L1B ``.SEN3`` granules.
- :class:`~pyramids_eo.drivers.slstr.SlstrDriver` — Sentinel-3 SLSTR L1 RBT ``.SEN3`` granules.

OLCI and SLSTR products share the ``.SEN3`` directory suffix, so both drivers claim it and their
``can_open`` predicates disambiguate on content — an OLCI granule holds ``Oa*_radiance.nc`` band
files, an SLSTR one holds ``S*_radiance_*.nc`` / ``*_BT_*.nc``. An ``olci://`` or ``slstr://``
prefix (or ``driver=``) names one explicitly.

The entry points in ``pyproject.toml`` are the whole integration; installing the package is
enough for ``read_file`` to find these. Each driver is a thin wrapper over an existing reader
(``from_earthengine`` / ``read_seviri`` / ``from_sentinel2`` / ``read_fci_l1c``), which stays the
ergonomic public API.

A single ``read_file`` source string can name a *set* of files — a directory, a glob, or an
``fci://`` path — which is how a multi-chunk FCI granule is addressed (see
:func:`~pyramids_eo.drivers._base.expand_sources`). The one reader reachable only via an explicit
``driver="s2"`` is Sentinel-2: GDAL ships its own ``SENTINEL2`` driver and opens a ``.SAFE`` on the
plain ``read_file`` path before any plugin, so a bare-path auto-route waits on the upstream
suffix-pre-emption feature requested in serapeum-org/pyramids#1344.
"""

from __future__ import annotations

from pyramids_eo.drivers.earthengine import EarthEngineDriver
from pyramids_eo.drivers.fci import FciDriver
from pyramids_eo.drivers.olci import OlciDriver
from pyramids_eo.drivers.sentinel2 import Sentinel2Driver
from pyramids_eo.drivers.seviri import SeviriDriver
from pyramids_eo.drivers.slstr import SlstrDriver

__all__ = [
    "EarthEngineDriver",
    "FciDriver",
    "OlciDriver",
    "Sentinel2Driver",
    "SeviriDriver",
    "SlstrDriver",
]
