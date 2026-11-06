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
prefix (or ``driver=``) names one explicitly. Deciding on content has two consequences worth
stating, since neither is obvious from the claim set:

- A ``.SEN3`` holding **both** kinds of file is claimed by both drivers, and ``select_driver``
  takes the first claimant in registration order, which is alphabetical — so ``olci`` wins. The
  outcome is stable rather than arbitrary, and a scheme prefix or ``driver=`` overrides it.
- An **empty or incomplete** ``.SEN3`` is claimed by *neither*, so ``read_file`` surfaces GDAL's
  own "not recognized" error instead of the readers' more useful "the product looks incomplete or
  is not the expected Sentinel-3 level/type". Calling ``read_olci`` / ``read_slstr`` directly, or
  passing ``driver=``, reaches that better message.

A boolean reader keyword passed through ``read_file`` arrives as a *string*, because
``Dataset.read_file`` normalises ``open_options`` to GDAL's ``KEY=VALUE`` form. The drivers
translate those spellings (see :func:`~pyramids_eo.drivers._base.coerce_booleans`) rather than
letting the truthy string ``"false"`` switch an option on.

The entry points in ``pyproject.toml`` are the whole integration; installing the package is
enough for ``read_file`` to find these. Each driver is a thin wrapper over an existing reader
(``from_earthengine`` / ``read_seviri`` / ``from_sentinel2`` / ``read_fci_l1c`` / ``read_olci`` /
``read_slstr``), which stays the ergonomic public API.

Only a driver's ``read`` is reachable this way: ``Dataset.read_file`` never dispatches
``read_collection``, so :meth:`~pyramids_eo.drivers.earthengine.EarthEngineDriver.read_collection`
is declared and implemented but unreachable through the facade — a temporal stack goes through
``collection_from_earthengine``.

A single ``read_file`` source string can name a *set* of files — a directory, a glob, or an
``fci://`` path — which is how a multi-chunk FCI granule is addressed (see
:func:`~pyramids_eo.drivers._base.expand_sources`). The one reader reachable only through an explicit
``driver="s2"`` is Sentinel-2's *zipped* form: GDAL opens a ``.SAFE.zip`` itself and hands back a
bare container. A product **directory** needs no ``driver=`` — GDAL cannot open one, so it arrives
through the ``can_open`` fallback. Because ESA shares ``.SAFE`` across missions, ``can_open``
identifies the mission from the product's contents rather than its name, so a Sentinel-1 product
is left to GDAL instead of failing in the Sentinel-2 reader.
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
