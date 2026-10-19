"""Raster-reader drivers for the ``pyramids.raster_drivers`` plugin hook.

pyramids-gis 0.70 added a ``pyramids.raster_drivers`` entry-point group: a third-party package
registers a driver class there, and ``Dataset.read_file(<source>)`` routes a source GDAL cannot
open into that driver with no import coupling. This subpackage registers pyramids-eo's readers
that fit the single-source model:

- :class:`~pyramids_eo.drivers.earthengine.EarthEngineDriver` — ``ee://<asset-id>`` sources.
- :class:`~pyramids_eo.drivers.seviri.SeviriDriver` — SEVIRI native ``.nat`` granules.

The entry points in ``pyproject.toml`` are the whole integration; installing the package is
enough for ``read_file`` to find these. Each driver is a thin wrapper over an existing reader
(``from_earthengine`` / ``read_seviri``), which stays the ergonomic public API.

Not registered, by design: Sentinel-2 (GDAL's own ``SENTINEL2`` driver opens a ``.SAFE`` before
the plugin fallback runs) and FCI / FCI-L1C (both read a *set* of chunk files, which the
single-source ``read_file`` cannot express). Those stay facade-only readers. See
``planning/scope/raster-driver-plugin-plan.md``.
"""

from __future__ import annotations

from pyramids_eo.drivers.earthengine import EarthEngineDriver
from pyramids_eo.drivers.seviri import SeviriDriver

__all__ = ["EarthEngineDriver", "SeviriDriver"]
