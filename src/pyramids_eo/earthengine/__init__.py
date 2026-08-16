"""Google Earth Engine reader for the pyramids stack.

pyramids-eo — the Earth-observation layer — turns an Earth Engine asset into a
pyramids ``Dataset`` without an ``earthengine-api`` dependency, using the bundled
GDAL ``EEDAI``/``EEDA`` drivers and Application Default Credentials. This mirrors
how :mod:`pyramids_eo.stac` reaches signed EO STAC assets: a provider-specific
integration that stays out of core pyramids but squarely in the EO layer
(serapeum-org/pyramids-eo#13).

* :class:`EarthEngineCredentials` — service-account JSON / ADC auth for the GDAL
  EE drivers.
* :func:`from_earthengine` — read a single EE ``Image`` asset (or a reduced
  ``ImageCollection`` composite) into a ``Dataset``.
* :func:`collection_from_earthengine` — read an EE ``ImageCollection`` over a date
  range into a ``DatasetCollection``.
"""

from __future__ import annotations

from pyramids_eo.earthengine.credentials import EarthEngineCredentials
from pyramids_eo.earthengine.reader import (
    collection_from_earthengine,
    from_earthengine,
)

__all__ = [
    "EarthEngineCredentials",
    "collection_from_earthengine",
    "from_earthengine",
]
