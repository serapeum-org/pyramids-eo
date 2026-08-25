"""Sentinel product readers built on GDAL's ``SENTINEL2`` / ``SAFE`` drivers.

The earth-observation reader family for ESA's Sentinel missions, layered on
pyramids-gis the same way :mod:`pyramids_eo.earthengine` layers on the EEDAI
driver: GDAL parses the product structure, pyramids does the raster ops, and
this package adds the instrument semantics — which subdataset is which band, how
to turn DN into reflectance, how to read the scene-classification mask.

Public surface:

* :func:`open_product` — open any supported Sentinel product into a typed model.
* :func:`from_sentinel2` — turnkey Sentinel-2 read → a pyramids ``Dataset``.
* :func:`scl_mask` / :class:`SclClass` — Level-2A cloud/shadow masking.

Sentinel-1 (``SAFE``) is planned as a later phase; :func:`open_product` raises a
clear error for it today.
"""

from __future__ import annotations

# isort: off
import pyramids as _pyramids_bootstrap  # noqa: F401  (activates the bundled osgeo)

# isort: on

from pyramids_eo.sentinel.product import SentinelProduct, open_product
from pyramids_eo.sentinel.s2 import (
    SclClass,
    collection_from_sentinel2,
    from_sentinel2,
    scl_mask,
)

__all__ = [
    "SclClass",
    "SentinelProduct",
    "collection_from_sentinel2",
    "from_sentinel2",
    "open_product",
    "scl_mask",
]
