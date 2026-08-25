"""Sentinel-2 MSI support — product model, reflectance scaling, SCL masking."""

from __future__ import annotations

from pyramids_eo.sentinel.s2.masks import SclClass, scl_mask
from pyramids_eo.sentinel.s2.product import S2Level, S2Product, S2Subdataset
from pyramids_eo.sentinel.s2.reader import from_sentinel2

__all__ = [
    "S2Level",
    "S2Product",
    "S2Subdataset",
    "SclClass",
    "from_sentinel2",
    "scl_mask",
]
