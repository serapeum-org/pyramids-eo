# Sentinel

The reader family for ESA's Sentinel missions, built on GDAL's `SENTINEL2` / `SAFE` drivers: GDAL parses the
product structure, pyramids does the raster ops, and this package adds the instrument semantics — which
subdataset is which band, how to turn DN into reflectance, how to read the scene-classification mask.

A Sentinel product is a *catalog of rasters*, not a raster, so it gets its own small model:

- `open_product` — sniff the GDAL driver behind a path (`.SAFE` / `MTD_*.xml` / `.zip`) and return the right
  typed `SentinelProduct` (currently `S2Product`; Sentinel-1 `SAFE` is a planned later phase and raises a clear
  error today).
- `from_sentinel2` — turnkey Sentinel-2 read into a pyramids `Dataset`: band selection, native-resolution
  choice, optional crop / reprojection, DN → reflectance tagging, and Level-2A SCL masking.
- `collection_from_sentinel2` — read several bands / products into a `DatasetCollection`.
- `scl_mask` / `SclClass` — Level-2A cloud / shadow masking via the scene-classification layer.

## Product model

::: pyramids_eo.sentinel
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sentinel.product
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

## Sentinel-2

::: pyramids_eo.sentinel.s2.reader
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sentinel.s2.product
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sentinel.s2.masks
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sentinel.s2.scaling
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
