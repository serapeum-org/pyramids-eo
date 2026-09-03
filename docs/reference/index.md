# API Reference

The pyramids-eo public API. The top-level package (`pyramids_eo`) re-exports the Earth Engine and Sentinel-2
entry points; the sensor readers, registry, composites, and STAC signers live under their subpackages.

- [Earth Engine](earthengine.md) — read EE `Image` / `ImageCollection` assets into a `Dataset` /
  `DatasetCollection` (`from_earthengine`, `collection_from_earthengine`, cost estimation, credentials).
- [Sentinel](sentinel.md) — ESA Sentinel product readers (`open_product`, `from_sentinel2`, SCL masking).
- **Sensors** — the L1 data-access layer:
    - [Readers](readers.md) — instrument readers (MTG-FCI, MSG-SEVIRI).
    - [Registry](registry.md) — sensor metadata (band → wavelength, native resolution, fill, calibration).
- [Composites](composites.md) — day/night compositing (solar / viewing geometry, Rayleigh correction, blend),
  plus the `stretch` / `to_area` display and resampling helpers.
- [STAC signers](stac.md) — EO provider asset signers (PC / Earthdata / CDSE / BDC).
- [Errors](errors.md) — the `EOError` exception family.

::: pyramids_eo
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
