# Earth Engine

The Google Earth Engine reader — turnkey entry points that pull an Earth Engine asset into a pyramids
`Dataset` / `DatasetCollection` with **no `earthengine-api` dependency**. Everything is built on the bundled
GDAL `EEDAI` (raster) and `EEDA` (catalog) drivers, with auth carried by
[`EarthEngineCredentials`](#pyramids_eo.earthengine.credentials.EarthEngineCredentials) over Application
Default Credentials.

- `from_earthengine` — read a single EE `Image` asset into a `Dataset`; or, given `start` / `end` + `reducer`,
  reduce an `ImageCollection` **client-side** to a single composite `Dataset` (server-side EE reducers are out of
  scope — compositing happens on the client over the fetched scene stack).
- `collection_from_earthengine` — read an `ImageCollection` over a date range into a `DatasetCollection`, one
  aligned `Dataset` per scene.
- `estimate_earthengine_cost` — estimate the read size (pixels / bytes / tiles) of a windowed request before
  fetching it, returning a `ReadCost`.
- `Window` — groups the AOI `bbox`, its `crs`, the output `scale` / `shape`, and the `resample` algorithm into
  the single `window` argument the readers take.
- `EarthEngineCredentials` — service-account JSON / ADC auth for the GDAL EE drivers.

## Readers

::: pyramids_eo.earthengine
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.earthengine.reader
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

## Credentials

::: pyramids_eo.earthengine.credentials
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
