# Sensors — Readers

Instrument readers that decode L1 formats into calibrated, geolocated pyramids
`Dataset`s. For real MTG-FCI L1C FDHSI granules use `read_fci_l1c`: it decodes
the packed radiance from the nested `data/<channel>/measured` groups, reads the
per-granule calibration coefficients, and stitches the chunks by their
geostationary geotransform Y origin (validated on real MTI1/Meteosat-12
granules — see issue #40). Read several channels in one call with
`channels=[...]`, which returns a `dict[str, Dataset]` and reads each chunk's
structure and coefficients once for the whole set (e.g. the red/blue/near-IR
bands of a true-colour composite); `available_channels(chunks)` lists which
channels the chunk set carries (it also accepts a single chunk path).
`open_fci_l1c_chunk` is the lower-level radiance opener for use with the generic
`read_fci` (which also accepts `channels=[...]`).

For MSG-SEVIRI Level-1.5 native (`.nat`) granules use `read_seviri`: it decodes
a VIS/IR channel's 10-bit packed counts, applies the granule's per-channel
`Cal_Slope` / `Cal_Offset` and the registry calibration, and returns the scene
north-up on the 3 km geostationary grid (validated on a real Meteosat-10
granule — see issue #40). `parse_seviri_native` is the underlying `.nat` decoder,
and a custom `parse` callable can still be injected.

::: pyramids_eo.sensors.readers
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.readers.fci_l1c
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        members:
          - read_fci_l1c
          - available_channels

::: pyramids_eo.sensors.readers.fci
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.readers.seviri
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.readers.harmonise
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

## Raster-driver plugins

The EO readers are also registered as `pyramids.raster_drivers` plugins (the hook
pyramids-core 0.70 added), so `pyramids.dataset.Dataset.read_file(<source>)` can
route a source straight into pyramids-eo — no import of this package needed at the
call site. Installing pyramids-eo is the whole integration.

| driver name | source | reader |
| --- | --- | --- |
| `earthengine` | `ee://<asset-id>` | `from_earthengine` |
| `seviri` | a `.nat` granule | `read_seviri` |
| `s2` | a `.SAFE` product (or `.SAFE.zip`) | `from_sentinel2` |
| `fci` | an FCI L1C granule: a chunk **directory**, a **glob**, or an `fci://` path | `read_fci_l1c` |

```python
from pyramids.dataset import Dataset

# auto-routed: GDAL cannot open these, so read_file falls through to the plugin
nat = Dataset.read_file("MSG_SEVIRI.nat", open_options={"channel": "IR_108"})
ee = Dataset.read_file("ee://USGS/SRTMGL1_003", open_options={"bbox": (0, 0, 1, 1)})
fci = Dataset.read_file("/data/fci_granule", open_options={"channel": "ir_105"})
fci2 = Dataset.read_file("fci:///data/fci_granule", open_options={"channel": "ir_105"})

# Sentinel-2: GDAL opens a .SAFE itself, so ask for the pyramids-eo reader explicitly
s2 = Dataset.read_file("S2A_MSIL2A.SAFE", driver="s2", open_options={"bands": ["B04", "B03", "B02"]})
```

`read_file` forwards only `open_options` to a driver, so reader keywords (channel,
bbox, bands, credentials, …) are passed there; options needing a non-string type
(e.g. S2 `resolution=10`) must be given in dict form, not a `KEY=VALUE` list. A
bare `read_file("…SAFE")` without `driver="s2"` returns GDAL's container until the
upstream suffix-pre-emption feature (serapeum-org/pyramids#1344) lands.

::: pyramids_eo.drivers
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
