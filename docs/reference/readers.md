# Sensors — Readers

Instrument readers that decode L1 formats into calibrated, geolocated pyramids
`Dataset`s. For real MTG-FCI L1C FDHSI granules use `read_fci_l1c`: it decodes
the packed radiance from the nested `data/<channel>/measured` groups, reads the
per-granule calibration coefficients, and stitches the chunks by their
geostationary geotransform Y origin (validated on real MTI1/Meteosat-12
granules — see issue #40). Read several channels in one call with
`channels=[...]`, which returns a `dict[str, Dataset]` and opens each chunk once
for the whole set (e.g. the red/blue/near-IR bands of a true-colour composite);
`available_channels(chunks)` lists which channels the chunk set carries.
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
