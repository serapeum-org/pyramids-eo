# Sensors — Readers

Instrument readers that decode L1 formats into calibrated, geolocated pyramids
`Dataset`s. For real MTG-FCI L1C FDHSI granules use `read_fci_l1c`: it decodes
the packed radiance from the nested `data/<channel>/measured` groups, reads the
per-granule calibration coefficients, and stitches the chunks by their
geostationary geotransform Y origin (validated on real MTI1/Meteosat-12
granules — see issue #40). `open_fci_l1c_chunk` is the lower-level radiance
opener for use with the generic `read_fci`.

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
