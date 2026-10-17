# spectral

Spectral transforms for EO rasters — operations that rotate or combine a
sensor's reflectance bands into a derived, physically meaningful space, using the
per-sensor knowledge in the [registry](registry.md).

Currently the Kauth-Thomas **tasseled cap** (`tasseled_cap`): brightness /
greenness / wetness from a multi-band reflectance `Dataset`, with the
coefficients selected by `sensor` and the band order handled by name rather than
assumed — a named stack is reordered to the coefficient order, an unnamed one is
taken in the sensor's documented order, and a stack whose names are ambiguous or
do not match the sensor is rejected — so a mis-ordered stack never silently
produces a plausible-looking but meaningless raster. The per-sensor coefficient
tables themselves live in the registry (`get_tasseled_cap`,
`sensors/registry/data/tasseled_cap.yaml`).

::: pyramids_eo.spectral
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.spectral.tasseled_cap
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
