# Sensors — Registry

Sensor / channel metadata and radiometric calibration for EO instruments
(MTG-FCI, MSG-SEVIRI). The channel tables live in `sensors/registry/data/*.yaml`;
the calibration functions turn raw L1 radiances into reflectance / brightness
temperature. The registry also holds the per-sensor **tasseled-cap** coefficient
tables (`get_tasseled_cap`, `sensors/registry/data/tasseled_cap.yaml`), consumed
by [`pyramids_eo.spectral.tasseled_cap`](spectral.md).

::: pyramids_eo.sensors.registry
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.registry.tasseled_cap
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.registry.sensors
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.registry.calibration
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
