# Sensors — Registry

Sensor / channel metadata and radiometric calibration for EO instruments
(MTG-FCI, MSG-SEVIRI). The channel tables live in `sensors/registry/data/*.yaml`;
the calibration functions turn raw L1 radiances into reflectance / brightness
temperature.

::: pyramids_eo.sensors.registry
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
