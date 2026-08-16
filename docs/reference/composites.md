# composites

Day/night compositing for EO imagery — pyramids-eo's port of satpy's
`true_color_with_night_ir` chain, over NumPy + pyramids-gis with no satpy /
PyTroll dependency. Currently exposes the solar-geometry primitive
(`solar_zenith_angle`); the blend, alpha-overlay, and static-image primitives
land here as they are implemented.

::: pyramids_eo.composites
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.composites.geometry
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
