# composites

Day/night compositing for EO imagery — pyramids-eo's port of satpy's
`true_color_with_night_ir` chain, over NumPy + pyramids-gis with no satpy /
PyTroll dependency. Covers the solar geometry (`solar_zenith_angle`,
`cos_solar_zenith_angle`), the SZA cross-fade (`day_night_blend` / `day_weight`),
the alpha overlay (`alpha_overlay`), the static background image (`static_image`),
the true-colour composite (`true_color`), and the full assembly (`night_ir`,
`true_color_with_night_ir`).

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

::: pyramids_eo.composites.blend
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.composites.overlay
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.composites.background
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.composites.true_color
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.composites.true_color_night
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
