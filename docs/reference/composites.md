# composites

Day/night compositing for EO imagery — the pyramids-eo `true_color_with_night_ir`
chain, over NumPy + pyramids-gis with no third-party compositing dependency.
Covers the solar and viewing geometry (`solar_zenith_angle`,
`cos_solar_zenith_angle`, `solar_zenith_azimuth`, `satellite_zenith_azimuth`,
`relative_azimuth`), the solar-zenith correction / reduction
(`sunz_correct` / `sunz_reduce`), a local Rayleigh atmospheric correction
(`rayleigh_correct`), the SZA cross-fade (`day_night_blend` / `day_weight`),
the alpha overlay (`alpha_overlay`), the static background image (`static_image`),
the true-colour composite (`true_color`), and the full assembly (`night_ir`,
`true_color_with_night_ir`).

The [display and resampling](#display-and-resampling) helpers at the bottom finish
the chain: `stretch` maps a physical composite onto a displayable range, and
`to_area` lands a georeferenced composite on an exact target grid.

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

::: pyramids_eo.composites.rayleigh
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

## Display and resampling

`stretch` (`pyramids_eo.enhance`) maps a composite's physical values — reflectance
in roughly `[0, 1]`, brightness temperature in kelvin — onto a display range and
dtype (uint8 by default), via a `"linear"`, `"crude"`, `"cira"`, or `"histogram"`
curve. `to_area` (`pyramids_eo.resample`) warps a georeferenced composite onto an
exact target grid (a CRS, an extent, and a pixel width/height) in a single GDAL
pass — the one step of the chain with no direct pyramids equivalent.

::: pyramids_eo.enhance
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.resample
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
