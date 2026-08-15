# stac

Provider-specific STAC asset signers for EO cloud providers. They implement
pyramids' `Signer` protocol, so they pass straight into
`pyramids.from_stac(signer=…)` / `pyramids.open_client(signer=…)`. See the
[STAC provider signers guide](../stac-signers.md) for per-provider usage.

## Signers

::: pyramids_eo.stac.signers
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

## CDSE S3 credentials

::: pyramids_eo.stac.auth_cdse
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
