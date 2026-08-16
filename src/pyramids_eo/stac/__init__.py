"""EO provider STAC asset signers for the pyramids stack.

pyramids owns the generic `pyramids.stac.Signer` protocol and the generic
`anonymous` / `aws-requester-pays` / bearer signers. pyramids-eo — the
Earth-observation layer — owns the **provider-specific** signers that implement
that same protocol for real EO cloud providers, so signed EO STAC assets stream
through GDAL `/vsicurl/` (and `/vsis3/`) for any pyramids consumer, without a
provider SDK:

* `PlanetaryComputerSigner` — Microsoft Planetary Computer (SAS).
* `EarthdataSigner` — NASA Earthdata (EDL bearer token).
* `CDSESigner` / `CdseS3Signer` — Copernicus Data Space (bearer / S3).
* `BdcTokenSigner` — Brazil Data Cube (`?access_token=…`).

They satisfy pyramids' `Signer` protocol, so they drop into
`pyramids.from_stac(signer=…)` / `pyramids.open_client(signer=…)` unchanged.
`build_signer` maps a catalog `signer:` string to the right object: the
`anonymous` case is the pyramids-eo-local `_AnonymousS3Signer` (which adds
`AWS_NO_SIGN_REQUEST` and region pinning), while `aws-requester-pays` reuses the
pyramids generic signer.
"""

from __future__ import annotations

from pyramids_eo.stac.signers import (
    BdcTokenSigner,
    CdseS3Signer,
    CDSESigner,
    EarthdataSigner,
    PlanetaryComputerSigner,
    build_signer,
)

__all__ = [
    "BdcTokenSigner",
    "CDSESigner",
    "CdseS3Signer",
    "EarthdataSigner",
    "PlanetaryComputerSigner",
    "build_signer",
]
