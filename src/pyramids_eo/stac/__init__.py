"""EO provider STAC asset signers for the pyramids stack.

pyramids owns the generic :class:`~pyramids.stac.Signer` protocol and the
generic ``anonymous`` / ``aws-requester-pays`` / bearer signers. pyramids-eo —
the Earth-observation layer — owns the **provider-specific** signers that
implement that same protocol for real EO cloud providers, so signed EO STAC
assets stream through GDAL ``/vsicurl/`` (and ``/vsis3/``) for any pyramids
consumer, without a provider SDK:

* :class:`PlanetaryComputerSigner` — Microsoft Planetary Computer (SAS).
* :class:`EarthdataSigner` — NASA Earthdata (EDL bearer token).
* :class:`CDSESigner` / :class:`CdseS3Signer` — Copernicus Data Space (bearer / S3).
* :class:`BdcTokenSigner` — Brazil Data Cube (``?access_token=…``).

They satisfy pyramids' ``Signer`` protocol, so they drop into
``pyramids.from_stac(signer=…)`` / ``pyramids.open_client(signer=…)`` unchanged.
:func:`build_signer` maps a catalog ``signer:`` string to the right object,
reusing the pyramids generic signers for the ``anonymous`` /
``aws-requester-pays`` cases.
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
