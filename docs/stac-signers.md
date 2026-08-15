# STAC provider signers

pyramids owns the generic STAC `Signer` protocol and the generic `anonymous` /
`aws-requester-pays` / bearer signers. **pyramids-eo** — the Earth-observation
layer — owns the **provider-specific** signers that implement that same protocol
for real EO cloud providers, so signed EO STAC assets stream through GDAL
`/vsicurl/` (and `/vsis3/`) for any pyramids consumer, with **no provider SDK**
(pure standard-library `urllib`).

Every signer here satisfies `pyramids.stac.Signer`, so you wire it in the same
way regardless of provider:

```python
from pyramids import from_stac, open_client

# either entry point accepts any signer below
ds = from_stac(item, signer=signer)
client = open_client(endpoint_url, signer=signer)
```

The `build_signer(name, **creds)` factory maps a catalog `signer:` string to the
right object; it reuses the pyramids generic signers for the `anonymous` /
`aws-requester-pays` cases.

!!! note "Credentials stay out of code"
    Every provider reads its secrets from environment variables (shown per
    provider below); pass them explicitly only when you must. Nothing here
    hard-codes a token.

## Microsoft Planetary Computer — `PlanetaryComputerSigner`

Mints a per-`(account, container)` Shared Access Signature (SAS) token from the
PC token endpoint and appends it to the blob href — the same algorithm as
`planetary_computer.sign`, over `urllib`, so no `planetary-computer` SDK is
needed. The credential rides the URL, so `gdal_env()` is empty. Non-PC hrefs,
the public `ai4edatasetspublicassets` bucket, and already-signed URLs pass
through unchanged.

Environment: `PC_SDK_SAS_URL` (optional endpoint override),
`PC_SDK_SUBSCRIPTION_KEY` (optional, raises rate limits).

```python
from pyramids import from_stac
from pyramids_eo.stac import PlanetaryComputerSigner

signer = PlanetaryComputerSigner()          # or build_signer("mpc-sas")
ds = from_stac(pc_item, signer=signer)
```

## NASA Earthdata (EDL) — `EarthdataSigner`

Uses a pre-minted bearer token when given, otherwise mints one from the EDL
`find_or_create_token` endpoint with HTTP Basic auth. The token is sent as a
GDAL `Authorization: Bearer` header for `/vsicurl/` reads of EDL-gated DAAC
assets (auth is header-side, so `sign_href` is identity).

Environment: `EARTHDATA_TOKEN` / `EARTHDATA_PAT` (a pre-minted token), or
`EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD` (to mint one).

```python
from pyramids import open_client
from pyramids_eo.stac import EarthdataSigner

signer = EarthdataSigner()                  # reads the env vars above
# or explicitly: EarthdataSigner(token="edl-token")
client = open_client("https://cmr.earthdata.nasa.gov/stac/…", signer=signer)
```

## Copernicus Data Space (CDSE) — `CDSESigner` / `CdseS3Signer`

CDSE assets are reachable two ways, so there are two signers:

- **`CDSESigner`** — the HTTPS/OData path. Mints an access token from the CDSE
  Keycloak endpoint (password grant, then refresh grant) and sends it as a GDAL
  `Authorization: Bearer` header. Environment: `CDSE_USERNAME` + `CDSE_PASSWORD`.
- **`CdseS3Signer`** — the S3 path. CDSE assets also live on an S3-compatible
  store at `eodata.dataspace.copernicus.eu`; this signer rewrites `s3://eodata/…`
  (and the equivalent `https://` host) to the `/vsis3/eodata/…` GDAL path and
  supplies the S3 credentials through `gdal_env()`. Environment:
  `CDSE_S3_ACCESS_KEY` + `CDSE_S3_SECRET_KEY` (generate at the
  [CDSE S3 keys dashboard](https://eodata-s3keysmanager.dataspace.copernicus.eu)).

```python
from pyramids import from_stac
from pyramids_eo.stac import CDSESigner, CdseS3Signer

bearer = CDSESigner()                        # CDSE_USERNAME / CDSE_PASSWORD
s3 = CdseS3Signer(access_key="…", secret_key="…")  # or the env vars above
ds = from_stac(cdse_item, signer=s3)
```

## Brazil Data Cube (BDC) — `BdcTokenSigner`

Rewrites each asset href to carry an `?access_token=…` query parameter, needed
only by the token-gated BDC tiers (most published BDC collections read
anonymously). The credential rides the URL, so `gdal_env()` is empty; a missing
token raises `AuthenticationError` naming the env var.

Environment: `BDC_ACCESS_TOKEN`.

```python
from pyramids import from_stac
from pyramids_eo.stac import BdcTokenSigner

signer = BdcTokenSigner()                    # reads $BDC_ACCESS_TOKEN
# or explicitly: BdcTokenSigner(token="…")
ds = from_stac(bdc_item, signer=signer)
```

## Selecting a signer by name — `build_signer`

When the signer is chosen from catalog configuration, resolve it by its
`signer:` string:

```python
from pyramids_eo.stac import build_signer

build_signer("mpc-sas")                      # PlanetaryComputerSigner
build_signer("earthdata", token="…")         # EarthdataSigner
build_signer("cdse", username="…", password="…")
build_signer("cdse-s3", access_key="…", secret_key="…")
build_signer("bdc-token", token="…")
build_signer("anonymous", region="af-south-1")     # pyramids-eo _AnonymousS3Signer
build_signer("aws-requester-pays", region="us-west-2")  # pyramids generic signer
```

See the [`stac` API reference](reference/stac.md) for the full signature of each
signer.
