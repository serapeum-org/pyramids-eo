# Real AHI granules for the `live` tests

The offline AHI suite builds its fixtures from the published Himawari Standard Data
specification. That proves the decoder is self-consistent with the document; it cannot prove the
document was read correctly. Three defects on this reader got through a green suite for exactly
that reason — in each case the fixture encoded the same misreading as the reader, so the two
agreed:

| What was misread | What a real granule carries | What the reader produced |
|---|---|---|
| block #5 central wavelength as metres | `10.4074` **micrometres** | a 280 K scene as `-1.8e44` K |
| block #10's length field as `I2` | an `I4` | big-endian granules failed the walk |
| block #1 satellite name as `H08` | `Himawari-9` | nothing — but the reality-check assertion itself encoded `H08`, so it would have failed on the first real granule |

A fourth would have rejected real data outright: the reader once refused a negative
count-to-radiance slope as "non-physical". **Real Himawari-9 band-13 granules declare
`slope = -0.0037525`.**

So the `live` tests are not a nicety. Point them at one real segment and they settle in a second
what the offline suite cannot settle at all.

## Getting a granule

Himawari HSD is in NOAA's Big Data Program bucket, public and **no credentials required**. JAXA's
own P-Tree service also serves it but needs registration; the bucket does not.

One band-13 full-disk segment is about 3 MB compressed, which is all these tests need:

```bash
curl -O "https://noaa-himawari9.s3.us-east-1.amazonaws.com/AHI-L1b-FLDK/2026/01/01/0000/HS_H09_20260101_0000_B13_FLDK_R20_S0510.DAT.bz2"
```

Note the **regional** endpoint host (`s3.us-east-1.amazonaws.com`). The virtual-hosted form
`noaa-himawari9.s3.amazonaws.com` is equivalent for AWS but is blocked by some corporate egress
filters, so the regional spelling is the more portable one to document.

Object layout in the bucket:

```
AHI-L1b-FLDK/<YYYY>/<MM>/<DD>/<hhmm>/HS_H09_<YYYYMMDD>_<hhmm>_B<bb>_FLDK_R<jj>_S<kk><ll>.DAT.bz2
```

`kk` is the segment number and `ll` the total (`10` for a full disk), so `S0510` is segment 5 of
10 — a mid-disk band, which is mostly Earth and therefore the most informative single segment.
`AHI-L1b-Japan/` and `AHI-L1b-Target/` hold the regional scans. Himawari-8 data (through 2022)
is in `noaa-himawari8`.

## Running the tests

The `live` **marker** decides whether these run; the environment variable only says where to look
(this repository gates test categories with markers, never with environment variables):

```bash
AHI_FIXTURES_DIR=/path/to/granules pytest tests/test_readers_ahi.py -m live -v
```

Dropping granules into this directory also works — it is the default location, and `.DAT` /
`.DAT.bz2` files here are git-ignored. Without a granule the tests skip, so an ordinary run is
unaffected.

## What a real granule measures, and what it does not

Verified against `HS_H09_20260101_0000_B13_FLDK_R20_S0510.DAT.bz2`:

```
satellite            'Himawari-9'
central wavelength   10.4074          (micrometres)
slope / intercept    -0.0037525 / 15.1977
valid_bits           12
columns x lines      5500 x 550       segment 5 of 10, first line 2201
grid step            2000.000 m x 2000.000 m
Tb p2 / p50 / p100   201.33 / 286.64 / 298.71 K      over 99.6% finite pixels
```

**One characteristic of the product worth knowing before you read a failure.** A tail of pixels
reads far below any physical scene temperature. Two things combine to produce it: off-limb space
beyond the Earth's edge is not always flagged with the outside-scan sentinel, and the granule's
affine calibration crosses zero radiance at count 4050 while its valid bit depth admits counts to
4095, so those high counts map to near-zero or negative radiance (the negative ones the reader
masks to NaN). Count 4050 is the `p0` every segment reports: it decodes to 69.9983 K, and 4051
and above go negative and so to NaN.

Measured across three adjacent segments of one full disk:

| segment | p0 | p2 | p50 | p100 | below 180 K | within 180–330 K |
|---|---|---|---|---|---|---|
| 4 | 70.0 | 117.4 | 291.2 | 302.9 | 4.46% | 95.54% |
| 5 | 70.0 | 201.3 | 286.6 | 298.7 | 1.29% | 98.71% |
| 6 | 70.0 | 199.8 | 279.8 | 299.7 | 1.28% | 98.72% |

The gradient is the explanation and the reassurance: segment 4 lies further from the disk centre,
so more of its row band is space. The tail therefore tracks *where the segment sits*, not the
decode. Masking it would be a radiometric calibration choice, which issue #136 places outside this
reader's scope, so the reader passes it through.

That is why the live assertions are written against the **median** and the bulk fraction rather
than the extremes. The failures that have actually occurred on this reader were unit and scale
errors, and those move the median by sixteen orders of magnitude — it is the discriminator that
matters, while the minimum only measures how much space the segment happens to contain.

## What a real granule does not settle

It confirms the decode, the units, the grid and the calibration chain for the band and segments
tested. It does not confirm the other fifteen bands, the 0.5 km and 1 km grids, the Japan and
Target area scans, a big-endian granule (every operational granule observed is little-endian), or
a granule carrying error rows in block #10. Those remain covered only by the offline fixtures.
