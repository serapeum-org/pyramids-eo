# Sensors — Readers

Instrument readers that decode L1 formats into calibrated, geolocated pyramids
`Dataset`s. For real MTG-FCI L1C FDHSI granules use `read_fci_l1c`: it decodes
the packed radiance from the nested `data/<channel>/measured` groups, reads the
per-granule calibration coefficients, and stitches the chunks by their
geostationary geotransform Y origin (validated on real MTI1/Meteosat-12
granules — see issue #40). Read several channels in one call with
`channels=[...]`, which returns a `dict[str, Dataset]` and reads each chunk's
structure and coefficients once for the whole set (e.g. the red/blue/near-IR
bands of a true-colour composite); `available_channels(chunks)` lists which
channels the chunk set carries (it also accepts a single chunk path).
`open_fci_l1c_chunk` is the lower-level radiance opener for use with the generic
`read_fci` (which also accepts `channels=[...]`).

For MSG-SEVIRI Level-1.5 native (`.nat`) granules use `read_seviri`: it decodes
a VIS/IR channel's 10-bit packed counts, applies the granule's per-channel
`Cal_Slope` / `Cal_Offset` and the registry calibration, and returns the scene
north-up on the 3 km geostationary grid (validated on a real Meteosat-10
granule — see issue #40). `parse_seviri_native` is the underlying `.nat` decoder,
and a custom `parse` callable can still be injected.

For Himawari-8/9 AHI Himawari Standard Data (`.DAT`) granules use `read_ahi`: it
walks the 11 self-describing HSD header blocks, decodes the band's 16-bit counts,
applies the granule's own count-to-radiance slope / intercept, masks the error
(65535) and outside-scan (65534) sentinel counts, and returns the band north-up
on its geostationary grid. Calibration prefers the coefficients the granule
carries over the registry's nominal table: a solar band (1-6) uses the granule's
`c'` as its band solar irradiance, and a thermal band (7-16) inverts Planck's law
with the granule's own `c` / `h` / `k` and applies its quadratic
`Tb = c0 + c1*Te + c2*Te^2` correction — which the registry's linear
`alpha` / `beta` form cannot express.

A granule names its own band, so `band` is optional and merely asserts the
expectation. A full disk ships as 10 segment files; pass the segment paths of one
band together and they are stitched by their block #7 first-line numbers, with a
gap or a duplicate reported rather than silently left blank. A **contiguous
subset** is accepted too and returns only the lines it covers, georeferenced
where they belong in the disk — so a partial download reads as the band of the
disk it actually covers; completeness is not required, only contiguity. A single
segment reads on its own the same way. `.DAT.bz2` granules are decompressed
transparently.

The segments must come from one acquisition. Line numbers alone do not establish
that, because every timeline numbers its segments identically — two scenes ten
minutes apart would otherwise stitch into a seamless-looking image of two
different moments. `read_ahi` therefore checks the satellite, observation time
and area, the declared segment count, and the projection block, and refuses a
mixed set.

```python
from pyramids_eo.sensors.readers import read_ahi

# one segment, calibrated to brightness temperature
bt = read_ahi("HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT")

# a stitched full disk from its 10 segments
import glob
disk = read_ahi(sorted(glob.glob("HS_H08_20150415_0230_B13_FLDK_R20_S*.DAT")))
```

`parse_ahi_hsd` is the underlying decoder, returning uncalibrated radiance, and
`parse_hsd_header` exposes the granule's grid, projection and calibration without
reading its pixels.

Because HSD carries no per-granule solar irradiance for the reflective bands in a
form the registry can hold, `ahi.yaml` deliberately omits `solar_irradiance`: a
solar calibration attempted without a granule raises `CalibrationError` rather
than quietly using a value from elsewhere. See the table's own header for the
provenance of every constant in it.

One unit detail matters if you supply your own thermal coefficients or hand
`read_ahi` a `Dataset` you built yourself. HSD defines radiance **per wavelength**
(`W / (m2 sr m)`), while the registry's inverse Planck takes it **per wavenumber**
(`mW / (m2 sr (cm-1)-1)`). `read_ahi` rebases between the two itself — by
`L * wavelength**2 * 1e5` — on every path that reaches the registry, so a
`Dataset` source and an explicit `coeffs=` both calibrate correctly. The two bases
differ by about four orders of magnitude at AHI's thermal wavelengths, so a
`Dataset` holding radiance in some *other* unit will not be rescued by this: pass
the granule, or convert before you call.

## Sentinel-3: swath readers

For Sentinel-3 `.SEN3` granules use `read_olci` (OLCI L1B EFR / ERR) and
`read_slstr` (SLSTR L1 RBT). Both decode through pyramids' bundled GDAL NetCDF
driver — there is no `netCDF4` dependency and nothing from the satpy stack — and
`available_olci_channels(granule)` lists the OLCI bands a granule carries.

These two differ from the geostationary readers above in three ways worth knowing:

- **They are swath instruments.** A pushbroom sensor sweeps a curved strip of
  ground, so no affine geotransform can describe where the pixels are; the product
  instead ships a longitude and a latitude for *every* pixel. The results therefore
  carry a GDAL **`GEOLOCATION` metadata domain** pointing at those arrays, a
  placeholder pixel/line geotransform, and no CRS.

    Sentinel-3 stores those arrays CF-packed as `int32` micro-degrees, and GDAL's
    geolocation transformer reads geolocation arrays *raw*, without applying
    `scale_factor`. The readers therefore point the domain at a small generated
    VRT that applies the packing, so the warp below receives real degrees rather
    than micro-degrees. The VRT is XML only — no geolocation array is read or
    copied until a warp pulls from it.

  Nothing is resampled — a caller
  who wants a map grid warps on demand:

    ```python
    from osgeo import gdal
    from pyramids_eo.sensors.readers import read_olci

    swath = read_olci("S3A_OL_1_EFR.SEN3", channel="Oa08")
    swath.geotransform          # the placeholder grid, not a geographic one
    gdal.Warp("oa08.tif", swath.raster, geoloc=True)   # user-driven reprojection
    ```

- **They decode only.** Radiometric calibration choices are out of scope for these
  readers (issue #136), so `read_olci` returns radiance and does not convert to
  reflectance; a caller converting it should use the granule's own per-detector
  `solar_flux` from `instrument_data.nc`. The SLSTR RBT product is already
  geophysical, so `S7`–`S9` / `F1`–`F2` arrive as brightness temperature in kelvin
  with no Planck conversion applied.

- **SLSTR is dual-view and multi-grid.** It images each scene twice, so
  `view="nadir"` / `"n"` or `"oblique"` / `"o"` selects one. `view` defaults to
  `None`, which is *not* a synonym for nadir: a channel name may spell the view
  itself (`"S8_BT_io"`), and a spelled view wins — nadir is the fallback only when
  neither the channel nor `view=` names one. Its channels sit on different
  resolutions — `S1`–`S6` on the 500 m `a` grid (with `S4`–`S6` also on the
  alternative 500 m `b`), `S7`–`S9` on the 1 km `i` grid, and of the fire channels
  **`F1` on `f` *or* `i` while `F2` is `i`-only** — they are asymmetric, and there
  is no `F2_BT_f*.nc`, so an `f`-less granule still reads `F1` from `i`. `grid=`
  restricts a read to one grid and never falls back to another, so an explicit
  `grid=` can't quietly hand back a different resolution.
  `available_slstr_channels(granule)` reports what a granule carries on the grids
  of **one view** per call — `view="nadir"` unless you ask for `"oblique"` — which
  is how to discover a non-nominal layout.
  Because the grids differ in shape, `channels=[...]` returns a
  `dict[str, Dataset]` of differently-sized arrays rather than a stack.

- **OLCI comes as EFR or ERR.** The two share every filename and variable name and
  differ only in pixel size (300 m against ~1200 m), so `read_olci` detects the
  product type from the granule name — falling back to `xfdumanifest.xml` — and
  reports `resolution_m` accordingly. A granule that identifies itself as neither
  carries no `resolution_m` at all, rather than a figure that could be wrong by a
  factor of four. (Nothing in this package reads that key today — `harmonise` takes
  an explicit reference grid — so it is provenance for a caller, not an input to
  the stack.)

A reader keyword passed through `read_file` as a `KEY=VALUE` list arrives as a
string, because `Dataset.read_file` normalises `open_options` to GDAL's form. The
drivers therefore translate the **string** spellings of a boolean
(`false`/`true`, `yes`/`no`, `on`/`off`, `1`/`0`, case-insensitively) and reject a
string they do not recognise, instead of treating the truthy `"false"` as
switching the option on. A value that is not a string reaches the reader
untouched, so the dict form with a real `False` is the unambiguous way to write
it. Note that a channel *set* is a direct-call feature:
`read_file` requires one `Dataset` back, so use `read_olci(..., channels=[...])`
when you want several at once.

Himawari AHI (HSD) remains unimplemented: its byte-level layout comes from JMA's
Himawari Standard Data User's Guide, which is needed before a correct decoder can
be written (see issue #136).

::: pyramids_eo.sensors.readers
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.readers.fci_l1c
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        members:
          - read_fci_l1c
          - available_channels

::: pyramids_eo.sensors.readers.fci
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.readers.seviri
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.readers.olci
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.readers.slstr
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

::: pyramids_eo.sensors.readers.harmonise
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source

## Raster-driver plugins

The EO readers are also registered as `pyramids.raster_drivers` plugins (the hook
pyramids-core 0.70 added), so `pyramids.dataset.Dataset.read_file(<source>)` can
route a source straight into pyramids-eo — no import of this package needed at the
call site. Installing pyramids-eo is the whole integration. The
[Reading EO sources via `read_file`](../read-file-plugin.md) guide covers the dispatch
rules, the `open_options` contract, and per-source recipes in full; the essentials:

| driver name | source | reader |
| --- | --- | --- |
| `earthengine` | `ee://<asset-id>` | `from_earthengine` |
| `seviri` | a `.nat` granule | `read_seviri` |
| `ahi` | an HSD granule named `HS_….DAT` (optionally `.bz2`) | `read_ahi` |
| `s2` | a `.SAFE` product (or `.SAFE.zip`) | `from_sentinel2` |
| `fci` | an FCI L1C granule: a chunk **directory**, a **glob**, or an `fci://` path | `read_fci_l1c` |
| `olci` | a Sentinel-3 OLCI `.SEN3` granule directory, or an `olci://` path | `read_olci` |
| `slstr` | a Sentinel-3 SLSTR `.SEN3` granule directory, or a `slstr://` path | `read_slstr` |

```python
from pyramids.dataset import Dataset

# auto-routed: GDAL cannot open these, so read_file falls through to the plugin
nat = Dataset.read_file("MSG_SEVIRI.nat", open_options={"channel": "IR_108"})
ee = Dataset.read_file("ee://USGS/SRTMGL1_003", open_options={"bbox": (0, 0, 1, 1)})
fci = Dataset.read_file("/data/fci_granule", open_options={"channel": "ir_105"})
fci2 = Dataset.read_file("fci:///data/fci_granule", open_options={"channel": "ir_105"})

# Himawari AHI: claimed by the HSD file-name convention, not by the generic .dat suffix,
# and the granule names its own band so open_options may be omitted entirely
ahi = Dataset.read_file("HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT")

# Sentinel-3: OLCI and SLSTR share the .SEN3 suffix, so the granule's contents decide
# which driver claims it (an olci:// / slstr:// prefix, or driver=, names one explicitly)
olci = Dataset.read_file("S3A_OL_1_EFR.SEN3", open_options={"channel": "Oa08"})
slstr = Dataset.read_file(
    "S3A_SL_1_RBT.SEN3", open_options={"channel": "S8", "view": "oblique"}
)

# Sentinel-2: GDAL opens a .SAFE itself, so ask for the pyramids-eo reader explicitly
s2 = Dataset.read_file("S2A_MSIL2A.SAFE", driver="s2", open_options={"bands": ["B04", "B03", "B02"]})
```

`read_file` forwards only `open_options` to a driver, so reader keywords (channel,
bbox, bands, credentials, …) are passed there; options needing a non-string type
(e.g. S2 `resolution=10`) must be given in dict form, not a `KEY=VALUE` list. A
bare `read_file("…SAFE")` without `driver="s2"` returns GDAL's container: pyramids-core
0.71 added driver pre-emption (serapeum-org/pyramids-core#1344), but it is opt-in and
this driver does not declare `preempts`, so GDAL still wins the bare path.

::: pyramids_eo.drivers
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
