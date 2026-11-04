# Reading EO sources through `Dataset.read_file`

pyramids-eo registers its readers as [`pyramids.raster_drivers`](#what-the-plugin-is) plugins, so you can open an
Earth-observation source **two ways**:

- the **typed facade** — `pyramids_eo.from_earthengine(...)`, `read_seviri(...)`, `from_sentinel2(...)`,
  `read_fci_l1c(...)` — the ergonomic, fully-typed entry points; or
- the **pyramids facade** — `pyramids.dataset.Dataset.read_file(<source>)` — which *routes* the source into the
  right pyramids-eo reader with **no `pyramids_eo` import at the call site**.

Both return the same object (a plain pyramids `Dataset` / `DatasetCollection`); which you reach for is a matter of
context, not capability. This guide covers the `read_file` route — how it finds a driver, how you pass reader
options through it, and the per-source recipes.

## What the plugin is

pyramids-core 0.70 added an entry-point group, `pyramids.raster_drivers`. A package registers a *driver* class
there; pyramids discovers it lazily the first time `read_file` needs a non-GDAL reader. **Installing pyramids-eo is
the whole integration** — there is nothing to import or configure. The six drivers it registers:

| driver name   | source it opens                               | reader it delegates to                         |
|---------------|-----------------------------------------------|------------------------------------------------|
| `earthengine` | an `ee://<asset-id>` URL                      | [`from_earthengine`](reference/earthengine.md) |
| `seviri`      | a SEVIRI native `.nat` granule                | [`read_seviri`](reference/readers.md)          |
| `s2`          | a Sentinel-2 `.SAFE` product (or `.SAFE.zip`) | [`from_sentinel2`](reference/sentinel.md)      |
| `fci`         | an MTG-FCI L1C FDHSI granule (a chunk set)    | [`read_fci_l1c`](reference/readers.md)         |
| `olci`        | a Sentinel-3 OLCI L1B `.SEN3` granule         | [`read_olci`](reference/readers.md)            |
| `slstr`       | a Sentinel-3 SLSTR L1 RBT `.SEN3` granule     | [`read_slstr`](reference/readers.md)           |

Every reader stays pure pyramids-eo over pyramids' bundled GDAL — no `earthengine-api`, no extra GDAL install.

## How `read_file` picks a driver

`Dataset.read_file(path, *, driver=None, open_options=None, ...)` dispatches in a fixed order:

1. **`driver="<name>"` given** → that registered driver is used and **GDAL is bypassed entirely**.
2. **a driver that *pre-empts* GDAL for this source** → one that lists the source's suffix or scheme in its
   `preempts` *and* confirms it via `can_open` is tried **before** GDAL. This is opt-in; of the six drivers only
   `s2` declares it, for `.zip`. An explicit `vsi=` outranks it.
3. **otherwise GDAL is tried** → if GDAL can open the source, you get GDAL's result and no plugin runs.
4. **only if GDAL fails** → each registered driver's `can_open(source)` is asked, and the first to claim it reads it.

The consequence is the single rule worth remembering:

- `ee://…`, a `.nat` granule, and the granule **directories** (FCI's chunk set, a Sentinel-3 `.SEN3`) auto-route
  to pyramids-eo, because GDAL cannot open any of them (step 3 fails, step 4 claims them).
- a Sentinel-2 `.SAFE` **product directory** auto-routes the same way: GDAL cannot open a product directory
  either, so it fails at step 3 and the driver claims it at step 4. `.safe` is deliberately *not* pre-empted —
  ESA shares the suffix with Sentinel-1, whose products GDAL reads itself, so pre-empting it would route those
  into the Sentinel-2 reader.
- a **zipped** product (`.SAFE.zip`) is the one form GDAL *does* open, handing back a bare *container* (0 bands
  plus a `ContainerRasterWarning`) and silently ignoring your `open_options`. The `s2` driver therefore declares
  `preempts = (".zip",)` and wins at **step 2**. An ordinary `.zip` is unaffected: `can_open` gates every
  pre-empted claim, so only a `.SAFE.zip` is taken.

!!! note "What pre-emption does and does not cover"

    Pre-emption shipped in pyramids-core 0.71
    ([serapeum-org/pyramids-core#1344](https://github.com/serapeum-org/pyramids-core/issues/1344)), below this
    package's floor, so every supported version has it. It is opt-in per driver and deliberately narrow here:
    only `s2` declares it, and only for `.zip`, the single form GDAL would otherwise claim. Everything else
    pyramids-eo reads — `ee://`, `.nat`, FCI chunk sets, `.SEN3`, and a `.SAFE` directory — is something GDAL
    cannot open at all, so it already arrives at step 4 and pre-emption would add nothing. An explicit `vsi=`
    skips the pre-empt scan entirely, since that is a deliberate "open member N of this archive" intent.

## Passing reader options

`read_file` forwards **only** its `open_options` to a driver — it has no `bbox` / `bands` / `crs` parameters of its
own. So every reader keyword travels inside `open_options`, which accepts either form:

```python
open_options = {"bands": ["B04", "B03", "B02"], "resolution": 10}  # a dict — preferred
open_options = ["bands=B04", "resolution=10"]  # a GDAL-style KEY=VALUE list
```

A **dict preserves Python types**; a `KEY=VALUE` list makes every value a **string**. So an option the reader needs
typed — Sentinel-2 `resolution` / `epsg` as `int`, a `bbox` tuple, a credentials object — must go in the dict form.
The drivers pull the window/band/credential keys out of `open_options` and pass the rest straight to the reader.

## By source

### Earth Engine — `ee://<asset-id>`

```python
from pyramids.dataset import Dataset


# A single EE image, windowed to an AOI:
srtm = Dataset.read_file(
    "ee://USGS/SRTMGL1_003",
    open_options={"bbox": (86.9, 27.9, 87.0, 28.0), "crs": 4326, "bands": ["elevation"]},
)

# A reduced composite over a date range (from_earthengine reduces the collection to one Dataset):
composite = Dataset.read_file(
    "ee://COPERNICUS/S2_SR_HARMONIZED",
    open_options={
        "bbox": (86.9, 27.9, 87.0, 28.0),
        "bands": ["B4", "B3", "B2"],
        "start": "2023-01-01",
        "end": "2023-03-01",
        "reducer": "median",
        "credentials": "path/to/service-account.json",  # or omit to use Application Default Credentials
    },
)
```

A **temporal** `DatasetCollection` (one timestep per scene, unreduced) is **not** reachable through `read_file` —
`read_file` dispatches a driver's `read` only, never `read_collection`. Use the typed
[`collection_from_earthengine`](reference/earthengine.md) for that.

### SEVIRI — a `.nat` granule

`read_seviri` reads **one channel**, so name it in `open_options`:

```python
from pyramids.dataset import Dataset


ir = Dataset.read_file("MSG4-SEVIRI-MSG15.nat", open_options={"channel": "IR_108"})
```

The read covers the whole granule, so `bbox` / `crs` do not apply.

### Sentinel-2 — a `.SAFE` product

```python
from pyramids.dataset import Dataset


# A product directory auto-routes - GDAL cannot open one, so the driver claims it
scene = Dataset.read_file(
    "S2A_MSIL2A.SAFE",
    open_options={"bands": ["B04", "B08"], "resolution": 10},
)

# A zipped product auto-routes too - the driver pre-empts GDAL for `.zip`
scene = Dataset.read_file("S2A_MSIL2A.SAFE.zip", open_options={"bands": ["B04"]})

# driver="s2" still works and short-circuits ahead of both - useful to be explicit
scene = Dataset.read_file(
    "S2A_MSIL2A.SAFE",
    driver="s2",
    open_options={"bands": ["B04", "B08"], "resolution": 10},
)
```

Scene-classification masking and the other `from_sentinel2` keywords (`reflectance`, `mask_scl`, `epsg`,
`resample`) go in `open_options` too.

### FCI — an L1C FDHSI granule (a chunk set)

A real FCI L1C granule is a *set* of NetCDF chunk files, so the source string names the **set** — a directory
holding the chunks, a glob matching them, or an `fci://` path — and the driver stitches them:

```python
from pyramids.dataset import Dataset


# A granule directory (GDAL cannot open a directory, so this auto-routes):
ir = Dataset.read_file("/data/MTI1_FCI_L1C_granule", open_options={"channel": "ir_105"})

# A glob of the chunks, or an explicit fci:// path:
ir = Dataset.read_file("/data/granule/*FDHSI*.nc", open_options={"channel": "ir_105"})
ir = Dataset.read_file("fci:///data/MTI1_FCI_L1C_granule", open_options={"channel": "ir_105"})

# A channel set is NOT available here - see the note below:
# Dataset.read_file("/data/granule", open_options={"channels": ["vis_06", "ir_105"]})  # TypeError
```

A channel **set** is a direct-call feature only. `read_file` requires a driver to hand back exactly one
`Dataset`, so `open_options={"channels": [...]}` raises `TypeError` from pyramids' plugin layer. Call
`read_fci_l1c(..., channels=[...])` when you want several channels at once.

### Sentinel-3 OLCI and SLSTR — a `.SEN3` granule

Both Sentinel-3 readers take a `.SEN3` granule **directory**, so both auto-route (GDAL cannot open a directory).
They are the one pair that **share a suffix**: `olci` and `slstr` both declare `.sen3`. Each one's
`can_open` inspects the granule's contents to decide whether to claim it — OLCI looks for
`Oa*_radiance.nc`, SLSTR for `S*_radiance_*.nc`, `S*_BT_*.nc` or `F*_BT_*.nc`. If a granule holds
**both** kinds, both claim it, and the tiebreak is the order pyramids discovers the entry points in:
`importlib.metadata` hands them over sorted by name, so `olci` precedes `slstr` and wins, whatever
order they appear in `pyproject.toml`. If it holds neither, nothing claims it and you get whatever
GDAL raises for that path (for a directory, typically a permission error) instead of the readers'
clearer "the product looks incomplete" message. Use the `olci://` / `slstr://` scheme, or `driver=`,
to name one explicitly and skip the guesswork.

Both read **one channel at a time** and the channel is required. Note that `read_file` forwards only
`open_options`, never a `bands` argument, so the channel must travel inside `open_options`:

```python
from pyramids.dataset import Dataset


# OLCI L1B (EFR or ERR) — the channel is required:
oa08 = Dataset.read_file("S3A_OL_1_EFR.SEN3", open_options={"channel": "Oa08"})

# Name the reader explicitly, by scheme or by driver:
oa08 = Dataset.read_file("olci:///data/S3A_OL_1_EFR.SEN3", open_options={"channel": "Oa08"})
oa08 = Dataset.read_file("S3A_OL_1_EFR.SEN3", driver="olci", open_options={"channel": "Oa08"})

# SLSTR L1 RBT — dual-view and multi-grid, so `view` and `grid` select which:
s8 = Dataset.read_file("S3A_SL_1_RBT.SEN3", open_options={"channel": "S8", "view": "oblique"})
s5 = Dataset.read_file("S3A_SL_1_RBT.SEN3", open_options={"channel": "S5", "grid": "a"})
```

`view` takes `"nadir"` / `"n"` or `"oblique"` / `"o"` (case-insensitive) and defaults to `None`. `None` does
**not** simply mean nadir: a channel name may spell the view itself (`"S8_BT_io"` ends in `io`, the
oblique `i` grid), and a spelled view wins. Nadir is only the fallback when neither the channel nor
`view=` names one, so `read_file("…SEN3", open_options={"channel": "S8_BT_io"})` reads **oblique** with
no `view=` at all. SLSTR's channels sit on different grids — `S1`–`S6` on the
500 m `a` grid (`S4`–`S6` also on the alternative `b`), `S7`–`S9` on the 1 km `i` grid, and of the fire channels
`F1` on `f` *or* `i` while `F2` is `i`-only — so an `f`-less granule still reads `F1`, from `i`. A
`grid=` restricts the read to that grid and never falls back to another, so it cannot quietly hand back
a different resolution.

These are **swath** readers, so the result carries a GDAL `GEOLOCATION` metadata domain, a placeholder
pixel/line geotransform and no CRS — nothing is resampled. Warp it yourself when you want a map grid:

```python
from osgeo import gdal

swath = Dataset.read_file("S3A_OL_1_EFR.SEN3", open_options={"channel": "Oa08"})
gdal.Warp("oa08.tif", swath.raster, geoloc=True)
```

Pass `open_options={"geolocation": False}` to skip attaching the geolocation arrays: the `GEOLOCATION`
domain is still present but empty, and the placeholder geotransform is unchanged, so the result stays
on its pixel grid. Because a `KEY=VALUE` list stringifies values, the two drivers translate the **string** spellings
(`"false"`, `"no"`, `"off"`, `"0"`, and their positive counterparts, case-insensitively) and reject a string they
do not recognise — so the truthy `"false"` cannot silently switch the option on. A value that is not a string is
passed through to the reader untouched, which is what makes the dict form with a real `False` the unambiguous
way to write it.

As with FCI, a channel **set** is a direct-call feature only — `open_options={"channels": [...]}` raises
`TypeError` from pyramids' plugin layer, so call `read_olci(..., channels=[...])` /
`read_slstr(..., channels=[...])` for several channels at once. `available_olci_channels` reports what a granule
carries; `available_slstr_channels` does the same for **one view** per call, taking `view="nadir"` by
default, so check the oblique side with `view="oblique"`.

## Facade or `read_file` — which to use

| use the typed facade when…                                                      | use `read_file` when…                                                                                       |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| you are writing pyramids-eo code and want the full typed signature and IDE help | generic pyramids code opens a source by **string/URL** (a pipeline, config-driven ingest, a STAC-like flow) |
| you need the **temporal collection** readers (`collection_from_*`)              | you want **zero import coupling** to pyramids-eo at the call site                                           |
| you want every reader keyword as a named argument                               | one code path must open GeoTIFFs, NetCDFs **and** EO products uniformly                                     |

The facades are the source of truth for the reader options; the driver simply maps `open_options` onto them.

## Troubleshooting

- **A `.SAFE.zip` gave a 0-band dataset and a `ContainerRasterWarning`, and ignored `open_options`.** GDAL
  opened the archive before the plugin, which means pyramids-eo was not discovered — pre-emption can only use a
  driver pyramids found. Check the install, or pass `driver="s2"` to force it.
- **A Sentinel-2 `resolution`/`epsg` was ignored or mistyped.** You passed a `KEY=VALUE` list, which keeps values as
  strings. Pass `open_options` as a **dict** so the `int` survives.
- **`FciDriver` raised "no chunk files found for FCI source …".** Point the source at the granule **directory**, a
  glob of its `*FDHSI*.nc` chunks, or an `fci://` path — not a single non-chunk file.
- **A `.SEN3` read raised "pass either a single channel or a channels list, not both".** `open_options` carried
  both keys; drop one. This is unrelated to which instrument claimed the granule, and `driver=` will not change it.
- **A `.SEN3` read returned the wrong instrument.** OLCI and SLSTR share the `.SEN3` suffix, and a granule holding
  both kinds of measurement file is claimed by whichever driver registers first (`olci`). Name the one you want
  with `driver="olci"` / `driver="slstr"`, or an `olci://` / `slstr://` path.
- **A `.SEN3` read complained that no channel was given.** `read_file` forwards only `open_options`, so the
  channel cannot come from a `bands` argument — pass `open_options={"channel": "Oa08"}`.
- **`open_options={"channels": [...]}` raised `TypeError`.** A channel set is direct-call only — `read_file`
  requires one `Dataset` back. Use the reader (`read_fci_l1c` / `read_olci` / `read_slstr`) with `channels=[...]`.
- **An Earth Engine read needs credentials.** Pass `open_options={"credentials": …}` (a key file or an
  `EarthEngineCredentials`), or rely on Application Default Credentials. The `earthengine` driver declares the
  `needs_auth` capability.

## See also

- [Earth Engine](reference/earthengine.md), [Sentinel](reference/sentinel.md), and
  [Readers](reference/readers.md) — the typed facades and their full option sets.
- The runnable example notebooks for the [Earth Engine](examples/earth_engine_reader.ipynb),
  [Sentinel-2](examples/sentinel2_reader.ipynb), and [FCI / SEVIRI](examples/readers_fci_seviri_harmonise.ipynb)
  readers.
