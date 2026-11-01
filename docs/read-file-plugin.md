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
the whole integration** — there is nothing to import or configure. The four drivers it registers:

| driver name | source it opens | reader it delegates to |
| --- | --- | --- |
| `earthengine` | an `ee://<asset-id>` URL | [`from_earthengine`](reference/earthengine.md) |
| `seviri` | a SEVIRI native `.nat` granule | [`read_seviri`](reference/readers.md) |
| `s2` | a Sentinel-2 `.SAFE` product (or `.SAFE.zip`) | [`from_sentinel2`](reference/sentinel.md) |
| `fci` | an MTG-FCI L1C FDHSI granule (a chunk set) | [`read_fci_l1c`](reference/readers.md) |

Every reader stays pure pyramids-eo over pyramids' bundled GDAL — no `earthengine-api`, no extra GDAL install.

## How `read_file` picks a driver

`Dataset.read_file(path, *, driver=None, open_options=None, ...)` dispatches in a fixed order:

1. **`driver="<name>"` given** → that registered driver is used and **GDAL is bypassed entirely**.
2. **otherwise GDAL is tried first** → if GDAL can open the source, you get GDAL's result and no plugin runs.
3. **only if GDAL fails** → each registered driver's `can_open(source)` is asked, and the first to claim it reads it.

The consequence is the single rule worth remembering:

- `ee://…`, a `.nat` granule, and an FCI granule **directory** auto-route to pyramids-eo, because GDAL cannot open
  any of them (step 2 fails, step 3 claims them).
- a Sentinel-2 `.SAFE` does **not** auto-route: GDAL ships its own `SENTINEL2` driver and opens it at step 2, so a
  bare `read_file("…SAFE")` returns GDAL's raw *container* (0 bands + a `ContainerRasterWarning`). Ask for the
  pyramids-eo reader explicitly with `driver="s2"` (step 1). A bare-path auto-route for S2 is tracked upstream in
  [serapeum-org/pyramids#1344](https://github.com/serapeum-org/pyramids/issues/1344).

## Passing reader options

`read_file` forwards **only** its `open_options` to a driver — it has no `bbox` / `bands` / `crs` parameters of its
own. So every reader keyword travels inside `open_options`, which accepts either form:

```python
open_options={"bands": ["B04", "B03", "B02"], "resolution": 10}   # a dict — preferred
open_options=["bands=B04", "resolution=10"]                        # a GDAL-style KEY=VALUE list
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
        "credentials": "path/to/service-account.json",   # or omit to use Application Default Credentials
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

### Sentinel-2 — a `.SAFE` product (needs `driver="s2"`)

```python
from pyramids.dataset import Dataset

# Band selection + DN -> reflectance (the pyramids-eo reader, not GDAL's container):
scene = Dataset.read_file(
    "S2A_MSIL2A.SAFE",
    driver="s2",
    open_options={"bands": ["B04", "B08"], "resolution": 10},
)

# A zipped product is also claimed:
scene = Dataset.read_file("S2A_MSIL2A.SAFE.zip", driver="s2", open_options={"bands": ["B04"]})
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

# Several channels in one call -> a dict[str, Dataset]:
bands = Dataset.read_file("/data/granule", open_options={"channels": ["vis_06", "ir_105"]})
```

## Facade or `read_file` — which to use

| use the typed facade when… | use `read_file` when… |
| --- | --- |
| you are writing pyramids-eo code and want the full typed signature and IDE help | generic pyramids code opens a source by **string/URL** (a pipeline, config-driven ingest, a STAC-like flow) |
| you need the **temporal collection** readers (`collection_from_*`) | you want **zero import coupling** to pyramids-eo at the call site |
| you want every reader keyword as a named argument | one code path must open GeoTIFFs, NetCDFs **and** EO products uniformly |

The facades are the source of truth for the reader options; the driver simply maps `open_options` onto them.

## Troubleshooting

- **A bare `read_file("X.SAFE")` gave a 0-band dataset and a `ContainerRasterWarning`.** GDAL opened the `.SAFE`
  before the plugin. Pass `driver="s2"` to reach the pyramids-eo reader.
- **A Sentinel-2 `resolution`/`epsg` was ignored or mistyped.** You passed a `KEY=VALUE` list, which keeps values as
  strings. Pass `open_options` as a **dict** so the `int` survives.
- **`FciDriver` raised "no chunk files found for FCI source …".** Point the source at the granule **directory**, a
  glob of its `*FDHSI*.nc` chunks, or an `fci://` path — not a single non-chunk file.
- **An Earth Engine read needs credentials.** Pass `open_options={"credentials": …}` (a key file or an
  `EarthEngineCredentials`), or rely on Application Default Credentials. The `earthengine` driver declares the
  `needs_auth` capability.

## See also

- [Earth Engine](reference/earthengine.md), [Sentinel](reference/sentinel.md), and
  [Readers](reference/readers.md) — the typed facades and their full option sets.
- The runnable example notebooks for the [Earth Engine](examples/earth_engine_reader.ipynb),
  [Sentinel-2](examples/sentinel2_reader.ipynb), and [FCI / SEVIRI](examples/readers_fci_seviri_harmonise.ipynb)
  readers.
