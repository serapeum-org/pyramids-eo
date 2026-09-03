# pyramids-eo

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**pyramids-eo** is the **Earth-observation layer** of the pyramids GIS stack. It is built on top of
[pyramids](https://github.com/serapeum-org/pyramids) (`pyramids-gis` — the generic raster/vector engine) and
adds the logic that is specific to **Earth-observation data**.

pyramids underneath knows only how to *move rasters around*. pyramids-eo knows what a pixel *means* for a given
instrument or provider: which subdataset is which channel, how to calibrate it, how to composite it, how to
resample a swath — and how to reach signed EO cloud assets.

```
pyramids-eo        (Earth-observation logic: sensor readers, calibration, composites,
    │               resample, masking, signed EO provider/STAC access)
    │  depends on
    ▼
pyramids           (generic raster / vector engine)
```

## Scope

pyramids-eo is scoped by *domain* — Earth-observation data — not by a restriction on what it may do. It covers
the EO-specific functionality that sits above the generic raster/vector operations pyramids-gis provides:
reading instrument L1/L2 formats, calibrating them, compositing them, resampling swaths, and reaching signed EO
cloud assets (Google Earth Engine, STAC providers). Everything a reader returns is a plain pyramids
`Dataset` / `DatasetCollection` — no engine-specific objects leak into the public types.

## Quickstart

Everything below uses the shipped public API. See the [Reference](reference/index.md) for the full surface.

### Google Earth Engine

Pull an Earth Engine asset into a pyramids `Dataset` with no `earthengine-api` dependency (the bundled GDAL
`EEDAI` / `EEDA` drivers plus Application Default Credentials — see [Earth Engine](reference/earthengine.md)).

```python
import pyramids_eo as eo
from pyramids_eo import Window

# A single EE Image, windowed to an AOI at a 0.01° pixel size.
srtm = eo.from_earthengine(
    "USGS/SRTMGL1_003",
    window=Window(bbox=(86.9, 27.9, 87.0, 28.0), scale=0.01),
)

# An ImageCollection reduced client-side to a median composite Dataset.
composite = eo.from_earthengine(
    "COPERNICUS/S2_SR_HARMONIZED",
    bands=["B4", "B3", "B2"],
    window=Window(bbox=(86.9, 27.9, 87.0, 28.0), scale=0.0001),
    start="2023-01-01",
    end="2023-03-01",
    reducer="median",
)
```

### Sentinel-2

Read an ESA Sentinel-2 product into a `Dataset` — band selection, DN → reflectance, and Level-2A
scene-classification masking (see [Sentinel](reference/sentinel.md)).

```python
import pyramids_eo as eo
from pyramids_eo import SclClass

scene = eo.from_sentinel2("S2A_..._MSIL2A.SAFE", bands=["B04", "B08"])
reflectance = scene.read_array(scaled=True)          # (DN + offset) / quantification

# Mask clouds and cloud shadows out via the L2A scene-classification layer.
clear = eo.from_sentinel2(
    "S2A_..._MSIL2A.SAFE",
    bands=["B04", "B03", "B02"],
    mask_scl=[SclClass.CLOUD_HIGH_PROBA, SclClass.CLOUD_SHADOW, SclClass.THIN_CIRRUS],
)
```

### Instrument readers (MTG-FCI, MSG-SEVIRI)

Decode geostationary L1 granules into calibrated, geolocated `Dataset`s (see
[Readers](reference/readers.md)).

```python
from pyramids_eo.sensors.readers import read_fci_l1c, read_seviri

# MTG-FCI L1C FDHSI: stitch a channel across its chunk set and calibrate.
fci = read_fci_l1c(["chunk_01.nc", "chunk_02.nc"], "ir_105")

# MSG-SEVIRI Level-1.5 native (.nat): decode + calibrate + geolocate one channel.
seviri = read_seviri("MSG4-...nat", "IR_108")
```

### Compositing

Assemble a true-colour composite over NumPy + pyramids-gis, then stretch it to a displayable frame (see
[Composites](reference/composites.md)).

```python
from pyramids_eo.composites import true_color
from pyramids_eo.enhance import stretch

rgb = true_color(red, green, blue)                   # physical reflectance in ~[0, 1]
frame = stretch(rgb, kind="cira")                    # → displayable uint8
```

## Next steps

- [Installation](installation.md) — install with pip, optional extras, development setup.
- [Reference](reference/index.md) — the full API surface.
- Examples — runnable notebooks for the [Earth Engine](examples/earth_engine_reader.ipynb) and
  [Sentinel-2](examples/sentinel2_reader.ipynb) readers, the
  [FCI / SEVIRI](examples/readers_fci_seviri_harmonise.ipynb) readers, and
  [day/night compositing](examples/day_night_compositing.ipynb).
