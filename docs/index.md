# pyramids-eo

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**pyramids-eo** is the **remote-sensing / Earth-observation tier** of the
pyramids GIS stack. It sits between [pyramids](https://github.com/serapeum-org/pyramids) — the generic
raster engine — and provider/orchestration layers.

pyramids underneath knows only how to *move rasters around*. pyramids-eo knows what a pixel *means* for a
given instrument: which subdataset is which channel, how to calibrate it, how to composite it, how to
resample a swath.

```
providers          (fetch granules, auth, catalogs, cache, pipelines)
    │  depends on
    ▼
pyramids-eo        (sensors: readers, calibration, indices, composites, resample, masking)
    │  depends on
    ▼
pyramids           (generic raster / vector engine)
```

## Design boundary

pyramids-eo takes a **local file / path / bytes** and decodes + processes it. It does **not** fetch from
providers (no HTTP, no auth, no product catalogs) — that stays in the orchestration layer above it.

- **Remote sensing** = offline compute.
- **Orchestration** = I/O.

## Public surface (readers)

> **Planned API — not yet implemented.** These reader entry points are the roadmap for the package;
> the current release ships the scaffold only, so the calls below are illustrative and will raise
> `AttributeError` until the readers land.

```python
import pyramids_eo as eo

scene = eo.read_fci("MTGI-FCI-...nc", channels=["ir_105", "vis_06"])   # -> Container
disc  = eo.read_seviri("MSG4-...nat", calibration="radiance")          # -> Dataset
swath = eo.read_olci("S3A_OL_1_...SEN3", bands=["Oa08_radiance"])      # -> curvilinear Container
```

See [Installation](installation.md) to get started and the [Reference](reference/index.md) for the API.
