# pyramids-eo

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**pyramids-eo** is the GDAL-native, xarray-free **remote-sensing / Earth-observation tier** of the
pyramids GIS stack. It sits between [pyramids](https://github.com/serapeum-org/pyramids) — the generic
GDAL raster engine — and provider/orchestration layers.

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
pyramids           (generic GDAL raster / vector engine)
```

## Design boundary

pyramids-eo takes a **local file / path / bytes** and decodes + processes it. It does **not** fetch from
providers (no HTTP, no auth, no product catalogs) — that stays in the orchestration layer above it.

## Installation

pyramids-eo is pure Python; GDAL is provided transitively by the `pyramids-gis` platform wheels, so no
system GDAL is required.

```console
pip install pyramids-eo
```

## Development

Environments are managed with [pixi](https://pixi.sh):

```console
pixi install -e dev
pixi run -e dev pytest        # run the tests
pixi run -e dev mypy          # type-check
pre-commit install            # enable git hooks
```

## Documentation

Full documentation: <https://serapeum-org.github.io/pyramids-eo>

## License

GNU General Public License v3 (GPLv3) — see [`LICENSE`](LICENSE).
