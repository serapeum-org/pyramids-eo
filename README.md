# pyramids-eo

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**pyramids-eo** is the **Earth-observation layer** of the pyramids GIS stack. It is built on top of
[pyramids](https://github.com/serapeum-org/pyramids) (`pyramids-gis`, the generic GDAL raster/vector
engine) and adds the logic that is specific to **Earth-observation data**.

pyramids underneath knows only how to *move rasters around*. pyramids-eo knows what a pixel *means* for a
given instrument or provider: which subdataset is which channel, how to calibrate it, how to composite it,
how to resample a swath — and how to reach signed EO cloud assets so they stream through GDAL.

```
pyramids-eo        (Earth-observation logic: sensor readers, calibration, indices, composites,
    │               resample, masking, signed EO provider/STAC access)
    │  depends on
    ▼
pyramids           (generic GDAL raster / vector engine)
```

## Scope

pyramids-eo is scoped by *domain* — Earth-observation data — not by a restriction on what it may do. It
covers the EO-specific functionality that sits above the generic raster/vector operations pyramids-gis
provides.

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
