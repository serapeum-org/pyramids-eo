# Installation

pyramids-eo is **pure Python**. Its native GIS stack (GDAL/OGR/PROJ/GEOS) is provided transitively by
[`pyramids-gis`](https://pypi.org/project/pyramids-gis/), whose self-contained platform wheels vendor
those libraries — so `pip install pyramids-eo` works out of the box on Linux, macOS, and Windows with
**no system GDAL** installation required.

**Package name:** `pyramids-eo`
**Import name:** `pyramids_eo`
**Supported Python versions:** 3.11–3.14 (`>=3.11,<4`)

## Install with pip

```console
pip install pyramids-eo
```

### Optional extras

- `viz`: `cleopatra[tiles]` — plotting and quick-look composites.

```console
pip install "pyramids-eo[viz]"
```

## Development install (pixi)

Environments are managed with [pixi](https://pixi.sh):

```console
pixi install -e dev
pixi run -e dev pytest
```

The `dev` environment installs pyramids-eo editable together with the test, lint, and type-checking
tooling; `docs` installs the MkDocs stack.
