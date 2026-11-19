# Installation

pyramids-eo is **pure Python**. Its native GIS stack is provided transitively by
[`pyramids-core`](https://pypi.org/project/pyramids-core/), whose self-contained platform wheels vendor
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

- `viz`: installs `pyramids-core[viz]` (plotting and quick-look composites via cleopatra).
- `lazy`: installs the [`pyramids-lazy`](https://pypi.org/project/pyramids-lazy/) plugin, which
  brings the dask array stack. Needed for any `DatasetCollection.data` access on a multi-scene
  collection — the `reduce_time` / `mean` / `min` / `max` / `sum` reducers all read `.data`, as
  does the Sentinel-2 reader's collection. A bare install raises a clear error there and nowhere
  else.

```console
pip install "pyramids-eo[viz]"
pip install "pyramids-eo[lazy]"
```

## Development install (pixi)

Environments are managed with [pixi](https://pixi.sh):

```console
pixi install -e dev
pixi run -e dev pytest
```

The `dev` environment installs pyramids-eo editable together with the test, lint, and type-checking
tooling; `docs` installs the MkDocs stack.
