"""pyramids-eo — GDAL-native, xarray-free remote-sensing tier for the pyramids stack.

``pyramids-eo`` sits between :mod:`pyramids` (the generic GDAL raster engine) and
provider/orchestration layers. It knows what a pixel *means* for a given
instrument: which subdataset is which channel, how to calibrate it, how to
composite it, how to resample a swath. It takes a local file / path / bytes and
decodes + processes it — it does **not** fetch from providers.

The instrument readers (``read_seviri``, ``read_fci``, ``read_abi``,
``read_olci``, ``read_slstr``) are re-exported here once implemented.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("pyramids-eo")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "__version__",
]
