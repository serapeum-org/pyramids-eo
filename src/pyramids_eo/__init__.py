"""pyramids-eo — the Earth-observation layer of the pyramids stack.

``pyramids-eo`` is built on top of :mod:`pyramids` (``pyramids-gis``, the generic
raster/vector engine) and adds the logic that is specific to
**Earth-observation data**. Where pyramids knows how to *move rasters around*,
pyramids-eo knows what a pixel *means* for a given instrument or provider: which
subdataset is which channel, how to calibrate it, how to composite it, how to
resample a swath — and how to reach signed EO cloud assets. It is scoped by
*domain* (EO data), not by a restriction on what it may do.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

from pyramids_eo.earthengine import (
    EarthEngineCredentials,
    collection_from_earthengine,
    from_earthengine,
)

try:
    __version__ = _get_version("pyramids-eo")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "EarthEngineCredentials",
    "__version__",
    "collection_from_earthengine",
    "from_earthengine",
]
