"""Static georeferenced background images for compositing.

`static_image` is the pyramids-eo port of satpy's `StaticImageCompositor`: it
loads a georeferenced raster (e.g. the NASA **Black Marble** city lights that
back the night-IR clouds), caching it locally when the source is a URL, and
optionally warps/crops it to another dataset's grid via pyramids' `align`.

Unlike satpy's compositor — which only uses a local file when `filename` is an
**absolute** path and otherwise falls back to a (now dead) bundled URL — this
takes a plain local **or** remote path and caches a remote one. A live
Black Marble mirror is:
``https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/BlackMarble_2016_3km_geo.tif``
(the older ``neo.gsfc.nasa.gov/archive/blackmarble/...`` URLs are 404).
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

#: Default cache directory for downloaded static images.
_DEFAULT_CACHE = Path.home() / ".cache" / "pyramids-eo"


def _download(url: str, target: Path, timeout: float) -> None:
    """Stream `url` to `target`, writing atomically via a `.part` temp file.

    Args:
        url: The http/https source URL.
        target: Destination path for the downloaded file.
        timeout: Per-request timeout in seconds.
    """
    request = urllib.request.Request(url)
    part = target.with_name(target.name + ".part")
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,  # nosec B310 - scheme restricted to http/https by _resolve_source
        open(part, "wb") as handle,
    ):
        shutil.copyfileobj(response, handle)
    os.replace(part, target)


def _resolve_source(source: Any, cache_dir: Any, timeout: float) -> Path:
    """Return a local path for `source`, downloading + caching a URL if needed.

    Args:
        source: A local filesystem path (relative or absolute) or an http/https
            URL.
        cache_dir: Directory to cache a downloaded URL (default `_DEFAULT_CACHE`).
        timeout: Per-request timeout in seconds for a download.

    Returns:
        The path to the local (possibly just-cached) file.

    Raises:
        FileNotFoundError: When a local `source` does not exist.
    """
    parsed = urlparse(str(source))
    if parsed.scheme in ("http", "https"):
        cache = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / Path(parsed.path).name
        if not (target.exists() and target.stat().st_size > 0):
            _download(str(source), target, timeout)
        return target
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"static image not found: {path}")
    return path


def static_image(
    source: Any,
    *,
    like: Any = None,
    cache_dir: Any = None,
    timeout: float = 60.0,
) -> Any:
    """Load a georeferenced image, caching a remote URL and warping to a grid.

    Port of satpy's `StaticImageCompositor`. `source` may be a local path
    (relative or absolute) or an http/https URL; a URL is downloaded once and
    cached under `cache_dir`. When `like` is given, the loaded image is warped
    and cropped onto that dataset's grid (CRS + rows/columns + cell size) via
    pyramids' `align`.

    Args:
        source: A local filesystem path or an http/https URL to a georeferenced
            raster (e.g. the Black Marble GeoTIFF).
        like: A pyramids `Dataset` whose grid the image is warped/cropped to. When
            `None`, the image is returned at its native grid.
        cache_dir: Directory used to cache a downloaded URL. Defaults to
            `~/.cache/pyramids-eo`. Ignored for a local `source`.
        timeout: Per-request download timeout in seconds (default 60).

    Returns:
        A pyramids `Dataset` — aligned to `like`'s grid when `like` is given,
        otherwise the image at its native grid.

    Raises:
        FileNotFoundError: When a local `source` does not exist.
    """
    path = _resolve_source(source, cache_dir, timeout)

    from pyramids.dataset import Dataset

    dataset = Dataset.read_file(str(path))
    if like is not None:
        dataset = dataset.align(like)
    return dataset
