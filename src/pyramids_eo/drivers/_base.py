"""Shared helpers for the ``pyramids.raster_drivers`` plugin drivers.

Each driver in this package is a thin class over an existing pyramids-eo reader. It adds the
two things :meth:`pyramids.dataset.Dataset.read_file` needs to route a source into the reader —
a class-level claim set plus a ``can_open`` predicate — and maps the generic ``read`` arguments
onto the reader's own keywords. The reader bodies are untouched; a driver only delegates.

The contract version is declared as the literal ``1`` on each driver, matching pyramids' own
"writing a raster driver" guidance (pyramids does not re-export ``RASTER_DRIVER_API_VERSION``
publicly — it lives in the private ``pyramids.dataset._drivers`` — so importing the constant
would couple to a private module). This should become ``api_version = RASTER_DRIVER_API_VERSION``
once pyramids exposes the contract version from a public module.
"""

from __future__ import annotations

from collections.abc import Mapping
from glob import glob as _glob
from pathlib import Path
from typing import Any

_GLOB_CHARS = "*?["

#: The driver-contract version every driver here declares. Must equal pyramids'
#: ``RASTER_DRIVER_API_VERSION`` (currently 1); hard-coded because pyramids does not export it
#: from a public module. A driver declaring a value pyramids no longer speaks is skipped at
#: discovery with a warning, so keep this in step with the installed pyramids.
RASTER_DRIVER_API_VERSION = 1


def options_as_dict(options: Any) -> dict[str, Any]:
    """Normalise ``read_file``'s ``open_options`` into a plain dict for a reader call.

    ``Dataset.read_file`` hands a driver whatever the caller passed as ``open_options``,
    uninterpreted — a dict, a GDAL-style ``["KEY=VALUE", ...]`` list/tuple, or ``None``. A
    pyramids-eo reader takes Python keyword arguments, so coerce every spelling to one dict.

    Args:
        options: The caller's ``open_options`` — a mapping (any ``collections.abc.Mapping``, not
            only an exact ``dict``), a sequence of ``"KEY=VALUE"`` strings, or ``None``.

    Returns:
        A dict of the options, empty when ``options`` is ``None`` or empty.

    Raises:
        ValueError: ``options`` is a bare ``str``/``bytes``, or a sequence item is not a
            ``"KEY=VALUE"`` string.

    Examples:
        - A dict is returned as a shallow copy, so the caller's object is not mutated:
            ```python
            >>> from pyramids_eo.drivers._base import options_as_dict
            >>> src = {"channel": "IR_108"}
            >>> options_as_dict(src)
            {'channel': 'IR_108'}
            >>> options_as_dict(src) is src
            False

            ```
        - A GDAL-style KEY=VALUE list is parsed, and ``None`` becomes an empty dict:
            ```python
            >>> options_as_dict(["reducer=median", "tile_size=256"])
            {'reducer': 'median', 'tile_size': '256'}
            >>> options_as_dict(None)
            {}

            ```
    """
    if options is None:
        return {}
    if isinstance(options, Mapping):
        return dict(options)
    if isinstance(options, (str, bytes)):
        raise ValueError(
            f"open_options must be a dict or a list of 'KEY=VALUE' strings, not a bare "
            f"{type(options).__name__} ({options!r}); a bare string would be read one "
            "character at a time."
        )
    result: dict[str, Any] = {}
    for item in options:
        text = str(item)
        if "=" not in text:
            raise ValueError(
                f"open_options item {item!r} is not a 'KEY=VALUE' string; pass a dict for "
                "reader keywords."
            )
        key, value = text.split("=", 1)
        result[key] = value
    return result


def expand_sources(source: Any, *, pattern: str = "*") -> list[str]:
    """Expand one ``read_file`` source string into the file list a multi-file reader needs.

    ``Dataset.read_file`` hands a driver a single source string, but a granule split across many
    files is named by one of three forms, each resolved here to a sorted list of path strings:

    - an empty string → ``[]`` (``Path("")`` is ``.``, which must not list the whole CWD);
    - an existing directory → its sorted *files* matching ``pattern`` (subdirectories excluded);
    - an existing file → ``[source]`` (even if its name contains a glob metacharacter);
    - a glob string (one containing ``*``, ``?`` or ``[``) → its sorted matches;
    - anything else → ``[source]`` (a single, possibly not-yet-existing path).

    Args:
        source: The source handed by ``Dataset.read_file`` (stringified).
        pattern: The glob applied inside a directory source (e.g. ``"*.nc"``); ignored for a glob
            string (which already carries its own pattern) and for a single file.

    Returns:
        A sorted list of path strings. Empty when a glob or directory matches nothing.

    Examples:
        - A single file is returned as a one-element list:
            ```python
            >>> from pyramids_eo.drivers._base import expand_sources
            >>> expand_sources("granule/chunk_0.nc")
            ['granule/chunk_0.nc']

            ```
    """
    text = str(source)
    if not text:
        # Guard the empty string: Path("") is ".", which would list the current directory.
        return []
    path = Path(text)
    # Resolve an existing path literally first, so a real directory or file whose own name contains
    # a glob metacharacter (e.g. ``granule[1]``) is not mistaken for a pattern and lost.
    if path.is_dir():
        return sorted(str(entry) for entry in path.glob(pattern) if entry.is_file())
    if path.exists():
        return [text]
    if any(character in text for character in _GLOB_CHARS):
        return sorted(_glob(text))
    return [text]
