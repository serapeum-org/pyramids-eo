"""A minimal NetCDF-3 (classic, CDF-1) writer for building faithful test granules.

The Sentinel-3 reader tests need fixture files that match the real products:
**several** variables in one file, each with its own CF attributes
(`scale_factor`, `_FillValue`, `valid_min`, `valid_max`). GDAL's netCDF *writer*
cannot express that -- a multi-band write suffixes the variable names
(`longitude1`, `latitude2`), `CreateMultiDimensional` produces files this GDAL
build cannot reopen through the classic `NETCDF:"file":var` path, and
`APPEND_SUBDATASET=YES` silently overwrites the file instead of appending (all
three measured). Writing the container format directly is simpler than working
around that, needs no new dependency, and lets a fixture be built from the
product specification rather than invented -- which is what makes a wrong
attribute name or a wrong packing visible to the suite.

The format is the published NetCDF classic layout: a big-endian header listing
dimensions, global attributes and variables (each with its attributes, type,
size and the byte offset of its data), followed by the non-record data in
variable order, every field padded to a 4-byte boundary.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

#: Header component tags from the NetCDF classic format.
_NC_DIMENSION = 10
_NC_VARIABLE = 11
_NC_ATTRIBUTE = 12

#: `nc_type` codes, and the big-endian numpy dtype each maps to.
NC_CHAR = 2
NC_SHORT = 3
NC_INT = 4
NC_FLOAT = 5
NC_DOUBLE = 6

_DTYPES: dict[int, str] = {
    NC_CHAR: "S1",
    NC_SHORT: ">i2",
    NC_INT: ">i4",
    NC_FLOAT: ">f4",
    NC_DOUBLE: ">f8",
}

#: numpy kind/itemsize -> nc_type, for inferring a variable's type from its array.
_NC_TYPE_OF: dict[tuple[str, int], int] = {
    ("i", 2): NC_SHORT,
    ("i", 4): NC_INT,
    ("f", 4): NC_FLOAT,
    ("f", 8): NC_DOUBLE,
}


def _pad(payload: bytes) -> bytes:
    """Pad `payload` with zero bytes up to a 4-byte boundary."""
    return payload + b"\x00" * (-len(payload) % 4)


def _name(text: str) -> bytes:
    """Encode a header name: its byte length, then the padded bytes."""
    encoded = text.encode("ascii")
    return struct.pack(">i", len(encoded)) + _pad(encoded)


def _attribute(name: str, value: Any) -> bytes:
    """Encode one attribute, inferring its `nc_type` from `value`.

    A `str` becomes `NC_CHAR`; an `int` becomes `NC_INT`; a `float` becomes
    `NC_DOUBLE`; a numpy array keeps its own type.
    """
    if isinstance(value, str):
        encoded = value.encode("ascii")
        return _name(name) + struct.pack(">ii", NC_CHAR, len(encoded)) + _pad(encoded)
    array = np.asarray(value)
    if array.dtype.kind in "iu":
        nc_type = NC_INT
    elif array.dtype.kind == "f":
        nc_type = NC_DOUBLE
    else:  # pragma: no cover - defensive; fixtures use numbers and strings only
        raise TypeError(f"unsupported attribute type for {name!r}: {array.dtype}")
    flat = np.atleast_1d(array).astype(_DTYPES[nc_type])
    return _name(name) + struct.pack(">ii", nc_type, flat.size) + _pad(flat.tobytes())


def _attribute_list(attributes: dict[str, Any] | None) -> bytes:
    """Encode an attribute list, or the ABSENT marker when there are none."""
    if not attributes:
        return struct.pack(">ii", 0, 0)
    body = b"".join(_attribute(key, value) for key, value in attributes.items())
    return struct.pack(">ii", _NC_ATTRIBUTE, len(attributes)) + body


def write_netcdf(
    path: Path,
    variables: dict[str, tuple[np.ndarray, dict[str, Any]]],
    *,
    dimensions: tuple[str, str] = ("rows", "columns"),
) -> Path:
    """Write a NetCDF-3 classic file holding `variables`, each with its attributes.

    Every variable shares one 2-D shape, which is how the Sentinel-3 annotation
    and measurement files are laid out (one grid per file).

    GDAL reads a NetCDF grid **bottom-up** (the CF convention is south-to-north),
    so each array is written with its rows reversed. A caller therefore gets back
    exactly the array it passed in, which keeps the fixtures readable; measured,
    rather than assumed.

    Args:
        path: The file to create (its parent must exist).
        variables: `{name: (array, attributes)}`. Arrays must all share a shape;
            `int16`/`int32`/`float32`/`float64` are supported. Attribute values
            may be `str`, `int`, `float` or a numpy array.
        dimensions: Names for the row and column dimensions.

    Returns:
        The written file's path.

    Raises:
        ValueError: When `variables` is empty or the arrays differ in shape.
    """
    if not variables:
        raise ValueError("write_netcdf needs at least one variable")
    arrays = [np.asarray(array) for array, _ in variables.values()]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"all variables must share one shape, got {sorted(shapes)}")
    height, width = arrays[0].shape

    header = bytearray(b"CDF\x01" + struct.pack(">i", 0))
    header += struct.pack(">ii", _NC_DIMENSION, 2)
    header += _name(dimensions[0]) + struct.pack(">i", height)
    header += _name(dimensions[1]) + struct.pack(">i", width)
    header += _attribute_list(None)
    header += struct.pack(">ii", _NC_VARIABLE, len(variables))

    # Each variable's entry ends with the byte offset of its data, so the header
    # must be sized before those offsets are known: build the entries once with
    # placeholder offsets to measure the header, then rewrite them with the real
    # ones. The layout is identical either way, so one measuring pass suffices.
    entries: list[bytes] = []
    payloads: list[bytes] = []
    for name, (array, attributes) in variables.items():
        values = np.asarray(array)
        nc_type = _NC_TYPE_OF.get((values.dtype.kind, values.dtype.itemsize))
        if nc_type is None:
            raise ValueError(f"unsupported dtype for variable {name!r}: {values.dtype}")
        # Rows reversed on write, so GDAL's bottom-up read returns the caller's
        # array in the order they wrote it.
        payload = _pad(values[::-1].astype(_DTYPES[nc_type]).tobytes())
        payloads.append(payload)
        entries.append(
            _name(name)
            + struct.pack(">i", 2)
            + struct.pack(">ii", 0, 1)
            + _attribute_list(attributes)
            + struct.pack(">iii", nc_type, len(payload), 0)
        )

    header_size = len(header) + sum(len(entry) for entry in entries)
    offset = header_size
    body = bytearray(header)
    for entry, payload in zip(entries, payloads):
        # Replace the trailing placeholder offset with this variable's real one.
        body += entry[:-4] + struct.pack(">i", offset)
        offset += len(payload)
    body += b"".join(payloads)

    path.write_bytes(bytes(body))
    return path
