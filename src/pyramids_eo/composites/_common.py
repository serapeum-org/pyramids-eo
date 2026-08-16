"""Shared helpers for the compositing primitives."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_array(value: Any) -> np.ndarray:
    """Return `value` as a float ndarray, reading a pyramids `Dataset` if given.

    Args:
        value: An ndarray-like, or a pyramids `Dataset` (read via `read_array`).

    Returns:
        A float ndarray.
    """
    if hasattr(value, "read_array"):
        return np.asarray(value.read_array(), dtype=float)
    return np.asarray(value, dtype=float)


def _wrap_like(out: np.ndarray, *candidates: Any) -> Any:
    """Wrap `out` in a pyramids `Dataset` cloned from the first Dataset candidate.

    Args:
        out: The result array to return.
        *candidates: Inputs to search for a georeferenced template; the first one
            exposing both `read_array` and `geotransform` supplies the
            geotransform + CRS.

    Returns:
        A pyramids `Dataset` carrying the template's geotransform + CRS when a
        candidate is a `Dataset`, otherwise `out` unchanged (an ndarray).
    """
    out = np.asarray(out, dtype=float)
    template = next(
        (
            candidate
            for candidate in candidates
            if hasattr(candidate, "read_array") and hasattr(candidate, "geotransform")
        ),
        None,
    )
    if template is None:
        return out

    from pyramids.dataset import Dataset

    # Composited data can legitimately hold NaN (masked terminator / gaps), so
    # declare NaN as the nodata value rather than the default -9999 sentinel.
    return Dataset.create_from_array(
        out, geo=template.geotransform, epsg=template.epsg, no_data_value=np.nan
    )
