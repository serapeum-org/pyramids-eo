"""Map composite physical values to a display range.

pyramids-eo composites return physical values — reflectance in roughly `[0, 1]`,
brightness temperature in kelvin. `stretch` maps those to a display range and
dtype (uint8 by default), the step between a raw composite and a displayable
frame.

Four curves are provided:

* `"linear"` — percentile-clip (the `cutoffs`) then rescale to `[0, 1]`.
* `"crude"` — fixed min/max linear rescale, `(x - lo) / (hi - lo)`.
* `"cira"` — a logarithmic, human-vision-tuned stretch; recommended for a
  true-colour composite (no gamma needed). Note it is not the signature default;
  pass `kind="cira"` explicitly.
* `"histogram"` — histogram equalisation.

The stretch is computed over the whole array (shared across bands), preserving
inter-band ratios — important for a true-colour image. NaN / nodata pixels are
excluded from the statistics and preserved (float output) or filled with 0
(integer output). For an RGBA input carrying a coverage / alpha band (e.g. from
`keep_alpha`), pass `preserve_alpha=True` so the trailing band is passed through
untouched instead of being curved like image data. Reflectance is expected in
`[0, 1]`; scale first if it is in percent.

For an integer frame, 0 is used for both a masked pixel and a valid-black pixel,
so the coverage / alpha band — not the nodata value — is the reliable mask.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Stretch curves accepted by :func:`stretch`.
_KINDS = ("linear", "crude", "cira", "histogram")

#: CIRA log-stretch constants (`log10(0.0223)` and its normalising denominator).
_CIRA_LOG_ROOT = float(np.log10(0.0223))
_CIRA_DENOM = (1.0 - _CIRA_LOG_ROOT) * 0.75


def _read(value: Any) -> tuple[np.ndarray, Any]:
    """Return `(float_array, template)` from an array-like or pyramids `Dataset`.

    Args:
        value: An ndarray-like, or a pyramids `Dataset` (read via `read_array`).

    Returns:
        The values as a float ndarray, and the source `Dataset` (or `None`) to
        carry georeferencing onto the result.
    """
    if hasattr(value, "read_array") and hasattr(value, "geotransform"):
        return np.asarray(value.read_array(), dtype=float), value
    return np.asarray(value, dtype=float), None


def _linear_bounds(
    finite: np.ndarray,
    min_stretch: float | None,
    max_stretch: float | None,
    cutoffs: tuple[float, float] | None,
) -> tuple[float, float]:
    """Resolve the `(lo, hi)` stretch bounds for the linear/crude curves.

    Args:
        finite: The finite (non-NaN) sample values.
        min_stretch: Explicit lower bound, or `None` to derive it.
        max_stretch: Explicit upper bound, or `None` to derive it.
        cutoffs: `(left, right)` percentile fractions to clip at when deriving a
            bound, or `None` to use the data min/max (the "crude" behaviour).

    Returns:
        The `(lo, hi)` bounds, with `hi > lo` guaranteed.
    """
    empty = finite.size == 0
    if min_stretch is not None:
        lo = float(min_stretch)
    elif empty:
        lo = 0.0
    elif cutoffs is not None:
        lo = float(np.percentile(finite, 100.0 * cutoffs[0]))
    else:
        lo = float(finite.min())

    if max_stretch is not None:
        hi = float(max_stretch)
    elif empty:
        hi = 1.0
    elif cutoffs is not None:
        hi = float(np.percentile(finite, 100.0 * (1.0 - cutoffs[1])))
    else:
        hi = float(finite.max())

    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _normalise(
    values: np.ndarray,
    kind: str,
    min_stretch: float | None,
    max_stretch: float | None,
    cutoffs: tuple[float, float],
) -> np.ndarray:
    """Map physical `values` to a `[0, 1]`-ish float, preserving NaN.

    Args:
        values: The float input array.
        kind: One of `_KINDS`.
        min_stretch: Optional fixed lower bound (linear/crude).
        max_stretch: Optional fixed upper bound (linear/crude).
        cutoffs: `(left, right)` percentile fractions for the linear curve.

    Returns:
        The normalised array (not yet clipped to `[0, 1]`); NaN where the input
        was NaN.
    """
    if kind == "cira":
        clipped = np.clip(values, np.finfo(float).eps, None)
        return np.asarray((np.log10(clipped) - _CIRA_LOG_ROOT) / _CIRA_DENOM)

    if kind == "histogram":
        return _histogram_equalise(values)

    use_cutoffs = cutoffs if kind == "linear" else None
    lo, hi = _linear_bounds(
        values[np.isfinite(values)], min_stretch, max_stretch, use_cutoffs
    )
    return np.asarray((values - lo) / (hi - lo))


def _histogram_equalise(values: np.ndarray, bins: int = 256) -> np.ndarray:
    """Histogram-equalise `values` to `[0, 1]` over the finite pixels.

    Args:
        values: The float input array.
        bins: Number of histogram bins.

    Returns:
        The equalised array, NaN where the input was NaN.
    """
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    sample = values[finite]
    if sample.size == 0:
        return out
    hist, edges = np.histogram(sample, bins=bins)
    cdf = np.cumsum(hist).astype(float)
    cdf /= cdf[-1]  # cdf[-1] == sample.size > 0 here (guarded above)
    out[finite] = np.interp(sample, edges[:-1], cdf)
    return out


def stretch(
    image: Any,
    *,
    kind: str = "linear",
    min_stretch: float | None = None,
    max_stretch: float | None = None,
    gamma: float | None = None,
    cutoffs: tuple[float, float] = (0.005, 0.005),
    preserve_alpha: bool = False,
    dtype: Any = "uint8",
) -> Any:
    """Map a composite's physical values to a display range and dtype.

    Args:
        image: The composite — an array (`(H, W)` or `(band, H, W)`) or a pyramids
            `Dataset`. Reflectance is expected in `[0, 1]`.
        kind: The stretch curve — `"linear"` (percentile-clip then rescale),
            `"crude"` (fixed min/max rescale), `"cira"` (logarithmic; the
            true-colour default), or `"histogram"` (equalisation).
        min_stretch: Fixed lower bound for `linear`/`crude`; `None` derives it.
        max_stretch: Fixed upper bound for `linear`/`crude`; `None` derives it.
        gamma: Optional power curve `x ** (1 / gamma)` applied after the stretch.
        cutoffs: `(left, right)` percentile fractions clipped by the `linear`
            curve (default 0.5% per side). Ignored by the other kinds. Each must
            be non-negative and the two must sum to less than 1.
        preserve_alpha: When `True` and `image` is `(band, H, W)` with 2+ bands,
            the trailing band is treated as a coverage / alpha band — passed
            through untouched (only cast to `dtype`) rather than curved. Use it
            when stretching an RGBA frame from `keep_alpha`.
        dtype: Output dtype. Integer dtypes scale `[0, 1]` onto `[0, max]` and
            fill NaN with 0; float dtypes keep `[0, 1]` and preserve NaN.

    Returns:
        The stretched image in `dtype` — a pyramids `Dataset` (carrying the
        input's geotransform + CRS, nodata 0 for integer / NaN for float) when
        `image` is a `Dataset`, otherwise an ndarray.

    Raises:
        ValueError: When `kind` is unknown, `gamma` is not positive, or `cutoffs`
            are negative or sum to 1 or more.

    Examples:
        - Crude-stretch reflectance onto the 8-bit display range:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.enhance import stretch
            >>> stretch(
            ...     np.array([[0.0, 0.5, 1.0]]),
            ...     kind="crude",
            ...     min_stretch=0.0,
            ...     max_stretch=1.0,
            ... ).tolist()
            [[0, 128, 255]]

            ```
        - The CIRA log stretch keeps NaN and lifts low reflectance (float out):
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.enhance import stretch
            >>> out = stretch(np.array([[0.02, 0.5]]), kind="cira", dtype="float64")
            >>> bool(out[0, 1] > out[0, 0])
            True

            ```
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}; got {kind!r}")
    if gamma is not None and gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if cutoffs[0] < 0 or cutoffs[1] < 0 or cutoffs[0] + cutoffs[1] >= 1:
        raise ValueError(
            f"cutoffs must be non-negative and sum to < 1; got {cutoffs!r}"
        )

    values, template = _read(image)
    if preserve_alpha and values.ndim >= 3 and values.shape[0] >= 2:
        rgb = _curve(values[:-1], kind, min_stretch, max_stretch, gamma, cutoffs)
        alpha = np.clip(values[-1], 0.0, 1.0)
        norm = np.concatenate([rgb, alpha[np.newaxis, ...]], axis=0)
    else:
        norm = _curve(values, kind, min_stretch, max_stretch, gamma, cutoffs)

    out = _to_dtype(norm, dtype)
    return _wrap(out, template, dtype)


def _curve(
    values: np.ndarray,
    kind: str,
    min_stretch: float | None,
    max_stretch: float | None,
    gamma: float | None,
    cutoffs: tuple[float, float],
) -> np.ndarray:
    """Normalise, gamma-correct and clip `values` to `[0, 1]` (NaN preserved).

    Args:
        values: The float input array (all image bands, no alpha).
        kind: One of `_KINDS`.
        min_stretch: Optional fixed lower bound (linear/crude).
        max_stretch: Optional fixed upper bound (linear/crude).
        gamma: Optional power curve applied after the stretch.
        cutoffs: `(left, right)` percentile fractions for the linear curve.

    Returns:
        The stretched array clipped to `[0, 1]`, NaN where the input was NaN.
    """
    norm = _normalise(values, kind, min_stretch, max_stretch, cutoffs)
    if gamma is not None:
        norm = np.clip(norm, 0.0, None) ** (1.0 / gamma)
    return np.asarray(np.clip(norm, 0.0, 1.0))


def _to_dtype(norm: np.ndarray, dtype: Any) -> np.ndarray:
    """Cast a `[0, 1]` (NaN-carrying) array to `dtype`.

    Args:
        norm: The normalised array, clipped to `[0, 1]`, NaN where masked.
        dtype: The target dtype.

    Returns:
        The array as `dtype` — integers scaled to `[0, max]` with NaN filled to
        0; floats keeping `[0, 1]` and NaN.
    """
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        scaled = norm * np.iinfo(dt).max
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        return np.round(scaled).astype(dt)
    return norm.astype(dt)


def _wrap(out: np.ndarray, template: Any, dtype: Any) -> Any:
    """Wrap `out` in a `Dataset` cloned from `template`, or return it unchanged.

    Args:
        out: The stretched result array.
        template: The source `Dataset` (or `None`).
        dtype: The output dtype, deciding the nodata value (0 for integer, NaN
            for float).

    Returns:
        A pyramids `Dataset` when `template` is given, otherwise `out`.
    """
    if template is None:
        return out
    from pyramids.dataset import Dataset

    nodata: Any = 0 if np.issubdtype(np.dtype(dtype), np.integer) else np.nan
    return Dataset.create_from_array(
        out, geo=template.geotransform, epsg=template.epsg, no_data_value=nodata
    )
