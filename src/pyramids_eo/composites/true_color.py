"""True-colour RGB from calibrated reflectance bands.

SEVIRI carries no native green band, so for it `true_color` synthesises one from
the red, blue, and near-IR ("veggie") reflectances using the CIMSS weighted
recipe (`green_mode="synthetic"`, the default), then stacks red / green / blue
into an RGB image.

FCI FDHSI *does* carry a native green band (`vis_05`, ~0.51 um). For FCI, pass
that band as `green=` and select `green_mode="native"` to use it directly, or
`green_mode="ndvi_hybrid"` to blend it with the NIR by an NDVI fraction (more NIR
over barren surfaces, less over vegetation) — the reference true-colour look.

This is the **no-Rayleigh** variant (per the pyramids-eo compositing decision):
by default it does the band combination only. Atmospheric / Rayleigh correction
is intentionally out of scope of the base install to keep the dependency
footprint small; the result is slightly flatter over ocean / haze than a
Rayleigh-corrected image. A correction can be opted in per call via the
`rayleigh=` callable, which is applied to each solar band before green synthesis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from pyramids_eo.composites._common import _as_array, _wrap_like

#: Default CIMSS synthetic-green weights (red, near-IR/veggie, blue).
_DEFAULT_GREEN_WEIGHTS = (0.45, 0.10, 0.45)

#: Green-synthesis modes accepted by :func:`true_color`.
_GREEN_MODES = ("synthetic", "native", "ndvi_hybrid")


def _ndvi_hybrid_green(
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    *,
    strength: float = 3.0,
    limits: tuple[float, float] = (0.15, 0.05),
    ndvi_min: float = 0.0,
    ndvi_max: float = 1.0,
) -> np.ndarray:
    """Blend a native green band with the NIR by a per-pixel NDVI fraction.

    Computes `NDVI = (nir - red) / (nir + red)`, optionally sharpens it with a
    non-linear `strength`, maps it linearly from `[ndvi_min, ndvi_max]` onto
    `[limits[0], limits[1]]` to get a NIR fraction, then returns
    `(1 - fraction) * green + fraction * nir`. With the default `limits`, barren
    pixels (low NDVI) take more NIR (0.15) and vegetated pixels take less (0.05),
    so the native green dominates (0.85-0.95).

    Args:
        green: The native green reflectance (e.g. FCI `vis_05`).
        red: The red reflectance (NDVI red term).
        nir: The near-IR reflectance (NDVI NIR term and blend source).
        strength: Non-linear sharpening exponent applied to the NDVI; `1.0`
            leaves it linear. Must be `> 0`.
        limits: The `(low_ndvi, high_ndvi)` NIR fractions the NDVI is mapped onto.
        ndvi_min: Lower NDVI clip / mapping bound.
        ndvi_max: Upper NDVI clip / mapping bound.

    Returns:
        The blended green channel, same shape as the inputs.

    Raises:
        ValueError: When `strength <= 0`.
    """
    if strength <= 0:
        raise ValueError(f"strength must be > 0, got {strength}")
    denom = nir + red
    ndvi = np.divide(nir - red, denom, out=np.zeros_like(denom), where=denom != 0)
    ndvi = np.clip(np.nan_to_num(ndvi), ndvi_min, ndvi_max)
    if strength != 1.0:
        powered = ndvi**strength
        ndvi = powered / (powered + (1.0 - ndvi) ** strength)
    span = ndvi_max - ndvi_min
    fraction = (ndvi - ndvi_min) / span * (limits[1] - limits[0]) + limits[0]
    return np.asarray((1.0 - fraction) * green + fraction * nir, dtype=float)


def true_color(
    red: Any,
    blue: Any,
    nir: Any,
    *,
    green: Any = None,
    green_mode: str = "synthetic",
    green_weights: tuple[float, float, float] = _DEFAULT_GREEN_WEIGHTS,
    ndvi_strength: float = 3.0,
    ndvi_limits: tuple[float, float] = (0.15, 0.05),
    rayleigh: Callable[[np.ndarray], Any] | None = None,
    gamma: float | None = None,
    clip: bool = False,
) -> Any:
    """Build a true-colour RGB from red, blue, and near-IR reflectances.

    The green channel is built per `green_mode`:

    * `"synthetic"` (default) — `green = wr*red + wn*nir + wb*blue` (the CIMSS
      recipe), for sensors with no native green band (SEVIRI). Byte-identical to
      the historical behaviour.
    * `"native"` — use the `green=` band directly (FCI `vis_05`).
    * `"ndvi_hybrid"` — blend the `green=` band with the NIR by an NDVI fraction
      (see :func:`_ndvi_hybrid_green`).

    When `rayleigh` is given it is applied to each solar band (red, blue, nir, and
    a native `green`) before green synthesis; `None` (default) leaves the bands
    untouched and adds no dependency.

    Args:
        red: Red-band reflectance — array-like or a pyramids `Dataset`.
        blue: Blue-band reflectance, same grid as `red`.
        nir: Near-IR ("veggie") reflectance — the synthetic-green source, the
            NDVI NIR term, and the `ndvi_hybrid` blend source.
        green: Native green reflectance (FCI `vis_05`). Required for
            `green_mode` `"native"` / `"ndvi_hybrid"`, ignored for `"synthetic"`.
        green_mode: How to build the green channel — `"synthetic"` (default),
            `"native"`, or `"ndvi_hybrid"`.
        green_weights: The `(red, nir, blue)` weights for the synthetic green
            (default the CIMSS `(0.45, 0.10, 0.45)`).
        ndvi_strength: Non-linear NDVI sharpening for `"ndvi_hybrid"` (default 3.0).
        ndvi_limits: The `(low, high)` NIR fractions for `"ndvi_hybrid"`.
        rayleigh: Optional callable applied to each solar band before green
            synthesis (an atmospheric correction), or `None`.
        gamma: Optional gamma to apply (`value ** (1 / gamma)`), or `None`.
        clip: When `True`, clip the output to `[0, 1]`.

    Returns:
        The `(3, H, W)` RGB image. A pyramids `Dataset` (carrying an input
        `Dataset`'s geotransform + CRS) when any input is a `Dataset`, otherwise
        an ndarray.

    Raises:
        ValueError: When `green_mode` is unknown, or a native/ndvi_hybrid mode is
            requested without a `green=` band.

    Examples:
        - Synthetic CIMSS green (SEVIRI-style) from red / blue / NIR reflectance:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import true_color
            >>> r, b, n = np.full((1, 1), 0.2), np.full((1, 1), 0.6), np.full((1, 1), 0.9)
            >>> true_color(r, b, n)[1].round(3).tolist()
            [[0.45]]

            ```
        - Use FCI's native green band directly (`green_mode="native"`):
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import true_color
            >>> out = true_color(
            ...     np.full((1, 1), 0.2),
            ...     np.full((1, 1), 0.6),
            ...     np.full((1, 1), 0.9),
            ...     green=np.full((1, 1), 0.44),
            ...     green_mode="native",
            ... )
            >>> out[1].tolist()
            [[0.44]]

            ```
    """
    if green_mode not in _GREEN_MODES:
        raise ValueError(
            f"green_mode must be one of {_GREEN_MODES}; got {green_mode!r}"
        )

    r = _as_array(red)
    b = _as_array(blue)
    n = _as_array(nir)
    g_native = _as_array(green) if green is not None else None

    if rayleigh is not None:
        r = np.asarray(rayleigh(r), dtype=float)
        b = np.asarray(rayleigh(b), dtype=float)
        n = np.asarray(rayleigh(n), dtype=float)
        if g_native is not None:
            g_native = np.asarray(rayleigh(g_native), dtype=float)

    if green_mode == "synthetic":
        wr, wn, wb = green_weights
        green_ch = wr * r + wn * n + wb * b
    elif green_mode == "native":
        if g_native is None:
            raise ValueError("green_mode='native' requires a `green=` band")
        green_ch = g_native
    else:  # "ndvi_hybrid"
        if g_native is None:
            raise ValueError("green_mode='ndvi_hybrid' requires a `green=` band")
        green_ch = _ndvi_hybrid_green(
            g_native, r, n, strength=ndvi_strength, limits=ndvi_limits
        )

    rgb = np.stack([r, green_ch, b], axis=0)

    if gamma is not None:
        rgb = np.where(rgb > 0, rgb, 0.0) ** (1.0 / gamma)
    if clip:
        rgb = np.clip(rgb, 0.0, 1.0)

    return _wrap_like(rgb, red, blue, nir, green)
