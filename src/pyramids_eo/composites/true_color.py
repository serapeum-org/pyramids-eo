"""True-colour RGB from calibrated reflectance bands.

FCI / SEVIRI carry no true green band, so `true_color` synthesises one from the
red, blue, and near-IR ("veggie") reflectances using the CIMSS weighted recipe,
then stacks red / synthetic-green / blue into an RGB image.

This is the **no-Rayleigh** variant (per the pyramids-eo compositing decision):
it does the band combination only. Atmospheric / Rayleigh correction — satpy
gets it from `pyspectral` — is intentionally out of scope here to keep the
dependency footprint free of the PyTroll stack; the result is slightly flatter
over ocean / haze than a Rayleigh-corrected image.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids_eo.composites._common import _as_array, _wrap_like

#: Default CIMSS synthetic-green weights (red, near-IR/veggie, blue).
_DEFAULT_GREEN_WEIGHTS = (0.45, 0.10, 0.45)


def true_color(
    red: Any,
    blue: Any,
    nir: Any,
    *,
    green_weights: tuple[float, float, float] = _DEFAULT_GREEN_WEIGHTS,
    gamma: float | None = None,
    clip: bool = False,
) -> Any:
    """Build a true-colour RGB from red, blue, and near-IR reflectances.

    Synthesises the green channel as
    `green = wr*red + wn*nir + wb*blue` (the CIMSS recipe), then stacks
    `[red, green, blue]` into a `(3, H, W)` image.

    Args:
        red: Red-band reflectance — array-like or a pyramids `Dataset`.
        blue: Blue-band reflectance, same grid as `red`.
        nir: Near-IR ("veggie") reflectance used to synthesise green.
        green_weights: The `(red, nir, blue)` weights for the synthetic green
            (default the CIMSS `(0.45, 0.10, 0.45)`).
        gamma: Optional gamma to apply (`value ** (1 / gamma)`), or `None`.
        clip: When `True`, clip the output to `[0, 1]`.

    Returns:
        The `(3, H, W)` RGB image. A pyramids `Dataset` (carrying an input
        `Dataset`'s geotransform + CRS) when any input is a `Dataset`, otherwise
        an ndarray.
    """
    r = _as_array(red)
    b = _as_array(blue)
    n = _as_array(nir)
    wr, wn, wb = green_weights
    green = wr * r + wn * n + wb * b
    rgb = np.stack([r, green, b], axis=0)

    if gamma is not None:
        rgb = np.where(rgb > 0, rgb, 0.0) ** (1.0 / gamma)
    if clip:
        rgb = np.clip(rgb, 0.0, 1.0)

    return _wrap_like(rgb, red, blue, nir)
