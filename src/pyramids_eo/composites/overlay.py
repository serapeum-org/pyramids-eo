"""Alpha overlay of one image over another.

`alpha_overlay` composites an RGBA *foreground* over an RGB or RGBA *background*
with the standard "over" operator. In the day/night chain the foreground is the
night-IR
cloud RGBA (transparent where there are no clouds) and the background is the
Black Marble city lights, so the lights show through the gaps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids_eo.composites._common import _as_array, _wrap_like


def alpha_overlay(foreground: Any, background: Any) -> Any:
    """Composite an RGBA foreground over an RGB(A) background.

    Implements the "over" operator. For an **opaque (RGB) background** the result
    is `fg_rgb * fg_a + bg_rgb * (1 - fg_a)` (a 3-band RGB image). For an **RGBA
    background** the full premultiplied over is used, yielding a 4-band RGBA image
    with `a_out = fg_a + bg_a * (1 - fg_a)`.

    Args:
        foreground: The RGBA foreground — a `(4, H, W)` array-like (R, G, B,
            alpha) or a pyramids `Dataset` with four bands. Alpha is expected in
            `[0, 1]`.
        background: The background — a `(3, H, W)` RGB or `(4, H, W)` RGBA
            array-like or `Dataset`, matching the foreground's `(H, W)`.

    Returns:
        The composited image: 3-band RGB for an RGB background, 4-band RGBA for an
        RGBA background. A pyramids `Dataset` (carrying the foreground's or
        background's geotransform + CRS) when either input is a `Dataset`,
        otherwise an ndarray.

    Raises:
        ValueError: When `foreground` is not a `(4, H, W)` RGBA array, or when
            `background` is not a `(3|4, H, W)` array.

    Examples:
        - A half-transparent red foreground over a blue background blends 50/50:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.composites import alpha_overlay
            >>> fg = np.array([[[1.0]], [[0.0]], [[0.0]], [[0.5]]])  # red, alpha 0.5
            >>> bg = np.array([[[0.0]], [[0.0]], [[1.0]]])           # blue
            >>> alpha_overlay(fg, bg).round(2).ravel().tolist()
            [0.5, 0.0, 0.5]

            ```
    """
    fg = _as_array(foreground)
    bg = _as_array(background)
    if fg.ndim != 3 or fg.shape[0] != 4:
        raise ValueError(
            f"foreground must be a (4, H, W) RGBA array; got shape {fg.shape}"
        )
    if bg.ndim != 3 or bg.shape[0] not in (3, 4):
        raise ValueError(
            f"background must be a (3, H, W) RGB or (4, H, W) RGBA array; "
            f"got shape {bg.shape}"
        )

    fg_rgb = fg[:3]
    fg_a = fg[3]

    if bg.shape[0] == 4:
        bg_rgb = bg[:3]
        bg_a = bg[3]
        out_a = fg_a + bg_a * (1.0 - fg_a)
        premultiplied = fg_rgb * fg_a + bg_rgb * bg_a * (1.0 - fg_a)
        out_rgb = np.divide(
            premultiplied,
            out_a,
            out=np.zeros_like(premultiplied),
            where=out_a > 0,
        )
        out = np.concatenate([out_rgb, out_a[np.newaxis, ...]], axis=0)
    else:
        out = fg_rgb * fg_a + bg[:3] * (1.0 - fg_a)

    return _wrap_like(out, foreground, background)
