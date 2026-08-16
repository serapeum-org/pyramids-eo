"""Harmonise multi-resolution L1 bands onto a common grid.

FCI / SEVIRI channels arrive at different native resolutions (FCI: 0.5 / 1 / 2
km). `harmonise` warps/crops each band onto one target grid — the equivalent of
satpy's `Scene.resample` — by delegating to pyramids' `align` (which copies the
reference's CRS, rows/columns, and cell size and resamples the band onto it). No
new reprojection code; the reference grid decides the output.
"""

from __future__ import annotations

from typing import Any

from pyramids_eo.errors import ReaderError


def harmonise(bands: Any, reference: Any) -> Any:
    """Align a set of bands onto a reference grid.

    Args:
        bands: The bands to harmonise — a mapping `{name: Dataset}` or an
            iterable of pyramids `Dataset`s. Each is aligned to `reference` via
            `Dataset.align` (nearest-neighbour resampling onto the reference's
            CRS + rows/columns + cell size).
        reference: A pyramids `Dataset` whose grid every band is aligned to.

    Returns:
        The aligned bands in the same container shape as the input: a `dict`
        `{name: Dataset}` for a mapping, else a `list[Dataset]`.

    Raises:
        ReaderError: When `reference` is `None`, or when `bands` is empty.
    """
    if reference is None:
        raise ReaderError("harmonise: a reference grid is required")

    if isinstance(bands, dict):
        if not bands:
            raise ReaderError("harmonise: no bands given")
        return {name: band.align(reference) for name, band in bands.items()}

    band_list = list(bands)
    if not band_list:
        raise ReaderError("harmonise: no bands given")
    return [band.align(reference) for band in band_list]
