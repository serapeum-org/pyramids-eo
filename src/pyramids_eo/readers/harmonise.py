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


def _resample_to(band: Any, reference: Any, method: str | None) -> Any:
    """Resample one band onto the reference grid with the chosen method.

    Args:
        band: The band `Dataset` to resample.
        reference: The reference-grid `Dataset`.
        method: `None` / `"nearest"` uses `Dataset.align` (exact reference grid,
            nearest-neighbour); any other value (e.g. `"bilinear"`, `"cubic"` for
            continuous radiometric bands) warps with `warped_view` and then snaps
            to the reference's exact grid via `align`.

    Returns:
        The resampled `Dataset`.
    """
    if method is None or method == "nearest":
        return band.align(reference)
    crs = reference.epsg if reference.epsg is not None else reference.crs
    warped = band.warped_view(
        crs,
        method=method,
        cell_size=reference.cell_size,
        bbox=tuple(reference.bbox),
    )
    # warped_view derives its grid from bbox + cell size, which can land off the
    # reference by a pixel for non-integer ratios / rounding; align snaps it to
    # the reference's exact rows/columns so multi-band harmonisation stays
    # co-registered.
    return warped.align(reference)


def harmonise(bands: Any, reference: Any, *, method: str | None = None) -> Any:
    """Align a set of bands onto a reference grid.

    Args:
        bands: The bands to harmonise — a mapping `{name: Dataset}` or an
            iterable of pyramids `Dataset`s. Each is resampled onto `reference`'s
            grid (CRS + rows/columns + cell size).
        reference: A pyramids `Dataset` whose grid every band is aligned to.
        method: Resampling method. `None` (default) / `"nearest"` uses
            `Dataset.align` (nearest-neighbour onto the exact reference grid); any
            other value (e.g. `"bilinear"`, `"cubic"`) uses `warped_view` onto the
            reference's CRS + cell size + bbox — preferable for continuous
            reflectance / brightness-temperature bands.

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
        return {
            name: _resample_to(band, reference, method) for name, band in bands.items()
        }

    band_list = list(bands)
    if not band_list:
        raise ReaderError("harmonise: no bands given")
    return [_resample_to(band, reference, method) for band in band_list]
