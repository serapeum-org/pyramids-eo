"""Tasseled-cap (Kauth-Thomas) transform for a multi-band reflectance raster.

`tasseled_cap(dataset, *, sensor, components=...)` rotates a stack of reflectance
bands into the tasseled-cap components -- brightness, greenness, wetness -- using
the per-sensor coefficients from the registry
(`pyramids_eo.sensors.registry.get_tasseled_cap`). It returns a `Dataset` with
the requested components as named bands, carrying the input's grid and CRS, so it
stays composable with the rest of the stack.

The coefficients are a fixed linear combination tied to a *specific band order*,
so a silently wrong band order yields a plausible-looking but meaningless raster
-- the failure this transform is designed against. When the input `Dataset`'s
bands are named (`blue`/`green`/... or a sensor spelling such as `B4` / `SR_B5`),
they are matched to the sensor's roles and reordered to the coefficient order;
when they are unnamed (generic), the stack is assumed to already be in the
sensor's documented band order and only its band *count* is checked.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from pyramids_eo.errors import TransformError
from pyramids_eo.sensors.registry.tasseled_cap import (
    TasseledCapCoefficients,
    get_tasseled_cap,
)

#: The three components of the classic tasseled-cap transform.
_DEFAULT_COMPONENTS: tuple[str, ...] = ("brightness", "greenness", "wetness")

#: A finite input value above this is implausible for reflectance (which is ~0-1,
#: a little above 1 for bright TOA) and signals a scaled-integer / DN stack the
#: coefficients were not derived for -- warned about, not rejected, since the
#: caller owns the domain.
_IMPLAUSIBLE_REFLECTANCE = 100.0


def _resolve_band_order(
    band_names: Sequence[str] | None, coeffs: TasseledCapCoefficients
) -> list[int] | None:
    """Return indices that reorder a band stack into the coefficient band order.

    Args:
        band_names: The input `Dataset`'s band names, or `None`.
        coeffs: The sensor's coefficient table.

    Returns:
        A list of indices `order` such that `stack[order]` is in `coeffs.bands`
        order, or `None` to use the stack as-is (positional) -- which happens
        when no band is named in a way the sensor recognises.

    Raises:
        TransformError: When some but not all bands are recognised (ambiguous), or
            the recognised names are not exactly the sensor's band set (a missing,
            extra, or duplicated role).
    """
    if not band_names:
        return None
    roles = [coeffs.role_of(name) for name in band_names]
    recognised = [r for r in roles if r is not None]
    if not recognised:
        # Generic names (e.g. "Band_1"): fall back to the documented positional
        # order rather than guessing. The band count is validated by the caller.
        return None
    if len(recognised) != len(roles):
        unknown = [name for name, role in zip(band_names, roles) if role is None]
        raise TransformError(
            f"tasseled_cap: cannot match band(s) {unknown} to {coeffs.sensor} roles "
            f"{list(coeffs.bands)}; name every band or none (aliases: "
            f"{dict(coeffs.aliases)})"
        )
    if sorted(recognised) != sorted(coeffs.bands):
        raise TransformError(
            f"tasseled_cap: band names resolve to roles {recognised}, which are not "
            f"the {coeffs.sensor} bands {list(coeffs.bands)} (a missing, extra, or "
            f"duplicated band)"
        )
    index_of_role = {role: i for i, role in enumerate(roles)}
    return [index_of_role[role] for role in coeffs.bands]


def tasseled_cap(
    dataset: Any,
    *,
    sensor: str,
    components: Sequence[str] = _DEFAULT_COMPONENTS,
) -> Any:
    """Rotate a reflectance band stack into its tasseled-cap components.

    Applies the sensor's tasseled-cap coefficients (see
    `pyramids_eo.sensors.registry.get_tasseled_cap`) to the bands of `dataset` and
    returns a `Dataset` whose bands are the requested `components`, in the order
    given, carrying `dataset`'s geotransform and CRS. The input bands must be in
    the sensor's documented reflectance domain (TOA / reflectance factor, not raw
    DN) and in the sensor's band order -- which is enforced by name when the bands
    are named and by count otherwise (see the module docstring).

    The whole band stack is read into memory and processed at full resolution
    (several float64 arrays at once: the input, and one per requested component);
    there is no windowing, so a full scene is sized accordingly.

    Args:
        dataset: A pyramids `Dataset` whose bands are the sensor's reflectance
            bands (a `(band, row, col)` stack). Read via `read_array`; its
            `geotransform`, `epsg` (and `crs`, when it has no EPSG code) and
            per-band `no_data_value` are carried onto / used for the result. Band
            names, when present, drive band-order validation.
        sensor: The sensor whose coefficient table to use (e.g. `"landsat8_oli"`,
            `"landsat5_tm"`, `"landsat7_etm"`); case-insensitive.
        components: The components to compute, in output-band order. Defaults to
            `("brightness", "greenness", "wetness")`. Each must be defined in the
            sensor's table.

    Returns:
        A pyramids `Dataset` of shape `(len(components), row, col)` with the
        components as its band names, `dataset`'s geotransform / CRS, and NaN as
        its nodata value. A pixel that is the input's nodata, or non-finite in any
        band, is NaN in every component.

    Raises:
        TransformError: When `components` is empty, names a component the sensor's
            table does not define, or when the input is not a band stack whose
            band count / order matches the sensor.
        UnknownSensorError: When `sensor` has no bundled coefficient table.

    Examples:
        - Brightness / greenness / wetness from a Landsat 8 OLI reflectance stack:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids_eo.spectral import tasseled_cap
            >>> stack = np.ones((6, 1, 1), dtype=float)  # blue..swir2, reflectance 1.0
            >>> ds = Dataset.from_array(
            ...     stack, geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
            ... )
            >>> ds.band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
            >>> tc = tasseled_cap(ds, sensor="landsat8_oli")
            >>> tc.band_names
            ['brightness', 'greenness', 'wetness']
            >>> float(tc.read_array()[0, 0, 0].round(4))  # sum of the brightness row
            2.3099

            ```
        - Pick a single component; its coefficients are a sensor-specific constant:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from pyramids_eo.spectral import tasseled_cap
            >>> stack = np.ones((6, 1, 1), dtype=float)
            >>> ds = Dataset.from_array(
            ...     stack, geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
            ... )
            >>> ds.band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
            >>> wet = tasseled_cap(ds, sensor="landsat8_oli", components=["wetness"])
            >>> wet.band_names
            ['wetness']

            ```
    """
    coeffs = get_tasseled_cap(sensor)
    requested = list(components)
    if not requested:
        raise TransformError("tasseled_cap: `components` is empty")
    unknown = [c for c in requested if c not in coeffs.components]
    if unknown:
        raise TransformError(
            f"tasseled_cap: {coeffs.sensor} has no component(s) {unknown}; "
            f"available: {coeffs.component_names}"
        )

    if not hasattr(dataset, "read_array"):
        raise TransformError(
            "tasseled_cap: `dataset` must be a pyramids Dataset (a band stack), "
            f"got {type(dataset).__name__}"
        )
    stack = np.asarray(dataset.read_array(), dtype=float)
    if stack.ndim != 3:
        raise TransformError(
            f"tasseled_cap: expected a (band, row, col) stack of {len(coeffs.bands)} "
            f"bands, got an array of shape {stack.shape}"
        )
    if stack.shape[0] != len(coeffs.bands):
        raise TransformError(
            f"tasseled_cap: {coeffs.sensor} expects {len(coeffs.bands)} bands "
            f"{list(coeffs.bands)}, got {stack.shape[0]}"
        )

    _warn_if_not_reflectance(stack, coeffs)

    order = _resolve_band_order(getattr(dataset, "band_names", None), coeffs)
    ordered = stack[order] if order is not None else stack

    out = np.stack(
        [
            np.tensordot(np.asarray(coeffs.components[name]), ordered, axes=(0, 0))
            for name in requested
        ],
        axis=0,
    )
    # The linear combination turns an input fill value into a plausible finite
    # number, so a pixel that is fill (or non-finite) in any band would otherwise
    # be emitted unflagged while the output declares NaN nodata. Mask those pixels
    # to NaN so the declared nodata is honest.
    out[:, _invalid_pixels(stack, dataset)] = np.nan
    return _wrap_components(out, dataset, requested)


def _warn_if_not_reflectance(
    stack: np.ndarray, coeffs: TasseledCapCoefficients
) -> None:
    """Warn when the stack's magnitude is implausible for reflectance.

    The coefficients are tied to a reflectance domain (see `coeffs.domain`), so a
    scaled-integer or DN stack (finite max far above 1) would yield a
    wrong-magnitude raster. This warns rather than raises -- negative or slightly
    above-1 reflectance is legitimate, and the caller owns the domain.

    Args:
        stack: The `(band, row, col)` input stack.
        coeffs: The sensor's coefficient table (for its `domain` in the message).
    """
    finite = stack[np.isfinite(stack)]
    if finite.size and float(finite.max()) > _IMPLAUSIBLE_REFLECTANCE:
        warnings.warn(
            f"tasseled_cap: input max {float(finite.max()):.4g} is far above the "
            f"reflectance range the {coeffs.sensor} coefficients expect "
            f"({coeffs.domain!r}); scale a DN / integer stack to reflectance first "
            "or the components will be wrong-magnitude",
            UserWarning,
            stacklevel=3,
        )


def _invalid_pixels(stack: np.ndarray, dataset: Any) -> np.ndarray:
    """Return a `(row, col)` mask, `True` where a pixel is fill or non-finite.

    A pixel is invalid when any band is non-finite, or equals that band's declared
    `no_data_value` -- so the component output over it can be set to NaN rather
    than a plausible finite value.

    Args:
        stack: The `(band, row, col)` input stack (original band order).
        dataset: The input `Dataset`, read for its per-band `no_data_value`.

    Returns:
        A boolean `(row, col)` array, `True` for an invalid pixel.
    """
    invalid = ~np.isfinite(stack).all(axis=0)
    nodata = getattr(dataset, "no_data_value", None)
    if nodata is not None:
        values = (
            nodata
            if isinstance(nodata, (list, tuple, np.ndarray))
            else [nodata] * stack.shape[0]
        )
        for band, value in zip(stack, values):
            if value is not None and np.isfinite(value):
                invalid |= band == value
    return invalid


def _wrap_components(out: np.ndarray, template: Any, component_names: list[str]) -> Any:
    """Wrap the component stack in a `Dataset` cloned from `template`.

    Args:
        out: The `(len(component_names), row, col)` component array.
        template: The input `Dataset` supplying the geotransform + CRS.
        component_names: The output band names, in band order.

    Returns:
        A pyramids `Dataset` carrying `template`'s geotransform / CRS, NaN nodata,
        and `component_names` as its band names.
    """
    from pyramids.base.crs import crs_spec
    from pyramids.dataset import Dataset, GeoReference

    result = Dataset.from_array(
        np.asarray(out, dtype=float),
        geo_ref=GeoReference(
            geo=template.geotransform,
            epsg=crs_spec(template.epsg, getattr(template, "crs", None)),
        ),
        no_data_value=np.nan,
    )
    result.band_names = component_names
    return result
