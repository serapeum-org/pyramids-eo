"""Shared helpers for the instrument readers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pyramids_eo.errors import CalibrationError, ReaderError
from pyramids_eo.sensors.registry import (
    get_sensor,
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)

#: Matches a signed integer/decimal number, optionally in scientific notation.
#: Written without overlapping quantifiers so it has linear (no-backtracking) time.
_NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def finite_float(text: Any) -> float | None:
    """Parse `text` to a float, rejecting non-finite values.

    A CF attribute of `"nan"` or `"inf"` parses happily as a float but is
    meaningless as a scale, offset or validity bound -- a NaN scale silently
    voids a whole array, and an infinite one produces values that evade a NaN
    nodata entirely. Both are treated as unparseable here.

    Args:
        text: The raw attribute value, or `None`.

    Returns:
        The finite float, or `None` when `text` is absent, not a number, or not
        finite.

    Examples:
        - A numeric attribute parses, in plain or scientific notation:
            ```python
            >>> from pyramids_eo.sensors.readers._common import finite_float
            >>> finite_float("0.5")
            0.5
            >>> finite_float("1e-06")
            1e-06

            ```
        - Absent, non-numeric and non-finite values are all rejected:
            ```python
            >>> finite_float(None) is None
            True
            >>> finite_float("nodata") is None
            True
            >>> finite_float("nan") is None, finite_float("inf") is None
            (True, True)

            ```
    """
    if text is None:
        return None
    try:
        value = float(str(text))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def valid_bounds(text: Any) -> tuple[float | None, float | None]:
    """Parse `(min, max)` from a combined CF `valid_range` attribute.

    This is MTG-FCI's spelling of the validity limits (GDAL renders it as a
    `"{lo, hi}"` string). A missing, unparseable or *reversed* range yields
    `(None, None)` so it masks nothing -- a `lo > hi` range would otherwise mask
    every pixel with no diagnostic.

    Args:
        text: The GDAL `valid_range` metadata value, or `None`.

    Returns:
        The `(valid_min, valid_max)` pair, or `(None, None)`.

    Examples:
        - A GDAL-rendered range parses to floats:
            ```python
            >>> from pyramids_eo.sensors.readers._common import valid_bounds
            >>> valid_bounds("{0, 4095}")
            (0.0, 4095.0)

            ```
        - Missing, malformed and reversed ranges all mask nothing:
            ```python
            >>> valid_bounds(None)
            (None, None)
            >>> valid_bounds("not-a-range")
            (None, None)
            >>> valid_bounds("{4, 1}")
            (None, None)

            ```
    """
    if not text:
        return None, None
    numbers = re.findall(_NUMBER, str(text))
    if len(numbers) < 2:
        return None, None
    low, high = finite_float(numbers[0]), finite_float(numbers[-1])
    if low is None or high is None or low > high:
        return None, None
    return low, high


def validity_bounds(meta: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Resolve a variable's validity limits from its CF attributes.

    Prefers the **separate** `valid_min` / `valid_max` scalars, which is what the
    Sentinel-3 L1 products carry (`valid_range` appears in neither the OLCI nor
    the SLSTR specification), and falls back to a combined `valid_range`, which
    is MTG-FCI's spelling. Either bound may be present alone. A reversed pair is
    discarded, as in `valid_bounds`.

    The limits are in the variable's **packed** units -- the published OLCI
    latitude bounds are literally `-90000000`, not `-90.0` -- so they must be
    compared against the raw values, before unpacking.

    Args:
        meta: The variable's GDAL attribute dict.

    Returns:
        The `(valid_min, valid_max)` pair, either element `None` when absent.

    Examples:
        - The Sentinel-3 spelling, as separate scalars:
            ```python
            >>> from pyramids_eo.sensors.readers._common import validity_bounds
            >>> validity_bounds({"valid_min": "-90000000", "valid_max": "90000000"})
            (-90000000.0, 90000000.0)

            ```
        - The MTG-FCI spelling is the fallback, and one bound alone is honoured:
            ```python
            >>> validity_bounds({"valid_range": "{0, 4095}"})
            (0.0, 4095.0)
            >>> validity_bounds({"valid_min": "0"})
            (0.0, None)
            >>> validity_bounds({"units": "K"})
            (None, None)

            ```
    """
    low = finite_float(meta.get("valid_min"))
    high = finite_float(meta.get("valid_max"))
    if low is None and high is None:
        return valid_bounds(meta.get("valid_range"))
    if low is not None and high is not None and low > high:
        return None, None
    return low, high


def invalid_mask(raw: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    """Return a boolean mask, `True` where a raw value is invalid per CF attributes.

    Masks anything outside the variable's validity limits (see
    `validity_bounds`) and anything equal to its `_FillValue`. The mask is
    derived from the **raw**, still-packed values, which is what CF requires
    because both the fill value and the limits are expressed in the packed type.

    Args:
        raw: The raw, still-packed array as read from the file.
        meta: The variable's GDAL attribute dict.

    Returns:
        A boolean array of `raw`'s shape, `True` for an invalid element.

    Examples:
        - Out-of-range values and the fill value are both flagged, against the
          Sentinel-3 `valid_min` / `valid_max` spelling:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.sensors.readers._common import invalid_mask
            >>> raw = np.array([-999, 1, 50, 500], dtype=np.int32)
            >>> meta = {"valid_min": "10", "valid_max": "100", "_FillValue": "-999"}
            >>> invalid_mask(raw, meta).tolist()
            [True, True, False, True]

            ```
        - With no validity attributes at all, nothing is masked:
            ```python
            >>> import numpy as np
            >>> from pyramids_eo.sensors.readers._common import invalid_mask
            >>> invalid_mask(np.array([1, 2, 3]), {}).tolist()
            [False, False, False]

            ```
    """
    invalid = np.zeros(raw.shape, dtype=bool)
    low, high = validity_bounds(meta)
    if low is not None:
        invalid |= raw < low
    if high is not None:
        invalid |= raw > high
    fill = finite_float(meta.get("_FillValue"))
    if fill is not None:
        invalid |= raw == fill
    return invalid


def resolve_channels(
    channel: str | None, channels: Sequence[str] | None, reader: str
) -> tuple[list[str], bool]:
    """Normalise the `channel` / `channels` arguments a multi-channel reader takes.

    A reader accepts exactly one of a single `channel` (returning a `Dataset`) or a
    `channels` sequence (returning a `dict[str, Dataset]`). This validates that
    exactly one was given and returns the channel list plus a flag for which shape
    the caller asked for.

    Args:
        channel: A single channel identifier, or `None`.
        channels: A sequence of channel identifiers, or `None`.
        reader: The calling reader's name, for error messages.

    Returns:
        A `(channel_list, single)` pair — `single` is `True` when a lone `channel`
        was given (the caller wants one `Dataset`), `False` for a `channels`
        sequence (the caller wants a `dict`).

    Raises:
        ReaderError: When both or neither of `channel` / `channels` are given;
            when `channels` is a bare `str` / `bytes` rather than a sequence of
            names (which would otherwise be read one character at a time); when
            `channels` is empty; or when `channels` names the same channel twice.
    """
    if (channel is None) == (channels is None):
        raise ReaderError(f"{reader}: pass exactly one of `channel` or `channels`")
    if channels is None:
        return [channel], True  # type: ignore[list-item]
    if isinstance(channels, (str, bytes)):
        raise ReaderError(
            f"{reader}: `channels` must be a sequence of channel names, not a single "
            f"string; use `channel=` for one channel"
        )
    requested = list(channels)
    if not requested:
        raise ReaderError(f"{reader}: `channels` is empty")
    if len(set(requested)) != len(requested):
        duplicates = sorted({c for c in requested if requested.count(c) > 1})
        raise ReaderError(f"{reader}: duplicate channels in `channels`: {duplicates}")
    return requested, False


def calibrate_channel(
    radiance: np.ndarray,
    channel: str,
    sensor: str,
    sun_earth_distance: float,
    cos_sza: Any,
    *,
    coeffs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Calibrate raw radiance for `channel` to a physical quantity.

    Applies the conversion the channel's radiometric kind needs: reflectance for
    a solar channel, brightness temperature for a thermal one. Each constant is
    taken from `coeffs` (the per-granule metadata) when present there, otherwise
    from the bundled registry table as a fallback — so a reader that carries the
    granule's own coefficients gets granule-accurate output, while the registry
    supplies nominal values when it does not.

    Args:
        radiance: Raw radiance array.
        channel: Channel identifier (registry key).
        sensor: Sensor name for the registry lookup.
        sun_earth_distance: Sun-earth distance (AU) for a solar channel.
        cos_sza: Cosine of the solar zenith angle, or `None`.
        coeffs: Per-granule calibration coefficients that override the registry —
            any of `kind`, `solar_irradiance`, `central_wavenumber_cm1`, `alpha`,
            `beta`. Missing keys fall back to the registry channel.

    Returns:
        Reflectance (solar) or brightness temperature (thermal).

    Raises:
        CalibrationError: When the channel lacks the constants its kind needs.
        UnknownSensorError: When the sensor / channel is not in the registry.
    """
    ch = get_sensor(sensor).get_channel(channel)
    overrides = coeffs or {}

    def _coef(key: str, fallback: Any) -> Any:
        return overrides[key] if key in overrides else fallback

    if _coef("kind", ch.kind) == "solar":
        solar_irradiance = _coef("solar_irradiance", ch.solar_irradiance)
        if solar_irradiance is None:
            raise CalibrationError(f"solar channel {channel!r} has no solar_irradiance")
        return radiance_to_reflectance(
            radiance, solar_irradiance, sun_earth_distance, cos_sza
        )

    central_wavenumber = _coef("central_wavenumber_cm1", ch.central_wavenumber_cm1)
    if central_wavenumber is None:
        raise CalibrationError(
            f"thermal channel {channel!r} has no central_wavenumber_cm1"
        )
    alpha = _coef("alpha", ch.alpha)
    beta = _coef("beta", ch.beta)
    return radiance_to_brightness_temperature(
        radiance,
        central_wavenumber,
        alpha if alpha is not None else 1.0,
        beta if beta is not None else 0.0,
    )
