"""MSG-SEVIRI native (`.nat`) reader.

`read_seviri` calibrates a SEVIRI channel to reflectance / brightness
temperature (via the sensor registry) and returns a geolocated pyramids
`Dataset`.

.. warning::
    The MSG native (`.nat`) format is a **packed binary** (a 15-record header
    followed by line records of 10-bit packed counts) documented in the EUMETSAT
    MSG Level-1.5 native-format spec. A faithful binary parser is **not** bundled
    here: the default `parse` raises `NotImplementedError`. Pass a `parse`
    callable (or a pyramids `Dataset` already holding the channel radiance) —
    the calibrate + geolocate orchestration below is real and unit-tested; only
    the raw `.nat` byte decoding needs the format spec + a real granule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids_eo.errors import ReaderError
from pyramids_eo.readers._common import calibrate_channel


def _default_parse(path: Any, channel: str) -> Any:
    """Placeholder `.nat` parser — not implemented (needs the format spec).

    Args:
        path: Path to a SEVIRI `.nat` file.
        channel: Channel identifier.

    Raises:
        NotImplementedError: Always — supply a `parse` callable to `read_seviri`.
    """
    raise NotImplementedError(
        "SEVIRI native (.nat) byte decoding is not bundled; pass read_seviri a "
        "`parse` callable that returns the channel radiance as a pyramids Dataset "
        "(see the module warning)."
    )


def read_seviri(
    source: Any,
    channel: str,
    *,
    sensor: str = "seviri",
    calibrate: bool = True,
    sun_earth_distance: float = 1.0,
    cos_sza: Any = None,
    coeffs: dict[str, Any] | None = None,
    parse: Any = None,
) -> Any:
    """Read one SEVIRI channel into a calibrated, geolocated `Dataset`.

    Calibrates the channel's radiance to reflectance (solar) or brightness
    temperature (thermal) via the registry and returns a pyramids `Dataset`
    carrying the source's CRS + geotransform.

    Args:
        source: A pyramids `Dataset` already holding the channel radiance, or a
            value accepted by `parse` (by default a `.nat` path).
        channel: Channel identifier (e.g. `"IR_108"`, `"VIS006"`).
        sensor: Registry sensor name (default `"seviri"`).
        calibrate: When `True` (default), calibrate to a physical quantity; when
            `False`, return the raw radiance.
        sun_earth_distance: Sun-earth distance (AU) for solar-channel reflectance.
        cos_sza: Cosine of the solar zenith angle for the reflectance sun-angle
            correction, or `None`.
        coeffs: Per-granule calibration coefficients preferred over the registry
            fallback (see `calibrate_channel`), or `None` to use the registry.
        parse: Callable `(source, channel) -> Dataset` used when `source` is not
            already a `Dataset`. Defaults to a not-implemented native parser (see
            the module warning).

    Returns:
        A pyramids `Dataset` of the calibrated (or raw) channel.

    Raises:
        ReaderError: When `source` is `None`.
        NotImplementedError: When a non-Dataset source is given without a `parse`.
        CalibrationError: When a channel lacks the constants its kind needs.
        UnknownSensorError: When the sensor / channel is not in the registry.
    """
    if source is None:
        raise ReaderError("read_seviri: source is required")

    parser = parse or _default_parse
    dataset = source if hasattr(source, "read_array") else parser(source, channel)
    radiance = np.asarray(dataset.read_array(), dtype=float)
    data = (
        calibrate_channel(
            radiance, channel, sensor, sun_earth_distance, cos_sza, coeffs=coeffs
        )
        if calibrate
        else radiance
    )

    from pyramids.dataset import Dataset

    return Dataset.create_from_array(data, geo=dataset.geotransform, epsg=dataset.epsg)
