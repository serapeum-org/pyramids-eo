"""Sensor / channel metadata loaded from the bundled registry tables.

`get_sensor(name)` loads a sensor's channel table from `registry/data/<name>.yaml`
and returns a `Sensor` of frozen `Channel` records — band wavelength, native
resolution, radiometric `kind` (`solar` / `thermal`), and the nominal calibration
constants a reader falls back to when the granule metadata does not carry them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from pyramids_eo.errors import UnknownSensorError

_DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Channel:
    """One instrument channel and its nominal calibration constants.

    Attributes:
        name: Channel identifier (e.g. `"ir_105"`).
        wavelength_um: Central wavelength in micrometres.
        resolution_m: Native ground sampling distance in metres.
        kind: `"solar"` (reflective) or `"thermal"` (emissive).
        solar_irradiance: Band-integrated solar irradiance `E0` (W m-2 um-1) for
            a solar channel, else `None`.
        central_wavenumber_cm1: Central wavenumber (cm-1) for a thermal channel,
            else `None`.
        alpha: Thermal band-correction slope, else `None`.
        beta: Thermal band-correction offset (kelvin), else `None`.
    """

    name: str
    wavelength_um: float
    resolution_m: int
    kind: str
    solar_irradiance: float | None = None
    central_wavenumber_cm1: float | None = None
    alpha: float | None = None
    beta: float | None = None


@dataclass(frozen=True)
class Sensor:
    """A named sensor and its channel table."""

    name: str
    channels: dict[str, Channel]

    def get_channel(self, name: str) -> Channel:
        """Return the channel named `name`.

        Args:
            name: Channel identifier.

        Returns:
            The matching `Channel`.

        Raises:
            UnknownSensorError: When the channel is not in this sensor's table.
        """
        try:
            return self.channels[name]
        except KeyError as exc:
            raise UnknownSensorError(
                f"{self.name!r} has no channel {name!r}; "
                f"known channels: {sorted(self.channels)}"
            ) from exc

    @property
    def channel_names(self) -> list[str]:
        """Return the sorted channel identifiers."""
        return sorted(self.channels)


@cache
def get_sensor(name: str) -> Sensor:
    """Load and return the `Sensor` named `name` from the bundled tables.

    Args:
        name: Sensor identifier (e.g. `"fci"`, `"seviri"`), matching a
            `registry/data/<name>.yaml` table.

    Returns:
        The `Sensor` with its channel table (cached across calls).

    Raises:
        UnknownSensorError: When no table exists for `name`.
    """
    path = _DATA_DIR / f"{name.lower()}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in _DATA_DIR.glob("*.yaml"))
        raise UnknownSensorError(f"unknown sensor {name!r}; available: {available}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    channels = {
        key: Channel(name=key, **value)
        for key, value in raw.get("channels", {}).items()
    }
    return Sensor(name=raw.get("name", name), channels=channels)
