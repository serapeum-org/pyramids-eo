"""Tasseled-cap (Kauth-Thomas) coefficient tables from the bundled registry.

`get_tasseled_cap(sensor)` loads a sensor's tasseled-cap coefficients from
`sensors/registry/data/tasseled_cap.yaml` and returns a frozen
`TasseledCapCoefficients` -- the band order the coefficients were derived for,
the band-name aliases that let a transform recognise a `Dataset`'s bands, and the
per-component coefficient rows. The coefficients are published constants; see the
`reference` on each record and the provenance note in the YAML.

The coefficient tables are sensor reference data (which bands, in which order,
with which constants), so they live here with the channel tables; the transform
that *applies* them to a raster is `pyramids_eo.spectral.tasseled_cap`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from pyramids_eo.errors import UnknownSensorError

_DATA_FILE = Path(__file__).parent / "data" / "tasseled_cap.yaml"


@dataclass(frozen=True)
class TasseledCapCoefficients:
    """One sensor's tasseled-cap coefficient table.

    Attributes:
        sensor: Sensor identifier (e.g. `"landsat8_oli"`).
        bands: The band roles, in the order the coefficients were derived for
            (e.g. `("blue", "green", "red", "nir", "swir1", "swir2")`).
        components: Maps each component name (e.g. `"brightness"`) to its
            coefficients, one per band in `bands` order.
        aliases: Maps each band role to the band-name spellings that denote it,
            so a transform can match a `Dataset`'s band names to roles.
        reference: Citation for the coefficients.
        domain: The reflectance domain the coefficients apply to (e.g.
            `"at-satellite reflectance"`).
    """

    sensor: str
    bands: tuple[str, ...]
    components: Mapping[str, tuple[float, ...]]
    aliases: Mapping[str, tuple[str, ...]]
    reference: str
    domain: str

    @property
    def component_names(self) -> list[str]:
        """Return the component names this table defines, sorted.

        Returns:
            The sorted component identifiers (e.g. `["brightness", "greenness",
            "wetness"]`).

        Examples:
            - The classic three components of a Landsat table, sorted:
                ```python
                >>> from pyramids_eo.sensors.registry import get_tasseled_cap
                >>> get_tasseled_cap("landsat8_oli").component_names
                ['brightness', 'greenness', 'wetness']

                ```
        """
        return sorted(self.components)

    def role_of(self, band_name: str) -> str | None:
        """Return the band role `band_name` denotes, or `None` if unrecognised.

        Matching is case-insensitive and spans the canonical role names in
        `bands` plus every spelling in `aliases`.

        Args:
            band_name: A band name from a `Dataset` (e.g. `"B4"`, `"red"`).

        Returns:
            The canonical role (e.g. `"red"`), or `None` when `band_name`
            matches no role of this sensor.

        Examples:
            - A canonical role, an alias, and an unknown name:
                ```python
                >>> from pyramids_eo.sensors.registry import get_tasseled_cap
                >>> oli = get_tasseled_cap("landsat8_oli")
                >>> oli.role_of("red")
                'red'
                >>> oli.role_of("B5")
                'nir'
                >>> oli.role_of("thermal") is None
                True

                ```
        """
        key = band_name.strip().lower()
        for role in self.bands:
            if key == role.lower() or any(
                key == alias.lower() for alias in self.aliases.get(role, ())
            ):
                return role
        return None


@cache
def _load() -> Mapping[str, Any]:
    """Read and cache the raw tasseled-cap YAML (parsed once per process).

    Returns:
        The parsed mapping of sensor identifier -> raw coefficient block.
    """
    return yaml.safe_load(_DATA_FILE.read_text(encoding="utf-8")) or {}


def get_tasseled_cap(sensor: str) -> TasseledCapCoefficients:
    """Load and return the `TasseledCapCoefficients` for `sensor`.

    Args:
        sensor: Sensor identifier (e.g. `"landsat8_oli"`, `"landsat5_tm"`),
            matching a top-level key in `sensors/registry/data/tasseled_cap.yaml`.
            Case-insensitive.

    Returns:
        The frozen coefficient table for `sensor` (cached across calls, so any
        casing of the same sensor returns the identical object).

    Raises:
        UnknownSensorError: When no block exists for `sensor`.
        ValueError: When the bundled table is malformed -- a component row whose
            length does not match the number of bands.

    Examples:
        - Load a sensor and read its band order and a coefficient:
            ```python
            >>> from pyramids_eo.sensors.registry import get_tasseled_cap
            >>> oli = get_tasseled_cap("landsat8_oli")
            >>> oli.bands
            ('blue', 'green', 'red', 'nir', 'swir1', 'swir2')
            >>> oli.components["brightness"][0]
            0.3029

            ```
        - The lookup is case-insensitive and cites its source:
            ```python
            >>> from pyramids_eo.sensors.registry import get_tasseled_cap
            >>> tm = get_tasseled_cap("LANDSAT5_TM")
            >>> tm.sensor
            'landsat5_tm'
            >>> tm.reference
            'Crist (1985), Remote Sensing of Environment 17(3):301-306.'

            ```
    """
    # Normalise before the cached lookup so different casings share one entry
    # (and return the same object) rather than caching a duplicate per spelling.
    return _get_tasseled_cap(sensor.lower())


@cache
def _get_tasseled_cap(key: str) -> TasseledCapCoefficients:
    """Build (and cache) the table for an already-lowercased sensor `key`.

    Args:
        key: The lowercased sensor identifier.

    Returns:
        The frozen coefficient table.

    Raises:
        UnknownSensorError: When no block exists for `key`.
        ValueError: When a component row length does not match the band count.
    """
    raw_all = _load()
    if key not in raw_all:
        raise UnknownSensorError(
            f"unknown tasseled-cap sensor {key!r}; available: {sorted(raw_all)}"
        )
    raw = raw_all[key]
    bands = tuple(raw["bands"])
    components = {
        name: tuple(float(c) for c in coeffs)
        for name, coeffs in raw["components"].items()
    }
    for name, coeffs in components.items():
        if len(coeffs) != len(bands):
            # Deliberately a bare ValueError, not an EOError: a row/band mismatch
            # can only arise from corrupt *bundled* data (a packaging bug, not a
            # user-recoverable condition), so it is signalled outside the
            # user-facing EOError family rather than caught via `except EOError`.
            raise ValueError(
                f"tasseled_cap[{key}]: component {name!r} has {len(coeffs)} "
                f"coefficients but the sensor has {len(bands)} bands {bands}"
            )
    aliases = {role: tuple(raw.get("aliases", {}).get(role, ())) for role in bands}
    return TasseledCapCoefficients(
        sensor=key,
        bands=bands,
        components=MappingProxyType(components),
        aliases=MappingProxyType(aliases),
        reference=raw.get("reference", ""),
        domain=raw.get("domain", ""),
    )


def available_tasseled_cap_sensors() -> list[str]:
    """Return the sensor identifiers that have a bundled tasseled-cap table.

    Returns:
        The sorted sensor keys from `tasseled_cap.yaml` (e.g. `["landsat5_tm",
        "landsat7_etm", "landsat8_oli"]`).

    Examples:
        - List the sensors that have a bundled tasseled-cap table:
            ```python
            >>> from pyramids_eo.sensors.registry import available_tasseled_cap_sensors
            >>> available_tasseled_cap_sensors()
            ['landsat5_tm', 'landsat7_etm', 'landsat8_oli']

            ```
    """
    return sorted(_load())
