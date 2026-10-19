"""Discovery/dispatch wiring for the pyramids-eo raster drivers.

These exercise pyramids' real entry-point discovery (`pyramids.raster_drivers`) and source
selection (`select_driver`) against this package's installed `pyramids.raster_drivers`
entry points, so a driver that is registered but silently skipped at discovery — e.g. gated on
an uninstalled backend — is caught here rather than only in production.

`raster_drivers` / `select_driver` / `clear_driver_cache` are pyramids-internal (no public
re-export yet); they are imported here purely to assert the discovery wiring.
"""

import pytest

pytest.importorskip("pyramids.dataset._drivers")
from pyramids.dataset._drivers import (  # noqa: E402
    clear_driver_cache,
    raster_drivers,
    select_driver,
)

from pyramids_eo.drivers.earthengine import EarthEngineDriver  # noqa: E402
from pyramids_eo.drivers.fci import FciDriver  # noqa: E402
from pyramids_eo.drivers.sentinel2 import Sentinel2Driver  # noqa: E402
from pyramids_eo.drivers.seviri import SeviriDriver  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_driver_cache():
    """Force pyramids to re-discover drivers from entry points for each test."""
    clear_driver_cache()
    yield
    clear_driver_cache()


def test_all_drivers_are_discovered():
    """All four pyramids-eo drivers are found by entry-point discovery (none skipped by a gate)."""
    found = raster_drivers()
    for name in ("earthengine", "seviri", "s2", "fci"):
        assert name in found, f"{name} missing from discovered drivers: {sorted(found)}"


def test_ee_scheme_selects_the_earthengine_driver():
    """A bare `ee://` source resolves to EarthEngineDriver via can_open discovery."""
    assert select_driver("ee://COPERNICUS/S2_SR_HARMONIZED") is EarthEngineDriver


def test_nat_suffix_selects_the_seviri_driver():
    """A bare `.nat` source resolves to SeviriDriver via can_open discovery."""
    assert select_driver("scene.nat") is SeviriDriver


def test_safe_suffix_selects_the_sentinel2_driver():
    """A `.SAFE` source resolves to Sentinel2Driver via can_open discovery."""
    assert select_driver("S2A_MSIL2A.SAFE") is Sentinel2Driver


def test_fci_scheme_selects_the_fci_driver():
    """An `fci://` source resolves to FciDriver via can_open discovery."""
    assert select_driver("fci://granule") is FciDriver


def test_explicit_name_resolves_each_driver():
    """`driver="<name>"` resolves each pyramids-eo driver by its registry name."""
    assert select_driver("anything", "earthengine") is EarthEngineDriver
    assert select_driver("anything", "seviri") is SeviriDriver
    assert select_driver("anything", "s2") is Sentinel2Driver
    assert select_driver("anything", "fci") is FciDriver
