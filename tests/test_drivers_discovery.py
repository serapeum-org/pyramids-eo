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
from pyramids_eo.drivers.olci import OlciDriver  # noqa: E402
from pyramids_eo.drivers.sentinel2 import Sentinel2Driver  # noqa: E402
from pyramids_eo.drivers.seviri import SeviriDriver  # noqa: E402
from pyramids_eo.drivers.slstr import SlstrDriver  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_driver_cache():
    """Force pyramids to re-discover drivers from entry points for each test."""
    clear_driver_cache()
    yield
    clear_driver_cache()


def test_all_drivers_are_discovered():
    """All six pyramids-eo drivers are found by entry-point discovery (none skipped by a gate)."""
    found = raster_drivers()
    for name in ("earthengine", "seviri", "s2", "fci", "olci", "slstr"):
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


def test_olci_scheme_selects_the_olci_driver():
    """A bare `olci://` source resolves to OlciDriver via can_open discovery."""
    assert select_driver("olci://granule.SEN3") is OlciDriver


def test_slstr_scheme_selects_the_slstr_driver():
    """A bare `slstr://` source resolves to SlstrDriver via can_open discovery."""
    assert select_driver("slstr://granule.SEN3") is SlstrDriver


def test_sen3_granule_content_disambiguates_olci_from_slstr(tmp_path):
    """OLCI and SLSTR share the .SEN3 suffix, so content decides which driver claims it."""
    olci = tmp_path / "S3A_OL_1_EFR.SEN3"
    olci.mkdir()
    (olci / "Oa08_radiance.nc").write_bytes(b"")
    slstr = tmp_path / "S3A_SL_1_RBT.SEN3"
    slstr.mkdir()
    (slstr / "S5_radiance_an.nc").write_bytes(b"")
    assert select_driver(str(olci)) is OlciDriver
    assert select_driver(str(slstr)) is SlstrDriver


def test_explicit_name_resolves_each_driver():
    """`driver="<name>"` resolves each pyramids-eo driver by its registry name."""
    assert select_driver("anything", "earthengine") is EarthEngineDriver
    assert select_driver("anything", "seviri") is SeviriDriver
    assert select_driver("anything", "s2") is Sentinel2Driver
    assert select_driver("anything", "fci") is FciDriver
    assert select_driver("anything", "olci") is OlciDriver
    assert select_driver("anything", "slstr") is SlstrDriver


def test_a_dual_content_sen3_resolves_to_the_olci_driver(tmp_path):
    """A `.SEN3` holding both OLCI and SLSTR files resolves to olci, stably.

    Both drivers claim the suffix and both `can_open` predicates return True for
    such a directory, so `select_driver` takes the first claimant in registration
    order -- which is alphabetical. The outcome is documented; this pins it, so a
    future registration change cannot flip it silently.
    """
    granule = tmp_path / "S3A_BOTH.SEN3"
    granule.mkdir()
    (granule / "Oa08_radiance.nc").write_bytes(b"")
    (granule / "S5_radiance_an.nc").write_bytes(b"")
    assert OlciDriver.can_open(str(granule)) is True
    assert SlstrDriver.can_open(str(granule)) is True
    assert select_driver(str(granule)) is OlciDriver


def test_an_empty_sen3_is_claimed_by_neither_driver(tmp_path):
    """An empty `.SEN3` is claimed by neither, so read_file falls back to GDAL.

    Documented behaviour: claiming on content means an incomplete granule gets
    GDAL's generic error rather than the readers' "product looks incomplete"
    message. Pinned so the trade-off is visible if it ever changes.
    """
    granule = tmp_path / "S3A_EMPTY.SEN3"
    granule.mkdir()
    assert OlciDriver.can_open(str(granule)) is False
    assert SlstrDriver.can_open(str(granule)) is False
    assert select_driver(str(granule)) is None
