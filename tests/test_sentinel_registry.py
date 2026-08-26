"""The Sentinel-2 MSI registry table loads and is self-consistent."""

from __future__ import annotations

from pyramids_eo.sensors.registry import get_sensor


def test_msi_registry_loads():
    sensor = get_sensor("msi")
    assert sensor.name == "msi"
    # 13 MSI bands, all reflective.
    assert len(sensor.channel_names) == 13
    for name in sensor.channel_names:
        assert sensor.get_channel(name).kind == "solar"


def test_msi_band_metadata_is_sane():
    sensor = get_sensor("msi")
    b04 = sensor.get_channel("B04")
    assert b04.resolution_m == 10
    assert 0.6 < b04.wavelength_um < 0.7  # red ~0.665 µm
    assert b04.solar_irradiance is not None

    # The three 60 m atmospheric bands.
    sixty = {
        n for n in sensor.channel_names if sensor.get_channel(n).resolution_m == 60
    }
    assert sixty == {"B01", "B09", "B10"}
