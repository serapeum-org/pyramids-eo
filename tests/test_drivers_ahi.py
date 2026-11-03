"""Tests for the Himawari AHI raster driver.

The driver delegates to `read_ahi`; the facade is monkeypatched to capture the call where
only the forwarding matters, and the real reader is used end-to-end where the result does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pyramids_eo.sensors.readers as readers
from pyramids_eo.drivers.ahi import AhiDriver
from tests import _hsd_granule as fixture


class TestCanOpen:
    """`can_open` claims HSD granules by name, not by the generic `.dat` suffix."""

    @pytest.mark.parametrize(
        "name",
        [
            "HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT",
            "HS_H09_20240101_0000_B03_FLDK_R05_S0110.DAT",
            "HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT.bz2",
            "HS_H08_20150415_0230_B07_JP01_R10_S0101.dat",
            "/data/himawari/HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT",
            r"C:\data\HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT",
        ],
    )
    def test_claims_hsd_names(self, name):
        assert AhiDriver.can_open(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "scene.dat",  # the generic suffix must NOT be claimed
            "data.DAT",
            "HS_H08_20150415_0230_B13_FLDK_R20_S0101.nc",
            "HS_H08_B13.DAT",  # incomplete convention
            "XX_H08_20150415_0230_B13_FLDK_R20_S0101.DAT",
            "HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT.gz",
            "scene.tif",
            "",
        ],
    )
    def test_does_not_claim_other_names(self, name):
        assert AhiDriver.can_open(name) is False

    @pytest.mark.parametrize(
        "source", [None, 3, 3.5, b"bytes", ["a"], {"a": 1}, Path("x.DAT")]
    )
    def test_non_string_sources_are_declined_not_raised(self, source):
        assert AhiDriver.can_open(source) is False

    def test_answers_for_a_path_that_does_not_exist(self, tmp_path):
        """The read_file fallback asks about paths that may not be there."""
        assert AhiDriver.can_open(str(tmp_path / fixture.hsd_filename())) is True

    def test_never_touches_the_filesystem(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("can_open must not touch the filesystem")

        monkeypatch.setattr(Path, "exists", _boom)
        monkeypatch.setattr(Path, "iterdir", _boom)
        monkeypatch.setattr("builtins.open", _boom)
        assert AhiDriver.can_open(fixture.hsd_filename()) is True


class TestContract:
    """The duck-typed raster-driver contract."""

    def test_conforms_to_the_contract(self):
        from pyramids.dataset.driver_conformance import assert_raster_driver_conforms

        assert_raster_driver_conforms(AhiDriver)

    def test_declares_its_identity(self):
        assert AhiDriver.name == "ahi"
        assert AhiDriver.suffixes == (".dat",)
        assert AhiDriver.schemes == ()
        assert AhiDriver.capabilities == frozenset({"read"})
        assert AhiDriver.extra is None


class TestRead:
    """Option and band forwarding."""

    def test_forwards_the_source_and_band(self, tmp_path, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            readers,
            "read_ahi",
            lambda s, b=None, **k: captured.update(source=s, band=b, **k),
        )
        path = str(tmp_path / fixture.hsd_filename())
        AhiDriver().read(path, bands=["B13"])
        assert captured["source"] == path
        assert captured["band"] == "B13"

    def test_band_from_options_is_equivalent(self, tmp_path, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            readers, "read_ahi", lambda s, b=None, **k: captured.update(band=b)
        )
        AhiDriver().read(
            str(tmp_path / fixture.hsd_filename()), options={"band": "B07"}
        )
        assert captured["band"] == "B07"

    def test_band_is_optional(self, tmp_path, monkeypatch):
        """A granule names its own band, so the driver need not be told."""
        captured: dict = {}
        monkeypatch.setattr(
            readers, "read_ahi", lambda s, b=None, **k: captured.update(band=b)
        )
        AhiDriver().read(str(tmp_path / fixture.hsd_filename()))
        assert captured["band"] is None

    def test_rejects_several_bands(self, tmp_path):
        with pytest.raises(ValueError, match="an AHI read is one band"):
            AhiDriver().read(
                str(tmp_path / fixture.hsd_filename()), bands=["B13", "B14"]
            )

    def test_rejects_a_bare_string_band(self, tmp_path):
        with pytest.raises(ValueError, match="must be a list of band names"):
            AhiDriver().read(str(tmp_path / fixture.hsd_filename()), bands="B13")

    def test_ignores_the_generic_read_parameters(self, tmp_path, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            readers, "read_ahi", lambda s, b=None, **k: captured.update(**k)
        )
        AhiDriver().read(
            str(tmp_path / fixture.hsd_filename()),
            bands=["B13"],
            bbox=(1, 2, 3, 4),
            crs="EPSG:4326",
            auth={"token": "x"},
            something_new=1,
        )
        assert captured == {}

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("false", False),
            ("FALSE", False),
            ("no", False),
            ("0", False),
            ("true", True),
            ("yes", True),
            ("1", True),
        ],
    )
    def test_calibrate_is_coerced_from_its_string_spelling(
        self, tmp_path, monkeypatch, spelling, expected
    ):
        """`read_file` normalises options to KEY=VALUE, so "false" must not read as truthy."""
        captured: dict = {}
        monkeypatch.setattr(
            readers, "read_ahi", lambda s, b=None, **k: captured.update(**k)
        )
        AhiDriver().read(
            str(tmp_path / fixture.hsd_filename()),
            bands=["B13"],
            options=[f"calibrate={spelling}"],
        )
        assert captured["calibrate"] is expected

    def test_sun_earth_distance_is_coerced_from_its_string_spelling(
        self, tmp_path, monkeypatch
    ):
        captured: dict = {}
        monkeypatch.setattr(
            readers, "read_ahi", lambda s, b=None, **k: captured.update(**k)
        )
        AhiDriver().read(
            str(tmp_path / fixture.hsd_filename()),
            options=["sun_earth_distance=1.015"],
        )
        assert captured["sun_earth_distance"] == pytest.approx(1.015)

    @pytest.mark.parametrize("bad", ["maybe", "", "2.5x"])
    def test_an_unreadable_boolean_is_reported(self, tmp_path, bad):
        with pytest.raises(ValueError, match="calibrate"):
            AhiDriver().read(
                str(tmp_path / fixture.hsd_filename()), options=[f"calibrate={bad}"]
            )

    def test_an_unreadable_float_is_reported(self, tmp_path):
        with pytest.raises(ValueError, match="sun_earth_distance"):
            AhiDriver().read(
                str(tmp_path / fixture.hsd_filename()),
                options=["sun_earth_distance=soon"],
            )


class TestEndToEnd:
    """The real reader through the driver, and through `read_file`."""

    def test_driver_reads_a_real_fixture(self, tmp_path):
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=4,
            lines=2,
            counts=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
        )
        dataset = AhiDriver().read(str(path), options={"calibrate": False})
        assert dataset.read_array().shape == (2, 4)
        assert "Geostationary" in dataset.crs

    def test_read_file_routes_a_granule_end_to_end(self, tmp_path):
        from pyramids.dataset import Dataset

        path = fixture.write_granule(
            tmp_path, band=13, columns=4, lines=2, slope=1.0, intercept=0.0
        )
        dataset = Dataset.read_file(
            str(path), driver="ahi", open_options={"calibrate": "false"}
        )
        assert dataset.read_array().shape == (2, 4)

    def test_read_file_falls_through_to_the_driver_without_being_told(self, tmp_path):
        """GDAL cannot open HSD, so a bare read_file reaches this driver by name."""
        from pyramids.dataset import Dataset

        path = fixture.write_granule(
            tmp_path, band=13, columns=4, lines=2, slope=1.0, intercept=0.0
        )
        dataset = Dataset.read_file(str(path), open_options={"calibrate": "false"})
        assert dataset.read_array().shape == (2, 4)
