"""Unit tests for `pyramids_eo.sensors.readers.slstr`.

As for OLCI, the granule fixtures are real NetCDF files written through pyramids' bundled
GDAL netCDF driver (see `tests/_sentinel3_granule`). These focus on what SLSTR adds over
OLCI: the dual view (nadir / oblique) and the per-channel grids, including the discovery
fallback when a channel sits on a non-default grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers.slstr import (
    _canonical_channel,
    _resolve_view,
    read_slstr,
)
from tests._sentinel3_granule import (
    expected_radiance,
    packed_counts,
    slstr_granule,
    write_variable,
)


class TestCanonicalChannel:
    """Tests for `_canonical_channel`."""

    @pytest.mark.parametrize(
        "given, expected",
        [("S5", "S5"), ("s5", "S5"), ("F1", "F1"), ("f2", "F2"), ("  s9  ", "S9")],
    )
    def test_accepted_spellings(self, given, expected):
        """Channel case is normalised to the canonical product spelling."""
        assert _canonical_channel(given, "read_slstr") == expected

    @pytest.mark.parametrize("given", ["S0", "S10", "F3", "Oa08", "", "X1", 5])
    def test_rejected_spellings(self, given):
        """A channel the instrument does not have is rejected."""
        with pytest.raises(ReaderError, match="is not an SLSTR channel"):
            _canonical_channel(given, "read_slstr")


class TestResolveView:
    """Tests for `_resolve_view`."""

    @pytest.mark.parametrize(
        "given, expected",
        [("nadir", "n"), ("n", "n"), ("NADIR", "n"), ("oblique", "o"), ("o", "o")],
    )
    def test_accepted_views(self, given, expected):
        """Both the long and short view spellings resolve, case-insensitively."""
        assert _resolve_view(given, "read_slstr") == expected

    @pytest.mark.parametrize("given", ["sideways", "", "nad", 1])
    def test_rejected_views(self, given):
        """An unknown view is rejected rather than guessed."""
        with pytest.raises(ReaderError, match="is not an SLSTR view"):
            _resolve_view(given, "read_slstr")


class TestReadSlstr:
    """Tests for `read_slstr` against synthetic `.SEN3` granules."""

    def test_decodes_a_solar_channel_on_the_a_grid(self, tmp_path):
        """A solar channel (S1-S6) is read from its 500 m `a` grid file and unpacked."""
        counts = packed_counts()
        granule = slstr_granule(tmp_path, channels={"S5": counts})
        array = np.asarray(read_slstr(granule, channel="S5").read_array(), dtype=float)
        assert np.allclose(array, expected_radiance(counts), equal_nan=True), array

    def test_decodes_a_thermal_channel_on_the_i_grid(self, tmp_path):
        """A thermal channel (S7-S9) is read from its 1 km `i` grid BT file."""
        granule = slstr_granule(tmp_path, channels={"S8": packed_counts()})
        dataset = read_slstr(granule, channel="S8")
        assert dataset.band_names == ["S8"], dataset.band_names
        assert dataset.raster.GetMetadata()["grid"] == "in", (
            dataset.raster.GetMetadata()
        )

    def test_fire_channel_on_the_f_grid(self, tmp_path):
        """A fire channel (F1-F2) is read from its `f` grid BT file."""
        granule = slstr_granule(tmp_path, channels={"F1": packed_counts()})
        dataset = read_slstr(granule, channel="F1")
        assert dataset.raster.GetMetadata()["grid"] == "fn", (
            dataset.raster.GetMetadata()
        )

    def test_oblique_view_reads_the_oblique_files(self, tmp_path):
        """`view="oblique"` reads the `*_ao` / `*_io` files and records the view."""
        granule = slstr_granule(tmp_path, channels={"S5": packed_counts()}, view="o")
        dataset = read_slstr(granule, channel="S5", view="oblique")
        meta = dataset.raster.GetMetadata()
        assert meta["view"] == "oblique", meta
        assert meta["grid"] == "ao", meta

    def test_view_mismatch_raises(self, tmp_path):
        """Asking for a view the granule does not hold is a clear error."""
        granule = slstr_granule(tmp_path, channels={"S5": packed_counts()}, view="n")
        with pytest.raises(ReaderError, match="oblique view"):
            read_slstr(granule, channel="S5", view="oblique")

    def test_geodetic_domain_matches_the_grid(self, tmp_path):
        """The GEOLOCATION domain points at the geodetic file for the grid actually read."""
        granule = slstr_granule(tmp_path, channels={"S8": packed_counts()})
        domain = read_slstr(granule, channel="S8").raster.GetMetadata("GEOLOCATION")
        assert domain["X_DATASET"].endswith('geodetic_in.nc":longitude_in'), domain
        assert domain["Y_DATASET"].endswith('geodetic_in.nc":latitude_in'), domain

    def test_discovery_fallback_finds_a_non_default_grid(self, tmp_path):
        """A channel written on the alternative `b` grid is still found by discovery."""
        granule = tmp_path / "S3A_SL_1_RBT.SEN3"
        granule.mkdir()
        write_variable(granule / "S5_radiance_bn.nc", "S5_radiance_bn", packed_counts())
        write_variable(
            granule / "geodetic_bn.nc",
            "longitude_bn",
            np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype="float32"),
            scale=None,
            offset=None,
            fill=None,
        )
        dataset = read_slstr(granule, channel="S5")
        assert dataset.raster.GetMetadata()["grid"] == "bn", (
            dataset.raster.GetMetadata()
        )

    def test_explicit_grid_selects_a_specific_file(self, tmp_path):
        """`grid=` picks one grid when a channel exists on several."""
        granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
        write_variable(granule / "S5_radiance_bn.nc", "S5_radiance_bn", packed_counts())
        dataset = read_slstr(granule, channel="S5", grid="a", geolocation=False)
        assert dataset.raster.GetMetadata()["grid"] == "an", (
            dataset.raster.GetMetadata()
        )

    def test_ambiguous_grid_raises(self, tmp_path):
        """A channel present on several grids without `grid=` is a clear error."""
        granule = tmp_path / "S3A_SL_1_RBT.SEN3"
        granule.mkdir()
        for grid in ("bn", "cn"):
            write_variable(
                granule / f"S5_radiance_{grid}.nc",
                f"S5_radiance_{grid}",
                packed_counts(),
            )
        with pytest.raises(ReaderError, match="ambiguous"):
            read_slstr(granule, channel="S5")

    def test_channels_returns_a_dict_across_grids(self, tmp_path):
        """A channel set spanning grids returns a dict (shapes may differ)."""
        granule = slstr_granule(
            tmp_path, channels={"S5": packed_counts(), "S8": packed_counts()}
        )
        result = read_slstr(granule, channels=["S5", "s8"])
        assert sorted(result) == ["S5", "S8"], sorted(result)

    def test_channels_rejects_aliases_of_one_channel(self, tmp_path):
        """Two spellings of one channel are rejected, not collapsed to one entry."""
        granule = slstr_granule(tmp_path)
        with pytest.raises(ReaderError, match="same channel more than once"):
            read_slstr(granule, channels=["S5", "s5"])

    def test_metadata_records_sensor_and_registry_fields(self, tmp_path):
        """The result carries CF units plus the registry wavelength / kind."""
        granule = slstr_granule(tmp_path, channels={"S8": packed_counts()})
        meta = read_slstr(granule, channel="S8").raster.GetMetadata()
        assert meta["sensor"] == "slstr", meta
        assert meta["units"] == "K", meta
        assert meta["kind"] == "thermal", meta
        assert meta["wavelength_um"] == "10.854", meta

    def test_missing_geodetic_file_raises(self, tmp_path):
        """Requesting geolocation without the grid's geodetic file is an error."""
        granule = slstr_granule(tmp_path, geodetic=False)
        with pytest.raises(ReaderError, match="geodetic_an.nc"):
            read_slstr(granule, channel="S5")

    def test_geolocation_false_omits_the_domain(self, tmp_path):
        """`geolocation=False` skips the geodetic file entirely."""
        granule = slstr_granule(tmp_path, geodetic=False)
        dataset = read_slstr(granule, channel="S5", geolocation=False)
        assert not dataset.raster.GetMetadata("GEOLOCATION"), "domain should be empty"

    def test_missing_channel_raises(self, tmp_path):
        """A channel the granule does not carry is a clear error naming the expectation."""
        granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
        with pytest.raises(ReaderError, match="no file for channel 'S8'"):
            read_slstr(granule, channel="S8")

    def test_requires_exactly_one_of_channel_or_channels(self, tmp_path):
        """Neither or both of channel / channels is an error."""
        granule = slstr_granule(tmp_path)
        with pytest.raises(ReaderError, match="exactly one"):
            read_slstr(granule)

    def test_explicit_grid_that_matches_nothing_raises(self, tmp_path):
        """An explicit `grid=` naming a grid the granule lacks reports what it does have."""
        granule = tmp_path / "S3A_SL_1_RBT.SEN3"
        granule.mkdir()
        for grid in ("bn", "cn"):
            write_variable(
                granule / f"S5_radiance_{grid}.nc",
                f"S5_radiance_{grid}",
                packed_counts(),
            )
        with pytest.raises(ReaderError, match="does not match any of"):
            read_slstr(granule, channel="S5", grid="a")

    def test_registry_lookup_failure_still_yields_metadata(self, tmp_path, monkeypatch):
        """An unavailable registry degrades to CF-only metadata rather than failing the read."""
        from pyramids_eo.sensors import registry

        def _boom(_sensor):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(registry, "get_sensor", _boom)
        granule = slstr_granule(tmp_path, channels={"S8": packed_counts()})
        meta = read_slstr(granule, channel="S8").raster.GetMetadata()
        assert meta["channel"] == "S8", meta
        assert "wavelength_um" not in meta, (
            f"registry fields should be absent when lookup fails: {meta}"
        )
