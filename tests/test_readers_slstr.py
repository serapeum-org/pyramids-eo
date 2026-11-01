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
    _resolve_grid,
    _resolve_view,
    available_slstr_channels,
    read_slstr,
)
from tests._sentinel3_granule import (
    expected_radiance,
    packed_counts,
    slstr_granule,
    slstr_pdfs_granule,
    write_geolocation,
    write_measurement,
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
        dataset = read_slstr(granule, channel="S8")
        domain = dataset.raster.GetMetadata("GEOLOCATION")
        assert "longitude_in" in domain["X_DATASET"], domain
        assert "latitude_in" in domain["Y_DATASET"], domain
        from osgeo import gdal

        origin_x = gdal.Warp(
            "", dataset.raster, format="MEM", geoloc=True
        ).GetGeoTransform()[0]
        assert 3.0 < origin_x < 6.0, (
            f"the i-grid domain must warp to degrees: {origin_x}"
        )

    def test_falls_back_to_the_next_documented_grid(self, tmp_path):
        """A channel absent from its first grid is found on the next documented one.

        Test scenario:
            `S5` is documented on `a` then `b`, so a granule carrying only the
            `b` stripe still reads -- without globbing, which is what used to let
            a quality-annotation file be selected as the measurement.
        """
        granule = tmp_path / "S3A_SL_1_RBT.SEN3"
        granule.mkdir()
        write_measurement(
            granule / "S5_radiance_bn.nc", "S5_radiance_bn", packed_counts()
        )
        write_geolocation(
            granule / "geodetic_bn.nc",
            longitude_variable="longitude_bn",
            latitude_variable="latitude_bn",
        )
        dataset = read_slstr(granule, channel="S5")
        assert dataset.raster.GetMetadata()["grid"] == "bn", (
            dataset.raster.GetMetadata()
        )

    def test_explicit_grid_selects_a_specific_file(self, tmp_path):
        """`grid=` picks one grid when a channel exists on several."""
        granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
        write_measurement(
            granule / "S5_radiance_bn.nc", "S5_radiance_bn", packed_counts()
        )
        dataset = read_slstr(granule, channel="S5", grid="a", geolocation=False)
        assert dataset.raster.GetMetadata()["grid"] == "an", (
            dataset.raster.GetMetadata()
        )

    def test_explicit_grid_never_falls_back_to_another(self, tmp_path):
        """`grid=` naming a grid the granule lacks raises instead of substituting one.

        Test scenario:
            With exactly one file matching the channel and view, the old
            discovery returned it unconditionally without consulting `grid`, so
            `grid="i"` (1 km thermal) silently handed back the 500 m `a` grid --
            a different array shape at a different ground resolution, with no
            warning. Branch coverage was satisfied by the two-file case, which
            reached the guard; the realistic single-grid case went untested.
        """
        granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
        for wanted in ("b", "i"):
            with pytest.raises(ReaderError, match="has no 'S5' measurement"):
                read_slstr(granule, channel="S5", grid=wanted, geolocation=False)

    @pytest.mark.parametrize("bad", ["", "..", "*", "zzz", "A/b"])
    def test_invalid_grid_is_rejected(self, tmp_path, bad):
        """An unknown `grid` is rejected rather than interpolated into a filename.

        Test scenario:
            The letter is substituted straight into the measurement filename, so
            an unvalidated value built nonsense paths -- `""` silently meant "the
            default grid", and `".."` / `"*"` produced separator- and
            wildcard-bearing names that then fell through to the glob.
        """
        granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
        with pytest.raises(ReaderError, match="is not an SLSTR grid"):
            read_slstr(granule, channel="S5", grid=bad, geolocation=False)

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
        with pytest.raises(ReaderError, match="has no 'S8' measurement"):
            read_slstr(granule, channel="S8")

    @pytest.mark.parametrize("view", ["nadir", "oblique"])
    def test_every_channel_reads_on_a_pdfs_accurate_granule(self, tmp_path, view):
        """All eleven channels read, in both views, off the real PDFS filename set.

        Test scenario:
            This is the case the invented fixtures could not express. The granule
            reproduces the product composition verbatim -- S1-S3 on `a`, S4-S6 on
            `a`+`b`, S7-S9 on `i`, F1 on `f`, **F2 on `i`** -- with a
            `*_quality_*` sibling beside every measurement. Against it, `F2` used
            to be unreadable: it was mapped to the `f` grid that does not carry
            it, and the discovery glob then matched its quality file, producing an
            ambiguity error for one of the eleven advertised channels.
        """
        granule = slstr_pdfs_granule(tmp_path, views=("n", "o"))
        expected = {
            "S1": "a",
            "S2": "a",
            "S3": "a",
            "S4": "a",
            "S5": "a",
            "S6": "a",
            "S7": "i",
            "S8": "i",
            "S9": "i",
            "F1": "f",
            "F2": "i",
        }
        suffix = "n" if view == "nadir" else "o"
        for channel, grid in expected.items():
            dataset = read_slstr(granule, channel=channel, view=view)
            assert dataset.raster.GetMetadata()["grid"] == f"{grid}{suffix}", (
                f"{channel} ({view}) read off the wrong grid: "
                f"{dataset.raster.GetMetadata()}"
            )

    def test_requires_exactly_one_of_channel_or_channels(self, tmp_path):
        """Neither or both of channel / channels is an error."""
        granule = slstr_granule(tmp_path)
        with pytest.raises(ReaderError, match="exactly one"):
            read_slstr(granule)

    def test_oblique_channels_set_across_grids(self, tmp_path):
        """`view="oblique"` applies to every channel in a `channels` set, across grids.

        Test scenario:
            The view is a per-call argument while the grid is per-channel, so a set
            spanning both grids must read the oblique file of each and report the
            oblique geodetic domain for each.
        """
        granule = slstr_granule(
            tmp_path, channels={"S5": packed_counts(), "S8": packed_counts()}, view="o"
        )
        result = read_slstr(granule, channels=["S5", "S8"], view="oblique")
        assert sorted(result) == ["S5", "S8"], sorted(result)
        assert result["S5"].raster.GetMetadata()["grid"] == "ao", result[
            "S5"
        ].raster.GetMetadata()
        assert result["S8"].raster.GetMetadata()["grid"] == "io", result[
            "S8"
        ].raster.GetMetadata()
        assert (
            "latitude_io" in result["S8"].raster.GetMetadata("GEOLOCATION")["Y_DATASET"]
        ), result["S8"].raster.GetMetadata("GEOLOCATION")

    def test_metadata_geolocation_and_band_names_coexist(self, tmp_path):
        """Setting the CF metadata does not clobber the band name or the GEOLOCATION domain."""
        dataset = read_slstr(slstr_granule(tmp_path), channel="S5")
        assert dataset.band_names == ["S5"], dataset.band_names
        assert dataset.raster.GetMetadata("GEOLOCATION").get("X_DATASET"), (
            "the GEOLOCATION domain must survive the default-domain SetMetadata"
        )
        assert dataset.raster.GetMetadata()["sensor"] == "slstr", (
            dataset.raster.GetMetadata()
        )

    def test_missing_grid_error_reports_what_the_granule_holds(self, tmp_path):
        """The error names the grid tried and lists the measurements that do exist.

        Test scenario:
            A caller asking for a grid the granule lacks needs to see what it
            does carry -- that is how someone discovers F2 is on `i` -- and the
            listing must never offer a `*_quality_*` annotation file as a choice.
        """
        granule = tmp_path / "S3A_SL_1_RBT.SEN3"
        granule.mkdir()
        write_measurement(
            granule / "S5_radiance_bn.nc", "S5_radiance_bn", packed_counts()
        )
        write_measurement(granule / "S5_quality_an.nc", "S5_dL_BB_an", packed_counts())
        with pytest.raises(ReaderError, match="S5_radiance_bn.nc") as error:
            read_slstr(granule, channel="S5", grid="a")
        assert "quality" not in str(error.value), (
            f"a quality file must not be offered as a grid choice: {error.value}"
        )

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


class TestResolveGrid:
    """Tests for `_resolve_grid`."""

    @pytest.mark.parametrize(
        "given, expected",
        [("a", "a"), ("B", "b"), ("i", "i"), ("f", "f"), (" I ", "i"), (None, None)],
    )
    def test_accepted_grids(self, given, expected):
        """The four real grid letters are accepted case-insensitively; None passes through."""
        assert _resolve_grid(given, "read_slstr") == expected

    @pytest.mark.parametrize("given", ["", "c", "zzz", "..", "*", 1])
    def test_rejected_grids(self, given):
        """Anything that is not a documented grid letter is rejected.

        Test scenario:
            `c` is included deliberately: SLSTR has no `c` stripe, though earlier
            fixtures invented one, which is part of why a wrong grid table went
            unnoticed.
        """
        with pytest.raises(ReaderError, match="is not an SLSTR grid"):
            _resolve_grid(given, "read_slstr")


class TestAvailableSlstrChannels:
    """Tests for `available_slstr_channels`."""

    def test_reports_the_real_product_layout(self, tmp_path):
        """A PDFS-accurate granule reports each channel's actual grid(s).

        Test scenario:
            This is how a caller discovers the fire channels are asymmetric --
            F1 on `f`, F2 on `i` -- and which of S4-S6 are duplicated onto `b`.
        """
        granule = slstr_pdfs_granule(tmp_path, views=("n",))
        assert available_slstr_channels(granule) == {
            "S1": ["a"],
            "S2": ["a"],
            "S3": ["a"],
            "S4": ["a", "b"],
            "S5": ["a", "b"],
            "S6": ["a", "b"],
            "S7": ["i"],
            "S8": ["i"],
            "S9": ["i"],
            "F1": ["f"],
            "F2": ["i"],
        }

    def test_reports_only_what_is_present(self, tmp_path):
        """A sparse granule lists only the channels it actually carries."""
        granule = slstr_granule(
            tmp_path, channels={"S5": packed_counts(), "S8": packed_counts()}
        )
        assert available_slstr_channels(granule) == {"S5": ["a"], "S8": ["i"]}

    def test_the_other_view_is_empty_when_absent(self, tmp_path):
        """A nadir-only granule reports nothing for the oblique view."""
        granule = slstr_pdfs_granule(tmp_path, views=("n",))
        assert available_slstr_channels(granule, view="oblique") == {}

    def test_rejects_an_unknown_view(self, tmp_path):
        """An unknown view is rejected, as in the readers."""
        granule = slstr_granule(tmp_path)
        with pytest.raises(ReaderError, match="is not an SLSTR view"):
            available_slstr_channels(granule, view="sideways")
