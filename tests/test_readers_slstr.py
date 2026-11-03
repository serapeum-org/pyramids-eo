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
    _parse_channel,
    _resolve_channel_target,
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

    def test_a_channel_without_its_geodetic_file_is_not_listed(self, tmp_path):
        """A measurement whose geodetic annotation is missing is omitted, not offered.

        Test scenario:
            Every other fixture writes the measurement and its `geodetic_*` sibling
            together, so the two halves of the condition always agree and dropping
            the geodetic half changes nothing that is asserted. Here `S8`'s `i`-grid
            annotation is absent while its measurement is present: listing it would
            advertise a channel `read_slstr` then refuses.
        """
        granule = slstr_granule(
            tmp_path, channels={"S5": packed_counts(), "S8": packed_counts()}
        )
        (granule / "geodetic_in.nc").unlink()
        listed = available_slstr_channels(granule)
        assert listed == {"S5": ["a"]}, (
            f"S8 has no geodetic_in.nc and must not be listed: {listed}"
        )
        with pytest.raises(ReaderError, match="geodetic_in.nc"):
            read_slstr(granule, channel="S8")

    def test_one_grid_of_a_duplicated_channel_drops_out_on_its_own(self, tmp_path):
        """Losing `geodetic_bn.nc` removes only the `b` grid from a channel's list.

        Test scenario:
            `S4`-`S6` are carried on both 500 m grids, so the per-grid filter has
            to act grid by grid rather than per channel -- a whole-channel test
            cannot tell those two readings apart.
        """
        granule = slstr_pdfs_granule(tmp_path, views=("n",))
        (granule / "geodetic_bn.nc").unlink()
        listed = available_slstr_channels(granule)
        assert listed["S5"] == ["a"], f"S5's b grid lost its annotation: {listed['S5']}"
        assert listed["S1"] == ["a"], f"an a-grid-only channel is unaffected: {listed}"

    def test_a_geodetic_file_without_a_measurement_lists_nothing(self, tmp_path):
        """An annotation file alone does not make a channel available.

        Test scenario:
            The complement of the case above: both halves of the condition must be
            required, so a granule carrying only annotations reports no channels.
        """
        granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
        (granule / "S5_radiance_an.nc").unlink()
        assert available_slstr_channels(granule) == {}, (
            "a geodetic file with no measurement must not be listed"
        )


def _geodetic(granule, grid, view="n"):
    """Write the `geodetic_<grid><view>.nc` annotation file for one grid."""
    suffix = f"{grid}{view}"
    return write_geolocation(
        granule / f"geodetic_{suffix}.nc",
        longitude_variable=f"longitude_{suffix}",
        latitude_variable=f"latitude_{suffix}",
    )


class TestFireChannelGrids:
    """The fire channels are asymmetric and F1's grid changed between baselines."""

    def test_f1_on_a_legacy_baseline_granule_falls_back_to_the_i_grid(self, tmp_path):
        """A granule predating the `f` baseline carries F1 on `i`, and must still read.

        Test scenario:
            F1 is documented on `f` *then* `i` precisely because products
            predating the fire-grid baseline ship `F1_BT_in.nc` and no
            `F1_BT_fn.nc`. The whole reason the candidate list is ordered rather
            than a single letter is this granule, and the PDFS fixture -- which
            writes F1 on `f` only -- cannot express it, so the fallback leg was
            reachable but never exercised against a product that needs it.
        """
        granule = tmp_path / "S3A_SL_1_RBT____legacy.SEN3"
        granule.mkdir()
        write_measurement(
            granule / "F1_BT_in.nc", "F1_BT_in", packed_counts(), units="K"
        )
        _geodetic(granule, "i")
        dataset = read_slstr(granule, channel="F1")
        meta = dataset.raster.GetMetadata()
        assert meta["grid"] == "in", f"F1 should fall back to the i grid: {meta}"
        assert meta["resolution_m"] == "1000", f"the i grid is 1 km: {meta}"
        assert (
            "latitude_in" in dataset.raster.GetMetadata("GEOLOCATION")["Y_DATASET"]
        ), "the geodetic file must match the grid actually read"
        assert available_slstr_channels(granule) == {"F1": ["i"]}, (
            "discovery should report the grid the granule really carries"
        )

    def test_f1_on_both_grids_prefers_f_and_honours_an_explicit_i(self, tmp_path):
        """With F1 on `f` and `i`, the documented order picks `f`; `grid="i"` overrides.

        Test scenario:
            A reprocessed granule can hold both stripes. The candidate order has
            to be a real preference rather than "whichever the filesystem lists
            first", and an explicit `grid=` must still win over it.
        """
        granule = tmp_path / "S3A_SL_1_RBT____both.SEN3"
        granule.mkdir()
        for grid in ("f", "i"):
            write_measurement(
                granule / f"F1_BT_{grid}n.nc",
                f"F1_BT_{grid}n",
                packed_counts(),
                units="K",
            )
            _geodetic(granule, grid)
        assert read_slstr(granule, channel="F1").raster.GetMetadata()["grid"] == "fn", (
            "the first documented grid must win when both are present"
        )
        forced = read_slstr(granule, channel="F1", grid="i").raster.GetMetadata()
        assert forced["grid"] == "in", (
            f"grid='i' must override the preference: {forced}"
        )
        assert available_slstr_channels(granule) == {"F1": ["f", "i"]}, (
            "discovery should list both stripes, in documented order"
        )

    def test_f2_is_never_looked_for_on_the_fire_grid(self, tmp_path):
        """`grid="f"` for F2 fails: there is no `F2_BT_f*.nc` in any baseline.

        Test scenario:
            F2 stays on the 1 km `i` grid while F1 moved to `f`. Asking for F2 on
            `f` must be an error naming the grid tried, not a silent fall back to
            `i` at a different nominal resolution.
        """
        granule = slstr_granule(tmp_path, channels={"F2": packed_counts()})
        with pytest.raises(ReaderError, match=r"grid 'f'") as error:
            read_slstr(granule, channel="F2", grid="f", geolocation=False)
        assert "F2_BT_in.nc" in str(error.value), (
            f"the error should point the caller at F2's real `i` stripe: {error.value}"
        )
        assert read_slstr(granule, channel="F2").raster.GetMetadata()["grid"] == "in", (
            "F2's own grid is `i`"
        )


class TestGridResolutionReporting:
    """`resolution_m` must describe the grid read, not the channel's nominal one."""

    @pytest.mark.parametrize(
        "channel, grid, expected",
        [
            ("S5", "a", "500"),
            ("S5", "b", "500"),
            ("S8", "i", "1000"),
            ("F1", "f", "1000"),
        ],
    )
    def test_each_grid_reports_its_own_ground_sampling_distance(
        self, tmp_path, channel, grid, expected
    ):
        """Every documented grid stamps its own GSD onto the result.

        Args:
            channel: The channel to write and read.
            grid: The grid letter to write it on.
            expected: The `resolution_m` the result must carry.

        Test scenario:
            The 500 m `a`/`b` and 1 km `i`/`f` grids are what makes a multi-grid
            read return a dict instead of a stack, so a consumer that resamples
            or harmonises depends on this number being the grid's, not the
            channel's usual one.
        """
        quantity = "radiance" if channel.startswith("S") and channel < "S7" else "BT"
        granule = tmp_path / "S3A_SL_1_RBT____gsd.SEN3"
        granule.mkdir(exist_ok=True)
        stem = f"{channel}_{quantity}_{grid}n"
        write_measurement(granule / f"{stem}.nc", stem, packed_counts())
        _geodetic(granule, grid)
        meta = read_slstr(granule, channel=channel, grid=grid).raster.GetMetadata()
        assert meta["resolution_m"] == expected, (
            f"{channel} on grid {grid!r} should report {expected} m: {meta}"
        )
        assert meta["grid"] == f"{grid}n", meta

    def test_a_channel_read_off_an_unusual_grid_is_labelled_with_that_grid(
        self, tmp_path
    ):
        """A thermal channel found on a 500 m grid reports 500 m, not its nominal 1 km.

        Test scenario:
            S7 is a 1 km `i`-grid channel, so a granule offering it on `a`
            (reprocessed or test data) is exactly the case where taking the
            resolution from the registry instead of the grid would be wrong by a
            factor of two -- and silently, since the array shape alone does not
            say which grid it came from.
        """
        granule = tmp_path / "S3A_SL_1_RBT____offgrid.SEN3"
        granule.mkdir()
        write_measurement(
            granule / "S7_BT_an.nc", "S7_BT_an", packed_counts(), units="K"
        )
        _geodetic(granule, "a")
        meta = read_slstr(granule, channel="S7", grid="a").raster.GetMetadata()
        assert meta["resolution_m"] == "500", (
            f"the grid read, not S7's nominal 1 km, must set the resolution: {meta}"
        )
        assert meta["grid"] == "an", meta


class TestExplicitGridIsNotSubstituted:
    """`grid=` naming a grid real for another channel must not be silently honoured."""

    @pytest.mark.parametrize(
        "channel, grid",
        [("S8", "a"), ("S8", "b"), ("S5", "i"), ("S5", "f"), ("F2", "f")],
    )
    def test_a_grid_real_elsewhere_in_the_product_still_raises(
        self, tmp_path, channel, grid
    ):
        """Every cross-channel grid request fails, naming the grid and what is held.

        Args:
            channel: The channel to request.
            grid: A grid letter that is real for the product but not this channel.

        Test scenario:
            `a`, `b`, `i` and `f` all pass `_resolve_grid`, so each is a
            plausible-looking request that reaches discovery. Since the granule
            holds exactly one file per channel, this is the single-candidate case
            where discovery used to return whatever it found without consulting
            `grid` -- handing back a different resolution and a different array
            shape with no warning.
        """
        granule = slstr_granule(
            tmp_path,
            channels={
                "S5": packed_counts(),
                "S8": packed_counts(),
                "F2": packed_counts(),
            },
        )
        with pytest.raises(ReaderError, match=f"grid '{grid}'") as error:
            read_slstr(granule, channel=channel, grid=grid, geolocation=False)
        assert channel in str(error.value), (
            f"the error should name the channel asked for: {error.value}"
        )
        assert "quality" not in str(error.value), (
            f"no annotation sibling may be offered as a grid choice: {error.value}"
        )


class TestChannelSpellingSymmetry:
    """SLSTR accepts the product's variable spelling, as OLCI accepts `Oa08_radiance`."""

    @pytest.mark.parametrize(
        "given, expected",
        [
            ("S5_radiance_an", "S5"),
            ("S8_BT_in", "S8"),
            ("F1_BT_fo", "F1"),
            ("s5_radiance_bn", "S5"),
        ],
    )
    def test_accepts_the_product_variable_spelling(self, given, expected):
        """A pasted variable or filename stem resolves to its channel."""
        assert _canonical_channel(given, "read_slstr") == expected

    def test_reads_through_the_product_spelling(self, tmp_path):
        """The product spelling works end to end, with grid/view still from the arguments."""
        granule = slstr_granule(tmp_path, channels={"S5": packed_counts()})
        dataset = read_slstr(granule, channel="S5_radiance_an")
        assert dataset.band_names == ["S5"], dataset.band_names

    @pytest.mark.parametrize("given", ["S5_radiance", "S5_reflectance_an", "S5_BT_zz"])
    def test_rejects_a_malformed_suffix(self, given):
        """A suffix that is not a real quantity/grid/view combination is still rejected."""
        with pytest.raises(ReaderError, match="is not an SLSTR channel"):
            _canonical_channel(given, "read_slstr")


class TestSpelledGridAndView:
    """A product variable spelling must select the file it names, not another."""

    @pytest.mark.parametrize(
        "spelling, grid, view",
        [
            ("S5_radiance_an", "an", "nadir"),
            ("S5_radiance_bn", "bn", "nadir"),
            ("S5_radiance_ao", "ao", "oblique"),
            ("S8_BT_io", "io", "oblique"),
            ("F1_BT_fo", "fo", "oblique"),
            ("F2_BT_in", "in", "nadir"),
        ],
    )
    def test_the_spelling_selects_the_file(self, tmp_path, spelling, grid, view):
        """The grid and view encoded in a spelling decide which file is read.

        Args:
            spelling: The product variable name passed as `channel`.
            grid: The grid+view suffix the result must report.
            view: The view the result must report.

        Test scenario:
            Accepting the spelling and then discarding its suffix read a
            *different* file than the caller named -- `S5_radiance_bn` returned
            the `an` grid and `S5_radiance_ao` returned the nadir view -- which is
            the silent-substitution class the `grid=` fix had just closed,
            re-entering through the spelling convenience.
        """
        granule = slstr_pdfs_granule(tmp_path, views=("n", "o"))
        metadata = read_slstr(granule, channel=spelling).raster.GetMetadata()
        assert metadata["grid"] == grid, metadata
        assert metadata["view"] == view, metadata

    def test_a_spelled_grid_conflicting_with_an_explicit_grid_raises(self, tmp_path):
        """A spelling and a `grid=` that disagree is an error, not a silent choice."""
        granule = slstr_pdfs_granule(tmp_path, views=("n",))
        with pytest.raises(ReaderError, match="names grid 'b' but grid='a'"):
            read_slstr(granule, channel="S5_radiance_bn", grid="a")

    def test_a_spelled_view_conflicting_with_an_explicit_view_raises(self, tmp_path):
        """A spelling and a `view=` that disagree is an error."""
        granule = slstr_pdfs_granule(tmp_path, views=("n", "o"))
        with pytest.raises(ReaderError, match="names the oblique view"):
            read_slstr(granule, channel="S5_radiance_ao", view="nadir")

    def test_agreeing_explicit_arguments_are_accepted(self, tmp_path):
        """A spelling plus a matching `grid=` is fine, not a spurious conflict."""
        granule = slstr_pdfs_granule(tmp_path, views=("n",))
        metadata = read_slstr(
            granule, channel="S5_radiance_bn", grid="b"
        ).raster.GetMetadata()
        assert metadata["grid"] == "bn", metadata

    def test_a_channel_set_keeps_each_spelling(self, tmp_path):
        """Each channel in a set keeps its own spelled grid and view."""
        granule = slstr_pdfs_granule(tmp_path, views=("n", "o"))
        result = read_slstr(granule, channels=["S5_radiance_bn", "S8_BT_io"])
        assert result["S5"].raster.GetMetadata()["grid"] == "bn"
        assert result["S8"].raster.GetMetadata()["grid"] == "io"

    @pytest.mark.parametrize(
        "spelling", ["S9_radiance_an", "S5_BT_an", "F2_radiance_in"]
    )
    def test_a_quantity_the_channel_does_not_ship_is_rejected(self, spelling):
        """A quantity/channel combination no product has is rejected, not swapped.

        Test scenario:
            `S9` is thermal, so `S9_radiance_an` names a file that cannot exist --
            and it used to resolve quietly to the `S9_BT_in` brightness
            temperature, returning kelvin for a request that said radiance.
        """
        with pytest.raises(ReaderError, match="names quantity"):
            _canonical_channel(spelling, "read_slstr")


class TestGeodeticGridConformance:
    """The geodetic grid must match the channel's grid, not merely exist."""

    @staticmethod
    def _granule(root, geo_shape):
        """Build a granule whose S5 band is 2x3 and whose geodetic grid is `geo_shape`."""
        granule = root / "S3A_SL_1_RBT____shape.SEN3"
        granule.mkdir(parents=True, exist_ok=True)
        write_measurement(
            granule / "S5_radiance_an.nc", "S5_radiance_an", packed_counts()
        )
        rows, columns = geo_shape
        write_geolocation(
            granule / "geodetic_an.nc",
            longitude_variable="longitude_an",
            latitude_variable="latitude_an",
            shape=(rows, columns),
        )
        return granule

    @pytest.mark.parametrize("geo_shape", [(4, 6), (1, 2), (2, 6), (4, 3)])
    def test_a_mismatched_geodetic_grid_raises(self, tmp_path, geo_shape):
        """A geodetic grid of the wrong size is rejected rather than silently used.

        Args:
            geo_shape: The geodetic grid's `(rows, columns)`, against a 2x3 band.

        Test scenario:
            Only the annotation file's *existence* and its variables' presence were
            checked, never that the grid conformed. An oversized grid was the worst
            case: GDAL emitted no warning at all and the warp simply landed
            elsewhere (origin 3.96667 against the correct 3.92857), which is a
            plausible-looking raster in the wrong place.
        """
        granule = self._granule(tmp_path, geo_shape)
        with pytest.raises(ReaderError, match="but the band is 2x3"):
            read_slstr(granule, channel="S5")

    def test_a_conforming_geodetic_grid_is_accepted(self, tmp_path):
        """The matching grid still reads, so the check is not over-eager."""
        granule = self._granule(tmp_path, (2, 3))
        assert read_slstr(granule, channel="S5").read_array().shape == (2, 3)

    def test_a_transposed_geodetic_grid_raises(self, tmp_path):
        """A geodetic grid that is the band's shape transposed is rejected.

        Test scenario:
            The mismatch cases above all differ from the band in *area*, so a
            conformance check comparing `(width, height)` instead of
            `(height, width)` would still raise on every one of them. A 3x2 grid
            against a 2x3 band is the one shape that tells the two comparisons
            apart -- and it is the realistic failure, since a transposed read of
            the annotation file is exactly what a row/column mix-up produces.
        """
        granule = self._granule(tmp_path, (3, 2))
        with pytest.raises(ReaderError, match="is 3x2 but the band is 2x3"):
            read_slstr(granule, channel="S5")

    @pytest.mark.parametrize("bad_scale", [float("nan"), float("inf"), 0.0])
    def test_an_unusable_geolocation_scale_raises(self, tmp_path, bad_scale):
        """A non-finite or zero geolocation scale is rejected at read time.

        Test scenario:
            The finite-packing guard was applied to the measurement decoder but
            not to the geolocation path, so these built an all-NaN or all-zero VRT
            that read fine and failed only inside the caller's warp, with GDAL's
            own `Int overflow : -nan(ind)`.
        """
        from tests._netcdf3 import write_netcdf

        granule = tmp_path / "S3A_SL_1_RBT____scale.SEN3"
        granule.mkdir()
        write_measurement(
            granule / "S5_radiance_an.nc", "S5_radiance_an", packed_counts()
        )
        values = np.full((2, 3), 4000000, dtype="int32")
        write_netcdf(
            granule / "geodetic_an.nc",
            {
                "longitude_an": (values, {"scale_factor": bad_scale}),
                "latitude_an": (values, {"scale_factor": bad_scale}),
            },
        )
        with pytest.raises(ReaderError, match="unusable scale_factor"):
            read_slstr(granule, channel="S5")


class TestChannelTargetMatrix:
    """Every spelled-against-explicit combination `_resolve_channel_target` must handle.

    The reader has two independent sources for the grid and the view -- the
    channel's own spelling and the `grid=` / `view=` arguments -- and three
    outcomes per axis: the spelling decides, the argument decides, or they
    disagree and it is an error. The single spelling the earlier tests used
    (`S5_radiance_an`) agrees with every default, so none of that was separable.
    Each cell below names which source must win, or that the pair is refused.
    """

    @pytest.mark.parametrize(
        "channel, view, grid, expected",
        [
            # A spelling alone decides both axes.
            ("S5_radiance_bn", None, None, ("S5", "b", "n")),
            ("S5_radiance_ao", None, None, ("S5", "a", "o")),
            ("S8_BT_io", None, None, ("S8", "i", "o")),
            ("F1_BT_fo", None, None, ("F1", "f", "o")),
            # A bare channel leaves the grid open and defaults the view to nadir.
            ("S5", None, None, ("S5", None, "n")),
            ("s5", None, None, ("S5", None, "n")),
            # With a bare channel the arguments decide, in either spelling.
            ("S5", "oblique", None, ("S5", None, "o")),
            ("S5", "o", None, ("S5", None, "o")),
            ("S5", "nadir", "b", ("S5", "b", "n")),
            ("S5", None, "i", ("S5", "i", "n")),
            ("S5", "oblique", "B", ("S5", "b", "o")),
            # Agreeing sources are accepted rather than flagged as a conflict,
            # including when only their case differs.
            ("S5_radiance_bn", "nadir", "b", ("S5", "b", "n")),
            ("S5_radiance_bn", "n", "B", ("S5", "b", "n")),
            ("S5_radiance_ao", "oblique", "a", ("S5", "a", "o")),
            # One source given, the other silent: no conflict either way round.
            ("S5_radiance_bn", "nadir", None, ("S5", "b", "n")),
            ("S5_radiance_bn", None, "b", ("S5", "b", "n")),
            # A grid that agrees while the view is left open is not a conflict.
            ("S8_BT_io", None, "i", ("S8", "i", "o")),
        ],
    )
    def test_the_agreed_target_is_resolved(self, channel, view, grid, expected):
        """A spelling and arguments that do not contradict resolve to one target.

        Args:
            channel: The channel spelling passed by the caller.
            view: The caller's `view` argument, or `None`.
            grid: The caller's `grid` argument, or `None`.
            expected: The `(name, grid, view_suffix)` triple required.

        Test scenario:
            A spelled suffix names one specific file, so it must survive to the
            discovery step; `an` -- the only spelling previously tested -- happens
            to equal both defaults, so discarding the suffix altogether looked
            correct. Each row here uses a value that differs from the default on at
            least one axis.
        """
        resolved = _resolve_channel_target(channel, view, grid, "read_slstr")
        assert resolved == expected, (
            f"channel={channel!r} view={view!r} grid={grid!r} should resolve to "
            f"{expected}, got {resolved}"
        )

    @pytest.mark.parametrize(
        "channel, view, grid, message",
        [
            ("S5_radiance_bn", None, "a", "names grid 'b' but grid='a'"),
            ("S5_radiance_an", None, "b", "names grid 'a' but grid='b'"),
            # The grid agrees while the view does not: the view check must still fire.
            ("S8_BT_io", "nadir", "i", "names the oblique view"),
            ("S5_radiance_ao", "nadir", None, "names the oblique view"),
            ("S5_radiance_bn", "oblique", None, "names the nadir view"),
            # Both axes disagree: the grid conflict is reported first.
            ("S5_radiance_bn", "oblique", "a", "names grid 'b' but grid='a'"),
        ],
    )
    def test_a_contradiction_is_refused(self, channel, view, grid, message):
        """A spelling that contradicts an argument is an error, not a silent choice.

        Args:
            channel: The channel spelling passed by the caller.
            view: The caller's `view` argument, or `None`.
            grid: The caller's `grid` argument, or `None`.
            message: A fragment the error must name.

        Test scenario:
            Preferring either source silently reads a file the caller did not name.
            The rows cover both axes and both directions -- a spelled `b` against
            `grid='a'` and a spelled `a` against `grid='b'` -- so a check that
            compared only one way round cannot pass.
        """
        with pytest.raises(ReaderError, match=message):
            _resolve_channel_target(channel, view, grid, "read_slstr")

    def test_an_invalid_explicit_argument_is_still_rejected_beside_a_spelling(self):
        """A nonsense `grid=` / `view=` is validated even when the spelling carries one."""
        with pytest.raises(ReaderError, match="is not an SLSTR grid"):
            _resolve_channel_target("S5_radiance_bn", None, "z", "read_slstr")
        with pytest.raises(ReaderError, match="is not an SLSTR view"):
            _resolve_channel_target("S5_radiance_bn", "sideways", None, "read_slstr")


class TestParseChannelQuantityMatrix:
    """`_parse_channel` over every quantity x grid x view combination.

    The quantity gate is per channel -- `S1`-`S6` ship radiance, `S7`-`S9` and
    `F1`-`F2` ship brightness temperature -- so the full cross-product is the only
    way to show it is the *channel* that decides and not, say, the letter or the
    grid. The grid and view are carried through untouched, including letters the
    channel is not documented on, which the discovery step then refuses.
    """

    @pytest.mark.parametrize("grid", ["a", "b", "i", "f"])
    @pytest.mark.parametrize("view", ["n", "o"])
    @pytest.mark.parametrize(
        "channel, shipped, other",
        [
            ("S1", "radiance", "BT"),
            ("S5", "radiance", "BT"),
            ("S6", "radiance", "BT"),
            ("S7", "BT", "radiance"),
            ("S9", "BT", "radiance"),
            ("F1", "BT", "radiance"),
            ("F2", "BT", "radiance"),
        ],
    )
    def test_the_shipped_quantity_parses_and_the_other_is_refused(
        self, channel, shipped, other, grid, view
    ):
        """The channel's own quantity parses; the other is rejected on every grid/view.

        Args:
            channel: The SLSTR channel.
            shipped: The radiometric quantity that channel really ships.
            other: The quantity it does not ship.
            grid: The grid letter in the spelling.
            view: The view letter in the spelling.

        Test scenario:
            `S9_radiance_an` used to resolve quietly to the `S9_BT_in` brightness
            temperature, handing back kelvin for a request that said radiance.
            Sweeping every combination shows the gate is driven by the channel and
            that the grid and view letters reach the caller unchanged, rather than
            being normalised to the channel's default.
        """
        accepted = f"{channel}_{shipped}_{grid}{view}"
        assert _parse_channel(accepted, "read_slstr") == (channel, grid, view), (
            f"{accepted!r} must carry its own grid and view through"
        )
        with pytest.raises(ReaderError, match="names quantity"):
            _parse_channel(f"{channel}_{other}_{grid}{view}", "read_slstr")

    @pytest.mark.parametrize(
        "spelling", ["S5_RADIANCE_an", "s8_bt_IO", "S5_Radiance_Bn"]
    )
    def test_the_quantity_and_suffix_are_case_insensitive(self, spelling):
        """Mixed case in any part of a product spelling still resolves.

        Args:
            spelling: A product variable name with unconventional casing.

        Test scenario:
            The pattern is compiled case-insensitively, so the grid and view must
            be lower-cased on the way out -- they are interpolated straight into a
            filename, where `IO` would build `geodetic_IO.nc`.
        """
        name, grid, view = _parse_channel(spelling, "read_slstr")
        assert name[0].isupper(), f"channel letter must be upper-cased: {name!r}"
        assert name[1:].isdigit(), f"channel number must be digits: {name!r}"
        cased = f"the suffix must be lower-cased for the filename, got {grid!r}{view!r}"
        assert grid == grid.lower(), cased
        assert view == view.lower(), cased


class TestSpelledGridOffTheChannelsTable:
    """A spelled grid a channel is not documented on must not be swapped for one."""

    @pytest.mark.parametrize(
        "spelling, channel",
        [
            ("S7_BT_an", "S7"),
            ("S8_BT_bn", "S8"),
            ("S1_radiance_in", "S1"),
            ("F2_BT_fn", "F2"),
        ],
    )
    def test_it_raises_instead_of_falling_back_to_a_documented_grid(
        self, tmp_path, spelling, channel
    ):
        """A grid the channel does not ship on fails rather than reading another file.

        Args:
            spelling: A product spelling naming a grid the channel is not on.
            channel: The channel that spelling names.

        Test scenario:
            `_parse_channel` validates the quantity against the channel but not the
            grid, so these reach discovery as a request for a file that cannot
            exist. The granule *does* hold each channel on its own grid, so a
            reader that discarded the spelled grid would return that file instead
            -- a different resolution or stripe than the caller named, silently.
        """
        granule = slstr_pdfs_granule(tmp_path, views=("n", "o"))
        with pytest.raises(ReaderError, match=f"has no {channel!r} measurement"):
            read_slstr(granule, channel=spelling)

    def test_the_same_channels_still_read_on_their_documented_grid(self, tmp_path):
        """The control: the rejections above are about the grid, not the channel.

        Test scenario:
            Without this the class could pass against a granule that simply lacked
            every channel, which would make the assertions above prove nothing.
        """
        granule = slstr_pdfs_granule(tmp_path, views=("n", "o"))
        for spelling in ("S7_BT_in", "S8_BT_in", "S1_radiance_an", "F2_BT_in"):
            metadata = read_slstr(granule, channel=spelling).raster.GetMetadata()
            assert metadata["grid"] == spelling[-2:], (
                f"{spelling} should read the grid it names: {metadata}"
            )
