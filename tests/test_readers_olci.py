"""Unit tests for `pyramids_eo.sensors.readers.olci`.

The granule fixtures are **real** NetCDF files written through pyramids' bundled GDAL
netCDF driver (see `tests/_sentinel3_granule`), so these exercise the actual GDAL
subdataset path and CF attribute names rather than a mock. The reader's geolocation
contract is that it *emits* a correct GDAL `GEOLOCATION` domain -- GDAL consumes it on a
warp -- so the domain is asserted as metadata, not by warping.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._common import valid_bounds
from pyramids_eo.sensors.readers._sentinel3 import PLACEHOLDER_GEOTRANSFORM
from pyramids_eo.sensors.readers.olci import (
    _canonical_channel,
    available_olci_channels,
    product_type,
    read_olci,
    read_olci_band,
)
from tests._sentinel3_granule import (
    expected_radiance,
    olci_granule,
    packed_counts,
    write_geolocation,
)


class TestCanonicalChannel:
    """Tests for `_canonical_channel`."""

    @pytest.mark.parametrize(
        "given, expected",
        [
            ("Oa08", "Oa08"),
            ("oa8", "Oa08"),
            ("OA08", "Oa08"),
            ("Oa08_radiance", "Oa08"),
            ("oa1", "Oa01"),
            ("Oa21", "Oa21"),
            ("  Oa08  ", "Oa08"),
        ],
    )
    def test_accepted_spellings(self, given, expected):
        """Every accepted spelling of a band canonicalises to its zero-padded name."""
        assert _canonical_channel(given, "read_olci") == expected, (
            f"{given!r} should canonicalise to {expected!r}"
        )

    @pytest.mark.parametrize(
        "given", ["Oa00", "Oa22", "Oa99", "S5", "", "oa", "Ob08", 8]
    )
    def test_rejected_spellings(self, given):
        """A name outside Oa01..Oa21 is rejected rather than silently read."""
        with pytest.raises(ReaderError, match="is not an OLCI band"):
            _canonical_channel(given, "read_olci")


class TestValidBounds:
    """Tests for the shared `valid_bounds` CF `valid_range` parser."""

    @pytest.mark.parametrize(
        "given, expected",
        [
            ("{0, 4095}", (0.0, 4095.0)),
            ("0, 4095", (0.0, 4095.0)),
            ("{-1.5e2, 4.0e3}", (-150.0, 4000.0)),
            # Three or more numbers take the first and last, which is the
            # long-standing FCI behaviour this parser was lifted from.
            ("{1, 2, 3}", (1.0, 3.0)),
            (None, (None, None)),
            ("not-a-range", (None, None)),
            ("{1}", (None, None)),
            ("{a, b}", (None, None)),
            # A reversed range is malformed in the same sense as junk, and must
            # mask nothing rather than mask every pixel.
            ("{4, 1}", (None, None)),
        ],
    )
    def test_parsing(self, given, expected):
        """A GDAL-rendered range parses; malformed or reversed yields no bounds."""
        assert valid_bounds(given) == expected, f"{given!r} should parse to {expected}"


class TestReadOlci:
    """Tests for `read_olci` against synthetic `.SEN3` granules."""

    def test_decodes_and_unpacks_a_band(self, tmp_path):
        """A band's counts are unpacked by scale/offset with the fill pixel NaN."""
        counts = packed_counts()
        granule = olci_granule(tmp_path, bands={"Oa08": counts})
        dataset = read_olci(granule, channel="Oa08")
        array = np.asarray(dataset.read_array(), dtype=float)
        assert np.allclose(array, expected_radiance(counts), equal_nan=True), (
            f"unpacked radiance mismatch: {array}"
        )

    def test_fill_pixel_is_nan(self, tmp_path):
        """The granule's `_FillValue` pixel becomes NaN, not a finite unpacked value."""
        granule = olci_granule(tmp_path)
        array = np.asarray(read_olci(granule, channel="Oa08").read_array(), dtype=float)
        assert np.isnan(array[0, 2]), f"fill pixel should be NaN, got {array[0, 2]}"
        assert np.isfinite(array[0, 0]), "a valid pixel should stay finite"

    def test_carries_a_usable_geolocation_domain(self, tmp_path):
        """The result's GEOLOCATION domain warps to degrees over the granule's extent.

        Test scenario:
            Asserting the domain *strings* is what let a completely unusable
            domain pass at full coverage, so this warps through it instead. The
            granule's geolocation is packed int32 micro-degrees, as the real
            products ship, over lon 4-5 / lat 51-52.
        """
        from osgeo import gdal

        granule = olci_granule(tmp_path)
        dataset = read_olci(granule, channel="Oa08")
        domain = dataset.raster.GetMetadata("GEOLOCATION")
        assert (domain["PIXEL_STEP"], domain["LINE_STEP"]) == ("1", "1"), domain
        warped = gdal.Warp("", dataset.raster, format="MEM", geoloc=True)
        origin_x, _, _, origin_y, _, _ = warped.GetGeoTransform()
        assert 3.0 < origin_x < 6.0, f"longitude origin not in degrees: {origin_x}"
        assert 50.0 < origin_y < 53.0, f"latitude origin not in degrees: {origin_y}"

    def test_swath_has_no_affine_georeference(self, tmp_path):
        """A swath result carries the placeholder pixel grid and no EPSG, by contract."""
        dataset = read_olci(olci_granule(tmp_path), channel="Oa08")
        assert dataset.geotransform == PLACEHOLDER_GEOTRANSFORM, (
            f"expected the placeholder grid, got {dataset.geotransform}"
        )
        assert dataset.epsg is None, f"a swath has no EPSG, got {dataset.epsg}"

    def test_geolocation_false_omits_the_domain(self, tmp_path):
        """`geolocation=False` returns the array with no GEOLOCATION domain."""
        granule = olci_granule(tmp_path, geolocation=False)
        dataset = read_olci(granule, channel="Oa08", geolocation=False)
        assert not dataset.raster.GetMetadata("GEOLOCATION"), "domain should be empty"

    def test_geolocation_false_tolerates_a_missing_geo_file(self, tmp_path):
        """A granule without geo_coordinates.nc still reads when geolocation=False."""
        granule = olci_granule(tmp_path, geolocation=False)
        dataset = read_olci(granule, channel="Oa08", geolocation=False)
        assert dataset.read_array().shape == (2, 3), "the band should still decode"

    def test_missing_geo_file_raises_when_geolocation_requested(self, tmp_path):
        """Requesting geolocation from a granule lacking geo_coordinates.nc is an error."""
        granule = olci_granule(tmp_path, geolocation=False)
        with pytest.raises(ReaderError, match="geo_coordinates.nc"):
            read_olci(granule, channel="Oa08")

    def test_band_name_and_metadata(self, tmp_path):
        """The band is named for its channel and carries CF + registry metadata."""
        dataset = read_olci(olci_granule(tmp_path), channel="Oa08")
        assert dataset.band_names == ["Oa08"], dataset.band_names
        meta = dataset.raster.GetMetadata()
        assert meta["sensor"] == "olci", meta
        assert meta["channel"] == "Oa08", meta
        assert meta["units"] == "mW.m-2.sr-1.nm-1", meta
        assert meta["wavelength_um"] == "0.665", meta
        assert meta["kind"] == "solar", meta

    def test_accepts_a_file_inside_the_granule(self, tmp_path):
        """A path to a file inside the granule resolves to the granule directory."""
        granule = olci_granule(tmp_path)
        dataset = read_olci(granule / "Oa08_radiance.nc", channel="Oa08")
        assert dataset.read_array().shape == (2, 3), "should read via the sibling file"

    def test_channels_returns_a_dict(self, tmp_path):
        """`channels=[...]` returns a dict keyed by canonical channel name."""
        granule = olci_granule(
            tmp_path, bands={"Oa08": packed_counts(), "Oa17": packed_counts()}
        )
        result = read_olci(granule, channels=["Oa08", "oa17"])
        assert sorted(result) == ["Oa08", "Oa17"], f"unexpected keys: {sorted(result)}"
        assert result["Oa17"].band_names == ["Oa17"], result["Oa17"].band_names

    def test_channels_rejects_aliases_of_one_band(self, tmp_path):
        """Two spellings of one band are rejected, not silently collapsed to one entry."""
        granule = olci_granule(tmp_path)
        with pytest.raises(ReaderError, match="same band more than once"):
            read_olci(granule, channels=["Oa08", "oa8"])

    def test_requires_exactly_one_of_channel_or_channels(self, tmp_path):
        """Neither or both of channel / channels is an error."""
        granule = olci_granule(tmp_path)
        with pytest.raises(ReaderError, match="exactly one"):
            read_olci(granule)
        with pytest.raises(ReaderError, match="exactly one"):
            read_olci(granule, channel="Oa08", channels=["Oa17"])

    def test_missing_band_file_raises(self, tmp_path):
        """A band the granule does not carry is a clear error."""
        granule = olci_granule(tmp_path, bands={"Oa08": packed_counts()})
        with pytest.raises(ReaderError, match="Oa17_radiance.nc"):
            read_olci(granule, channel="Oa17")

    def test_non_granule_source_raises(self, tmp_path):
        """A source that is neither a directory nor a file in one is rejected."""
        with pytest.raises(ReaderError, match="not a Sentinel-3 granule"):
            read_olci(tmp_path / "nope.SEN3", channel="Oa08")

    def test_metadata_geolocation_and_band_names_coexist(self, tmp_path):
        """Setting the CF metadata does not clobber the band name or the GEOLOCATION domain.

        Test scenario:
            The reader sets `band_names`, attaches the GEOLOCATION domain, and then calls
            `SetMetadata` on the *default* domain. `SetMetadata` replaces a whole domain,
            so this pins that all three survive together -- a reordering or a change of
            target domain in a future refactor would otherwise silently drop the band
            name or the geolocation, which no coverage number would catch.
        """
        dataset = read_olci(olci_granule(tmp_path), channel="Oa08")
        assert dataset.band_names == ["Oa08"], dataset.band_names
        assert dataset.raster.GetMetadata("GEOLOCATION").get("X_DATASET"), (
            "the GEOLOCATION domain must survive the default-domain SetMetadata"
        )
        assert dataset.raster.GetMetadata()["sensor"] == "olci", (
            dataset.raster.GetMetadata()
        )

    def test_declares_nan_nodata(self, tmp_path):
        """The result declares NaN nodata, matching the masked fill pixels."""
        dataset = read_olci(olci_granule(tmp_path), channel="Oa08")
        assert np.isnan(dataset.no_data_value[0]), (
            f"nodata should be NaN, got {dataset.no_data_value}"
        )

    def test_repeated_reads_are_stable(self, tmp_path):
        """Reading the same band twice yields identical values (no cached-state drift)."""
        granule = olci_granule(tmp_path)
        first = np.asarray(read_olci(granule, channel="Oa08").read_array(), dtype=float)
        second = np.asarray(
            read_olci(granule, channel="Oa08").read_array(), dtype=float
        )
        assert np.allclose(first, second, equal_nan=True), (
            f"repeated reads differ:\n{first}\n{second}"
        )

    def test_channels_with_geolocation_disabled(self, tmp_path):
        """`channels=` and `geolocation=False` combine: a dict with no domains."""
        granule = olci_granule(
            tmp_path,
            bands={"Oa08": packed_counts(), "Oa17": packed_counts()},
            geolocation=False,
        )
        result = read_olci(granule, channels=["Oa08", "Oa17"], geolocation=False)
        assert sorted(result) == ["Oa08", "Oa17"], sorted(result)
        for name, dataset in result.items():
            assert not dataset.raster.GetMetadata("GEOLOCATION"), (
                f"{name} should carry no GEOLOCATION domain"
            )

    def test_registry_lookup_failure_still_yields_metadata(self, tmp_path, monkeypatch):
        """An unavailable registry degrades to CF-only metadata rather than failing the read."""
        from pyramids_eo.sensors import registry

        def _boom(_sensor):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(registry, "get_sensor", _boom)
        meta = read_olci(olci_granule(tmp_path), channel="Oa08").raster.GetMetadata()
        assert meta["channel"] == "Oa08", meta
        assert "wavelength_um" not in meta, (
            f"registry fields should be absent when lookup fails: {meta}"
        )


class TestAvailableOlciChannels:
    """Tests for `available_olci_channels`."""

    def test_lists_only_present_bands(self, tmp_path):
        """Only the bands whose files exist are listed, in canonical sorted order."""
        granule = olci_granule(
            tmp_path, bands={"Oa17": packed_counts(), "Oa01": packed_counts()}
        )
        assert available_olci_channels(granule) == ["Oa01", "Oa17"]

    def test_empty_granule_lists_nothing(self, tmp_path):
        """A granule with no band files lists no channels (rather than raising)."""
        granule = olci_granule(tmp_path, bands={}, geolocation=False)
        assert available_olci_channels(granule) == []

    def test_non_granule_source_raises(self, tmp_path):
        """A missing source is rejected with the granule error."""
        with pytest.raises(ReaderError, match="not a Sentinel-3 granule"):
            available_olci_channels(tmp_path / "missing")


class TestProductType:
    """Tests for `product_type`, which decides the reported resolution."""

    @pytest.mark.parametrize(
        "granule_name, expected",
        [
            ("S3A_OL_1_EFR____20260101T101500.SEN3", "EFR"),
            ("S3B_OL_1_ERR____20260101T101500.SEN3", "ERR"),
            ("s3a_ol_1_efr____lowercase.SEN3", "EFR"),
            ("something_unrelated.SEN3", None),
        ],
    )
    def test_detects_from_the_granule_name(self, tmp_path, granule_name, expected):
        """EFR and ERR are read off the granule name, which is the only place they differ."""
        granule = tmp_path / granule_name
        granule.mkdir()
        assert product_type(granule) == expected

    def test_falls_back_to_the_manifest(self, tmp_path):
        """An unconventionally named granule is identified from xfdumanifest.xml."""
        granule = tmp_path / "renamed.SEN3"
        granule.mkdir()
        (granule / "xfdumanifest.xml").write_text(
            '<xfdu><product>S3A_OL_1_ERR____x</product></xfdu>', encoding="utf-8"
        )
        assert product_type(granule) == "ERR"

    def test_reduced_resolution_is_not_labelled_as_full(self, tmp_path):
        """An ERR granule reports ~1200 m, not the registry's full-resolution 300 m.

        Test scenario:
            EFR and ERR share every filename and variable name, so the reader used
            to stamp the registry's single 300 m figure on both -- wrong by a
            factor of four for an ERR granule, and consumed as authoritative by
            resolution-aware code such as `harmonise`.
        """
        granule = olci_granule(tmp_path, product="ERR")
        meta = read_olci(granule, channel="Oa08").raster.GetMetadata()
        assert meta["product"] == "ERR", meta
        assert meta["resolution_m"] == "1200", meta

    def test_full_resolution_reports_300(self, tmp_path):
        """An EFR granule reports the full-resolution 300 m."""
        meta = read_olci(
            olci_granule(tmp_path, product="EFR"), channel="Oa08"
        ).raster.GetMetadata()
        assert meta["resolution_m"] == "300", meta

    def test_an_unidentifiable_granule_omits_the_resolution(self, tmp_path):
        """A granule that does not state its product type reports no resolution at all.

        Test scenario:
            Omitting the key is the honest outcome: asserting either 300 or 1200
            would be a guess, and a wrong resolution is consumed silently.
        """
        granule = olci_granule(tmp_path, name="mystery.SEN3")
        meta = read_olci(granule, channel="Oa08").raster.GetMetadata()
        assert "resolution_m" not in meta, meta
        assert "product" not in meta, meta


class TestGeolocationMetadataKey:
    """The `geolocation` metadata key must describe what the result really carries."""

    def test_says_swath_when_a_domain_is_attached(self, tmp_path):
        """With geolocation on, the key points at the GEOLOCATION domain."""
        meta = read_olci(olci_granule(tmp_path), channel="Oa08").raster.GetMetadata()
        assert "GEOLOCATION" in meta["geolocation"], meta

    def test_says_none_when_no_domain_was_attached(self, tmp_path):
        """With geolocation off, the key must not advertise a domain that is absent.

        Test scenario:
            It used to claim "swath (see the GEOLOCATION metadata domain)"
            unconditionally, so code branching on it took the geolocated path on a
            Dataset carrying only the placeholder (0, 1, 0, 0, 0, -1) grid -- i.e.
            treating pixel indices as coordinates.
        """
        granule = olci_granule(tmp_path, geolocation=False)
        dataset = read_olci(granule, channel="Oa08", geolocation=False)
        assert not dataset.raster.GetMetadata("GEOLOCATION"), "domain should be empty"
        assert dataset.raster.GetMetadata()["geolocation"] == (
            "none (placeholder pixel grid)"
        ), dataset.raster.GetMetadata()


class TestReadOlciBand:
    """Tests for the single-band entry point `read_olci_band`."""

    def test_decodes_one_band(self, tmp_path):
        """The single-band helper decodes and geolocates like `read_olci(channel=...)`."""
        dataset = read_olci_band(olci_granule(tmp_path), "Oa08")
        assert dataset.band_names == ["Oa08"], dataset.band_names
        assert dataset.raster.GetMetadata("GEOLOCATION"), "should carry a domain"

    def test_without_geolocation(self, tmp_path):
        """`geolocation=False` skips the annotation file entirely."""
        granule = olci_granule(tmp_path, geolocation=False)
        dataset = read_olci_band(granule, "oa8", geolocation=False)
        assert dataset.read_array().shape == (2, 3), "the band should still decode"


class TestProductTypeEdgeCases:
    """Coverage for `product_type`'s manifest branches."""

    def test_a_manifest_without_a_product_type_yields_none(self, tmp_path):
        """A manifest that names no product type is as good as no manifest."""
        granule = tmp_path / "mystery.SEN3"
        granule.mkdir()
        (granule / "xfdumanifest.xml").write_text("<xfdu/>", encoding="utf-8")
        assert product_type(granule) is None

    @pytest.mark.parametrize(
        "granule_name",
        [
            "S3A_OL_1_EFR_OL_1_ERR____20260101T101500.SEN3",
            "S3A_OL_1_ERR_OL_1_EFR____20260101T101500.SEN3",
        ],
    )
    def test_a_name_carrying_both_tokens_is_unknown(self, tmp_path, granule_name):
        """A name containing both product tokens is reported unknown, not guessed.

        Args:
            granule_name: A granule directory name holding both `_OL_1_EFR` and
                `_OL_1_ERR`.

        Test scenario:
            An ERR granule's own naming or manifest routinely references the EFR
            product it was derived from, so both tokens co-occurring is a real
            case, not a pathological one. Picking the first match in a fixed order
            is deterministic but still a guess, and it labelled such a granule
            300 m when it is ~1200 m. `None` is the honest answer -- the same one
            the function gives for a granule that states nothing -- and it is what
            the "omit rather than assert" rule requires.
        """
        granule = tmp_path / granule_name
        granule.mkdir()
        assert product_type(granule) is None, (
            f"{granule_name} is ambiguous and must not be resolved to a guess"
        )

    def test_the_granule_name_takes_precedence_over_the_manifest(self, tmp_path):
        """A name that states its product type is trusted over a disagreeing manifest.

        Test scenario:
            The manifest is a fallback for granules whose directory was renamed,
            so a granule whose name *does* say EFR must not be relabelled ERR by
            a stale or copied manifest -- which would quadruple the reported
            resolution. Pinning the precedence is what makes the two sources
            unambiguous.
        """
        granule = tmp_path / "S3A_OL_1_EFR____20260101T101500.SEN3"
        granule.mkdir()
        (granule / "xfdumanifest.xml").write_text(
            "<xfdu><product>S3A_OL_1_ERR____other</product></xfdu>", encoding="utf-8"
        )
        assert product_type(granule) == "EFR", (
            "the granule name must win over a disagreeing manifest"
        )

    def test_a_manifest_naming_both_products_is_unknown(self, tmp_path):
        """A manifest mentioning both product types is unknown, not resolved to one.

        Test scenario:
            This is the realistic shape of the ambiguity: a real
            `xfdumanifest.xml` references the source product and processing
            baseline, so an ERR granule's manifest legitimately names EFR too.
        """
        granule = tmp_path / "renamed-both.SEN3"
        granule.mkdir()
        (granule / "xfdumanifest.xml").write_text(
            "<xfdu><a>OL_1_ERR</a><b>OL_1_EFR</b></xfdu>", encoding="utf-8"
        )
        assert product_type(granule) is None, (
            "a manifest naming both products must not be resolved to a guess"
        )

    def test_an_err_granule_referencing_its_efr_source_reports_no_resolution(
        self, tmp_path
    ):
        """The realistic ambiguity yields no `resolution_m` rather than a wrong one."""
        granule = olci_granule(
            tmp_path, name="S3A_OL_1_ERR____from_OL_1_EFR_source.SEN3"
        )
        meta = read_olci(granule, channel="Oa08").raster.GetMetadata()
        assert "resolution_m" not in meta, (
            f"an ambiguous granule must not assert a resolution: {meta}"
        )

    def test_an_unreadable_manifest_yields_none(self, tmp_path, monkeypatch):
        """An unreadable manifest degrades to None rather than failing the read."""
        from pathlib import Path as _Path

        granule = tmp_path / "mystery.SEN3"
        granule.mkdir()
        (granule / "xfdumanifest.xml").write_text("<xfdu/>", encoding="utf-8")

        def _boom(self, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr(_Path, "read_text", _boom)
        assert product_type(granule) is None


class TestBandGridConsistency:
    """A multi-band read requires every requested band on one grid."""

    def test_bands_of_differing_shapes_raise(self, tmp_path):
        """Bands on different grids are rejected rather than sharing one domain.

        Test scenario:
            The geolocation domain is resolved once for a channel set and
            validated against the first band's shape, which is only sound because
            OLCI bands of a granule share a grid. A granule whose bands disagree
            would otherwise have the rest silently validated against the wrong
            shape, so it is caught explicitly.
        """
        granule = olci_granule(
            tmp_path,
            bands={
                "Oa08": packed_counts(),
                "Oa17": np.array([[1, 2], [3, 4]], dtype="int32"),
            },
        )
        with pytest.raises(ReaderError, match="not on one grid"):
            read_olci(granule, channels=["Oa08", "Oa17"])

    def test_bands_on_one_grid_still_read(self, tmp_path):
        """The usual case -- every band on the same grid -- is unaffected."""
        granule = olci_granule(
            tmp_path, bands={"Oa08": packed_counts(), "Oa17": packed_counts()}
        )
        result = read_olci(granule, channels=["Oa08", "Oa17"])
        assert sorted(result) == ["Oa08", "Oa17"], sorted(result)


class TestGeolocationGridConformance:
    """OLCI's geolocation grid must match the band, not merely exist.

    The conformance check lives in the shared `geolocation_source`, but OLCI
    reaches it through its own `_resolve_domain` and only SLSTR's path was
    covered. Both OLCI entry points matter: `read_olci_band` validates against the
    single band's shape, while `read_olci` validates one shared domain against the
    first band of a set.
    """

    @staticmethod
    def _granule(root, geo_shape, *, bands=None):
        """Build an OLCI granule whose bands are 2x3 and whose geolocation is `geo_shape`."""
        granule = olci_granule(root, bands=bands, geolocation=False)
        write_geolocation(granule / "geo_coordinates.nc", shape=geo_shape)
        return granule

    @pytest.mark.parametrize("geo_shape", [(4, 6), (1, 2), (2, 6), (4, 3), (3, 2)])
    def test_a_mismatched_geolocation_grid_raises(self, tmp_path, geo_shape):
        """A geolocation grid of the wrong size is rejected rather than silently used.

        Args:
            geo_shape: The geolocation grid's `(rows, columns)`, against a 2x3 band.

        Test scenario:
            Oversized is the dangerous case: GDAL emits no warning and the warp
            simply lands elsewhere, which looks like a plausible raster in the
            wrong place. `(3, 2)` is the band's shape transposed, which is the only
            case that distinguishes a `(height, width)` comparison from a
            `(width, height)` one -- every other row here differs in area too.
        """
        granule = self._granule(tmp_path / f"m{geo_shape[0]}{geo_shape[1]}", geo_shape)
        with pytest.raises(ReaderError, match="but the band is 2x3"):
            read_olci(granule, channel="Oa08")

    @pytest.mark.parametrize("geo_shape", [(4, 6), (3, 2)])
    def test_a_channel_set_validates_its_shared_domain_too(self, tmp_path, geo_shape):
        """The one domain a channel set shares is validated against the bands' grid.

        Args:
            geo_shape: The geolocation grid's `(rows, columns)`, against 2x3 bands.

        Test scenario:
            `read_olci` resolves the domain once for the whole set, so the
            validation runs on a different code path than the single-band read.
            Covering only `channel=` would leave the multi-band path unchecked.
        """
        granule = self._granule(
            tmp_path / f"s{geo_shape[0]}{geo_shape[1]}",
            geo_shape,
            bands={"Oa08": packed_counts(), "Oa17": packed_counts()},
        )
        with pytest.raises(ReaderError, match="but the band is 2x3"):
            read_olci(granule, channels=["Oa08", "Oa17"])

    def test_the_conforming_grid_still_reads(self, tmp_path):
        """The matching grid reads normally, so the check is not simply always-on.

        Test scenario:
            Without this control a check that rejected every grid would pass every
            assertion above.
        """
        granule = self._granule(tmp_path / "ok", (2, 3))
        assert read_olci(granule, channel="Oa08").read_array().shape == (2, 3)

    def test_geolocation_false_skips_the_conformance_check(self, tmp_path):
        """With geolocation off a mismatched annotation file is never consulted."""
        granule = self._granule(tmp_path / "off", (4, 6))
        dataset = read_olci(granule, channel="Oa08", geolocation=False)
        assert dataset.read_array().shape == (2, 3), "the band must still decode"


class TestProductTypeTokenSources:
    """Which text `product_type` is allowed to read the product token out of."""

    def test_a_token_in_a_parent_directory_is_not_adopted(self, tmp_path):
        """Only the granule's own name counts, not the path it happens to sit under.

        Test scenario:
            Granules are routinely unpacked into a directory named after the
            download -- `S3A_OL_1_ERR____.../mystery.SEN3` -- so matching against
            the whole path would let a sibling's or an archive's name decide this
            granule's resolution. The existing tests all place the granule
            directly in `tmp_path`, where the parent can never contain a token, so
            the two readings agree there.
        """
        parent = tmp_path / "S3A_OL_1_ERR____20260101T101500"
        granule = parent / "mystery.SEN3"
        granule.mkdir(parents=True)
        assert product_type(granule) is None, (
            "a token in a parent directory must not identify the granule"
        )

    def test_the_manifest_product_type_element_is_matched_without_a_prefix(
        self, tmp_path
    ):
        """A bare `OL_1_EFR` in the manifest resolves, as the real element spells it.

        Test scenario:
            The granule name carries `_OL_1_EFR` but a manifest's own
            `<productType>` element is `OL_1_EFR___`, with no leading underscore --
            which is why the two sources are searched with different prefixes. The
            existing manifest fixtures all embed the full product *name*, which
            contains `_OL_1_`, so they match under either prefix and cannot tell
            the two apart.
        """
        granule = tmp_path / "renamed.SEN3"
        granule.mkdir()
        (granule / "xfdumanifest.xml").write_text(
            "<xfdu><metadataObject><productType>OL_1_EFR___</productType>"
            "</metadataObject></xfdu>",
            encoding="utf-8",
        )
        assert product_type(granule) == "EFR", (
            "the manifest token is not underscore-prefixed and must still match"
        )

    def test_a_lower_cased_manifest_token_resolves(self, tmp_path):
        """The manifest is case-folded before matching, as the granule name is."""
        granule = tmp_path / "renamed-lower.SEN3"
        granule.mkdir()
        (granule / "xfdumanifest.xml").write_text(
            "<xfdu><t>ol_1_err</t></xfdu>", encoding="utf-8"
        )
        assert product_type(granule) == "ERR", (
            "the manifest must be matched case-folded"
        )

    def test_a_large_but_valid_manifest_still_resolves(self, tmp_path):
        """A token far into a realistically sized manifest is found, not truncated.

        Test scenario:
            A real `xfdumanifest.xml` is hundreds of kilobytes of dataObject
            entries, and the whole file is read as text and upper-cased. A reader
            that sampled only the head -- or that choked on the size -- would
            report `None` and silently drop the resolution for every real granule
            whose name had been changed.
        """
        granule = tmp_path / "renamed-big.SEN3"
        granule.mkdir()
        body = "<dataObject><byteStream>0</byteStream></dataObject>" * 20000
        (granule / "xfdumanifest.xml").write_text(
            f"<xfdu>{body}<productType>OL_1_ERR___</productType></xfdu>",
            encoding="utf-8",
        )
        manifest_size = (granule / "xfdumanifest.xml").stat().st_size
        assert manifest_size > 500_000, f"the fixture must be large: {manifest_size}"
        assert product_type(granule) == "ERR", (
            "a token late in a large manifest must still be found"
        )

    def test_a_manifest_is_only_consulted_when_the_name_says_nothing(
        self, tmp_path, monkeypatch
    ):
        """A granule naming its type never reads the manifest at all.

        Test scenario:
            The precedence is already asserted through a disagreeing manifest; this
            pins the stronger claim the docstring makes -- that a self-identifying
            name needs *no file access* -- by making any read of the manifest fail.
        """
        from pathlib import Path as _Path

        granule = tmp_path / "S3A_OL_1_EFR____20260101T101500.SEN3"
        granule.mkdir()
        (granule / "xfdumanifest.xml").write_text("<xfdu/>", encoding="utf-8")

        def _forbidden(self, *args, **kwargs):
            raise AssertionError(f"the manifest must not be read: {self}")

        monkeypatch.setattr(_Path, "read_text", _forbidden)
        assert product_type(granule) == "EFR", (
            "a self-identifying granule name must resolve without any file access"
        )
