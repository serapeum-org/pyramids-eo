"""Unit tests targeting the helpers and error/edge branches of `pyramids_eo.sentinel`.

Complements the feature-level tests (`test_sentinel_s2.py`) with per-function
coverage of the connection grammar, product dispatch, scaling, and masking
internals — the raise branches, dunders, and defensive paths that the
happy-path tests do not reach.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids_eo.errors import ProductError, UnsupportedProductError
from pyramids_eo.sentinel import _connection, open_product, scl_mask
from pyramids_eo.sentinel.product import (
    _find_metadata_in_dir,
    _resolve_container_path,
    open_connection,
)
from pyramids_eo.sentinel.s2 import masks as _masks
from pyramids_eo.sentinel.s2 import scaling as _scaling
from pyramids_eo.sentinel.s2.masks import SclClass
from pyramids_eo.sentinel.s2.product import (
    S2Level,
    _as_float,
    _bands_of,
    _resolution_metres,
)

_DATA = Path(__file__).parent / "data" / "sentinel2"
_L2A = _DATA / "fake_l2a" / "S2A_USER_PRD_MSIL2A.SAFE" / "S2A_USER_MTD_SAFL2A.xml"
_L1C = _DATA / "fake_l1c" / "S2A_OPER_PRD_MSIL1C.SAFE" / "S2A_OPER_MTD_SAFL1C.xml"
_S1_SAFE = Path(__file__).parent / "data" / "sentinel1" / "test.SAFE"


class _FakeSD:
    """Minimal pyramids ``SubDataset`` stand-in (name + description)."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description


# -- _connection -----------------------------------------------------------


class TestEpsgFromToken:
    """Tests for `_connection._epsg_from_token`."""

    @pytest.mark.parametrize(
        "token, expected",
        [
            ("EPSG_32632", 32632),
            ("epsg_4326", 4326),
            ("EPSG_notanint", None),
            ("60m", None),
            ("EPSG_", None),
        ],
    )
    def test_parses_or_rejects(self, token, expected):
        """Return the int for an ``EPSG_<code>`` token, else ``None``.

        Test scenario:
            Valid codes parse (case-insensitively); a non-numeric tail or a
            non-EPSG token yields ``None`` (the except / guard branches).
        """
        assert _connection._epsg_from_token(token) == expected, token


class TestParseS2:
    """Tests for `_connection.parse_s2`."""

    def test_full_string(self):
        """A full ``:RES:EPSG_<code>`` string parses every field."""
        conn = _connection.parse_s2("SENTINEL2_L2A:C:/x/MTD.xml:60m:EPSG_32632")
        assert (conn.level, conn.resolution, conn.epsg) == ("L2A", "60m", 32632)

    def test_without_epsg_token(self):
        """A string ending at the resolution leaves ``epsg`` ``None``.

        Test scenario:
            ``SENTINEL2_L1B:…:10m`` has no trailing ``EPSG_`` token, so the
            resolution is the final token and epsg is ``None``.
        """
        conn = _connection.parse_s2("SENTINEL2_L1B:C:/x/MTD.xml:10m")
        assert conn.resolution == "10m"
        assert conn.epsg is None

    def test_rejects_non_s2(self):
        """A non-``SENTINEL2_`` string raises ``ProductError``."""
        with pytest.raises(ProductError, match="not a SENTINEL2"):
            _connection.parse_s2("NETCDF:file.nc:var")

    def test_malformed_missing_resolution(self):
        """A string with an empty tail raises ``ProductError``."""
        with pytest.raises(ProductError, match="malformed"):
            _connection.parse_s2("SENTINEL2_L2A:onlypath")


class TestParseS1:
    """Tests for `_connection.parse_s1`."""

    def test_with_polarisation(self):
        """A ``SWATH_POL`` token splits swath and polarisation."""
        conn = _connection.parse_s1(
            "SENTINEL1_CALIB:SIGMA0:C:/x/manifest.safe:IW_VV:INTENSITY"
        )
        assert (conn.calibration, conn.swath, conn.polarisation, conn.unit) == (
            "SIGMA0",
            "IW",
            "VV",
            "INTENSITY",
        )

    def test_without_polarisation(self):
        """A bare swath token leaves ``polarisation`` ``None``."""
        conn = _connection.parse_s1(
            "SENTINEL1_CALIB:UNCALIB:C:/x/manifest.safe:IW:AMPLITUDE"
        )
        assert conn.polarisation is None

    def test_rejects_non_s1(self):
        """A non-``SENTINEL1_CALIB:`` string raises ``ProductError``."""
        with pytest.raises(ProductError, match="not a SENTINEL1"):
            _connection.parse_s1("SENTINEL2_L2A:x:60m:EPSG_32632")

    def test_malformed(self):
        """A string missing the swath/unit tail raises ``ProductError``."""
        with pytest.raises(ProductError, match="malformed"):
            _connection.parse_s1("SENTINEL1_CALIB:UNCALIB:onlypath")


class TestSelect:
    """Tests for `_connection.select`."""

    def test_matches_s1_by_swath_pol(self):
        """Select an S1 subdataset by swath + polarisation + calibration."""
        subs = [
            _FakeSD("SENTINEL1_CALIB:UNCALIB:m.safe:IW_VV:AMPLITUDE"),
            _FakeSD("SENTINEL1_CALIB:UNCALIB:m.safe:IW_VH:AMPLITUDE"),
        ]
        chosen = _connection.select(subs, swath="IW", polarisation="VV")
        assert chosen is subs[0]

    def test_skips_non_sentinel_names(self):
        """A subdataset whose name is neither S2 nor S1 is ignored, not matched.

        Test scenario:
            A ``NETCDF:`` entry parses to ``None`` and is skipped; with no
            remaining match, ``select`` raises.
        """
        subs = [_FakeSD("NETCDF:x.nc:var")]
        with pytest.raises(ProductError, match="no subdataset matches"):
            _connection.select(subs, resolution="60m")

    def test_ambiguous_raises(self):
        """More than one match raises ``ProductError``."""
        subs = [
            _FakeSD("SENTINEL2_L2A:x:10m:EPSG_32631"),
            _FakeSD("SENTINEL2_L2A:x:10m:EPSG_32632"),
        ]
        with pytest.raises(ProductError, match="match"):
            _connection.select(subs, resolution="10m")

    def test_field_eq_missing_attr_is_not_a_match(self):
        """A token naming an attribute the parsed connection lacks never matches.

        Test scenario:
            Asking for ``polarisation`` on S2 subdatasets (which have no such
            field) yields no match rather than a false positive.
        """
        subs = [_FakeSD("SENTINEL2_L2A:x:10m:EPSG_32632")]
        with pytest.raises(ProductError):
            _connection.select(subs, polarisation="VV")


# -- product dispatch ------------------------------------------------------


class TestOpenProduct:
    """Tests for `product.open_product` dispatch and resolution."""

    def test_unsupported_driver_raises(self, tmp_path):
        """A plain GeoTIFF (GTiff driver) is not a Sentinel product.

        Test scenario:
            A raster whose driver is not SENTINEL2 / SAFE raises
            ``UnsupportedProductError``.
        """
        from pyramids.dataset import Dataset

        out = tmp_path / "plain.tif"
        Dataset.create_from_array(
            arr=np.zeros((1, 4, 4), dtype="uint8"),
            geo=(0, 1, 0, 4, 0, -1),
            epsg=4326,
        ).to_file(str(out))
        with pytest.raises(UnsupportedProductError, match="not a Sentinel"):
            open_product(out)

    def test_sentinel1_safe_is_unsupported_phase(self):
        """A Sentinel-1 SAFE product raises the Phase-2 not-yet-modelled error.

        The S1 fixture is committed to the repo, so this always runs (no skip
        guard) — a missing fixture should fail loudly, not silently skip.
        """
        with pytest.raises(UnsupportedProductError, match="Sentinel-1"):
            open_product(_S1_SAFE)

    def test_directory_without_metadata_raises(self, tmp_path):
        """A directory with no product metadata raises ``ProductError``."""
        empty = tmp_path / "empty.SAFE"
        empty.mkdir()
        with pytest.raises(ProductError, match="no product metadata"):
            _resolve_container_path(empty)

    def test_zip_without_product_raises(self, tmp_path):
        """A ``.zip`` holding no Sentinel metadata raises ``ProductError``."""
        import zipfile

        z = tmp_path / "junk.zip"
        with zipfile.ZipFile(z, "w") as archive:
            archive.writestr("readme.txt", "nothing here")
        with pytest.raises(ProductError, match="no Sentinel product metadata"):
            _resolve_container_path(z)

    def test_bad_path_raises_product_error(self, tmp_path):
        """An unreadable path is wrapped as ``ProductError``."""
        missing = tmp_path / "does_not_exist.xml"
        with pytest.raises(ProductError, match="cannot open product"):
            open_product(missing)

    def test_find_metadata_in_dir_returns_xml(self):
        """The `.SAFE` directory resolves to its product MTD file."""
        safe_dir = _L2A.parent
        resolved = _find_metadata_in_dir(safe_dir)
        assert resolved.endswith(".xml")


class TestOpenConnection:
    """Tests for `product.open_connection`."""

    def test_plain_path_uses_read_file(self):
        """A non-``/vsi`` connection opens through ``Dataset.read_file``."""
        sub = f"SENTINEL2_L2A:{_L2A}:60m:EPSG_32632"
        ds = open_connection(sub)
        assert ds.band_count == 17

    def test_bad_vsi_connection_raises(self):
        """A ``/vsi`` connection GDAL cannot open raises ``ProductError``."""
        with pytest.raises(ProductError, match="could not open"):
            open_connection("/vsizip/nope.zip/inner.tif")


# -- s2.product ------------------------------------------------------------


class TestS2ProductInternals:
    """Tests for `s2.product` helpers and edge cases."""

    def test_subdataset_connection_and_repr(self):
        """`S2Subdataset` exposes the connection string and a readable repr."""
        product = open_product(_L2A)
        sd = product.image_subdatasets[0]
        assert isinstance(sd.connection, str)
        assert sd.connection.startswith("SENTINEL2_")
        assert "S2Subdataset" in repr(sd)

    def test_product_repr(self):
        """`S2Product` repr names the driver and subdataset count."""
        product = open_product(_L2A)
        text = repr(product)
        assert "S2Product" in text
        assert "SENTINEL2" in text

    @pytest.mark.parametrize(
        "token, expected",
        [("60m", 60), ("10m", 10), ("PREVIEW", None), ("bogus", None)],
    )
    def test_resolution_metres(self, token, expected):
        """`_resolution_metres` parses ``<int>m`` and rejects other tokens."""
        assert _resolution_metres(token) == expected

    def test_bands_of_parses_description(self):
        """`_bands_of` extracts the band list from a driver description."""
        desc = "Bands B1, B2, B8A with 60m resolution, UTM 32N"
        assert _bands_of(desc) == ["B1", "B2", "B8A"]

    def test_bands_of_unexpected_shape_is_empty(self):
        """A description without ``Bands `` yields an empty list."""
        assert _bands_of("RGB preview, UTM 32N") == []

    @pytest.mark.parametrize(
        "value, expected",
        [("1000", 1000.0), (None, None), ("not-a-number", None)],
    )
    def test_as_float(self, value, expected):
        """`_as_float` parses numbers and tolerates absent / malformed values."""
        assert _as_float(value) == expected

    def test_subdataset_for_miss_raises(self):
        """Requesting a resolution the product lacks raises ``ProductError``."""
        product = open_product(_L2A)  # 60 m only
        with pytest.raises(ProductError, match="no 10m subdataset"):
            product.subdataset_for(10)

    def test_subdataset_for_multi_zone_ambiguity(self):
        """Two UTM zones at one resolution is ambiguous without ``epsg``.

        Test scenario:
            Injecting a second-zone 60 m subdataset makes ``subdataset_for(60)``
            ambiguous (raises), while passing ``epsg=`` disambiguates it — the
            multi-zone contract a single-zone fixture cannot exercise end-to-end.
        """
        from pyramids_eo.sentinel.s2.product import S2Subdataset

        product = open_product(_L2A)  # one zone: EPSG:32632
        other_zone = S2Subdataset(_FakeSD("fake:60m:EPSG_32631"), 60, 32631, ["B1"])
        product._image_subdatasets.append(other_zone)

        assert sorted(product.epsg_codes) == [32631, 32632]
        with pytest.raises(ProductError, match="ambiguous"):
            product.subdataset_for(60)
        assert product.subdataset_for(60, epsg=32632).epsg == 32632

    def test_level_enum_values(self):
        """The `S2Level` enum carries the three processing levels."""
        assert {level.value for level in S2Level} == {"L1B", "L1C", "L2A"}

    def test_product_type_fallback_is_canonical(self):
        """Without ``PRODUCT_TYPE`` metadata the fallback is ``S2MSI2A``, not ``S2MSIL2A``."""
        product = open_product(_L2A)
        product.metadata.pop("PRODUCT_TYPE", None)
        assert product.product_type == "S2MSI2A"


# -- scaling ---------------------------------------------------------------


class TestBandOffset:
    """Tests for `scaling._band_offset`."""

    @pytest.mark.parametrize(
        "meta, expected",
        [
            ({"BOA_ADD_OFFSET": "-1000"}, -1000.0),
            ({"RADIO_ADD_OFFSET": "-1000"}, -1000.0),
            ({"BOA_ADD_OFFSET": "junk"}, 0.0),
            ({}, 0.0),
        ],
    )
    def test_reads_or_defaults(self, meta, expected):
        """Read the baseline offset from band metadata, defaulting to 0.

        Test scenario:
            BOA / RADIO offsets parse; a malformed or absent value is 0.0.
        """
        assert _scaling._band_offset(meta) == expected


class TestTagReflectance:
    """Tests for `scaling.tag_reflectance`."""

    def test_zero_quantification_raises(self):
        """A product with a zero quantification value raises ``ProductError``."""
        product = open_product(_L2A)
        object.__setattr__(product, "quantification", 0.0)
        sub = product.subdataset_for(60).open()
        ds = sub.bands.select([1])
        with pytest.raises(ProductError, match="quantification"):
            _scaling.tag_reflectance(ds, product)

    def test_non_spectral_band_keeps_identity_scale(self):
        """Auxiliary bands (SCL / AOT / …) are left at scale 1, offset 0.

        Test scenario:
            Reading a subdataset that mixes spectral and SCL bands and tagging
            it leaves the SCL band's scale at 1.0.
        """
        product = open_product(_L2A)
        sub = product.subdataset_for(60).open()
        # B1 (spectral) + SCL (auxiliary) as a writable 2-band selection.
        ds = sub.bands.select([1, 15])
        tagged = _scaling.tag_reflectance(ds, product)
        assert tagged.scale[0] == pytest.approx(1.0 / product.quantification)
        assert tagged.scale[1] == pytest.approx(1.0)
        assert tagged.offset[1] == pytest.approx(0.0)

    def test_offsets_override_applied(self):
        """An explicit offsets list overrides the metadata-derived offset."""
        product = open_product(_L2A)
        sub = product.subdataset_for(60).open()
        ds = sub.bands.select([1, 2])
        _scaling.tag_reflectance(ds, product, offsets=[-1000.0, 0.0])
        assert ds.offset[0] == pytest.approx(-1000.0 / product.quantification)

    def test_stamp_baseline_swallows_write_failure(self):
        """`_stamp_baseline` never raises when the metadata write fails.

        Test scenario:
            A dataset whose ``meta_data`` setter raises leaves scaling
            unaffected (the defensive except path).
        """

        class _Stubborn:
            baseline = "05.09"

            @property
            def meta_data(self):
                return {}

            @meta_data.setter
            def meta_data(self, value):
                raise RuntimeError("read only")

        product = open_product(_L2A)
        # Should not raise despite the setter blowing up.
        _scaling._stamp_baseline(_Stubborn(), product)


# -- masks -----------------------------------------------------------------


class TestResolveClasses:
    """Tests for `masks._resolve_classes`."""

    def test_mixed_selectors(self):
        """Enum members, names, and raw ints all resolve to codes."""
        codes = _masks._resolve_classes([SclClass.WATER, "CLOUD_HIGH_PROBA", 3])
        assert codes == {6, 9, 3}

    def test_unknown_name_raises(self):
        """An unknown class name raises ``ProductError``."""
        with pytest.raises(ProductError, match="unknown SCL class"):
            _masks._resolve_classes(["NOPE"])


class TestSclMask:
    """Tests for `masks.scl_mask` and its SCL-source resolution."""

    def test_explicit_ndarray_scl(self):
        """An explicit SCL ndarray masks the matching pixels to no-data.

        Test scenario:
            A one-band dataset masked by a hand-built SCL array where one pixel
            is WATER sets that pixel to the no-data value.
        """
        from pyramids.dataset import Dataset

        arr = np.arange(16, dtype="uint16").reshape(1, 4, 4)
        ds = Dataset.create_from_array(arr=arr, geo=(0, 1, 0, 4, 0, -1), epsg=4326)
        ds.no_data_value = [0]
        scl = np.zeros((4, 4), dtype="uint8")
        scl[0, 0] = int(SclClass.WATER)
        masked = scl_mask(ds, [SclClass.WATER], scl=scl)
        assert masked.read_array()[0, 0] == 0

    def test_scl_from_single_band_dataset(self):
        """An explicit single-band SCL ``Dataset`` is read via band 0."""
        from pyramids.dataset import Dataset

        data = Dataset.create_from_array(
            arr=np.ones((1, 4, 4), dtype="uint16"),
            geo=(0, 1, 0, 4, 0, -1),
            epsg=4326,
        )
        scl_ds = Dataset.create_from_array(
            arr=np.full((1, 4, 4), int(SclClass.CLOUD_HIGH_PROBA), dtype="uint16"),
            geo=(0, 1, 0, 4, 0, -1),
            epsg=4326,
        )
        masked = scl_mask(data, [SclClass.CLOUD_HIGH_PROBA], scl=scl_ds)
        assert masked.band_count == 1

    def test_shape_mismatch_raises(self):
        """An SCL grid that does not match the data grid raises ``ProductError``."""
        from pyramids.dataset import Dataset

        ds = Dataset.create_from_array(
            arr=np.ones((1, 4, 4), dtype="uint16"),
            geo=(0, 1, 0, 4, 0, -1),
            epsg=4326,
        )
        wrong_grid = np.zeros((3, 3), dtype="uint8")
        with pytest.raises(ProductError, match="does not match"):
            scl_mask(ds, [SclClass.WATER], scl=wrong_grid)

    def test_no_scl_available_raises(self):
        """Masking a dataset with no SCL band and no ``scl=`` raises."""
        from pyramids.dataset import Dataset

        ds = Dataset.create_from_array(
            arr=np.ones((1, 4, 4), dtype="uint16"),
            geo=(0, 1, 0, 4, 0, -1),
            epsg=4326,
        )
        with pytest.raises(ProductError, match="no SCL band"):
            scl_mask(ds, [SclClass.WATER])

    def test_masked_pixel_takes_resolved_nodata(self):
        """A masked pixel is set to the dataset's resolved no-data value."""
        from pyramids.dataset import Dataset

        ds = Dataset.create_from_array(
            arr=np.ones((1, 4, 4), dtype="uint16"),
            geo=(0, 1, 0, 4, 0, -1),
            epsg=4326,
        )
        scl = np.zeros((4, 4), dtype="uint8")
        scl[1, 1] = int(SclClass.WATER)
        masked = scl_mask(ds, [SclClass.WATER], scl=scl)
        assert masked.read_array()[1, 1] == _masks._nodata_of(ds)

    @pytest.mark.parametrize(
        "nodata_values, expected",
        [((None, None), 0.0), ((5.0,), 5.0), ((None, 7.0), 7.0)],
    )
    def test_nodata_of_default(self, nodata_values, expected):
        """`_nodata_of` returns the first defined value, else the S2 default (0).

        Args:
            nodata_values: The dataset's ``no_data_value`` tuple.
            expected: The value ``_nodata_of`` should return.
        """

        class _Fake:
            no_data_value = nodata_values

        assert _masks._nodata_of(_Fake()) == expected

    def test_carry_band_state_swallows_band_names_but_not_tags(self):
        """Band-name copy failures are swallowed; scale/offset failures propagate.

        Test scenario:
            The scale/offset tags are the reflectance calibration, so a failure
            to carry them must surface (not silently drop reflectance), while a
            display-only band-names failure is ignored.
        """

        class _Source:
            band_names = ["a"]
            scale = [1.0]
            offset = [0.0]

        class _RejectsNames:
            scale = [1.0]
            offset = [0.0]

            def __setattr__(self, name, value):
                if name == "band_names":
                    raise RuntimeError("no band names")
                object.__setattr__(self, name, value)

        # band_names rejected -> swallowed, scale/offset copied fine.
        dest = _RejectsNames()
        _masks._carry_band_state(_Source(), dest)
        assert dest.scale == [1.0]

        class _RejectsTags:
            band_names = ["a"]

            def __setattr__(self, name, value):
                if name in ("scale", "offset"):
                    raise RuntimeError("read only")
                object.__setattr__(self, name, value)

        with pytest.raises(RuntimeError, match="read only"):
            _masks._carry_band_state(_Source(), _RejectsTags())


# -- reader edge cases -----------------------------------------------------


class TestReaderEdges:
    """Tests for `s2.reader` branches not hit by the happy path."""

    def test_bbox_crops_output_and_keeps_reflectance(self):
        """A ``bbox`` crops the output AND the reflectance tags survive the crop.

        Test scenario:
            pyramids ``crop`` resets per-band scale/offset, so the reader must
            re-tag after cropping — a cropped read must still scale to
            reflectance (``1/quantification``), not raw DN.
        """
        from pyramids_eo.sentinel import from_sentinel2

        product = open_product(_L2A)
        full = from_sentinel2(product, bands=["B04"])
        bb = full.bbox
        window = (
            bb[0],
            bb[1],
            bb[0] + (bb[2] - bb[0]) / 2,
            bb[1] + (bb[3] - bb[1]) / 2,
        )
        cropped = from_sentinel2(_L2A, bands=["B04"], bbox=window)
        assert cropped.shape[2] < full.shape[2]
        assert cropped.scale[0] == pytest.approx(1.0 / product.quantification)

    def test_bbox_and_reproject_keep_reflectance(self):
        """Reflectance survives a combined crop + reprojection."""
        from pyramids_eo.sentinel import from_sentinel2

        product = open_product(_L2A)
        bb = from_sentinel2(product, bands=["B04"]).bbox
        window = (
            bb[0],
            bb[1],
            bb[0] + (bb[2] - bb[0]) / 2,
            bb[1] + (bb[3] - bb[1]) / 2,
        )
        out = from_sentinel2(_L2A, bands=["B04"], bbox=window, crs=4326)
        assert out.epsg == 4326
        assert out.scale[0] == pytest.approx(1.0 / product.quantification)

    def test_mask_scl_on_product_without_scl_raises(self):
        """Requesting an SCL mask on L1C (no SCL band) raises ``ProductError``."""
        from pyramids_eo.sentinel import from_sentinel2

        with pytest.raises(ProductError, match="SCL"):
            from_sentinel2(_L1C, bands=["B04"], mask_scl=[SclClass.WATER])

    def test_empty_bands_list_raises(self):
        """An explicit ``bands=[]`` is rejected, not silently read as 'all'."""
        from pyramids_eo.sentinel import from_sentinel2

        with pytest.raises(ProductError, match="empty"):
            from_sentinel2(_L2A, bands=[])

    def test_collection_rejects_path_out_kwarg(self, tmp_path):
        """`collection_from_sentinel2` rejects a ``path_out`` kwarg with a clear error."""
        from pyramids_eo.sentinel import collection_from_sentinel2

        root = tmp_path / "s"
        out = tmp_path / "x.tif"
        with pytest.raises(ProductError, match="path_out"):
            collection_from_sentinel2([_L2A], root_dir=root, path_out=out)


class TestReaderHelpers:
    """Direct unit tests for `s2.reader` pure helpers (fake products/datasets)."""

    def test_resolve_epsg_multi_zone_raises(self):
        """A product spanning multiple UTM zones requires an explicit ``epsg``."""
        from pyramids_eo.sentinel.s2 import reader as _reader

        class _MultiZone:
            epsg_codes = [32631, 32632]

        with pytest.raises(ProductError, match="spans UTM zones"):
            _reader._resolve_epsg(_MultiZone(), None)

    def test_resolve_epsg_explicit_wins(self):
        """An explicit ``epsg`` is returned regardless of the product's zones."""
        from pyramids_eo.sentinel.s2 import reader as _reader

        class _MultiZone:
            epsg_codes = [32631, 32632]

        assert _reader._resolve_epsg(_MultiZone(), 32631) == 32631

    def test_resolve_epsg_zero_zones_returns_none(self):
        """A product with no declared UTM zone resolves to ``None``."""
        from pyramids_eo.sentinel.s2 import reader as _reader

        class _NoZone:
            epsg_codes: list[int] = []

        assert _reader._resolve_epsg(_NoZone(), None) is None

    def test_band_index_not_found_raises(self):
        """`_band_index` raises when the band is not in the list."""
        from pyramids_eo.sentinel.s2 import reader as _reader

        with pytest.raises(ProductError, match="not found"):
            _reader._band_index(["B1", "B2"], "B99")

    def test_native_subdataset_missing_band_raises(self):
        """`_native_subdataset` raises for a band absent from the product."""
        from pyramids_eo.sentinel.s2 import reader as _reader

        product = open_product(_L2A)
        with pytest.raises(ProductError, match="not in product"):
            _reader._native_subdataset(product, "B99", 32632)

    def test_set_nodata_swallows_failure(self):
        """`_set_nodata` never raises when the no-data write fails."""
        from pyramids_eo.sentinel.s2 import reader as _reader

        class _NoWrite:
            band_count = 1

            @property
            def no_data_value(self):
                return (None,)

            @no_data_value.setter
            def no_data_value(self, value):
                raise RuntimeError("read only")

        product = open_product(_L2A)
        _reader._set_nodata(_NoWrite(), product)  # must not raise

    def test_set_nodata_reads_special_value(self):
        """`_set_nodata` applies the product's ``SPECIAL_VALUE_NODATA``."""
        from pyramids.dataset import Dataset

        from pyramids_eo.sentinel.s2 import reader as _reader

        product = open_product(_L2A)  # fixture carries SPECIAL_VALUE_NODATA=1
        ds = Dataset.create_from_array(
            arr=np.ones((1, 4, 4), dtype="uint16"), geo=(0, 1, 0, 4, 0, -1), epsg=4326
        )
        _reader._set_nodata(ds, product)
        assert ds.no_data_value[0] == float(product.metadata["SPECIAL_VALUE_NODATA"])
