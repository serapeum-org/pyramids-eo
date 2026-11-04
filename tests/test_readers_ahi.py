"""Tests for the Himawari AHI Himawari Standard Data reader."""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers.ahi import (
    HSD_SUFFIX,
    HsdHeader,
    _byte_order,
    _canonical_band,
    _unpack,
    band_from_filename,
    band_name,
    geotransform,
    parse_ahi_hsd,
    parse_hsd_header,
    pixel_size_m,
    read_ahi,
    walk_header_blocks,
)
from tests import _hsd_granule as fixture


def _planck_radiance(temperature: float, wavelength_m: float) -> float:
    """Forward Planck function, written independently of the reader's inverse.

    The reader inverts Planck's law with ``log1p``; this builds the radiance from
    a chosen temperature with the exponential form instead, so a round-trip test
    compares two independent expressions rather than one against itself.

    Args:
        temperature: Brightness temperature in kelvin.
        wavelength_m: Wavelength in metres.

    Returns:
        Spectral radiance in ``W m-2 sr-1 m-1``.
    """
    numerator = 2.0 * fixture.PLANCK_CONSTANT * fixture.LIGHT_SPEED**2
    exponent = (fixture.PLANCK_CONSTANT * fixture.LIGHT_SPEED) / (
        wavelength_m * fixture.BOLTZMANN_CONSTANT * temperature
    )
    return numerator / (wavelength_m**5 * (math.exp(exponent) - 1.0))


class TestBandNames:
    """Band-key helpers."""

    def test_band_name_zero_pads(self):
        assert band_name(1) == "B01"
        assert band_name(16) == "B16"

    @pytest.mark.parametrize(
        "spelling", [13, "13", "b13", "B13", "band13", "Band_13", " B13 ", "013"]
    )
    def test_canonical_band_accepts_every_spelling(self, spelling):
        assert _canonical_band(spelling) == "B13"

    @pytest.mark.parametrize("bad", ["B17", 0, 17, -1, "red", "", "B1.5", None, True])
    def test_canonical_band_rejects_non_bands(self, bad):
        with pytest.raises(ReaderError, match="band"):
            _canonical_band(bad)

    def test_band_from_filename_reads_the_convention(self):
        assert band_from_filename(fixture.hsd_filename(band=7)) == "B07"
        assert (
            band_from_filename("/a/b/HS_H09_20240101_0000_B03_FLDK_R05_S0110.DAT")
            == "B03"
        )

    def test_band_from_filename_returns_none_for_other_names(self):
        assert band_from_filename("scene.dat") is None
        assert band_from_filename("HS_H08_bad.DAT") is None

    def test_suffix_constant(self):
        assert HSD_SUFFIX == ".dat"


class TestHeaderWalk:
    """The self-describing header walk."""

    def test_walks_to_the_data_block(self):
        buffer = fixture.granule_bytes(columns=8, lines=4)
        blocks = walk_header_blocks(buffer, "<")
        assert set(blocks) == set(range(1, 13))
        # Every header block lands where the previous one ended.
        offset = 0
        for number in range(1, 12):
            assert blocks[number][0] == offset
            offset += blocks[number][1]
        assert blocks[12] == (offset, 8 * 4 * 2)

    def test_variable_length_blocks_are_not_the_fixed_ones(self):
        """A fixed offset table would miss the data block in these fixtures."""
        buffer = fixture.granule_bytes()
        blocks = walk_header_blocks(buffer, "<")
        for number, length in fixture.VARIABLE_BLOCK_LENGTHS.items():
            assert blocks[number][1] == length
            assert length not in fixture.FIXED_BLOCK_LENGTHS.values()

    @pytest.mark.parametrize("order", ["<", ">"])
    def test_block_ten_declares_its_length_as_an_i4(self, order):
        """Block #10 is the one block the spec gives a 4-byte length field.

        A little-endian ``I4`` under 65536 reads identically to an ``I2``, so a
        reader using one width for every block passes on a little-endian granule
        and fails on a big-endian one. Both orders are asserted here so the
        coincidence cannot hide the mistake.
        """
        buffer = fixture.granule_bytes(order=order)
        blocks = walk_header_blocks(buffer, order)
        assert blocks[10][1] == fixture.VARIABLE_BLOCK_LENGTHS[10]
        # The I4 is what the fixture wrote, so the two high bytes are zero and an
        # I2 read of a big-endian granule would see 0 rather than the length.
        offset = blocks[10][0]
        assert struct.unpack_from(f"{order}I", buffer, offset + 1)[0] == blocks[10][1]

    def test_only_block_ten_has_a_long_length_field(self):
        """Every other block's length is an I2; widening them all would misparse."""
        from pyramids_eo.sensors.readers.ahi import _LONG_LENGTH_BLOCKS

        assert _LONG_LENGTH_BLOCKS == frozenset({10})
        assert _LONG_LENGTH_BLOCKS == fixture.LONG_LENGTH_BLOCKS

    def test_fixed_block_lengths_match_the_specification(self):
        buffer = fixture.granule_bytes()
        blocks = walk_header_blocks(buffer, "<")
        for number, length in fixture.FIXED_BLOCK_LENGTHS.items():
            assert blocks[number][1] == length

    def test_rejects_a_block_declaring_the_wrong_number(self):
        buffer = bytearray(fixture.granule_bytes())
        blocks = walk_header_blocks(bytes(buffer), "<")
        buffer[blocks[5][0]] = 9  # block #5 claims to be #9
        payload = bytes(buffer)
        with pytest.raises(ReaderError, match="declares number 9, expected #5"):
            walk_header_blocks(payload, "<")

    def test_rejects_a_zero_length_block(self):
        buffer = bytearray(fixture.granule_bytes())
        offset = walk_header_blocks(bytes(buffer), "<")[8][0]
        struct.pack_into("<H", buffer, offset + 1, 0)
        payload = bytes(buffer)
        with pytest.raises(ReaderError, match="zero length"):
            walk_header_blocks(payload, "<")

    def test_rejects_a_fixed_block_of_the_wrong_length(self):
        buffer = bytearray(fixture.granule_bytes())
        offset = walk_header_blocks(bytes(buffer), "<")[7][0]
        struct.pack_into("<H", buffer, offset + 1, 48)
        payload = bytes(buffer)
        with pytest.raises(ReaderError, match=r"block #7 declares 48 bytes"):
            walk_header_blocks(payload, "<")

    def test_rejects_a_block_running_past_the_end(self):
        buffer = bytearray(fixture.granule_bytes())
        offset = walk_header_blocks(bytes(buffer), "<")[9][0]
        struct.pack_into("<H", buffer, offset + 1, 60000)
        payload = bytes(buffer)
        with pytest.raises(ReaderError, match="runs past the end"):
            walk_header_blocks(payload, "<")

    def test_rejects_a_header_truncated_mid_block(self):
        """Blocks #1 + #2 end at byte 332, so 333 bytes cut block #3's own header."""
        buffer = fixture.granule_bytes()[:333]
        with pytest.raises(ReaderError, match="truncated inside header block #3"):
            walk_header_blocks(buffer, "<")

    def test_rejects_a_header_cut_after_a_block_boundary(self):
        """A cut past the length field is reported as the block overrunning."""
        buffer = fixture.granule_bytes()[:400]
        with pytest.raises(ReaderError, match="runs past the end"):
            walk_header_blocks(buffer, "<")


class TestByteOrder:
    """Byte-order detection from block #1."""

    @pytest.mark.parametrize("order", ["<", ">"])
    def test_detects_both_orders(self, order):
        assert _byte_order(fixture.granule_bytes(order=order)) == order

    @pytest.mark.parametrize("order", ["<", ">"])
    def test_header_decodes_identically_either_way(self, order):
        header = parse_hsd_header(fixture.granule_bytes(order=order, band=13))
        assert (header.columns, header.lines, header.band) == (8, 4, "B13")
        assert header.cfac == fixture.CFAC_BY_RESOLUTION_M[2000]
        assert header.sub_lon == pytest.approx(140.7)

    @pytest.mark.parametrize("order", ["<", ">"])
    def test_pixels_decode_identically_either_way(self, order):
        counts = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        buffer = fixture.granule_bytes(order=order, columns=2, lines=2, counts=counts)
        values = parse_ahi_hsd_array(buffer)
        assert values == pytest.approx(counts * 0.5 - 10.0)

    def test_rejects_a_file_not_starting_with_block_one(self):
        buffer = bytearray(fixture.granule_bytes())
        buffer[0] = 7
        payload = bytes(buffer)
        with pytest.raises(ReaderError, match="does not start with header block #1"):
            _byte_order(payload)

    def test_rejects_a_file_shorter_than_block_one(self):
        with pytest.raises(ReaderError, match="shorter than the 282-byte"):
            _byte_order(b"\x01\x1a\x01")

    def test_rejects_a_byte_order_flag_contradicting_the_block_length(self):
        buffer = bytearray(fixture.granule_bytes(order="<"))
        buffer[5] = 1  # claim big-endian while the length is little-endian
        payload = bytes(buffer)
        with pytest.raises(ReaderError, match="unexpected byte order or layout"):
            _byte_order(payload)


def parse_ahi_hsd_array(buffer: bytes) -> np.ndarray:
    """Decode a granule held in memory to its radiance array (test helper)."""
    from pyramids_eo.sensors.readers.ahi import _counts, _radiance

    header = parse_hsd_header(buffer)
    blocks = walk_header_blocks(buffer, _byte_order(buffer))
    return _radiance(_counts(buffer, header, blocks), header)


class TestHeaderFields:
    """Field-by-field header decode."""

    def test_decodes_every_field(self):
        header = parse_hsd_header(
            fixture.granule_bytes(
                band=13,
                columns=6,
                lines=3,
                satellite="H09",
                sub_lon=140.7,
                slope=0.25,
                intercept=-5.0,
                wavelength_m=10.4e-6,
                total_segments=10,
                segment_number=4,
                first_line=1651,
            )
        )
        assert isinstance(header, HsdHeader)
        assert header.satellite == "H09"
        assert (header.band, header.band_number, header.kind) == ("B13", 13, "thermal")
        assert (header.columns, header.lines) == (6, 3)
        assert (header.bits_per_pixel, header.valid_bits) == (16, 14)
        assert header.compression == 0
        assert (header.slope, header.intercept) == (0.25, -5.0)
        assert (header.error_count, header.outside_count) == (65535, 65534)
        assert header.central_wavelength_m == pytest.approx(10.4e-6)
        assert (header.total_segments, header.segment_number) == (10, 4)
        assert header.first_line == 1651
        assert header.rs_km == pytest.approx(42164.0)
        assert header.req_km == pytest.approx(6378.1370)
        assert header.rpol_km == pytest.approx(6356.7523)

    def test_satellite_height_is_the_distance_less_the_equatorial_radius(self):
        header = parse_hsd_header(fixture.granule_bytes())
        assert header.satellite_height_m == pytest.approx(
            (42164.0 - 6378.1370) * 1000.0
        )

    @pytest.mark.parametrize(
        ("band", "kind"),
        [(1, "solar"), (6, "solar"), (7, "thermal"), (16, "thermal")],
    )
    def test_kind_follows_the_band_number(self, band, kind):
        header = parse_hsd_header(fixture.granule_bytes(band=band))
        assert header.kind == kind

    def test_solar_bands_carry_the_albedo_coefficient_and_no_planck(self):
        header = parse_hsd_header(
            fixture.granule_bytes(band=3, albedo_coefficient=0.004)
        )
        assert header.albedo_coefficient == pytest.approx(0.004)
        assert header.planck_c0 is None
        assert header.light_speed is None

    def test_thermal_bands_carry_planck_and_no_albedo(self):
        header = parse_hsd_header(
            fixture.granule_bytes(band=13, planck=(1.5, 0.998, 2.0e-6))
        )
        assert header.albedo_coefficient is None
        assert header.planck_c0 == pytest.approx(1.5)
        assert header.planck_c1 == pytest.approx(0.998)
        assert header.planck_c2 == pytest.approx(2.0e-6)
        assert header.light_speed == pytest.approx(fixture.LIGHT_SPEED)
        assert header.planck_constant == pytest.approx(fixture.PLANCK_CONSTANT)
        assert header.boltzmann_constant == pytest.approx(fixture.BOLTZMANN_CONSTANT)

    def test_a_recalibrated_solar_granule_prefers_the_updated_gain(self):
        header = parse_hsd_header(
            fixture.granule_bytes(
                band=3, slope=0.5, intercept=-10.0, updated=(0.75, -2.0)
            )
        )
        assert (header.slope, header.intercept) == (0.75, -2.0)

    def test_an_unrecalibrated_solar_granule_keeps_the_original_gain(self):
        """Fields 12/13 are zero until a recalibration; reading them would zero the gain."""
        header = parse_hsd_header(
            fixture.granule_bytes(band=3, slope=0.5, intercept=-10.0, updated=None)
        )
        assert (header.slope, header.intercept) == (0.5, -10.0)

    def test_rejects_an_out_of_range_band_number(self):
        buffer = bytearray(fixture.granule_bytes())
        offset = walk_header_blocks(bytes(buffer), "<")[5][0]
        struct.pack_into("<H", buffer, offset + 3, 42)
        payload = bytes(buffer)
        with pytest.raises(ReaderError, match="band number 42"):
            parse_hsd_header(payload)

    def test_rejects_an_empty_grid(self):
        buffer = fixture.granule_bytes(columns=0, lines=4)
        with pytest.raises(ReaderError, match="empty grid"):
            parse_hsd_header(buffer)

    @pytest.mark.parametrize("slope", [0.0, float("nan"), float("inf")])
    def test_rejects_a_non_physical_slope(self, slope):
        buffer = fixture.granule_bytes(band=13, slope=slope)
        with pytest.raises(ReaderError, match="non-physical count-to-radiance slope"):
            parse_hsd_header(buffer)


class TestProjection:
    """The normalised geostationary projection."""

    @pytest.mark.parametrize(
        ("resolution_m", "expected"), [(2000, 2000.0), (1000, 1000.0), (500, 500.0)]
    )
    def test_published_scaling_factors_give_the_published_grids(
        self, resolution_m, expected
    ):
        """The published CFAC reproduces the real AHI grid to five significant figures.

        CFAC is an integer, so it cannot encode the grid exactly; the residual is
        under a millimetre. A degrees/radians mix-up would be out by a factor 57,
        so this tolerance still discriminates the convention decisively.
        """
        header = parse_hsd_header(fixture.granule_bytes(resolution_m=resolution_m))
        dx, dy = pixel_size_m(header)
        assert dx == pytest.approx(expected, rel=1e-5)
        assert dy == pytest.approx(expected, rel=1e-5)

    def test_the_grid_spans_the_earth_disk(self):
        """A sanity check using only in-granule values: the disk must fit the grid."""
        header = parse_hsd_header(fixture.granule_bytes(resolution_m=2000))
        step_deg = 65536.0 / header.cfac
        span = 5500 * step_deg
        earth = 2.0 * math.degrees(math.asin(header.req_km / header.rs_km))
        assert earth < span < earth * 1.1

    def test_a_full_disk_is_symmetric_about_the_sub_satellite_point(self):
        header = parse_hsd_header(fixture.granule_bytes(resolution_m=2000))
        x_min, dx, _, y_max, _, dy = geotransform(header, 1)
        # COFF/LOFF sit at the grid centre, so the west edge and north edge are
        # equal and opposite, and 5500 pixels wide.
        assert x_min == pytest.approx(-y_max)
        assert x_min + 5500 * dx == pytest.approx(-x_min)
        assert y_max + 5500 * dy == pytest.approx(-y_max)

    def test_a_segment_is_offset_to_its_own_first_line(self):
        header = parse_hsd_header(
            fixture.granule_bytes(resolution_m=2000, first_line=1651)
        )
        _, _, _, y_full, _, dy = geotransform(header, 1)
        _, _, _, y_segment, _, _ = geotransform(header, 1651)
        # 1650 lines further south, each one pixel high (dy is negative).
        assert y_segment == pytest.approx(y_full + 1650 * dy)

    @pytest.mark.parametrize("factor", [0, 1, 10**9])
    def test_rejects_a_scaling_factor_implying_an_implausible_pixel(self, factor):
        buffer = bytearray(fixture.granule_bytes())
        offset = walk_header_blocks(bytes(buffer), "<")[3][0]
        struct.pack_into("<I", buffer, offset + 11, factor)
        header = parse_hsd_header(bytes(buffer))
        with pytest.raises(ReaderError, match="zero CFAC|outside the plausible"):
            pixel_size_m(header)

    def test_rejects_a_non_physical_satellite_height(self):
        buffer = bytearray(fixture.granule_bytes())
        offset = walk_header_blocks(bytes(buffer), "<")[3][0]
        struct.pack_into("<d", buffer, offset + 27, 1.0)  # Rs below the Earth's radius
        header = parse_hsd_header(bytes(buffer))
        with pytest.raises(ReaderError, match="non-physical satellite height"):
            pixel_size_m(header)

    def test_the_crs_is_geostationary_at_the_sub_satellite_longitude(self, tmp_path):
        path = fixture.write_granule(tmp_path, band=13, sub_lon=140.7)
        dataset = parse_ahi_hsd(path)
        assert "Geostationary" in dataset.crs
        assert "140.7" in dataset.crs


class TestRadiance:
    """Count-to-radiance conversion and sentinel masking."""

    def test_applies_the_granule_slope_and_intercept(self, tmp_path):
        counts = np.array([[0, 1], [2, 3]], dtype=np.uint16)
        path = fixture.write_granule(
            tmp_path, columns=2, lines=2, counts=counts, slope=0.5, intercept=-10.0
        )
        values = parse_ahi_hsd(path).read_array()
        assert values.ravel() == pytest.approx([-10.0, -9.5, -9.0, -8.5])

    def test_masks_both_sentinel_counts(self, tmp_path):
        counts = np.array(
            [[10, fixture.ERROR_COUNT], [fixture.OUTSIDE_COUNT, 20]], dtype=np.uint16
        )
        path = fixture.write_granule(tmp_path, columns=2, lines=2, counts=counts)
        values = parse_ahi_hsd(path).read_array()
        assert np.isnan(values[0, 1]), "the error-count pixel must be masked"
        assert np.isnan(values[1, 0]), "the outside-scan pixel must be masked"
        assert np.isfinite(values[0, 0]), f"valid pixel masked: {values[0, 0]}"
        assert np.isfinite(values[1, 1]), f"valid pixel masked: {values[1, 1]}"

    def test_rows_are_stored_north_up_and_not_flipped(self, tmp_path):
        """Positive LFAC means line 1 is northernmost, so the rows stand as stored."""
        counts = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.uint16)
        path = fixture.write_granule(
            tmp_path, columns=2, lines=3, counts=counts, slope=1.0, intercept=0.0
        )
        values = parse_ahi_hsd(path).read_array()
        assert values[:, 0] == pytest.approx([1.0, 2.0, 3.0])

    def test_reads_a_bzip2_granule(self, tmp_path):
        counts = np.array([[7, 8]], dtype=np.uint16)
        path = fixture.write_granule(
            tmp_path,
            columns=2,
            lines=1,
            counts=counts,
            slope=1.0,
            intercept=0.0,
            compress=True,
        )
        assert path.name.endswith(".DAT.bz2")
        assert parse_ahi_hsd(path).read_array().ravel() == pytest.approx([7.0, 8.0])

    def test_rejects_a_compressed_data_block(self, tmp_path):
        path = fixture.write_granule(tmp_path, compression=2)
        with pytest.raises(ReaderError, match="compression flag 2"):
            parse_ahi_hsd(path)

    def test_rejects_a_non_sixteen_bit_data_block(self, tmp_path):
        path = fixture.write_granule(tmp_path)
        buffer = bytearray(path.read_bytes())
        offset = walk_header_blocks(bytes(buffer), "<")[2][0]
        struct.pack_into("<H", buffer, offset + 3, 8)
        path.write_bytes(bytes(buffer))
        with pytest.raises(ReaderError, match="8 bits per pixel"):
            parse_ahi_hsd(path)

    def test_rejects_a_truncated_data_block(self, tmp_path):
        path = fixture.write_granule(tmp_path, columns=8, lines=4)
        path.write_bytes(path.read_bytes()[:-10])
        with pytest.raises(ReaderError, match="data block holds"):
            parse_ahi_hsd(path)

    def test_reports_an_unreadable_path(self, tmp_path):
        with pytest.raises(ReaderError, match="cannot read"):
            parse_ahi_hsd(tmp_path / "absent.DAT")

    def test_reports_a_corrupt_bzip2_granule(self, tmp_path):
        path = tmp_path / "HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT.bz2"
        path.write_bytes(b"not bzip2 at all")
        with pytest.raises(ReaderError, match="cannot read"):
            parse_ahi_hsd(path)


class TestCalibration:
    """Calibration to brightness temperature and reflectance."""

    @pytest.mark.parametrize("temperature", [200.0, 250.0, 300.0, 330.0])
    def test_thermal_round_trips_an_independently_built_planck_radiance(
        self, tmp_path, temperature
    ):
        """Build the radiance forward from a known temperature, recover it back.

        The expected value comes from an exponential-form Planck function written
        in this test, while the reader inverts with ``log1p`` — so this compares
        two independent expressions, not one against itself.
        """
        wavelength = 10.4e-6
        radiance = _planck_radiance(temperature, wavelength)
        # Choose a gain that lands this radiance on an exact integer count.
        count = 1000
        slope = radiance / count
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=1,
            lines=1,
            counts=np.array([[count]], dtype=np.uint16),
            slope=slope,
            intercept=0.0,
            wavelength_m=wavelength,
            planck=(0.0, 1.0, 0.0),  # no correction: Tb == Te
        )
        result = read_ahi(path, "B13").read_array()
        assert float(result[0, 0]) == pytest.approx(temperature, rel=1e-9)

    def test_thermal_applies_the_granule_quadratic_correction(self, tmp_path):
        wavelength = 10.4e-6
        temperature = 280.0
        radiance = _planck_radiance(temperature, wavelength)
        c0, c1, c2 = 1.5, 0.995, 3.0e-6
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=1,
            lines=1,
            counts=np.array([[1000]], dtype=np.uint16),
            slope=radiance / 1000,
            intercept=0.0,
            wavelength_m=wavelength,
            planck=(c0, c1, c2),
        )
        expected = c0 + c1 * temperature + c2 * temperature**2
        assert float(read_ahi(path, "B13").read_array()[0, 0]) == pytest.approx(
            expected, rel=1e-9
        )

    def test_thermal_leaves_masked_pixels_nan(self, tmp_path):
        counts = np.array([[1000, fixture.ERROR_COUNT]], dtype=np.uint16)
        radiance = _planck_radiance(300.0, 10.4e-6)
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=2,
            lines=1,
            counts=counts,
            slope=radiance / 1000,
            intercept=0.0,
        )
        result = read_ahi(path, "B13").read_array()
        assert np.isfinite(result[0, 0]), f"valid pixel lost: {result[0, 0]}"
        assert np.isnan(result[0, 1]), "the error-count pixel must stay masked"

    def test_solar_uses_the_granule_irradiance_in_the_standard_convention(
        self, tmp_path
    ):
        """Reflectance is pi * L * d^2 / S0, with S0 = 1 / c' from the granule."""
        albedo_coefficient = 0.0025  # c', so S0 = 400
        path = fixture.write_granule(
            tmp_path,
            band=3,
            columns=1,
            lines=1,
            counts=np.array([[100]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
            albedo_coefficient=albedo_coefficient,
        )
        result = read_ahi(path, "B03", sun_earth_distance=1.0).read_array()
        expected = math.pi * 100.0 * 1.0 / (1.0 / albedo_coefficient)
        assert float(result[0, 0]) == pytest.approx(expected)

    def test_solar_scales_with_the_sun_earth_distance(self, tmp_path):
        path = fixture.write_granule(
            tmp_path,
            band=3,
            columns=1,
            lines=1,
            counts=np.array([[100]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
        )
        near = float(read_ahi(path, "B03", sun_earth_distance=1.0).read_array()[0, 0])
        far = float(read_ahi(path, "B03", sun_earth_distance=1.1).read_array()[0, 0])
        assert far == pytest.approx(near * 1.1**2)

    def test_calibrate_false_returns_the_raw_radiance(self, tmp_path):
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=2,
            lines=1,
            counts=np.array([[10, 20]], dtype=np.uint16),
            slope=0.5,
            intercept=-1.0,
        )
        values = read_ahi(path, "B13", calibrate=False).read_array()
        assert values.ravel() == pytest.approx([4.0, 9.0])

    def test_explicit_coeffs_override_the_granule(self, tmp_path):
        """A caller's coeffs route a thermal band through the registry's linear form."""
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=1,
            lines=1,
            counts=np.array([[1000]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
        )
        granule = float(read_ahi(path, "B13").read_array()[0, 0])
        overridden = float(
            read_ahi(
                path,
                "B13",
                coeffs={"central_wavenumber_cm1": 961.538, "alpha": 1.0, "beta": 0.0},
            ).read_array()[0, 0]
        )
        assert not math.isclose(granule, overridden, rel_tol=1e-6)

    def test_thermal_missing_a_constant_is_reported(self, tmp_path):
        path = fixture.write_granule(tmp_path, band=13)
        buffer = bytearray(path.read_bytes())
        offset = walk_header_blocks(bytes(buffer), "<")[5][0]
        struct.pack_into("<d", buffer, offset + 83, float("nan"))  # speed of light
        path.write_bytes(bytes(buffer))
        with pytest.raises(ReaderError, match="missing the calibration constant"):
            read_ahi(path, "B13")

    def test_thermal_with_a_non_physical_wavelength_is_reported(self, tmp_path):
        path = fixture.write_granule(tmp_path, band=13, wavelength_m=0.0)
        with pytest.raises(ReaderError, match="non-physical central wavelength"):
            read_ahi(path, "B13")


class TestSegments:
    """Segment stitching."""

    def test_stitches_segments_in_order(self, tmp_path):
        paths = fixture.segment_set(
            tmp_path,
            segments=3,
            lines_per_segment=4,
            columns=5,
            slope=1.0,
            intercept=0.0,
        )
        values = read_ahi(paths, "B13", calibrate=False).read_array()
        assert values.shape == (12, 5)
        # Each segment wrote its own constant count, so the rows prove placement.
        assert values[:, 0] == pytest.approx([1.0] * 4 + [2.0] * 4 + [3.0] * 4)

    def test_stitches_regardless_of_the_order_given(self, tmp_path):
        paths = fixture.segment_set(
            tmp_path,
            segments=3,
            lines_per_segment=2,
            columns=3,
            slope=1.0,
            intercept=0.0,
        )
        shuffled = [paths[2], paths[0], paths[1]]
        values = read_ahi(shuffled, "B13", calibrate=False).read_array()
        assert values[:, 0] == pytest.approx([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

    def test_the_stitched_grid_starts_at_the_first_segment(self, tmp_path):
        paths = fixture.segment_set(
            tmp_path, segments=2, lines_per_segment=3, columns=4
        )
        stitched = read_ahi(paths, "B13", calibrate=False)
        single = parse_ahi_hsd(paths[0])
        assert stitched.geotransform == pytest.approx(single.geotransform)

    def test_a_lone_segment_is_georeferenced_where_it_belongs(self, tmp_path):
        paths = fixture.segment_set(
            tmp_path, segments=3, lines_per_segment=4, columns=4
        )
        first = parse_ahi_hsd(paths[0])
        third = parse_ahi_hsd(paths[2])
        dy = first.geotransform[5]
        assert third.geotransform[3] == pytest.approx(first.geotransform[3] + 8 * dy)

    def test_rejects_a_gap_between_segments(self, tmp_path):
        paths = fixture.segment_set(
            tmp_path, segments=3, lines_per_segment=4, columns=4
        )
        with pytest.raises(ReaderError, match="not contiguous"):
            read_ahi([paths[0], paths[2]], "B13")

    def test_rejects_a_duplicated_segment(self, tmp_path):
        paths = fixture.segment_set(
            tmp_path, segments=2, lines_per_segment=4, columns=4
        )
        with pytest.raises(ReaderError, match="not contiguous"):
            read_ahi([paths[0], paths[0]], "B13")

    def test_rejects_segments_of_different_bands(self, tmp_path):
        first = fixture.write_granule(
            tmp_path,
            band=13,
            columns=4,
            lines=4,
            first_line=1,
            segment_number=1,
            total_segments=2,
        )
        second = fixture.write_granule(
            tmp_path,
            band=14,
            columns=4,
            lines=4,
            first_line=5,
            segment_number=2,
            total_segments=2,
        )
        with pytest.raises(ReaderError, match="mix bands"):
            read_ahi([first, second])

    def test_rejects_segments_of_different_widths(self, tmp_path):
        first = fixture.write_granule(
            tmp_path,
            band=13,
            columns=4,
            lines=4,
            first_line=1,
            segment_number=1,
            total_segments=2,
        )
        second = fixture.write_granule(
            tmp_path,
            band=13,
            columns=6,
            lines=4,
            first_line=5,
            segment_number=2,
            total_segments=2,
        )
        with pytest.raises(ReaderError, match="disagree on width"):
            read_ahi([first, second])

    def test_rejects_segments_with_different_projections(self, tmp_path):
        first = fixture.write_granule(
            tmp_path,
            band=13,
            columns=4,
            lines=4,
            first_line=1,
            segment_number=1,
            total_segments=2,
        )
        second = fixture.write_granule(
            tmp_path,
            band=13,
            columns=4,
            lines=4,
            first_line=5,
            segment_number=2,
            total_segments=2,
            coff=10.0,
        )
        with pytest.raises(ReaderError, match="disagree on the projection"):
            read_ahi([first, second])

    def test_rejects_an_empty_segment_list(self):
        with pytest.raises(ReaderError, match="no HSD segment paths"):
            read_ahi([], "B13")


class TestReadAhi:
    """The `read_ahi` entry point."""

    def test_band_is_optional_because_the_granule_names_it(self, tmp_path):
        path = fixture.write_granule(
            tmp_path, band=7, columns=2, lines=1, slope=1.0, intercept=1.0
        )
        assert read_ahi(path, calibrate=False).read_array().shape == (1, 2)

    def test_a_requested_band_matching_the_granule_is_accepted(self, tmp_path):
        path = fixture.write_granule(tmp_path, band=7)
        for spelling in (7, "B07", "band7"):
            assert read_ahi(path, spelling, calibrate=False) is not None

    def test_a_requested_band_contradicting_the_granule_is_rejected(self, tmp_path):
        path = fixture.write_granule(tmp_path, band=7)
        with pytest.raises(
            ReaderError, match="B13 was requested but the granule carries B07"
        ):
            read_ahi(path, "B13")

    def test_requires_a_source(self):
        with pytest.raises(ReaderError, match="source is required"):
            read_ahi(None, "B13")

    def test_a_dataset_source_requires_the_band(self, tmp_path):
        dataset = parse_ahi_hsd(fixture.write_granule(tmp_path, band=13))
        with pytest.raises(
            ReaderError, match="band is required when source is already a Dataset"
        ):
            read_ahi(dataset)

    def test_a_dataset_source_is_calibrated_on_its_own_grid(self, tmp_path):
        path = fixture.write_granule(
            tmp_path,
            band=3,
            columns=2,
            lines=1,
            counts=np.array([[100, 200]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
        )
        dataset = parse_ahi_hsd(path)
        result = read_ahi(
            dataset, "B03", coeffs={"kind": "solar", "solar_irradiance": 400.0}
        )
        assert result.geotransform == pytest.approx(dataset.geotransform)
        assert result.read_array()[0, 0] == pytest.approx(math.pi * 100.0 / 400.0)

    def test_the_result_keeps_the_geostationary_grid(self, tmp_path):
        path = fixture.write_granule(tmp_path, band=13, columns=3, lines=2)
        radiance = parse_ahi_hsd(path)
        calibrated = read_ahi(path, "B13")
        assert calibrated.geotransform == pytest.approx(radiance.geotransform)
        assert "Geostationary" in calibrated.crs


class TestInternals:
    """Two paths the public API cannot reach, covered directly."""

    def test_unpack_reports_a_field_running_past_the_buffer(self):
        """Defensive depth in a shared helper.

        `walk_header_blocks` proves every header block lies wholly inside the
        granule before any field is read, so no public path can overrun. That
        guarantee is non-local, though, and `_unpack` is used for every field, so
        it carries its own bound check — exercised here against its contract
        rather than left to a caller's invariant.
        """
        with pytest.raises(ReaderError, match="granule is truncated"):
            _unpack(b"", "<", "d", 0)

    def test_a_dataset_source_with_an_epsg_keeps_its_own_crs(self, tmp_path):
        """A `Dataset` on a projected grid skips the geostationary CRS carry-over."""
        from pyramids.dataset import Dataset, GeoReference

        source = Dataset.from_array(
            np.array([[100.0, 200.0]]),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), epsg=4326),
            no_data_value=np.nan,
        )
        result = read_ahi(
            source, "B03", coeffs={"kind": "solar", "solar_irradiance": 400.0}
        )
        assert result.epsg == 4326
        assert result.read_array()[0, 0] == pytest.approx(math.pi * 100.0 / 400.0)


@pytest.mark.live
def test_read_ahi_real_granule():
    """End-to-end decode of a real Himawari AHI HSD granule into brightness temperature.

    The offline suite builds its fixtures from the published HSD layout, so it
    proves the decoder is self-consistent but cannot prove it against reality.
    This is the CI-independent ground truth: point it at a real band-13 granule
    and the brightness temperatures, grid step and orientation must all land in
    their physical ranges.

    Skips unless a directory holding real HSD granules is provided via
    `AHI_FIXTURES_DIR` (the marker, not the env var, gates whether this runs).
    """
    fixtures = Path(os.environ.get("AHI_FIXTURES_DIR", "tests/data/ahi"))
    granules = sorted(fixtures.glob("HS_H0?_*_B13_*.DAT")) + sorted(
        fixtures.glob("HS_H0?_*_B13_*.DAT.bz2")
    )
    if not granules:
        pytest.skip("real AHI fixtures not available (set AHI_FIXTURES_DIR)")
    scene = read_ahi(str(granules[0]))
    array = np.asarray(scene.read_array(), dtype=float)
    finite = array[np.isfinite(array)]
    assert finite.size, "a real granule must decode some finite pixels"
    assert finite.min() > 180.0, f"BT min too low: {finite.min()}"
    assert finite.max() < 340.0, f"BT max too high: {finite.max()}"
    assert np.median(finite) > 220.0, f"BT median implausibly cold: {np.median(finite)}"
    assert "Geostationary" in str(scene.crs), "should carry the geostationary CRS"
    assert abs(scene.geotransform[1]) == pytest.approx(2000.0, abs=1.0), (
        "band 13 is the 2 km grid"
    )
    assert scene.geotransform[5] < 0, "the grid must be north-up (gt[5] < 0)"
    assert np.isnan(scene.no_data_value[0]), "nodata should be NaN"


@pytest.mark.live
def test_read_ahi_real_granule_matches_the_published_registration():
    """A real granule's grid agrees with the published Himawari full-disk navigation.

    Checks the decoded projection against values the granule does NOT supply:
    the sub-satellite longitude of the operational Himawari slot, and the fact
    that a full disk is registered symmetrically about that point.
    """
    fixtures = Path(os.environ.get("AHI_FIXTURES_DIR", "tests/data/ahi"))
    granules = sorted(fixtures.glob("HS_H0?_*_FLDK_*.DAT")) + sorted(
        fixtures.glob("HS_H0?_*_FLDK_*.DAT.bz2")
    )
    if not granules:
        pytest.skip("real AHI fixtures not available (set AHI_FIXTURES_DIR)")
    from pyramids_eo.sensors.readers.ahi import _read_bytes

    header = parse_hsd_header(_read_bytes(str(granules[0])))
    assert 139.0 < header.sub_lon < 142.0, (
        f"Himawari sits near 140.7E, got {header.sub_lon}"
    )
    assert header.satellite in {"H08", "H09"}, header.satellite
    assert header.rs_km == pytest.approx(42164.0, abs=1.0)
    assert header.req_km == pytest.approx(6378.137, abs=0.01)
    assert header.total_segments >= 1
    # A full disk's sub-satellite point sits at the centre of the whole image.
    full_lines = header.total_segments * header.lines
    assert header.coff == pytest.approx(header.columns / 2.0 + 0.5, abs=1.0)
    assert header.loff == pytest.approx(full_lines / 2.0 + 0.5, abs=1.0)


class TestGranuleGovernsDecoding:
    """Values the reader must take FROM the granule, not from its own constants.

    Line coverage cannot distinguish "reads the granule's field" from "happens to
    agree with the module constant", because the fixtures use the specification's
    own values. These cases make the granule disagree with the constant and check
    the granule wins — which is what a real product with a different declaration
    depends on.
    """

    def test_sentinel_counts_come_from_the_granule(self, tmp_path):
        """A granule declaring non-standard sentinels must still mask correctly."""
        counts = np.array([[10, 60000], [60001, 20]], dtype=np.uint16)
        path = fixture.write_granule(
            tmp_path,
            columns=2,
            lines=2,
            counts=counts,
            slope=1.0,
            intercept=0.0,
            error_count=60000,
            outside_count=60001,
        )
        values = parse_ahi_hsd(path).read_array()
        assert np.isnan(values[0, 1]), "the granule's declared error count must mask"
        assert np.isnan(values[1, 0]), "the granule's declared outside count must mask"
        assert values[0, 0] == pytest.approx(10.0)
        assert values[1, 1] == pytest.approx(20.0)

    def test_the_standard_sentinels_are_not_masked_when_not_declared(self, tmp_path):
        """65535 is only special because block #5 says so, not inherently."""
        counts = np.array([[65535, 65534]], dtype=np.uint16)
        path = fixture.write_granule(
            tmp_path,
            columns=2,
            lines=1,
            counts=counts,
            slope=1.0,
            intercept=0.0,
            error_count=60000,
            outside_count=60001,
        )
        values = parse_ahi_hsd(path).read_array()
        assert np.isfinite(values[0, 0]), "65535 must be ordinary data here"
        assert values[0, 0] == pytest.approx(65535.0)
        assert values[0, 1] == pytest.approx(65534.0)

    @pytest.mark.parametrize("order", ["<", ">"])
    def test_a_granule_decodes_end_to_end_in_either_byte_order(self, tmp_path, order):
        """The whole file path, not just the header, must honour block #1's flag."""
        counts = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        path = tmp_path / f"HS_H08_20150415_0230_B13_FLDK_R20_S0101_{order == '>'}.DAT"
        path.write_bytes(
            fixture.granule_bytes(
                order=order,
                columns=2,
                lines=2,
                counts=counts,
                slope=0.5,
                intercept=-10.0,
            )
        )
        values = parse_ahi_hsd(path).read_array()
        assert values.ravel() == pytest.approx([40.0, 90.0, 140.0, 190.0])

    @pytest.mark.parametrize(
        ("resolution_m", "band"), [(2000, 13), (1000, 1), (500, 3)]
    )
    def test_the_grid_step_comes_from_the_granule(self, tmp_path, resolution_m, band):
        """Each native grid must register its own step, not a built-in default."""
        path = fixture.write_granule(
            tmp_path, resolution_m=resolution_m, band=band, columns=2, lines=2
        )
        dataset = parse_ahi_hsd(path)
        assert abs(dataset.geotransform[1]) == pytest.approx(resolution_m, rel=1e-5)
        assert abs(dataset.geotransform[5]) == pytest.approx(resolution_m, rel=1e-5)

    def test_solar_bands_declare_eleven_valid_bits(self, tmp_path):
        header = parse_hsd_header(fixture.granule_bytes(band=3))
        assert header.valid_bits == 11, header.valid_bits


class TestSourceForms:
    """The shapes `source` may legitimately take."""

    def test_a_single_element_list_reads_as_one_granule(self, tmp_path):
        paths = fixture.segment_set(
            tmp_path,
            segments=1,
            lines_per_segment=3,
            columns=4,
            slope=1.0,
            intercept=0.0,
        )
        values = read_ahi(paths, "B13", calibrate=False).read_array()
        assert values.shape == (3, 4)

    def test_an_iterator_of_segments_is_accepted(self, tmp_path):
        """`source` is consumed once, so a generator must not be exhausted early."""
        paths = fixture.segment_set(
            tmp_path,
            segments=3,
            lines_per_segment=2,
            columns=3,
            slope=1.0,
            intercept=0.0,
        )
        values = read_ahi(iter(paths), "B13", calibrate=False).read_array()
        assert values.shape == (6, 3)
        assert values[:, 0] == pytest.approx([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])

    def test_a_tuple_of_segments_is_accepted(self, tmp_path):
        paths = fixture.segment_set(
            tmp_path,
            segments=2,
            lines_per_segment=2,
            columns=3,
            slope=1.0,
            intercept=0.0,
        )
        values = read_ahi(tuple(paths), "B13", calibrate=False).read_array()
        assert values.shape == (4, 3)

    def test_a_string_path_is_accepted(self, tmp_path):
        path = fixture.write_granule(tmp_path, band=13, columns=2, lines=1)
        assert read_ahi(str(path), calibrate=False).read_array().shape == (1, 2)

    def test_a_band_contradicting_a_stitched_set_is_rejected(self, tmp_path):
        """The band assertion must also apply on the multi-segment path."""
        paths = fixture.segment_set(
            tmp_path, segments=2, lines_per_segment=2, columns=3
        )
        with pytest.raises(
            ReaderError, match="B14 was requested but the granule carries B13"
        ):
            read_ahi(paths, "B14")


class TestCalibrationExtras:
    """Calibration paths not exercised above."""

    def test_cos_sza_applies_the_sun_angle_correction(self, tmp_path):
        path = fixture.write_granule(
            tmp_path,
            band=3,
            columns=1,
            lines=1,
            counts=np.array([[100]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
            albedo_coefficient=0.0025,
        )
        plain = float(read_ahi(path, "B03").read_array()[0, 0])
        corrected = float(read_ahi(path, "B03", cos_sza=0.5).read_array()[0, 0])
        assert corrected == pytest.approx(plain / 0.5)

    def test_cos_sza_at_the_terminator_is_nan(self, tmp_path):
        path = fixture.write_granule(
            tmp_path,
            band=3,
            columns=1,
            lines=1,
            counts=np.array([[100]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
        )
        result = read_ahi(path, "B03", cos_sza=0.0).read_array()
        assert np.isnan(result[0, 0]), "a non-positive cos(sza) has no reflectance"

    def test_non_positive_thermal_radiance_is_nan_not_an_error(self, tmp_path):
        """Planck has no inverse at or below zero radiance; those pixels go NaN."""
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=3,
            lines=1,
            counts=np.array([[0, 5, 1000]], dtype=np.uint16),
            slope=1e-8,
            intercept=-1e-8,  # the first two pixels land <= 0
        )
        result = read_ahi(path, "B13").read_array()
        assert np.isnan(result[0, 0]), "zero radiance must be NaN"
        assert np.isfinite(result[0, 2]), f"positive radiance lost: {result[0, 2]}"

    @pytest.mark.parametrize("flag", [1, 2])
    def test_both_compressed_data_block_flags_are_reported(self, tmp_path, flag):
        path = fixture.write_granule(tmp_path, compression=flag)
        with pytest.raises(ReaderError, match=f"compression flag {flag}"):
            parse_ahi_hsd(path)
