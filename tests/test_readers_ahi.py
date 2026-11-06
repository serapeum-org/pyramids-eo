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
    _read_header_bytes,
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


def _planck_radiance(temperature: float, wavelength_um: float) -> float:
    """Forward Planck function, written independently of the reader's inverse.

    The reader inverts Planck's law with ``log1p``; this builds the radiance from
    a chosen temperature with the exponential form instead, so a round-trip test
    compares two independent expressions rather than one against itself.

    Args:
        temperature: Brightness temperature in kelvin.
        wavelength_um: Wavelength in micrometres.

    Returns:
        Spectral radiance in ``W m-2 sr-1 um-1`` — HSD's own basis.
    """
    wavelength = wavelength_um * 1e-6
    numerator = 2.0 * fixture.PLANCK_CONSTANT * fixture.LIGHT_SPEED**2
    exponent = (fixture.PLANCK_CONSTANT * fixture.LIGHT_SPEED) / (
        wavelength * fixture.BOLTZMANN_CONSTANT * temperature
    )
    # Returned per MICROMETRE of wavelength, which is the basis HSD itself
    # uses, so a fixture can carry it as a granule's slope / intercept do.
    return numerator / (wavelength**5 * (math.exp(exponent) - 1.0)) / 1e6


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

    def test_rejects_a_granule_cut_inside_block_tens_length_field(self):
        """Block #10's length field is an I4, so four bytes of it is a cut block.

        A walk assuming the three-byte preamble every other block has would read
        a length out of the two bytes that are there and report the wrong fault
        (an overrun) instead of the truncation.
        """
        buffer = fixture.granule_bytes()
        offset = walk_header_blocks(buffer, "<")[10][0]
        with pytest.raises(ReaderError, match="truncated inside header block #10"):
            walk_header_blocks(buffer[: offset + 4], "<")

    def test_rejects_a_walk_disagreeing_with_the_declared_total(self):
        """Block #1's total header length is the walk's independent check.

        Every block can be self-consistent while the walk still drifted, so where
        the walk ends is compared with the total block #1 declares.
        """
        buffer = fixture.granule_bytes(columns=8, lines=4, header_length=12345)
        with pytest.raises(
            ReaderError, match=r"header walk ended at byte \d+ but block #1"
        ):
            walk_header_blocks(buffer, "<")

    @pytest.mark.parametrize("declared", [1, 10, 12, 255])
    def test_rejects_a_declared_block_count_other_than_eleven(self, declared):
        """Block #1 field 3 states how many header blocks the granule carries.

        Test scenario:
            The walk steps over exactly eleven blocks because that is what HSD
            defines, so a granule declaring any other count is a layout this
            reader cannot be walking correctly — even when every individual
            block happens to be self-consistent. Field 3 is patched in an
            otherwise valid granule and the count is expected in the message.
        """
        buffer = bytearray(fixture.granule_bytes(columns=8, lines=4))
        struct.pack_into("<H", buffer, 3, declared)
        payload = bytes(buffer)
        with pytest.raises(
            ReaderError, match=rf"block #1 declares {declared} header blocks"
        ):
            walk_header_blocks(payload, "<")

    def test_a_zero_declared_block_count_is_tolerated(self):
        """A granule leaving field 3 unset is walked rather than refused.

        Test scenario:
            The check is guarded on a truthy declaration, so field 3 patched to
            0 — the "not stated" spelling — must leave the walk's own result
            intact rather than raising about "0 header blocks".
        """
        buffer = bytearray(fixture.granule_bytes(columns=8, lines=4))
        struct.pack_into("<H", buffer, 3, 0)
        blocks = walk_header_blocks(bytes(buffer), "<")
        assert len(blocks) == 12, (
            f"expected 11 header blocks plus the data block, got {blocks}"
        )
        assert blocks[12][1] == 8 * 4 * 2, (
            f"the data block should hold the 8 x 4 payload, got {blocks[12]}"
        )

    def test_the_block_count_is_read_in_the_granule_byte_order(self):
        """Field 3 is an I2, so a big-endian granule's count must not read as 2816.

        Test scenario:
            A big-endian granule declaring 11 walks cleanly; the same granule
            with field 3 patched big-endian to 12 is refused with 12 in the
            message, which a little-endian read of the field could not produce.
        """
        assert len(walk_header_blocks(fixture.granule_bytes(order=">"), ">")) == 12
        buffer = bytearray(fixture.granule_bytes(order=">"))
        struct.pack_into(">H", buffer, 3, 12)
        payload = bytes(buffer)
        with pytest.raises(ReaderError, match="block #1 declares 12 header blocks"):
            walk_header_blocks(payload, ">")


class TestHeaderOnlyRead:
    """The header fetch a stitch's first pass runs on.

    A full-disk segment is tens of megabytes and sizing a stitch needs every
    segment's header before any pixel is wanted, so `_read_header_bytes` takes
    the header in two small reads guided by block #1's declared total instead of
    pulling whole granules into memory.
    """

    def test_reads_the_header_without_the_pixels(self, tmp_path):
        """The returned prefix decodes as a header and stops short of the payload."""
        path = fixture.write_granule(tmp_path, columns=64, lines=16)
        head = _read_header_bytes(path)
        whole = path.read_bytes()
        assert len(head) < len(whole), (
            f"read {len(head)} of {len(whole)} bytes — the payload came too"
        )
        assert head == whole[: len(head)], "the header is not the file's own prefix"
        assert parse_hsd_header(head).columns == 64

    def test_reads_a_compressed_granule_through_bz2(self, tmp_path):
        """A ``.DAT.bz2`` segment is decompressed, not read as raw bytes."""
        path = fixture.write_granule(tmp_path, columns=8, lines=4, compress=True)
        assert parse_hsd_header(_read_header_bytes(path)).columns == 8

    @pytest.mark.parametrize("declared", [0, (1 << 20) + 1])
    def test_an_unusable_declared_length_reads_the_whole_file(self, tmp_path, declared):
        """Zero means "not stated" and a huge figure means "not a length".

        Args:
            declared: The value written into block #1's total-header-length field.

        Test scenario:
            Neither value can size a second read, so the read falls back to the
            whole granule rather than guessing a header size.
        """
        path = fixture.write_granule(
            tmp_path, columns=8, lines=4, header_length=declared
        )
        assert _read_header_bytes(path) == path.read_bytes(), (
            f"a declared length of {declared} should fall back to the whole file"
        )

    def test_the_ceiling_is_far_above_a_real_header(self):
        """The ceiling has to separate "not a length" from a large real header."""
        from pyramids_eo.sensors.readers.ahi import _MAX_HEADER_BYTES

        header_end = walk_header_blocks(fixture.granule_bytes(), "<")[12][0]
        assert _MAX_HEADER_BYTES == 1 << 20
        assert header_end * 100 < _MAX_HEADER_BYTES, (
            f"a {header_end}-byte header is not comfortably under the ceiling"
        )

    @pytest.mark.parametrize("declared", [1, fixture.FIXED_BLOCK_LENGTHS[1]])
    def test_a_declaration_inside_the_first_read_needs_no_second_read(
        self, tmp_path, declared
    ):
        """A total no longer than block #1 is already satisfied by the first read.

        Args:
            declared: The value written into block #1's total-header-length field.

        Test scenario:
            Both values are covered by the 282 bytes already in hand, so the read
            returns them unchanged instead of asking the file for more.
        """
        path = fixture.write_granule(
            tmp_path, columns=8, lines=4, header_length=declared
        )
        head = _read_header_bytes(path)
        assert len(head) == fixture.FIXED_BLOCK_LENGTHS[1], (
            f"expected only the 282-byte first read, got {len(head)}"
        )

    def test_reports_an_unreadable_path(self, tmp_path):
        """A missing granule is a `ReaderError`, not a bare `OSError`."""
        with pytest.raises(ReaderError, match="cannot read"):
            _read_header_bytes(tmp_path / "absent.DAT")

    def test_reports_a_corrupt_bzip2_granule(self, tmp_path):
        """A ``.bz2`` name over non-bzip2 bytes fails on the first read."""
        path = tmp_path / "HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT.bz2"
        path.write_bytes(b"not bzip2 at all")
        with pytest.raises(ReaderError, match="cannot read"):
            _read_header_bytes(path)


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

    order = _byte_order(buffer)
    blocks = walk_header_blocks(buffer, order)
    header = parse_hsd_header(buffer)
    return _radiance(_counts(buffer, header, blocks, order), header)


class TestHeaderFields:
    """Field-by-field header decode."""

    def test_decodes_every_field(self):
        header = parse_hsd_header(
            fixture.granule_bytes(
                band=13,
                columns=6,
                lines=3,
                satellite="Himawari-9",
                sub_lon=140.7,
                slope=0.25,
                intercept=-5.0,
                wavelength_um=10.4,
                total_segments=10,
                segment_number=4,
                first_line=1651,
            )
        )
        assert isinstance(header, HsdHeader)
        assert header.satellite == "Himawari-9"
        assert (header.band, header.band_number, header.kind) == ("B13", 13, "thermal")
        assert (header.columns, header.lines) == (6, 3)
        assert (header.bits_per_pixel, header.valid_bits) == (16, 14)
        assert header.compression == 0
        assert (header.slope, header.intercept) == (0.25, -5.0)
        assert (header.error_count, header.outside_count) == (65535, 65534)
        assert header.central_wavelength_um == pytest.approx(10.4)
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

    @pytest.mark.parametrize(
        "slope", [0.0, 1e-20, -1e-20, float("nan"), float("inf"), float("-inf")]
    )
    def test_rejects_an_unusable_slope(self, slope):
        buffer = fixture.granule_bytes(band=13, slope=slope)
        with pytest.raises(ReaderError, match="unusable count-to-radiance slope"):
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

    def test_rejects_a_declared_data_length_disagreeing_with_the_grid(self, tmp_path):
        """Block #1 field 14 sizes the payload independently of block #2's grid.

        Test scenario:
            A granule whose declared total data length is not
            ``lines * columns * 2`` means one of the two blocks was misread, so
            the read is refused rather than decoding a plausible-looking array
            from whichever happened to be right. The payload itself is the full
            size here, so the later "data block holds" truncation check cannot
            be what fires.
        """
        path = fixture.write_granule(tmp_path, columns=8, lines=4, data_length=999)
        with pytest.raises(
            ReaderError,
            match=r"block #1 declares a 999-byte data block but the 4 x 8 grid",
        ):
            parse_ahi_hsd(path)

    def test_a_declared_data_length_matching_the_grid_is_accepted(self, tmp_path):
        """The cross-check passes when field 14 agrees with the grid.

        Test scenario:
            Field 14 set explicitly to the grid's own ``4 * 8 * 2`` bytes must
            read exactly as the default does, which pins the check to
            disagreement rather than to the field being written at all.
        """
        counts = np.arange(32, dtype=np.uint16).reshape(4, 8)
        path = fixture.write_granule(
            tmp_path,
            columns=8,
            lines=4,
            counts=counts,
            slope=1.0,
            intercept=0.0,
            data_length=4 * 8 * 2,
        )
        assert parse_ahi_hsd(path).read_array() == pytest.approx(counts.astype(float))

    def test_a_zero_declared_data_length_is_tolerated(self, tmp_path):
        """A granule leaving field 14 unset is decoded from block #2's grid.

        Test scenario:
            The check is guarded on a truthy declaration, so ``data_length=0``
            must still decode the full grid instead of being compared against
            the payload size.
        """
        counts = np.arange(32, dtype=np.uint16).reshape(4, 8)
        path = fixture.write_granule(
            tmp_path,
            columns=8,
            lines=4,
            counts=counts,
            slope=1.0,
            intercept=0.0,
            data_length=0,
        )
        assert parse_ahi_hsd(path).read_array() == pytest.approx(counts.astype(float))

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
        wavelength = 10.4
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
            wavelength_um=wavelength,
            planck=(0.0, 1.0, 0.0),  # no correction: Tb == Te
        )
        result = read_ahi(path, "B13").read_array()
        assert float(result[0, 0]) == pytest.approx(temperature, rel=1e-9)

    def test_thermal_applies_the_granule_quadratic_correction(self, tmp_path):
        wavelength = 10.4
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
            wavelength_um=wavelength,
            planck=(c0, c1, c2),
        )
        expected = c0 + c1 * temperature + c2 * temperature**2
        assert float(read_ahi(path, "B13").read_array()[0, 0]) == pytest.approx(
            expected, rel=1e-9
        )

    def test_thermal_leaves_masked_pixels_nan(self, tmp_path):
        counts = np.array([[1000, fixture.ERROR_COUNT]], dtype=np.uint16)
        radiance = _planck_radiance(300.0, 10.4)
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

    def test_explicit_coeffs_route_through_the_registry_and_stay_physical(
        self, tmp_path
    ):
        """A caller's coeffs use the registry's linear form - in the right units.

        Asserting only that the two paths *differ* is too weak: an unconverted
        per-wavelength radiance fed to the per-wavenumber inverse Planck differs
        from the granule answer by six orders of magnitude and would satisfy it.
        Both results must be physical temperatures and agree closely.
        """
        wavelength = 10.4
        temperature = 280.0
        radiance = _planck_radiance(temperature, wavelength)
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=1,
            lines=1,
            counts=np.array([[1000]], dtype=np.uint16),
            slope=radiance / 1000,
            intercept=0.0,
            wavelength_um=wavelength,
        )
        granule = float(read_ahi(path, "B13").read_array()[0, 0])
        overridden = float(
            read_ahi(
                path,
                "B13",
                coeffs={"central_wavenumber_cm1": 961.538, "alpha": 1.0, "beta": 0.0},
            ).read_array()[0, 0]
        )
        assert granule == pytest.approx(temperature, rel=1e-9)
        assert overridden == pytest.approx(temperature, abs=0.05), (
            f"registry path must be physical, got {overridden}"
        )


class TestThermalUnits:
    """HSD radiance is per WAVELENGTH; the registry's Planck wants per WAVENUMBER.

    HSD block #5 defines radiance as ``W / (m2 sr m)`` - per metre of wavelength.
    `radiance_to_brightness_temperature` documents ``mW m-2 sr-1 (cm-1)-1``.
    Feeding one to the other unconverted yields ~921,613 K for a 280 K scene, so
    every path that reaches the registry must convert first.
    """

    @pytest.mark.parametrize(
        ("wavelength_um", "temperature"),
        [(10.4, 280.0), (10.4, 200.0), (3.9, 300.0), (13.3, 250.0), (6.2, 230.0)],
    )
    def test_a_dataset_source_calibrates_to_a_physical_temperature(
        self, tmp_path, wavelength_um, temperature
    ):
        """A `Dataset` source has no granule, so it takes the registry path."""
        radiance = _planck_radiance(temperature, wavelength_um)
        band = {10.4: "B13", 3.9: "B07", 13.3: "B16", 6.2: "B08"}[wavelength_um]
        path = fixture.write_granule(
            tmp_path,
            band=int(band[1:]),
            columns=1,
            lines=1,
            counts=np.array([[1000]], dtype=np.uint16),
            slope=radiance / 1000,
            intercept=0.0,
            wavelength_um=wavelength_um,
        )
        source = parse_ahi_hsd(path)
        result = float(read_ahi(source, band).read_array()[0, 0])
        assert result == pytest.approx(temperature, abs=0.6), (
            f"{band} at {temperature} K came back as {result} K"
        )

    def test_skipping_the_rebasing_looks_plausible_and_is_wrong(self):
        """Pins the defect this conversion exists to prevent.

        In HSD's native per-micrometre units the unconverted value is not
        absurd, it is merely wrong: a 280 K scene reads as about 189 K, which
        passes for a cold cloud top. That is harder to notice than an obviously
        broken number, so it is pinned by value rather than by magnitude.
        """
        from pyramids_eo.sensors.registry.calibration import (
            radiance_to_brightness_temperature,
        )

        radiance = _planck_radiance(280.0, 10.4)
        unconverted = float(
            radiance_to_brightness_temperature(radiance, 961.538, 1.0, 0.0)
        )
        assert unconverted == pytest.approx(189.1, abs=0.5), unconverted
        assert 150.0 < unconverted < 250.0, (
            "the wrong answer is a plausible-looking temperature, which is "
            "precisely why the rebasing cannot be left to inspection"
        )

    def test_the_conversion_is_the_wavelength_squared_scale(self):
        """L_nu = L_lambda * lambda^2, carried through in micrometre units."""
        from pyramids_eo.sensors.readers.ahi import (
            _WAVENUMBER_RADIANCE_SCALE,
            _to_wavenumber_radiance,
        )

        assert _WAVENUMBER_RADIANCE_SCALE == 1e-1
        radiance = _planck_radiance(280.0, 10.4)
        converted = _to_wavenumber_radiance(radiance, 10.4)
        assert float(converted) == pytest.approx(radiance * 10.4**2 * 1e-1)
        assert 10.0 < float(converted) < 1000.0, (
            "a terrestrial scene is order 10-100 in the registry's units"
        )

    def test_a_solar_band_is_not_unit_converted(self, tmp_path):
        """Reflectance divides radiance by an irradiance in the SAME basis."""
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
        source = parse_ahi_hsd(path)
        result = float(
            read_ahi(
                source, "B03", coeffs={"kind": "solar", "solar_irradiance": 400.0}
            ).read_array()[0, 0]
        )
        assert result == pytest.approx(math.pi * 100.0 / 400.0)

    def test_coeffs_declaring_solar_on_a_thermal_band_skip_the_conversion(
        self, tmp_path
    ):
        """The effective kind decides, so an override is honoured either way."""
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=1,
            lines=1,
            counts=np.array([[100]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
        )
        result = float(
            read_ahi(
                path, "B13", coeffs={"kind": "solar", "solar_irradiance": 400.0}
            ).read_array()[0, 0]
        )
        assert result == pytest.approx(math.pi * 100.0 / 400.0)

    def test_thermal_missing_a_constant_is_reported(self, tmp_path):
        path = fixture.write_granule(tmp_path, band=13)
        buffer = bytearray(path.read_bytes())
        offset = walk_header_blocks(bytes(buffer), "<")[5][0]
        struct.pack_into("<d", buffer, offset + 83, float("nan"))  # speed of light
        path.write_bytes(bytes(buffer))
        with pytest.raises(ReaderError, match="missing the calibration constant"):
            read_ahi(path, "B13")

    def test_thermal_with_a_non_physical_wavelength_is_reported(self, tmp_path):
        path = fixture.write_granule(tmp_path, band=13, wavelength_um=0.0)
        with pytest.raises(ReaderError, match="non-physical central wavelength"):
            read_ahi(path, "B13")

    @pytest.mark.parametrize("bad", [0.0, -1.0e-6, float("nan"), float("inf")])
    def test_the_conversion_refuses_a_non_physical_wavelength(self, bad):
        """The conversion carries its own check, not just the header parser's.

        Args:
            bad: A wavelength that is zero, negative, or not finite.

        Test scenario:
            A granule's wavelength is already validated at parse time, so this
            guard is reachable only directly — it belongs to the conversion,
            which scales by the square of the wavelength and would otherwise
            return a silently meaningless radiance.
        """
        from pyramids_eo.sensors.readers.ahi import _to_wavenumber_radiance

        radiance = np.array([[1.0]])
        with pytest.raises(ReaderError, match="non-physical central wavelength"):
            _to_wavenumber_radiance(radiance, bad)


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

    def test_the_stitch_refuses_an_empty_set_of_its_own(self):
        """`read_ahi` screens an empty source, so this is the stitch's own guard.

        Test scenario:
            Called directly with no paths, the stitch reports the same error
            rather than failing later on an empty sort or an unsized output.
        """
        from pyramids_eo.sensors.readers.ahi import _stitch

        with pytest.raises(ReaderError, match="no HSD segment paths"):
            _stitch([])


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
    """Paths the public API cannot reach, covered directly."""

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

    def test_a_dataset_source_without_a_crs_is_wrapped_unchanged(self):
        """The CRS carry-over is unconditional, so a source without one still works.

        Test scenario:
            A `Dataset` built with no EPSG has an empty CRS. The wrap copies the
            source's CRS either way, so the result must come back with the values
            registered on the same grid and no CRS invented for it.
        """
        from pyramids.dataset import Dataset, GeoReference

        source = Dataset.from_array(
            np.array([[100.0, 200.0]]),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), epsg=None),
            no_data_value=np.nan,
        )
        assert not source.crs, "the source must carry no CRS for this to be the test"
        result = read_ahi(
            source, "B03", coeffs={"kind": "solar", "solar_irradiance": 400.0}
        )
        assert not result.crs, f"a CRS was invented for the result: {result.crs!r}"
        assert result.geotransform == pytest.approx(source.geotransform)
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
    # Asserted on the MEDIAN alone, because every other summary statistic
    # measured here turned out to depend on where the segment sits in the disk
    # rather than on whether the decode is right. Off-limb space beyond the
    # Earth's edge is not always flagged with the outside-scan sentinel, and the
    # granule's affine calibration maps those high counts to near-zero radiance,
    # so limb segments carry a large cold tail. Measured on five real band-13
    # segments of one full disk:
    #
    #   seg  first_line  median   within 180-330 K   below 180 K
    #     1        1     237.31        68.24%           31.76%
    #     4     1651     291.16        95.54%            4.46%
    #     5     2201     286.64        98.71%            1.29%
    #     6     2751     279.77        98.72%            1.28%
    #    10     4951     250.49        67.97%           32.03%
    #
    # An earlier version of this test required the minimum above 180 K, and the
    # version that replaced it required 90% of pixels inside 180-330 K. Both
    # fail on segments 1 and 10 -- the limb segments -- because both measure how
    # much space a segment contains. The median does not: it stays between 237
    # and 292 K across all five.
    #
    # Nor does a bulk-fraction test buy anything. Across ten deliberate
    # perturbations (wavelength as metres, x10, /10; slope sign-flipped, x1e6;
    # intercept zeroed; two neighbouring-band wavelengths; flipped rows) the
    # fraction never failed where the median did not, so it added no detection
    # power while introducing a false failure on real data.
    median = float(np.median(finite))
    assert 180.0 < median < 330.0, (
        f"BT median {median} is not a terrestrial temperature. A unit or scale "
        "error on the central wavelength or the slope lands here: taking the "
        "wavelength as metres rather than micrometres gives about -1.8e44 K "
        "with this granule's quadratic terms."
    )
    assert float(finite.max()) < 340.0, f"BT max too high: {finite.max()}"
    assert "Geostationary" in str(scene.crs), "should carry the geostationary CRS"
    assert abs(scene.geotransform[1]) == pytest.approx(2000.0, abs=1.0), (
        "band 13 is the 2 km grid"
    )
    assert scene.geotransform[5] < 0, "the grid must be north-up (gt[5] < 0)"
    assert np.isnan(scene.no_data_value[0]), "nodata should be NaN"


@pytest.mark.live
def test_read_ahi_real_segments_stitch():
    """Stitch real segments -- the defect this change fixes, on real data.

    Every other live test reads `granules[0]`, so none of them exercises the
    multi-granule path at all. That is the path the start-time equality check
    broke: it rejected a legitimate full-disk set outright, and no live test
    would have noticed.

    Needs two or more contiguous segments of one acquisition in the fixture
    directory; skips otherwise rather than asserting on whatever is present.
    """
    fixtures = Path(os.environ.get("AHI_FIXTURES_DIR", "tests/data/ahi"))
    granules = sorted(fixtures.glob("HS_H0?_*_B??_*.DAT")) + sorted(
        fixtures.glob("HS_H0?_*_B??_*.DAT.bz2")
    )
    if not granules:
        pytest.skip("real AHI fixtures not available (set AHI_FIXTURES_DIR)")
    from pyramids_eo.sensors.readers.ahi import _read_bytes

    # Group by acquisition and band, then keep the longest contiguous run.
    headers = {str(g): parse_hsd_header(_read_bytes(str(g))) for g in granules}
    groups: dict[tuple, list] = {}
    for path, header in headers.items():
        key = (
            header.satellite,
            header.band_number,
            header.observation_timeline,
            header.observation_area,
        )
        groups.setdefault(key, []).append((header.first_line, path, header))
    run: list = []
    for members in groups.values():
        members.sort()
        current: list = []
        for first_line, path, header in members:
            if (
                current
                and first_line != current[-1][2].first_line + current[-1][2].lines
            ):
                if len(current) > len(run):
                    run = current
                current = []
            current.append((first_line, path, header))
        if len(current) > len(run):
            run = current
    if len(run) < 2:
        pytest.skip(
            "need two or more contiguous segments of one acquisition; "
            f"longest run found was {len(run)}"
        )

    paths = [path for _, path, _ in run]
    starts = {headers[p].observation_start_mjd for p in paths}
    assert len(starts) == len(paths), (
        "these segments should each carry their own start time, which is what "
        f"makes this a regression test; got {len(starts)} distinct among "
        f"{len(paths)} segments"
    )
    spread = (max(starts) - min(starts)) * 86400.0
    assert spread > 1.0, (
        f"the start times span only {spread:.3f} s, so this fixture cannot "
        "demonstrate that differing starts are accepted"
    )

    stitched = read_ahi(paths, calibrate=False)
    array = np.asarray(stitched.read_array(), dtype=float)
    expected_rows = sum(headers[p].lines for p in paths)
    assert array.shape == (expected_rows, headers[paths[0]].columns), (
        f"expected {expected_rows} rows from {len(paths)} segments, got {array.shape}"
    )
    # The stitch must start where its northernmost segment starts.
    single = read_ahi(paths[0], calibrate=False)
    assert stitched.geotransform[3] == pytest.approx(single.geotransform[3]), (
        "the stitched origin must equal the first segment's north edge"
    )
    # No blank seam between segments: every row holds some finite data.
    finite_per_row = np.isfinite(array).mean(axis=1)
    assert float(finite_per_row.max()) > 0.0
    seams = [
        sum(headers[p].lines for p in paths[: i + 1]) for i in range(len(paths) - 1)
    ]
    for seam in seams:
        assert float(finite_per_row[seam - 1 : seam + 1].max()) > 0.0, (
            f"rows around the seam at {seam} are entirely non-finite, which is "
            "what a mis-placed segment looks like"
        )


@pytest.mark.live
def test_read_ahi_real_granule_band_identity():
    """The decoded band must be the band the file name declares.

    No live assertion checked this, and the brightness-temperature statistics
    cannot: substituting a neighbouring band's wavelength leaves the median
    comfortably terrestrial. Measured on a real band-13 segment, swapping in
    8.6 um gives a median of 289.9 K and 12.4 um gives 293.1 K, against 286.6 K
    for the correct 10.4074 um -- all three indistinguishable from a plausible
    scene. The granule's own central wavelength against the registry's nominal
    value for the band its name declares is what separates them.
    """
    fixtures = Path(os.environ.get("AHI_FIXTURES_DIR", "tests/data/ahi"))
    granules = sorted(fixtures.glob("HS_H0?_*_B??_*.DAT")) + sorted(
        fixtures.glob("HS_H0?_*_B??_*.DAT.bz2")
    )
    if not granules:
        pytest.skip("real AHI fixtures not available (set AHI_FIXTURES_DIR)")
    from pyramids_eo.sensors.readers.ahi import _read_bytes
    from pyramids_eo.sensors.registry.sensors import get_sensor

    for granule in granules:
        named = band_from_filename(granule.name)
        assert named is not None, f"{granule.name} does not follow the HSD convention"
        header = parse_hsd_header(_read_bytes(str(granule)))
        assert header.band == named, (
            f"{granule.name} declares band {named} in its name but {header.band} "
            "in calibration block #5"
        )
        nominal = get_sensor("ahi").get_channel(named).wavelength_um
        assert abs(header.central_wavelength_um - nominal) < 0.25, (
            f"{named}: granule wavelength {header.central_wavelength_um} um is "
            f"not the registry's nominal {nominal} um; a neighbouring band's "
            "value would pass every brightness-temperature check"
        )


@pytest.mark.live
def test_read_ahi_real_limb_segment_is_north_up():
    """Orientation checked from the pixels, not from the sign of `dy`.

    `geotransform[5] < 0` only restates what the reader wrote; it cannot catch a
    flipped array, and the brightness statistics are invariant under a flip
    (median 286.6 K either way, measured). The Earth's limb curves, so a segment
    north of the sub-satellite line carries more off-limb space in its top rows
    than its bottom rows, and a southern one the reverse. Measured:

        seg  first_line  off-limb top 10%  bottom 10%
          1        1          60.05%         31.22%
          4     1651          2.92%           0.40%
          5     2201          0.38%           0.36%
          6     2751          0.36%           0.38%
         10     4951         30.56%          59.71%

    Near the centre the gradient vanishes, so this asserts only on segments far
    enough from the sub-satellite line for it to be unambiguous.
    """
    fixtures = Path(os.environ.get("AHI_FIXTURES_DIR", "tests/data/ahi"))
    granules = sorted(fixtures.glob("HS_H0?_*_B??_*.DAT")) + sorted(
        fixtures.glob("HS_H0?_*_B??_*.DAT.bz2")
    )
    if not granules:
        pytest.skip("real AHI fixtures not available (set AHI_FIXTURES_DIR)")
    from pyramids_eo.sensors.readers.ahi import _read_bytes

    tested = 0
    for granule in granules:
        header = parse_hsd_header(_read_bytes(str(granule)))
        centre = header.first_line + header.lines / 2.0
        offset = centre - header.loff
        # Only the outer part of the disk shows a usable gradient; `lines * 2`
        # keeps this to segments whose middle is at least two segment-heights
        # from the sub-satellite line.
        if abs(offset) < header.lines * 2:
            continue
        tested += 1
        array = np.asarray(read_ahi(str(granule)).read_array(), dtype=float)
        edge = max(1, array.shape[0] // 10)
        top = 1.0 - float(np.isfinite(array[:edge]).mean())
        bottom = 1.0 - float(np.isfinite(array[-edge:]).mean())
        if offset < 0:  # north of the sub-satellite line
            assert top > bottom, (
                f"{granule.name} is north of the sub-satellite line, so its top "
                f"rows should hold more off-limb space; got top {top:.2%} and "
                f"bottom {bottom:.2%}, which is what a vertically flipped array "
                "looks like"
            )
        else:
            assert bottom > top, (
                f"{granule.name} is south of the sub-satellite line, so its "
                f"bottom rows should hold more off-limb space; got top {top:.2%} "
                f"and bottom {bottom:.2%}"
            )
    if not tested:
        pytest.skip(
            "no limb segment available; mid-disk segments show no usable "
            "off-limb gradient (set AHI_FIXTURES_DIR to a set including "
            "segment 1 or 10 of a full disk)"
        )


@pytest.mark.live
def test_read_ahi_real_granule_block_five_units():
    """Assert the two block #5 fields whose units were misread, on a real granule.

    These are the fields the offline suite cannot adjudicate, because the
    fixture necessarily encodes whatever reading the reader uses. Both were in
    fact wrong at one point and the suite stayed green through it:

    * the central wavelength is in MICROMETRES -- band 13 carries about 10.4,
      not 1.04e-05. With both unit conversions dropped, as they were, a 280 K
      scene calibrates to -1.80e44 K through this granule's own quadratic terms
      (measured; 9.99e24 K without them -- the figure depends on which
      coefficients are applied, so the conditions are stated rather than a bare
      number).
    * the slope is per micrometre. Per-metre radiance is 1e6 times larger, so
      whatever the operational magnitude turns out to be, the per-metre figure
      would be 1e6 times it -- which is why the assertion below brackets the
      magnitude loosely rather than pinning a value this repository has never
      seen a granule state.

    A real granule settles both in one read, which is the whole point of this
    test existing.
    """
    fixtures = Path(os.environ.get("AHI_FIXTURES_DIR", "tests/data/ahi"))
    granules = sorted(fixtures.glob("HS_H0?_*_B13_*.DAT")) + sorted(
        fixtures.glob("HS_H0?_*_B13_*.DAT.bz2")
    )
    if not granules:
        pytest.skip("real AHI fixtures not available (set AHI_FIXTURES_DIR)")
    from pyramids_eo.sensors.readers.ahi import _read_bytes

    header = parse_hsd_header(_read_bytes(str(granules[0])))
    assert 9.0 < header.central_wavelength_um < 12.0, (
        f"band 13 should be about 10.4 um; got {header.central_wavelength_um!r}. "
        "A value near 1.04e-05 means the field is metres after all and the "
        "reader's unit is wrong."
    )
    assert 1e-4 < abs(header.slope) < 1e2, (
        f"a per-micrometre slope should be of order 1e-2; got {header.slope!r}. "
        "A magnitude near 1e2-1e5 would mean the radiance is per metre."
    )
    assert math.isfinite(header.intercept), header.intercept
    # The two together must put a real scene in a terrestrial range.
    scene = read_ahi(str(granules[0]))
    finite = np.asarray(scene.read_array(), dtype=float)
    finite = finite[np.isfinite(finite)]
    assert 180.0 < float(np.median(finite)) < 320.0, (
        f"median brightness temperature {np.median(finite)} is not terrestrial, "
        "which is what a unit error on either field would produce"
    )


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
    assert header.satellite in {"Himawari-8", "Himawari-9"}, header.satellite
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

    @pytest.mark.parametrize("source", [42, 4.5])
    def test_a_non_path_non_iterable_source_is_reported(self, source):
        """Anything that is neither a path, an iterable of paths, nor a Dataset.

        Args:
            source: A value `list()` cannot consume.

        Test scenario:
            Without the guard `list(source)` raises a bare `TypeError`, which is
            not the documented failure; the reader names the three forms it takes
            and the type it was handed instead.
        """
        with pytest.raises(
            ReaderError, match="source must be a granule path"
        ) as exc_info:
            read_ahi(source, "B13")
        assert type(source).__name__ in str(exc_info.value), (
            f"the error should name the type it was given, got: {exc_info.value}"
        )

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


class TestSegmentIdentity:
    """Segments must come from ONE acquisition of one band on one satellite.

    Contiguous line numbers are not enough: two timelines ten minutes apart have
    the same per-segment first lines, so they stitch into a seamless-looking
    image made of two different scenes.
    """

    def _segment(self, directory, index, **kwargs):
        """Write segment `index` of a four-segment band-13 set."""
        return fixture.write_granule(
            directory,
            band=kwargs.pop("band", 13),
            columns=4,
            lines=3,
            counts=np.full((3, 4), index + 1, dtype=np.uint16),
            total_segments=4,
            segment_number=index + 1,
            first_line=index * 3 + 1,
            slope=1.0,
            intercept=0.0,
            **kwargs,
        )

    def test_rejects_segments_from_two_timelines(self, tmp_path):
        """Set the timeline, which is what actually identifies the slot.

        This case previously varied only `start_mjd` while leaving the timeline
        at its default, so it never tested what its name claimed. A real
        full-disk set varies the start time *within* one acquisition -- each
        segment is scanned at its own moment -- so varying it is not a
        difference of acquisition at all.
        """
        first = self._segment(tmp_path / "a", 0, start_mjd=57127.1041666667, timeline=0)
        second = self._segment(
            tmp_path / "b", 1, start_mjd=57127.1111111111, timeline=10
        )
        with pytest.raises(ReaderError, match="different observation"):
            read_ahi([first, second])

    def test_rejects_segments_from_two_satellites(self, tmp_path):
        first = self._segment(tmp_path / "a", 0, satellite="Himawari-8")
        second = self._segment(tmp_path / "b", 1, satellite="Himawari-9")
        with pytest.raises(ReaderError, match="different satellite"):
            read_ahi([first, second])

    def test_rejects_segments_from_two_observation_areas(self, tmp_path):
        first = self._segment(tmp_path / "a", 0, area="FLDK")
        second = self._segment(tmp_path / "b", 1, area="JP01")
        with pytest.raises(ReaderError, match="different observation"):
            read_ahi([first, second])

    def test_rejects_segments_disagreeing_on_the_sub_satellite_longitude(
        self, tmp_path
    ):
        first = self._segment(tmp_path / "a", 0, sub_lon=140.7)
        second = self._segment(tmp_path / "b", 1, sub_lon=104.0)
        with pytest.raises(ReaderError, match="projection"):
            read_ahi([first, second])

    def test_rejects_segments_disagreeing_on_the_segment_total(self, tmp_path):
        """A set whose members disagree on how many segments exist is incoherent."""
        first = self._segment(tmp_path / "a", 0)
        second = fixture.write_granule(
            tmp_path / "b",
            band=13,
            columns=4,
            lines=3,
            counts=np.full((3, 4), 2, dtype=np.uint16),
            total_segments=7,
            segment_number=2,
            first_line=4,
            slope=1.0,
            intercept=0.0,
        )
        with pytest.raises(ReaderError, match="how many segments"):
            read_ahi([first, second])

    def test_accepts_a_contiguous_subset_of_the_disk(self, tmp_path):
        """A partial read is in contract and georeferenced where it belongs.

        One segment already reads on its own, so a contiguous run of two is not
        a different kind of request. What matters is that the result is placed
        correctly rather than passed off as the whole disk.
        """
        paths = [self._segment(tmp_path / f"s{i}", i) for i in range(4)]
        middle = read_ahi(paths[1:3], calibrate=False)
        assert middle.read_array().shape == (6, 4)
        assert middle.read_array()[:, 0] == pytest.approx([2.0] * 3 + [3.0] * 3)
        whole = read_ahi(paths, calibrate=False)
        dy = whole.geotransform[5]
        assert middle.geotransform[3] == pytest.approx(whole.geotransform[3] + 3 * dy)

    def test_segments_of_one_acquisition_still_stitch(self, tmp_path):
        """The identity checks must not reject a legitimate set."""
        paths = [self._segment(tmp_path / f"s{i}", i) for i in range(4)]
        values = read_ahi(paths, calibrate=False).read_array()
        assert values.shape == (12, 4)
        assert values[:, 0] == pytest.approx(
            [1.0] * 3 + [2.0] * 3 + [3.0] * 3 + [4.0] * 3
        )


class TestDecodedFloatValidation:
    """Every calibration float the reader relies on is checked, not assumed."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_albedo_coefficient_is_reported(self, tmp_path, bad):
        """Left unchecked this produced an all-NaN reflectance image, no error."""
        buffer = fixture.granule_bytes(band=3, albedo_coefficient=bad)
        with pytest.raises(ReaderError, match="non-finite albedo coefficient"):
            parse_hsd_header(buffer)

    def test_a_zero_albedo_coefficient_falls_back_to_the_registry(self, tmp_path):
        """Zero means the granule has none; the registry then refuses honestly."""
        from pyramids_eo.errors import CalibrationError

        path = fixture.write_granule(
            tmp_path,
            band=3,
            columns=1,
            lines=1,
            counts=np.array([[100]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
            albedo_coefficient=0.0,
        )
        with pytest.raises(CalibrationError, match="solar_irradiance"):
            read_ahi(path, "B03")

    def test_a_recalibrated_slope_below_the_floor_is_ignored(self, tmp_path):
        """A denormal-ish updated slope is a misparse, not a very small gain."""
        header = parse_hsd_header(
            fixture.granule_bytes(
                band=3, slope=0.5, intercept=-10.0, updated=(1e-20, -2.0)
            )
        )
        assert header.slope == 0.5
        assert header.intercept == -10.0

    def test_the_slope_floor_is_many_orders_below_a_real_gain(self):
        """The floor separates "unset" from a real gain without clipping one."""
        from pyramids_eo.sensors.readers.ahi import _MIN_SLOPE

        assert _MIN_SLOPE == 1e-12
        assert _MIN_SLOPE < 1e-2, "real AHI slopes are order 1e-2 and above"


class TestSlopeSign:
    """The specification constrains the slope's magnitude, not its sign.

    Table 6 presents the count-to-radiance conversion as a plain affine
    relation and says nothing about the direction. An earlier version refused a
    negative gain as "non-physical by construction", which would have rejected
    every granule of any band JMA chose to ship that way, on the strength of an
    assumption the specification does not make and that no granule was
    available to test.
    """

    @pytest.mark.parametrize("slope", [-0.5, -1.0, -42.0])
    def test_a_negative_slope_is_accepted(self, slope):
        header = parse_hsd_header(fixture.granule_bytes(band=13, slope=slope))
        assert header.slope == slope

    def test_a_negative_slope_produces_decreasing_radiance(self, tmp_path):
        """What the granule asked for: higher counts, lower radiance."""
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=3,
            lines=1,
            counts=np.array([[10, 20, 30]], dtype=np.uint16),
            slope=-1.0,
            intercept=100.0,
        )
        values = parse_ahi_hsd(path).read_array()
        assert values.ravel() == pytest.approx([90.0, 80.0, 70.0])

    @pytest.mark.parametrize("magnitude", [1e-20, -1e-20, 0.0])
    def test_a_slope_magnitude_below_the_floor_is_still_refused(self, magnitude):
        """Too small to be a gain, either sign: the field is unset."""
        buffer = fixture.granule_bytes(band=13, slope=magnitude)
        with pytest.raises(ReaderError, match="unusable count-to-radiance slope"):
            parse_hsd_header(buffer)

    def test_a_recalibrated_negative_slope_is_honoured(self):
        header = parse_hsd_header(
            fixture.granule_bytes(
                band=3, slope=0.5, intercept=-10.0, updated=(-0.75, -2.0)
            )
        )
        assert header.slope == -0.75


class TestKindValidation:
    """A caller's `coeffs["kind"]` must match one of two spellings exactly.

    This reader branches on ``"thermal"``; the shared `calibrate_channel`
    branches on ``"solar"``. A third spelling therefore used to satisfy neither
    as intended -- taking the solar path here and the thermal path there -- and
    returned a brightness temperature computed from radiance that had never been
    rebased onto the per-wavenumber basis that calculation needs.
    """

    @pytest.mark.parametrize(
        "spelling",
        ["SOLAR", "Solar", "THERMAL", "Thermal", "reflectance", "", "banana"],
    )
    def test_an_unrecognised_kind_is_refused(self, tmp_path, spelling):
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=1,
            lines=1,
            counts=np.array([[100]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
        )
        with pytest.raises(ReaderError, match=r"coeffs\['kind'\] must be one of"):
            read_ahi(path, "B13", coeffs={"kind": spelling, "solar_irradiance": 400.0})

    @pytest.mark.parametrize("spelling", ["solar", "thermal"])
    def test_the_two_legal_spellings_are_accepted(self, tmp_path, spelling):
        path = fixture.write_granule(
            tmp_path,
            band=13,
            columns=1,
            lines=1,
            counts=np.array([[100]], dtype=np.uint16),
            slope=1.0,
            intercept=0.0,
        )
        coeffs = {"kind": spelling}
        if spelling == "solar":
            coeffs["solar_irradiance"] = 400.0
        else:
            coeffs["central_wavenumber_cm1"] = 961.538
        result = read_ahi(path, "B13", coeffs=coeffs).read_array()
        assert np.isfinite(result).all(), result

    def test_the_kind_set_is_exactly_the_two_the_registry_knows(self):
        from pyramids_eo.sensors.readers.ahi import _KINDS

        assert _KINDS == frozenset({"solar", "thermal"})


class TestHeaderReadIsBounded:
    """The header-only read must stay bounded even when the granule lies.

    Its whole purpose is to let a stitch size its output without pulling every
    segment into memory. Falling back to reading the entire file when the
    declared length is unusable defeated that: pass one read the granule whole
    and pass two read it again, so a ten-segment disk was read twice over.
    """

    def test_an_unusable_declaration_is_capped_not_widened(self, tmp_path):
        from pyramids_eo.sensors.readers.ahi import (
            _MAX_HEADER_BYTES,
            _read_bytes,
            _read_header_bytes,
        )

        # A payload comfortably larger than the ceiling, and no declaration.
        path = fixture.write_granule(
            tmp_path, band=13, columns=2000, lines=500, header_length=0
        )
        whole = len(_read_bytes(path))
        assert whole > _MAX_HEADER_BYTES, "fixture must exceed the ceiling to be a test"
        read = len(_read_header_bytes(path))
        assert read <= _MAX_HEADER_BYTES, read
        assert read < whole, "the fallback must not read the whole granule"

    @pytest.mark.parametrize("declared", [0, (1 << 20) + 1, 4_000_000_000])
    def test_every_unusable_declaration_takes_the_capped_path(self, tmp_path, declared):
        from pyramids_eo.sensors.readers.ahi import (
            _MAX_HEADER_BYTES,
            _read_header_bytes,
        )

        path = fixture.write_granule(
            tmp_path / str(declared),
            band=13,
            columns=2000,
            lines=500,
            header_length=declared,
        )
        assert len(_read_header_bytes(path)) <= _MAX_HEADER_BYTES

    def test_a_valid_declaration_reads_only_the_header(self, tmp_path):
        """Exactly the header, derived from the block lengths rather than pinned.

        A literal would have to be edited whenever a variable block's fixture
        length changes, which says nothing about the behaviour under test.
        """
        from pyramids_eo.sensors.readers.ahi import _read_header_bytes

        expected = sum(fixture.FIXED_BLOCK_LENGTHS.values()) + sum(
            fixture.VARIABLE_BLOCK_LENGTHS.values()
        )
        path = fixture.write_granule(tmp_path, band=13, columns=2000, lines=500)
        assert len(_read_header_bytes(path)) == expected


class TestSegmentsOfOneAcquisitionDifferInStartTime:
    """AHI scans the disk progressively, so each segment has its own start time.

    Measured on a real Himawari-9 band-13 full-disk set (segments 4, 5, 6 of 10),
    `observation_start_mjd` varies across them, while every field that identifies
    the acquisition is constant:

        seg 4  start MJD 61041.00183583374    +0.0 s
        seg 5  start MJD 61041.00283434366   +86.3 s
        seg 6  start MJD 61041.00353695286  +147.0 s

    `observation_timeline` (0), `observation_area` ('FLDK'), the satellite, the
    segment total, the band, the grid and the whole projection block are all
    constant. An equality check on the start time therefore rejects a legitimate
    full-disk set -- the reader's primary advertised use -- which is what a real
    granule exposed and no synthetic fixture had, because the fixture wrote one
    start time to every segment.

    `observation_start_mjd` is not the only varying field, and an earlier version
    of this docstring claimed it was. Comparing all 35 decoded fields across the
    five real segments, four vary: `segment_number` and `first_line` (which is
    the point of a segment), `observation_start_mjd`, and
    `total_header_length` -- 1523 bytes for most segments and 1533 for one,
    because block #10 carries a per-segment number of error rows. The other 31
    are identical.
    """

    def _set(self, directory, *, starts, timeline=0, area="FLDK"):
        """Write a 3-segment band-13 set with the given per-segment start times."""
        return [
            fixture.write_granule(
                directory / f"s{index}",
                band=13,
                columns=4,
                lines=3,
                counts=np.full((3, 4), index + 1, dtype=np.uint16),
                total_segments=3,
                segment_number=index + 1,
                first_line=index * 3 + 1,
                slope=1.0,
                intercept=0.0,
                start_mjd=start,
                timeline=timeline,
                area=area,
            )
            for index, start in enumerate(starts)
        ]

    def test_segments_with_their_own_start_times_still_stitch(self, tmp_path):
        """The real spacing: ~86 s and ~147 s after the first segment."""
        base = 61041.00183583374
        paths = self._set(
            tmp_path, starts=[base, base + 86.3 / 86400.0, base + 147.0 / 86400.0]
        )
        values = read_ahi(paths, calibrate=False).read_array()
        assert values.shape == (9, 4)
        assert values[:, 0] == pytest.approx([1.0] * 3 + [2.0] * 3 + [3.0] * 3)

    def test_a_different_timeline_is_still_refused(self, tmp_path):
        """The timeline field, not the start time, separates acquisition slots."""
        base = 61041.00183583374
        first = self._set(tmp_path / "a", starts=[base], timeline=0)[0]
        second = fixture.write_granule(
            tmp_path / "b",
            band=13,
            columns=4,
            lines=3,
            counts=np.full((3, 4), 2, dtype=np.uint16),
            total_segments=3,
            segment_number=2,
            first_line=4,
            slope=1.0,
            intercept=0.0,
            start_mjd=base + 600.0 / 86400.0,
            timeline=10,
        )
        with pytest.raises(ReaderError, match="different observation"):
            read_ahi([first, second])

    def test_a_different_area_is_still_refused(self, tmp_path):
        base = 61041.00183583374
        first = self._set(tmp_path / "a", starts=[base], area="FLDK")[0]
        second = fixture.write_granule(
            tmp_path / "b",
            band=13,
            columns=4,
            lines=3,
            counts=np.full((3, 4), 2, dtype=np.uint16),
            total_segments=3,
            segment_number=2,
            first_line=4,
            slope=1.0,
            intercept=0.0,
            start_mjd=base,
            area="JP01",
        )
        with pytest.raises(ReaderError, match="different observation"):
            read_ahi([first, second])

    def test_the_same_timeline_on_a_different_day_is_refused(self, tmp_path):
        """Timeline 0000 recurs daily, so the date has to separate them too."""
        base = 61041.00183583374
        first = self._set(tmp_path / "a", starts=[base])[0]
        second = fixture.write_granule(
            tmp_path / "b",
            band=13,
            columns=4,
            lines=3,
            counts=np.full((3, 4), 2, dtype=np.uint16),
            total_segments=3,
            segment_number=2,
            first_line=4,
            slope=1.0,
            intercept=0.0,
            start_mjd=base + 1.0,  # same timeline, 24 hours later
        )
        with pytest.raises(ReaderError, match="different observation"):
            read_ahi([first, second])


class TestAcquisitionWindow:
    """The start-time window has to be bracketed, not merely chosen.

    Nothing pinned its size: any value from the real scan span (526 s) up to a
    full day passed the suite, so the constant could drift either into rejecting
    real scans or into accepting two days' worth. These two cases bracket it.
    """

    def _pair(self, directory, *, apart_seconds, timeline=0):
        base = 61041.00183583374
        first = fixture.write_granule(
            directory / "a",
            band=13,
            columns=4,
            lines=3,
            counts=np.full((3, 4), 1, dtype=np.uint16),
            total_segments=2,
            segment_number=1,
            first_line=1,
            slope=1.0,
            intercept=0.0,
            start_mjd=base,
            timeline=timeline,
        )
        second = fixture.write_granule(
            directory / "b",
            band=13,
            columns=4,
            lines=3,
            counts=np.full((3, 4), 2, dtype=np.uint16),
            total_segments=2,
            segment_number=2,
            first_line=4,
            slope=1.0,
            intercept=0.0,
            start_mjd=base + apart_seconds / 86400.0,
            timeline=timeline,
        )
        return [first, second]

    def test_a_real_scan_span_is_inside_the_window(self, tmp_path):
        """Measured segment 1 to segment 10 is 526.0 s, so that must be accepted."""
        paths = self._pair(tmp_path, apart_seconds=526.0)
        assert read_ahi(paths, calibrate=False).read_array().shape == (6, 4)

    def test_twice_the_real_scan_span_is_still_inside(self, tmp_path):
        """Headroom for a slower scan than the one disk measured here."""
        paths = self._pair(tmp_path, apart_seconds=1052.0)
        assert read_ahi(paths, calibrate=False).read_array().shape == (6, 4)

    def test_two_hours_apart_is_outside_the_window(self, tmp_path):
        """A window wide enough to admit this would admit unrelated scenes."""
        paths = self._pair(tmp_path, apart_seconds=7200.0)
        with pytest.raises(ReaderError, match="hours apart"):
            read_ahi(paths)

    def test_the_same_slot_a_day_later_is_outside_the_window(self, tmp_path):
        paths = self._pair(tmp_path, apart_seconds=86400.0)
        with pytest.raises(ReaderError, match="hours apart"):
            read_ahi(paths)

    def test_the_window_brackets_the_measured_scan_span(self):
        """The constant must sit between a real scan and the daily recurrence."""
        from pyramids_eo.sensors.readers.ahi import _MAX_ACQUISITION_SPAN_DAYS

        window_s = _MAX_ACQUISITION_SPAN_DAYS * 86400.0
        assert window_s > 526.0 * 2, (
            f"window {window_s} s leaves no headroom over the measured 526 s scan"
        )
        assert window_s < 86400.0, "a window of a day or more cannot separate slots"
