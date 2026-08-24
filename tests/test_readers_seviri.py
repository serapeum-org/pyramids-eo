"""Unit + live tests for `pyramids_eo.sensors.readers.seviri` (MSG L1.5 native)."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.errors import CalibrationError, ReaderError, UnknownSensorError
from pyramids_eo.sensors.readers import read_seviri
from pyramids_eo.sensors.readers._common import calibrate_channel
from pyramids_eo.sensors.readers.seviri import (
    _CALIBRATION_OFFSET,
    _GRID_STEP_OFFSET,
    _LINE_NUMBER_OFFSET,
    _LINE_SIDE_INFO_BYTES,
    _REF_GRID_OFFSET,
    _geostationary_wkt,
    _segments,
    _unpack_10bit,
    parse_seviri_native,
)
from pyramids_eo.sensors.registry import (
    Channel,
    Sensor,
    get_sensor,
    radiance_to_brightness_temperature,
    radiance_to_reflectance,
)

#: IR_108's index in the on-disk VIS/IR channel order (== its calibration index).
_IR108_POSITION = 8


def _ds(arr: np.ndarray, tlc=(0.0, 4.0)) -> Dataset:
    """A pyramids Dataset holding raw radiance (a geographic source)."""
    return Dataset.create_from_array(arr, top_left_corner=tlc, cell_size=1.0, epsg=4326)


def _pack_10bit(counts: np.ndarray) -> bytes:
    """Pack `uint16` 10-bit counts (4 pixels per 5 bytes) — inverse of the reader."""
    pixels = counts.reshape(-1, 4).astype(np.uint16)
    packed = np.empty((pixels.shape[0], 5), dtype=np.uint8)
    packed[:, 0] = (pixels[:, 0] >> 2) & 0xFF
    packed[:, 1] = ((pixels[:, 0] & 0x3) << 6) | ((pixels[:, 1] >> 4) & 0x3F)
    packed[:, 2] = ((pixels[:, 1] & 0xF) << 4) | ((pixels[:, 2] >> 6) & 0xF)
    packed[:, 3] = ((pixels[:, 2] & 0x3F) << 2) | ((pixels[:, 3] >> 8) & 0x3)
    packed[:, 4] = pixels[:, 3] & 0xFF
    return packed.reshape(-1).tobytes()


def _build_nat(
    *,
    records,
    position=_IR108_POSITION,
    columns=1024,
    lines_declared=None,
    slope=0.205,
    offset=-10.456,
    stride=None,
    header_length=None,
    data_length=None,
    header_offset=200,
    write_segment_table=True,
    trailer_bytes=0,
) -> bytes:
    """Build a minimal but format-faithful MSG L1.5 native `.nat` byte string.

    `records` is a list of `(line_number, counts_row)`; only the target channel's
    block is populated (the reader reads just that block). The fixed 15Header
    offsets are poked with the reference grid, grid step and calibration. The grid
    is square (`lines_declared` defaults to `columns`), as the real VIS/IR grid is.
    """
    if lines_declared is None:
        lines_declared = columns
    packed_bytes = columns * 10 // 8
    visir_block = _LINE_SIDE_INFO_BYTES + packed_bytes
    if stride is None:
        stride = 11 * visir_block
    if header_length is None:
        header_length = _CALIBRATION_OFFSET + 12 * 16
    if data_length is None:
        data_length = stride * lines_declared

    header = bytearray(header_length)
    if header_length >= _CALIBRATION_OFFSET + 12 * 16:
        struct.pack_into(">ii", header, _REF_GRID_OFFSET, lines_declared, columns)
        struct.pack_into(">ff", header, _GRID_STEP_OFFSET, 3.0004032, 3.0004032)
        for index in range(12):
            pair = (slope, offset) if index == position else (1.0, 0.0)
            struct.pack_into(">dd", header, _CALIBRATION_OFFSET + index * 16, *pair)

    data_offset = header_offset + header_length
    product_header = bytearray(b" " * header_offset)
    if write_segment_table:
        table = (
            f"15Header : {header_length} {header_offset}\n"
            f"15Data : {data_length} {data_offset}\n"
        ).encode()
        product_header[: len(table)] = table

    data = bytearray(stride * len(records))
    block_start = position * visir_block
    for index, (line_number, counts) in enumerate(records):
        record = index * stride
        struct.pack_into(
            ">i", data, record + block_start + _LINE_NUMBER_OFFSET, line_number
        )
        start = record + block_start + _LINE_SIDE_INFO_BYTES
        data[start : start + packed_bytes] = _pack_10bit(np.asarray(counts, np.uint16))
    # A real product appends a 15Trailer segment after 15Data.
    return bytes(product_header) + bytes(header) + bytes(data) + b"\x00" * trailer_bytes


def _write(tmp_path: Path, name: str, payload: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


class TestUnpack10Bit:
    """`_unpack_10bit` reverses the 4-pixels-per-5-bytes packing."""

    def test_round_trips_known_counts(self):
        """Packing then unpacking returns the original 10-bit counts."""
        counts = np.array([0, 1023, 512, 7, 300, 900, 4, 1000], dtype=np.uint16)
        out = _unpack_10bit(_pack_10bit(counts), counts.size)
        assert np.array_equal(out, counts), "10-bit pack/unpack should round-trip"

    def test_extremes(self):
        """The all-ones and all-zeros quartets unpack to 1023 and 0."""
        counts = np.array([1023, 1023, 1023, 1023, 0, 0, 0, 0], dtype=np.uint16)
        out = _unpack_10bit(_pack_10bit(counts), counts.size)
        assert out[0] == 1023 and out[4] == 0, "extreme counts should survive"


class TestSegments:
    """`_segments` parses the NUL-padded ASCII segment table."""

    def test_parses_nul_padded_rows(self):
        """The header pads fields with NUL bytes, which parse as separators."""
        header = (
            b"15Header\x00\x00 : \x00 445286\x00\x00 5114\n15Data : 270344960 450400\n"
        )
        segs = _segments(header)
        assert segs["15Header"] == (445286, 5114), "15Header (length, offset)"
        assert segs["15Data"] == (270344960, 450400), "15Data (length, offset)"

    def test_missing_rows_yield_empty(self):
        """A header without the segment rows yields an empty mapping."""
        assert _segments(b"nothing relevant here") == {}


class TestGeostationaryWkt:
    """`_geostationary_wkt` builds a geostationary CRS carrying +h."""

    def test_carries_height_and_projection(self):
        """The WKT names the geostationary projection and its satellite height."""
        wkt = _geostationary_wkt(0.0)
        assert "Geostationary" in wkt, "should be a geostationary CRS"
        from osgeo import osr

        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        assert "+h=35785831" in srs.ExportToProj4(), "should carry the MSG height"


class TestParseSeviriNative:
    """`parse_seviri_native` decodes a VIS/IR channel from a native `.nat`."""

    def test_decodes_radiance_grid_and_crs(self, tmp_path):
        """A uniform-count granule decodes to radiance on the geostationary grid."""
        columns = 1024
        row = np.full(columns, 500, dtype=np.uint16)
        payload = _build_nat(
            records=[(600, row), (601, row)],
            columns=columns,
            slope=0.205,
            offset=-10.456,
        )
        scene = parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")
        array = np.asarray(scene.read_array(), dtype=float)
        assert array.shape == (2, columns), "one row per present line record"
        assert np.allclose(array, 500 * 0.205 - 10.456), (
            "count -> radiance via slope/offset"
        )
        assert scene.epsg is None, "a geostationary grid has no EPSG code"
        assert "Geostationary" in str(scene.crs), "should carry the geostationary CRS"
        px = 3.0004032 * 1000.0
        gt = scene.geotransform
        assert gt[1] == pytest.approx(px), "px is the grid step in metres"
        assert gt[5] == pytest.approx(-px), "north-up grid (negative row step)"
        # SSP at the centre of reference pixel N/2 (CGMS COFF=LOFF): the west edge
        # is a half-pixel in from -(N/2)*px, and line 601 sits above the reference.
        assert gt[0] == pytest.approx(-(columns / 2 - 0.5) * px), (
            "west edge (SSP at reference pixel N/2)"
        )
        reference = columns / 2  # square grid: COFF = LOFF = N/2 (1-based)
        assert gt[3] == pytest.approx((601 - reference + 0.5) * px), (
            "north edge of line 601"
        )

    def test_orients_north_up_and_west_left(self, tmp_path):
        """The South-East-origin storage is flipped to north-up / west-left."""
        columns = 1024
        south = np.full(columns, 100, dtype=np.uint16)  # line 600 (south)
        north = np.concatenate(  # line 601 (north): east half 100, west half 300
            [
                np.full(columns // 2, 100, np.uint16),
                np.full(columns // 2, 300, np.uint16),
            ]
        )
        payload = _build_nat(
            records=[(600, south), (601, north)], columns=columns, slope=1.0, offset=0.0
        )
        scene = parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")
        array = np.asarray(scene.read_array(), dtype=float)
        assert np.allclose(array[-1], 100.0), "the south line ends up on the bottom row"
        assert array[0, 0] == pytest.approx(300.0), (
            "west (storage-right) is on the left"
        )
        assert array[0, -1] == pytest.approx(100.0), (
            "east (storage-left) is on the right"
        )

    def test_masks_zero_count_corners(self, tmp_path):
        """Zero counts (off-earth space corners) are masked to NaN."""
        columns = 1024
        row = np.full(columns, 500, dtype=np.uint16)
        row[:4] = 0  # a space corner
        payload = _build_nat(records=[(600, row), (601, row)], columns=columns)
        scene = parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")
        array = np.asarray(scene.read_array(), dtype=float)
        assert np.isnan(array).sum() == 8, "the two zero-count runs map to NaN"

    def test_hrv_rejected(self):
        """HRV is out of scope and rejected before any file access."""
        with pytest.raises(ReaderError, match="HRV"):
            parse_seviri_native("unused.nat", "HRV")

    def test_unknown_channel_rejected(self):
        """An unknown VIS/IR channel is rejected."""
        with pytest.raises(ReaderError, match="unknown SEVIRI"):
            parse_seviri_native("unused.nat", "NOPE")

    def test_missing_file_raises(self, tmp_path):
        """A path that cannot be opened raises ReaderError."""
        with pytest.raises(ReaderError, match="cannot open"):
            parse_seviri_native(str(tmp_path / "absent.nat"), "IR_108")

    def test_no_segment_table_raises(self, tmp_path):
        """A file lacking the segment table is not a native granule."""
        payload = _build_nat(
            records=[(600, np.full(1024, 5, np.uint16))], write_segment_table=False
        )
        with pytest.raises(ReaderError, match="segment table"):
            parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")

    def test_truncated_header_raises(self, tmp_path):
        """A 15Header too short for the fixed layout raises ReaderError."""
        payload = _build_nat(
            records=[(600, np.full(1024, 5, np.uint16))], header_length=100
        )
        with pytest.raises(ReaderError, match="truncated"):
            parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")

    def test_bad_reference_grid_raises(self, tmp_path):
        """A 15Header whose reference grid fails the sanity check is rejected."""
        columns = 1024
        payload = bytearray(
            _build_nat(records=[(600, np.full(columns, 5, np.uint16))], columns=columns)
        )
        struct.pack_into(
            ">ii", payload, 200 + _REF_GRID_OFFSET, 7, 9
        )  # lines != cols, tiny
        with pytest.raises(ReaderError, match="unexpected 15Header layout"):
            parse_seviri_native(_write(tmp_path, "s.nat", bytes(payload)), "IR_108")

    def test_non_physical_slope_raises(self, tmp_path):
        """A non-positive calibration slope signals a wrong header layout."""
        payload = _build_nat(
            records=[(600, np.full(1024, 5, np.uint16))], slope=-1.0, offset=0.0
        )
        with pytest.raises(ReaderError, match="calibration slope"):
            parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")

    def test_data_length_not_whole_records_raises(self, tmp_path):
        """A 15Data length that is not a whole number of records is rejected."""
        payload = _build_nat(
            records=[(600, np.full(1024, 5, np.uint16))], data_length=99999
        )
        with pytest.raises(ReaderError, match="whole number of line records"):
            parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")

    def test_channel_block_exceeds_stride_raises(self, tmp_path):
        """A stride too small to hold the channel's block is rejected."""
        columns = 1024
        visir_block = _LINE_SIDE_INFO_BYTES + columns * 10 // 8
        payload = _build_nat(records=[], columns=columns, stride=3 * visir_block)
        with pytest.raises(ReaderError, match="exceeds the line-record stride"):
            parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")

    def test_no_complete_record_raises(self, tmp_path):
        """A file with the headers but no image record is rejected."""
        payload = _build_nat(records=[])
        with pytest.raises(ReaderError, match="no complete image line record"):
            parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")

    def test_non_contiguous_lines_raise(self, tmp_path):
        """Line records that skip a line signal a wrong layout / selection."""
        row = np.full(1024, 5, np.uint16)
        payload = _build_nat(records=[(600, row), (602, row)])  # gap at 601
        with pytest.raises(ReaderError, match="not contiguous"):
            parse_seviri_native(_write(tmp_path, "s.nat", payload), "IR_108")

    def test_full_disk_with_trailer_decodes(self, tmp_path):
        """A full product (all declared lines) plus a 15Trailer decodes to the disk.

        The record count is capped at the declared line count, not derived from
        the file size — otherwise the trailer bytes are read as extra records and
        the contiguity guard rejects a valid full-disk product.
        """
        columns = 1000  # smallest square grid that passes the layout sanity check
        block = _LINE_SIDE_INFO_BYTES + columns * 10 // 8
        rows = [
            (line, np.full(columns, 300, np.uint16))
            for line in range(1, columns + 1)  # every declared line, contiguous
        ]
        payload = _build_nat(
            records=rows,
            columns=columns,
            position=0,
            stride=block,
            trailer_bytes=3 * block + 17,  # several strides of trailer, as a real product has
        )
        scene = parse_seviri_native(_write(tmp_path, "full.nat", payload), "VIS006")
        assert scene.read_array().shape == (columns, columns), "the full disk decodes"

    def test_subsatellite_longitude_sets_crs(self, tmp_path):
        """A non-zero sub-satellite longitude is carried into the geostationary CRS."""
        row = np.full(1024, 400, np.uint16)
        payload = _build_nat(records=[(600, row), (601, row)])
        scene = parse_seviri_native(
            _write(tmp_path, "iodc.nat", payload), "IR_108", subsatellite_longitude=41.5
        )
        from osgeo import osr

        srs = osr.SpatialReference()
        srs.ImportFromWkt(str(scene.crs))
        assert "lon_0=41.5" in srs.ExportToProj4(), "SSP longitude should set lon_0"


class TestReadSeviri:
    """`read_seviri` calibrates and geolocates a single channel."""

    def test_none_source_raises(self):
        """A missing source is a ReaderError."""
        with pytest.raises(ReaderError, match="source is required"):
            read_seviri(None, "IR_108")

    def test_thermal_channel_calibrated_to_bt(self):
        """A thermal channel (Dataset source) is calibrated to brightness temperature."""
        radiance = np.full((2, 2), 80.0)
        out = read_seviri(_ds(radiance), "IR_108")
        ch = get_sensor("seviri").get_channel("IR_108")
        expected = radiance_to_brightness_temperature(
            radiance, ch.central_wavenumber_cm1, ch.alpha, ch.beta
        )
        assert np.allclose(out.read_array(), expected), "BT calibration mismatch"

    def test_solar_channel_calibrated_to_reflectance(self):
        """A solar channel (Dataset source) is calibrated to reflectance."""
        radiance = np.full((2, 2), 120.0)
        out = read_seviri(_ds(radiance), "VIS006")
        ch = get_sensor("seviri").get_channel("VIS006")
        expected = radiance_to_reflectance(radiance, ch.solar_irradiance)
        assert np.allclose(out.read_array(), expected), "reflectance mismatch"

    def test_calibrate_false_returns_raw(self):
        """With calibrate=False the raw radiance is returned."""
        out = read_seviri(_ds(np.full((2, 2), 42.0)), "IR_108", calibrate=False)
        assert np.allclose(out.read_array(), 42.0), "raw radiance should pass through"

    def test_geolocation_preserved(self):
        """A geographic (EPSG) source keeps its CRS + geotransform."""
        src = _ds(np.ones((2, 2)))
        out = read_seviri(src, "IR_108")
        assert out.epsg == 4326, f"CRS should be preserved, got {out.epsg}"
        assert out.geotransform == src.geotransform, "geotransform should be preserved"

    def test_native_path_preserves_geostationary_crs(self, tmp_path):
        """A `.nat` path is decoded, calibrated, and keeps its geostationary CRS."""
        columns = 1024
        row = np.full(columns, 500, dtype=np.uint16)
        payload = _build_nat(
            records=[(600, row), (601, row)],
            columns=columns,
            slope=0.205,
            offset=-10.456,
        )
        out = read_seviri(_write(tmp_path, "s.nat", payload), "IR_108")
        radiance = 500 * 0.205 - 10.456
        ch = get_sensor("seviri").get_channel("IR_108")
        expected = radiance_to_brightness_temperature(
            radiance, ch.central_wavenumber_cm1, ch.alpha, ch.beta
        )
        assert np.allclose(out.read_array(), expected), (
            "native path should calibrate to BT"
        )
        assert out.epsg is None and "Geostationary" in str(out.crs), (
            "geos CRS preserved"
        )
        assert np.isnan(out.no_data_value[0]), "nodata should be NaN"

    def test_native_path_threads_subsatellite_longitude(self, tmp_path):
        """read_seviri forwards subsatellite_longitude to the default native parser."""
        from osgeo import osr

        row = np.full(1024, 400, np.uint16)
        payload = _build_nat(records=[(600, row), (601, row)])
        out = read_seviri(
            _write(tmp_path, "iodc.nat", payload),
            "IR_108",
            calibrate=False,
            subsatellite_longitude=9.5,
        )
        srs = osr.SpatialReference()
        srs.ImportFromWkt(str(out.crs))
        assert "lon_0=9.5" in srs.ExportToProj4(), "SSP longitude should reach the CRS"

    def test_parse_used_for_non_dataset(self):
        """A non-Dataset source is decoded via the injected `parse` callable."""
        captured = {}

        def _parser(path, channel):
            captured["path"], captured["channel"] = path, channel
            return _ds(np.full((2, 2), 7.0))

        out = read_seviri("scene.nat", "IR_108", calibrate=False, parse=_parser)
        assert captured == {"path": "scene.nat", "channel": "IR_108"}, captured
        assert np.allclose(out.read_array(), 7.0), "parsed radiance not used"

    def test_unknown_channel_raises(self):
        """An unknown channel surfaces UnknownSensorError."""
        src = _ds(np.ones((2, 2)))
        with pytest.raises(UnknownSensorError, match="has no channel"):
            read_seviri(src, "NOPE")

    def test_coeffs_override_thermal_constants(self):
        """Per-granule coeffs override the registry Planck constants."""
        radiance = np.full((2, 2), 80.0)
        out = read_seviri(
            _ds(radiance),
            "IR_108",
            coeffs={"central_wavenumber_cm1": 900.0, "alpha": 0.99, "beta": 0.5},
        )
        expected = radiance_to_brightness_temperature(radiance, 900.0, 0.99, 0.5)
        assert np.allclose(out.read_array(), expected), "coeffs not preferred"


class TestCalibrateChannelCoeffs:
    """`calibrate_channel` lets coeffs override the channel's radiometric kind."""

    def test_kind_override_forces_solar_path(self):
        """A coeffs kind='solar' routes a registry-thermal channel to reflectance."""
        radiance = np.full((2, 2), 50.0)
        out = calibrate_channel(
            radiance,
            "IR_108",
            "seviri",
            1.0,
            None,
            coeffs={"kind": "solar", "solar_irradiance": 100.0},
        )
        assert np.allclose(out, radiance_to_reflectance(radiance, 100.0)), (
            "kind override should take the solar path"
        )

    def test_missing_calibration_constant_raises(self, monkeypatch):
        """A channel missing its constants raises CalibrationError."""
        broken = Sensor(
            name="seviri",
            channels={
                "z": Channel("z", 3.9, 3000, "thermal", central_wavenumber_cm1=None)
            },
        )
        monkeypatch.setattr(
            "pyramids_eo.sensors.readers._common.get_sensor", lambda name: broken
        )
        src = _ds(np.ones((2, 2)))
        with pytest.raises(CalibrationError, match="central_wavenumber"):
            read_seviri(src, "z")


@pytest.mark.live
def test_read_seviri_real_granule():
    """End-to-end decode of a real MSG SEVIRI `.nat` into IR_108 brightness temperature.

    Skips unless a directory holding a real `.nat` granule is provided via
    `SEVIRI_FIXTURES_DIR` (the marker, not the env var, gates whether this runs).
    """
    fixtures = Path(os.environ.get("SEVIRI_FIXTURES_DIR", "tests/data/seviri"))
    granules = sorted(fixtures.glob("*.nat"))
    if not granules:
        pytest.skip("real SEVIRI fixtures not available (set SEVIRI_FIXTURES_DIR)")
    scene = read_seviri(str(granules[0]), "IR_108")
    array = np.asarray(scene.read_array(), dtype=float)
    finite = array[np.isfinite(array)]
    assert finite.min() > 180.0, f"BT min too low: {finite.min()}"
    assert finite.max() < 340.0, f"BT max too high: {finite.max()}"
    assert np.median(finite) > 260.0, f"BT median implausibly cold: {np.median(finite)}"
    assert "Geostationary" in str(scene.crs), "should carry the geostationary CRS"
    assert abs(scene.geotransform[1]) == pytest.approx(3000.4, abs=1.0), (
        "SEVIRI is 3 km"
    )
    assert scene.geotransform[5] < 0, "the grid must be north-up (gt[5] < 0)"
    assert np.isnan(scene.no_data_value[0]), "nodata should be NaN"


@pytest.mark.live
def test_read_seviri_real_granule_subsatellite_geolocation():
    """A real granule's grid places the sub-satellite point at a pixel centre (0N/0E).

    Independent absolute-geolocation check: transform every pixel centre to
    lon/lat and confirm the pixel nearest the sub-satellite point sits at
    (0, 0) to well within half a pixel — i.e. the CGMS reference-pixel
    convention, not the naive grid centre (which would be a half-pixel off).
    """
    fixtures = Path(os.environ.get("SEVIRI_FIXTURES_DIR", "tests/data/seviri"))
    granules = sorted(fixtures.glob("*.nat"))
    if not granules:
        pytest.skip("real SEVIRI fixtures not available (set SEVIRI_FIXTURES_DIR)")
    from pyproj import Transformer

    scene = read_seviri(str(granules[0]), "IR_108")
    gt = scene.geotransform
    rows, cols = np.asarray(scene.read_array()).shape
    col_c, row_c = np.meshgrid(np.arange(cols), np.arange(rows))
    xs = gt[0] + (col_c + 0.5) * gt[1]
    ys = gt[3] + (row_c + 0.5) * gt[5]
    lon, lat = Transformer.from_crs(str(scene.crs), "EPSG:4326", always_xy=True).transform(xs, ys)
    on_disk = np.isfinite(lon) & np.isfinite(lat)
    nearest = np.nanargmin(np.where(on_disk, np.abs(lon) + np.abs(lat), np.inf))
    assert abs(lon.ravel()[nearest]) < 0.005, "a pixel centre coincides with 0 deg E"
    assert abs(lat.ravel()[nearest]) < 0.005, "a pixel centre coincides with 0 deg N"
