"""MSG-SEVIRI Level-1.5 native (`.nat`) reader.

`read_seviri` decodes one SEVIRI VIS/IR channel from a real MSG Level-1.5 native
(`.nat`) granule and returns a calibrated, geolocated pyramids `Dataset`
(reflectance for a solar channel, brightness temperature for a thermal one).

The `.nat` format is a packed binary: an ASCII Main/Secondary Product Header
carrying the segment table, a fixed-layout binary ``15Header`` (the Level-1.5
prologue, with the per-granule calibration and the reference grid), then the
``15Data`` image records — one record per scan line, holding a 65-byte side-info
header plus **10-bit packed** counts for each selected channel in turn (the 11
VIS/IR channels first, then HRV). `parse_seviri_native` reads the segment table
and the ``15Header``, then for the requested channel unpacks the counts, applies
the granule's ``Cal_Slope`` / ``Cal_Offset`` to get radiance, masks the
off-earth (zero-count) corners, and orients the scene north-up / west-left (the
grid origin is South-East). `read_seviri` then calibrates that radiance via the
sensor registry.

Validated against a real Meteosat-10 (MSG3) full-disk granule (see issue #40):
``IR_108`` decodes to brightness temperature in the expected range on the 3 km
geostationary grid. HRV (a different, higher-resolution geometry) is out of
scope.
"""

from __future__ import annotations

import mmap
import os
import re
import struct
from typing import Any

import numpy as np

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._common import calibrate_channel

#: The 11 VIS/IR channels in their fixed on-disk order, which is also the order
#: of the ``Level15ImageCalibration`` slope/offset array. HRV follows them in the
#: record but is a different geometry and is not decoded here.
_VISIR_CHANNELS = (
    "VIS006",
    "VIS008",
    "IR_016",
    "IR_039",
    "WV_062",
    "WV_073",
    "IR_087",
    "IR_097",
    "IR_108",
    "IR_120",
    "IR_134",
)

# --- 15Data image-record layout (fixed MSG Level-1.5 native format) ----------
#: Bytes of per-channel line side-info before that line's packed pixels.
_LINE_SIDE_INFO_BYTES = 65
#: Offset of the big-endian int32 ``line_number_in_grid`` within a line block.
_LINE_NUMBER_OFFSET = 51

# --- 15Header layout (byte offsets from the 15Header segment start) ----------
# The Level-1.5 header is a fixed-size, fixed-layout binary structure, so these
# offsets are format constants. They were reverse-engineered from and validated
# against a real Meteosat-10 (MSG3) full-disk granule: the calibration block
# decodes IR_108 to a physical brightness-temperature range and the reference
# grid / grid step read 3712 / 3.0004 km at the offsets below. The offline suite
# cannot prove them against reality (its fixtures are built from these same
# constants); the `live` `test_read_seviri_real_granule*` cases are the
# CI-independent ground truth, and `test_layout_offsets_match_expected_values`
# pins them so an accidental edit is caught. They are cross-checked at read time
# against the reference grid / grid step, which sit just before the calibration.
#: ``ReferenceGridVIS_IR``: NumberOfLines (int32) then NumberOfColumns (int32).
_REF_GRID_OFFSET = 386936
#: ``ReferenceGridVIS_IR``: LineDirGridStep (float32 km) then ColumnDirGridStep.
_GRID_STEP_OFFSET = 386944
#: ``RadiometricProcessing.Level15ImageCalibration``: 12 x (Cal_Slope float64,
#: Cal_Offset float64), indexed by the channel's position in ``_VISIR_CHANNELS``.
_CALIBRATION_OFFSET = 387104

# --- MSG SEVIRI geostationary projection constants ---------------------------
#: Perspective-point (satellite) height above the ellipsoid, metres.
_SATELLITE_HEIGHT_M = 35785831.0
#: Ellipsoid semi-axes (metres) of the MSG geostationary projection.
_EARTH_EQUATORIAL_M = 6378169.0
_EARTH_POLAR_M = 6356583.8

#: Matches a ``15Header`` / ``15Data`` segment-table row: ``name : length offset``
#: (the product header pads fields with NUL bytes, normalised to spaces first).
_SEGMENT_RE = re.compile(r"(15Header|15Data)\s*:\s*(\d+)\s+(\d+)")
#: Leading bytes scanned for the ASCII segment table (MPH + SPH are ~5114 bytes).
_PRODUCT_HEADER_SCAN_BYTES = 6000


def _segments(product_header: bytes) -> dict[str, tuple[int, int]]:
    """Parse the ASCII product header for the ``15Header`` / ``15Data`` segments.

    Args:
        product_header: The leading bytes of the file (Main + Secondary Product
            Header), which carry the ASCII segment table.

    Returns:
        A mapping of segment name to its ``(length, offset)`` in bytes.
    """
    # The header pads fields with NUL bytes; treat them as whitespace so the
    # ``name : length offset`` rows parse.
    text = product_header.replace(b"\x00", b" ").decode("latin-1", "replace")
    return {
        match.group(1): (int(match.group(2)), int(match.group(3)))
        for match in _SEGMENT_RE.finditer(text)
    }


def _unpack_10bit(packed: bytes, columns: int) -> np.ndarray:
    """Unpack big-endian 10-bit counts (4 pixels per 5 bytes) to ``uint16``.

    Args:
        packed: The packed pixel bytes for one line (``columns * 10 / 8`` bytes).
        columns: The number of pixels in the line (a multiple of 4).

    Returns:
        The `columns` unpacked counts as a 1-D ``uint16`` array.
    """
    quintets = np.frombuffer(packed, dtype=np.uint8).reshape(-1, 5).astype(np.uint16)
    out = np.empty((quintets.shape[0], 4), dtype=np.uint16)
    out[:, 0] = (quintets[:, 0] << 2) | (quintets[:, 1] >> 6)
    out[:, 1] = ((quintets[:, 1] & 0x3F) << 4) | (quintets[:, 2] >> 4)
    out[:, 2] = ((quintets[:, 2] & 0x0F) << 6) | (quintets[:, 3] >> 2)
    out[:, 3] = ((quintets[:, 3] & 0x03) << 8) | quintets[:, 4]
    return out.reshape(-1)[:columns]


def _geostationary_wkt(longitude_of_projection_origin: float) -> str:
    """Build the MSG SEVIRI geostationary CRS as WKT.

    Args:
        longitude_of_projection_origin: Sub-satellite longitude (degrees east).

    Returns:
        The geostationary CRS as a WKT string (its PROJ form carries ``+h=``).
    """
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromProj4(
        f"+proj=geos +h={_SATELLITE_HEIGHT_M} +a={_EARTH_EQUATORIAL_M} "
        f"+b={_EARTH_POLAR_M} +lon_0={longitude_of_projection_origin} "
        f"+sweep=y +units=m +no_defs"
    )
    return str(srs.ExportToWkt())


def parse_seviri_native(
    path: Any, channel: str, *, subsatellite_longitude: float = 0.0
) -> Any:
    """Decode one VIS/IR channel of an MSG Level-1.5 native (`.nat`) granule.

    Reads the segment table and the fixed-layout ``15Header`` (reference grid,
    grid step and the per-granule calibration), then unpacks the channel's 10-bit
    counts from every image line record, applies the granule ``Cal_Slope`` /
    ``Cal_Offset`` to get radiance, masks the zero-count (off-earth) corners to
    NaN, and orients the scene north-up / west-left. Only the image lines
    actually present are read, so a header-preserved subset (whose header still
    declares the full disk) decodes correctly rather than reading past end-of-file.

    Args:
        path: Path to an MSG SEVIRI Level-1.5 native `.nat` file.
        channel: A VIS/IR channel identifier (e.g. `"IR_108"`, `"VIS006"`); HRV
            is not supported.
        subsatellite_longitude: Sub-satellite longitude (degrees east) for the
            geostationary CRS. Defaults to `0.0`, the nominal prime service
            (`MSG15`); pass the service longitude for IODC (41.5) or rapid-scan
            (9.5) granules, whose `.nat` header the reader does not parse for it.

    Returns:
        A pyramids `Dataset` holding the channel radiance on the geostationary
        grid, with the geostationary CRS, a metre geotransform, and NaN nodata.

    Raises:
        ReaderError: When the channel is unknown / HRV, the file is not a
            recognised MSG Level-1.5 native granule (no segment table, or the
            ``15Header`` layout / calibration fails its sanity checks), the file
            carries no complete line record, or the decoded line records are not
            contiguous (an unexpected channel selection or layout).
    """
    if channel == "HRV":
        raise ReaderError("read_seviri: HRV is not supported (different geometry)")
    try:
        position = _VISIR_CHANNELS.index(channel)
    except ValueError as exc:
        raise ReaderError(
            f"read_seviri: unknown SEVIRI VIS/IR channel {channel!r}"
        ) from exc

    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise ReaderError(f"read_seviri: cannot open {path} ({exc})") from exc
    with handle:
        segments = _segments(handle.read(_PRODUCT_HEADER_SCAN_BYTES))
        try:
            header_length, header_offset = segments["15Header"]
            data_length, data_offset = segments["15Data"]
        except KeyError as exc:
            raise ReaderError(
                "read_seviri: no 15Header/15Data segment table "
                f"(not an MSG Level-1.5 native file?) — {path}"
            ) from exc
        handle.seek(header_offset)
        header = handle.read(header_length)
        if len(header) < _CALIBRATION_OFFSET + 12 * 16:
            raise ReaderError(
                "read_seviri: 15Header is truncated "
                "(not a supported MSG Level-1.5 native file)"
            )
        file_size = os.fstat(handle.fileno()).st_size

    lines_declared, columns = struct.unpack_from(">ii", header, _REF_GRID_OFFSET)
    line_step_km, column_step_km = struct.unpack_from(">ff", header, _GRID_STEP_OFFSET)
    # The reference grid and grid step sit immediately before the calibration
    # block: sane values here confirm the fixed 15Header layout was found. The
    # line and column steps must match (the VIS/IR grid is isotropic 3 km).
    if not (
        lines_declared == columns
        and 1000 <= columns <= 12000
        and 2.9 < column_step_km < 3.1
        and np.isclose(line_step_km, column_step_km)
        and columns % 4 == 0
    ):
        raise ReaderError(
            "read_seviri: unexpected 15Header layout (reference grid / grid step "
            "sanity check failed) — not a supported MSG Level-1.5 native file"
        )
    pixel_m = column_step_km * 1000.0

    slope, offset = struct.unpack_from(
        ">dd", header, _CALIBRATION_OFFSET + position * 16
    )
    if slope <= 0:
        raise ReaderError(
            "read_seviri: non-physical calibration slope (unexpected header layout)"
        )

    if data_length % lines_declared:
        raise ReaderError(
            "read_seviri: 15Data length is not a whole number of line records"
        )
    stride = data_length // lines_declared
    packed_bytes = columns * 10 // 8
    # The 11 VIS/IR channels precede HRV in every record; assume the standard
    # all-channel product so channel `position` maps straight to block `position`.
    # The contiguity check below rejects a file where that assumption does not hold.
    block_start = position * (_LINE_SIDE_INFO_BYTES + packed_bytes)
    data_start = block_start + _LINE_SIDE_INFO_BYTES
    if data_start + packed_bytes > stride:
        raise ReaderError("read_seviri: channel block exceeds the line-record stride")

    # The declared 15Data holds exactly `lines_declared` records; a full product
    # has a 15Trailer segment after it, so cap the count at the declared line
    # count. Without this cap the trailer bytes get read as extra "line records",
    # break the +1 contiguity of `line_numbers`, and reject a valid full disk.
    # The file-size term still handles a header-preserved subset (fewer records).
    available = min((file_size - data_offset) // stride, lines_declared)
    if available < 1:
        raise ReaderError("read_seviri: file carries no complete image line record")

    counts = np.empty((available, columns), dtype=np.uint16)
    line_numbers = np.empty(available, dtype=np.int64)
    with (
        open(path, "rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        for index in range(available):
            record = data_offset + index * stride
            line_numbers[index] = struct.unpack_from(
                ">i", mapped, record + block_start + _LINE_NUMBER_OFFSET
            )[0]
            start = record + data_start
            counts[index] = _unpack_10bit(mapped[start : start + packed_bytes], columns)

    if not np.all(np.diff(line_numbers) == 1):
        raise ReaderError(
            "read_seviri: image line records are not contiguous "
            "(unexpected channel selection or record layout)"
        )

    radiance = counts.astype(float) * slope + offset
    radiance[counts == 0] = np.nan  # zero counts mark the off-earth space corners
    # The grid origin is South-East: line numbers increase northward and the
    # stored columns run east -> west (the first column is East). Reverse both
    # axes for a north-up, west-left raster. Copy the flipped view to a
    # C-contiguous array so no downstream consumer trips on the negative strides.
    radiance = np.ascontiguousarray(radiance[::-1, ::-1])

    # CGMS/EUMETSAT navigation places the sub-satellite point at the CENTRE of the
    # reference pixel N/2 (0-based, both axes; e.g. pixel 1856 of the 3712 grid) —
    # the COFF = LOFF convention, which reproduces the published MSG full-disk area
    # extent (west edge -(N/2 + 0.5) * px). All columns are present (full width) so
    # the SSP column is N/2; only a subset of lines may be present, so the SSP row
    # is N/2 measured from the northmost present line (row 0 after the north-up
    # flip). Register the metre geotransform so that reference pixel's centre is
    # (x=0, y=0).
    reference = columns / 2.0  # == lines_declared / 2.0 for the square VIS/IR grid
    ssp_row = int(line_numbers[-1]) - reference
    x_west_edge = -(reference + 0.5) * pixel_m
    y_north_edge = (ssp_row + 0.5) * pixel_m
    geo = (x_west_edge, pixel_m, 0.0, y_north_edge, 0.0, -pixel_m)

    from pyramids.dataset import Dataset

    dataset = Dataset.create_from_array(
        radiance, geo=geo, epsg=None, no_data_value=np.nan
    )
    dataset.crs = _geostationary_wkt(subsatellite_longitude)
    return dataset


def read_seviri(
    source: Any,
    channel: str,
    *,
    sensor: str = "seviri",
    calibrate: bool = True,
    sun_earth_distance: float = 1.0,
    cos_sza: Any = None,
    coeffs: dict[str, Any] | None = None,
    parse: Any = None,
    subsatellite_longitude: float = 0.0,
) -> Any:
    """Read one SEVIRI channel into a calibrated, geolocated `Dataset`.

    Decodes the channel radiance (from a `.nat` granule via `parse`, or straight
    from a `Dataset` source), then calibrates it to reflectance (solar) or
    brightness temperature (thermal) via the registry and returns a pyramids
    `Dataset` carrying the source's CRS + geotransform.

    Args:
        source: A path to an MSG Level-1.5 native `.nat` file (decoded by
            `parse`), or a pyramids `Dataset` already holding the channel radiance.
        channel: Channel identifier (e.g. `"IR_108"`, `"VIS006"`).
        sensor: Registry sensor name (default `"seviri"`).
        calibrate: When `True` (default), calibrate to a physical quantity; when
            `False`, return the raw radiance.
        sun_earth_distance: Sun-earth distance (AU) for solar-channel reflectance.
            The default `1.0` leaves reflectance up to ~3.4% off (`d` ranges
            ~0.983-1.017 AU over the year); pass the granule's `d` for absolute
            accuracy.
        cos_sza: Cosine of the solar zenith angle for the reflectance sun-angle
            correction, or `None`.
        coeffs: Per-granule calibration coefficients preferred over the registry
            fallback (see `calibrate_channel`), or `None` to use the registry.
        parse: Callable `(source, channel) -> Dataset` used when `source` is not
            already a `Dataset`. Defaults to the native `.nat` parser
            (`parse_seviri_native`).
        subsatellite_longitude: Sub-satellite longitude (degrees east) passed to
            the default native parser for the geostationary CRS (default `0.0`,
            the prime service); ignored when a custom `parse` is supplied.

    Returns:
        A pyramids `Dataset` of the calibrated (or raw) channel.

    Raises:
        ReaderError: When `source` is `None`, or the native parse fails.
        CalibrationError: When a channel lacks the constants its kind needs.
        UnknownSensorError: When the sensor / channel is not in the registry.
    """
    if source is None:
        raise ReaderError("read_seviri: source is required")

    if hasattr(source, "read_array"):
        dataset = source
    elif parse is not None:
        dataset = parse(source, channel)
    else:
        dataset = parse_seviri_native(
            source, channel, subsatellite_longitude=subsatellite_longitude
        )
    radiance = np.asarray(dataset.read_array(), dtype=float)
    data = (
        calibrate_channel(
            radiance, channel, sensor, sun_earth_distance, cos_sza, coeffs=coeffs
        )
        if calibrate
        else radiance
    )

    from pyramids.dataset import Dataset

    # Calibration can produce NaN (terminator reflectance / non-positive
    # radiance), so declare NaN as nodata rather than the default -9999.
    epsg = dataset.epsg if dataset.epsg else None
    result = Dataset.create_from_array(
        data, geo=dataset.geotransform, epsg=epsg, no_data_value=np.nan
    )
    if epsg is None:
        # A non-EPSG CRS (e.g. the geostationary grid) must be carried explicitly.
        result.crs = dataset.crs
    return result
