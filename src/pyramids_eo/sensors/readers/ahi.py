"""Himawari-8/9 AHI Himawari Standard Data (HSD, ``.DAT``) reader.

`read_ahi` decodes one AHI band from a real Himawari Standard Data granule and
returns a calibrated, geolocated pyramids `Dataset` (reflectance for a solar
band, brightness temperature for a thermal one).

HSD is a packed binary of **12 blocks**: 11 fixed-layout header blocks followed
by the image data block. Every header block opens with its own block number
(`I1`) and block length (`I2`), so the header is *self-describing* and is walked
rather than assumed — blocks #8, #9 and #10 are variable length (they carry
per-line repeats), so a fixed offset table would not reach the data. All integer
fields are **unsigned**, and block #1 field 4 declares the file's byte order.

`parse_ahi_hsd` walks the header, reads the grid from block #2, the normalised
geostationary projection from block #3, the per-granule count-to-radiance
calibration from block #5 and the segment placement from block #7, then decodes
the `uint16` counts of block #12, masks the two sentinel count values, and
registers the metre geotransform of the geostationary grid. `read_ahi` then
calibrates that radiance, preferring the granule's own coefficients over the
registry.

A full disk is distributed as 10 segment files (`Skkll`); pass the segment paths
together and they are stitched into the full image by their block #7 first-line
numbers. A single segment reads on its own, georeferenced at its true position
in the disk.

Format reference: *Himawari Standard Data User's Guide*, Version 1.3 (3 July
2017), Japan Meteorological Agency — Table 6 block structures. The projection is
the Normalized Geostationary Projection of the CGMS *LRIT/HRIT Global
Specification* v2.8, Section 4.4, which HSD Section 3 adopts.
"""

from __future__ import annotations

import bz2
import math
import mmap
import re
import struct
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._common import calibrate_channel

#: A `struct` / numpy byte-order prefix. HSD declares its own order in block #1, so the
#: two spellings are the only ones a granule can ask for.
ByteOrder = Literal["<", ">"]

#: The Himawari Standard Data file extension (segments may add ``.bz2``).
HSD_SUFFIX = ".dat"

#: Number of header blocks preceding the data block (HSD has 12 blocks total).
_HEADER_BLOCK_COUNT = 11
#: Block number of the image data block, which follows the 11 header blocks.
_DATA_BLOCK = 12

# --- Header block numbers (each block asserts its own number in field 1) -----
_BASIC = 1
_DATA_INFO = 2
_PROJECTION = 3
_CALIBRATION = 5
_SEGMENT = 7

# --- Fixed block lengths, used as layout cross-checks ------------------------
#: Declared lengths of the fixed-size header blocks, by block number. Blocks #8,
#: #9 and #10 are variable length (per-line repeats) and so are absent here.
_FIXED_BLOCK_LENGTHS = {
    1: 282,
    2: 50,
    3: 127,
    4: 139,
    5: 147,
    6: 259,
    7: 47,
    11: 259,
}

# --- #1 Basic information block (282 bytes) ----------------------------------
#: ``Byte order``: 0 little-endian, 1 big-endian. An ``I1``, so it reads the same
#: either way — which is what breaks the chicken-and-egg of needing to know the
#: byte order in order to read the byte order.
_B1_BYTE_ORDER = 5
#: ``Satellite name`` (``C`` x 16), e.g. ``H08`` / ``H09``.
_B1_SATELLITE = 6
_B1_SATELLITE_BYTES = 16

# --- #2 Data information block (50 bytes) ------------------------------------
_B2_BITS_PER_PIXEL = 3  # I2
_B2_COLUMNS = 5  # I2
_B2_LINES = 7  # I2
_B2_COMPRESSION = 9  # I1: 0 none, 1 gzip, 2 bzip2

# --- #3 Projection information block (127 bytes) -----------------------------
_B3_SUB_LON = 3  # R8, degrees east
_B3_CFAC = 11  # I4, column scaling factor
_B3_LFAC = 15  # I4, line scaling factor
_B3_COFF = 19  # R4, column offset (sub-satellite column, 1-based)
_B3_LOFF = 23  # R4, line offset (sub-satellite line, 1-based)
_B3_RS_KM = 27  # R8, Earth centre to satellite distance
_B3_REQ_KM = 35  # R8, equatorial radius (WGS84)
_B3_RPOL_KM = 43  # R8, polar radius (WGS84)

# --- #5 Calibration information block (147 bytes) ----------------------------
_B5_BAND_NUMBER = 3  # I2
_B5_CENTRAL_WAVELENGTH_M = 5  # R8, metres
_B5_VALID_BITS = 13  # I2
_B5_ERROR_COUNT = 15  # I2, sentinel 65535
_B5_OUTSIDE_COUNT = 17  # I2, sentinel 65534
_B5_SLOPE = 19  # R8, radiance from slope times count plus intercept
_B5_INTERCEPT = 27  # R8
# Infrared variant (bands 7-16), from offset 35: the Planck correction
# coefficients and the physical constants the granule was calibrated with.
_B5_IR_C0 = 35  # R8, quadratic Te-to-Tb correction, constant term
_B5_IR_C1 = 43  # R8
_B5_IR_C2 = 51  # R8
_B5_IR_LIGHT_SPEED = 83  # R8, m/s
_B5_IR_PLANCK = 91  # R8, J s
_B5_IR_BOLTZMANN = 99  # R8, J/K
# Visible / near-infrared variant (bands 1-6), from offset 35.
_B5_VIS_ALBEDO_COEFF = 35  # R8, the albedo coefficient c-prime, i.e. 1 over S0
_B5_VIS_SLOPE = 51  # R8, updated (calibrated) slope
_B5_VIS_INTERCEPT = 59  # R8, updated (calibrated) intercept

# --- #7 Segment information block (47 bytes) ---------------------------------
_B7_TOTAL_SEGMENTS = 3  # I1
_B7_SEGMENT_NUMBER = 4  # I1
_B7_FIRST_LINE = 5  # I2, 1-based line of this segment in the full image

#: Highest band number carried by AHI; bands 1-6 are solar, 7-16 thermal.
_SOLAR_BAND_MAX = 6
_BAND_MAX = 16

#: Plausible range (metres) for the ground sampling distance derived from
#: ``CFAC`` / ``LFAC``. AHI's grids are 0.5 / 1 / 2 km; a value outside this
#: window means the projection block was misread (wrong byte order or offsets)
#: rather than an exotic grid, and is reported instead of silently producing a
#: wrong geotransform.
_MIN_PIXEL_M = 100.0
_MAX_PIXEL_M = 20000.0

#: Absolute tolerance for treating a decoded float as zero. HSD leaves an unused
#: calibration field as literal zero bytes, so this only has to separate "the
#: field is unset" from any real gain, which is many orders of magnitude larger.
_ZERO_TOLERANCE = 1e-30

#: Scaling denominator of the normalised geostationary projection: the
#: ``2**-16`` factor of ``c = COFF + nint(x * 2**-16 * CFAC)`` (LRIT/HRIT 4.4).
_PROJECTION_SCALE = 65536.0

#: ``HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT`` — the HSD file-name
#: convention (Guide Table 4).
_FILENAME_RE = re.compile(
    r"^HS_(?P<satellite>H\d{2})_(?P<date>\d{8})_(?P<time>\d{4})_B(?P<band>\d{2})"
    # `A-Z` alone: the IGNORECASE flag already covers the lowercase spellings, so
    # adding `a-z` would just duplicate the class.
    r"_(?P<area>[A-Z0-9]{4})_R(?P<resolution>\d{2})_S(?P<segment>\d{4})",
    re.IGNORECASE,
)


def band_name(number: int) -> str:
    """Return the registry channel key for an AHI band number.

    Args:
        number: Band number, 1-16.

    Returns:
        The zero-padded ``"B<nn>"`` key used by the bundled registry table and by
        the HSD file name (e.g. ``3 -> "B03"``).

    Examples:
        - Band numbers are zero-padded to two digits:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import band_name
            >>> band_name(3), band_name(13)
            ('B03', 'B13')

            ```
    """
    return f"B{int(number):02d}"


def _canonical_band(band: Any) -> str:
    """Normalise a caller's band spelling to the registry ``"B<nn>"`` key.

    Args:
        band: A band number (``13``), or a string spelling of one (``"13"``,
            ``"b13"``, ``"B13"``, ``"band13"``).

    Returns:
        The canonical ``"B<nn>"`` key.

    Raises:
        ReaderError: When `band` does not name an AHI band in 1-16.

    Examples:
        - Several spellings canonicalise to the same band:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import _canonical_band
            >>> _canonical_band(13)
            'B13'
            >>> _canonical_band("b13")
            'B13'
            >>> _canonical_band("band13")
            'B13'

            ```
        - A single-digit band is zero-padded, and a padded spelling is accepted:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import _canonical_band
            >>> _canonical_band(3)
            'B03'
            >>> _canonical_band("3")
            'B03'
            >>> _canonical_band("B03")
            'B03'

            ```
        - A band outside AHI's 1-16 is refused rather than silently clamped:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import _canonical_band
            >>> _canonical_band(17)
            Traceback (most recent call last):
                ...
            pyramids_eo.errors.ReaderError: read_ahi: AHI band number 17 is out of range 1-16

            ```
    """
    if isinstance(band, (int, np.integer)) and not isinstance(band, bool):
        number = int(band)
    else:
        match = re.fullmatch(r"(?:b|band)?_?0*(\d{1,2})", str(band).strip().lower())
        if match is None:
            raise ReaderError(
                f"read_ahi: unknown AHI band {band!r}; expected a band number "
                f"1-{_BAND_MAX} or a spelling such as 'B13'"
            )
        number = int(match.group(1))
    if not 1 <= number <= _BAND_MAX:
        raise ReaderError(
            f"read_ahi: AHI band number {number} is out of range 1-{_BAND_MAX}"
        )
    return band_name(number)


def band_from_filename(path: Any) -> str | None:
    """Return the band key named by an HSD file name, when it follows Table 4.

    Args:
        path: Path to an HSD granule.

    Returns:
        The ``"B<nn>"`` band key the file name declares, or `None` when the name
        does not follow the HSD convention.

    Examples:
        - A conventional name yields its band; anything else yields `None`:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import band_from_filename
            >>> band_from_filename("HS_H08_20150415_0230_B13_FLDK_R20_S0101.DAT")
            'B13'
            >>> band_from_filename("scene.dat") is None
            True

            ```
    """
    match = _FILENAME_RE.match(str(path).replace("\\", "/").rsplit("/", 1)[-1])
    return None if match is None else band_name(int(match.group("band")))


def _read_bytes(path: Any) -> bytes:
    """Read an HSD granule, transparently decompressing a ``.bz2`` segment.

    Args:
        path: Path to a ``.DAT`` or ``.DAT.bz2`` granule.

    Returns:
        The granule bytes.

    Raises:
        ReaderError: When the file cannot be opened or decompressed.
    """
    name = str(path)
    try:
        if name.lower().endswith(".bz2"):
            with bz2.open(name, "rb") as compressed:
                return compressed.read()
        with (
            open(name, "rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            return bytes(mapped)
    except (OSError, EOFError, ValueError) as exc:
        raise ReaderError(f"read_ahi: cannot read {name} ({exc})") from exc


def _byte_order(buffer: bytes) -> ByteOrder:
    """Determine the granule's byte order from block #1.

    The order flag is itself a single byte, so it reads identically either way;
    the block length is then checked against its fixed 282 bytes to confirm the
    flag (and the whole layout) rather than trusting it.

    Args:
        buffer: The granule bytes.

    Returns:
        A `struct` byte-order prefix, ``"<"`` or ``">"``.

    Raises:
        ReaderError: When the file does not open with block #1, is truncated, or
            the declared order disagrees with the fixed block length.
    """
    if len(buffer) < _FIXED_BLOCK_LENGTHS[_BASIC]:
        raise ReaderError(
            "read_ahi: file is shorter than the 282-byte basic information block "
            "(not a Himawari Standard Data granule?)"
        )
    if buffer[0] != _BASIC:
        raise ReaderError(
            f"read_ahi: file does not start with header block #1 (first byte is "
            f"{buffer[0]}) — not a Himawari Standard Data granule"
        )
    order: ByteOrder = "<" if buffer[_B1_BYTE_ORDER] == 0 else ">"
    length = struct.unpack_from(f"{order}H", buffer, 1)[0]
    if length != _FIXED_BLOCK_LENGTHS[_BASIC]:
        raise ReaderError(
            f"read_ahi: basic information block declares {length} bytes, expected "
            f"{_FIXED_BLOCK_LENGTHS[_BASIC]} — unexpected byte order or layout"
        )
    return order


def walk_header_blocks(buffer: bytes, order: ByteOrder) -> dict[int, tuple[int, int]]:
    """Walk the 11 self-describing HSD header blocks.

    Each block opens with its block number (``I1``) and length (``I2``), so the
    header is traversed by following those lengths. This is what makes the
    variable-length blocks (#8 navigation correction, #9 observation time, #10
    error information — each carrying per-line repeats) safe to step over: their
    sizes are not knowable in advance, so a fixed offset table would not find
    the data block.

    Args:
        buffer: The granule bytes.
        order: The `struct` byte-order prefix from :func:`_byte_order`.

    Returns:
        A mapping of block number to its ``(offset, length)`` in bytes. The data
        block (#12) is included, with the remaining byte count as its length.

    Raises:
        ReaderError: When a block is truncated, reports a zero length, declares a
            number other than its position, or a fixed-size block's length
            disagrees with the specification.
    """
    blocks: dict[int, tuple[int, int]] = {}
    offset = 0
    for expected in range(1, _HEADER_BLOCK_COUNT + 1):
        if offset + 3 > len(buffer):
            raise ReaderError(
                f"read_ahi: granule is truncated inside header block #{expected}"
            )
        number = buffer[offset]
        length = struct.unpack_from(f"{order}H", buffer, offset + 1)[0]
        if number != expected:
            raise ReaderError(
                f"read_ahi: header block at offset {offset} declares number "
                f"{number}, expected #{expected} — unexpected HSD layout"
            )
        if length == 0:
            raise ReaderError(
                f"read_ahi: header block #{expected} declares a zero length"
            )
        fixed = _FIXED_BLOCK_LENGTHS.get(expected)
        if fixed is not None and length != fixed:
            raise ReaderError(
                f"read_ahi: header block #{expected} declares {length} bytes, "
                f"expected the fixed {fixed} — unexpected HSD layout"
            )
        if offset + length > len(buffer):
            raise ReaderError(
                f"read_ahi: header block #{expected} ({length} bytes at offset "
                f"{offset}) runs past the end of the granule"
            )
        blocks[expected] = (offset, length)
        offset += length
    blocks[_DATA_BLOCK] = (offset, len(buffer) - offset)
    return blocks


@dataclass(frozen=True)
class HsdHeader:
    """The HSD header fields a read needs, decoded from one granule.

    Attributes:
        satellite: Satellite name from block #1 (e.g. ``"H08"``).
        band_number: AHI band number, 1-16 (block #5).
        columns: Pixels per line in this granule / segment (block #2).
        lines: Lines in this granule / segment (block #2).
        bits_per_pixel: Stored bit depth, always 16 for HSD (block #2).
        valid_bits: Valid bits per pixel for the band, 11 / 12 / 14 (block #5).
        compression: Data-block compression flag — 0 none, 1 gzip, 2 bzip2.
        sub_lon: Sub-satellite longitude, degrees east (block #3).
        cfac: Column scaling factor of the normalised projection (block #3).
        lfac: Line scaling factor of the normalised projection (block #3).
        coff: Column offset — the sub-satellite column, 1-based (block #3).
        loff: Line offset — the sub-satellite line, 1-based (block #3).
        rs_km: Distance from the Earth's centre to the satellite (block #3).
        req_km: Ellipsoid equatorial radius, WGS84 (block #3).
        rpol_km: Ellipsoid polar radius, WGS84 (block #3).
        slope: Count-to-radiance slope actually in force (block #5); the
            calibrated (updated) value for a solar band when the granule
            carries one.
        intercept: Count-to-radiance intercept in force (block #5).
        error_count: Count value marking an error pixel (block #5, 65535).
        outside_count: Count value marking a pixel outside the scan area
            (block #5, 65534).
        central_wavelength_m: Band central wavelength in metres (block #5).
        albedo_coefficient: Solar bands only — ``c'`` of ``albedo = c' * radiance``,
            so ``1 / c'`` is the band solar irradiance; `None` for a thermal band.
        planck_c0: Thermal bands only — constant term of the effective-to-
            brightness-temperature correction; `None` for a solar band.
        planck_c1: Thermal bands only — linear term, else `None`.
        planck_c2: Thermal bands only — quadratic term, else `None`.
        light_speed: Thermal bands only — speed of light the granule was
            calibrated with (m/s), else `None`.
        planck_constant: Thermal bands only — Planck constant (J s), else `None`.
        boltzmann_constant: Thermal bands only — Boltzmann constant (J/K), else
            `None`.
        total_segments: Number of segments the full image is split into (#7).
        segment_number: This granule's segment sequence number (block #7).
        first_line: 1-based line of this segment within the full image (#7).

    Examples:
        - A decoded band-13 full-disk header reports its band, its radiometric
          kind, and the satellite height the projection is built from:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import HsdHeader
            >>> header = HsdHeader(
            ...     satellite="H08", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_m=10.4e-06, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ... )
            >>> header.band
            'B13'
            >>> header.kind
            'thermal'
            >>> round(header.satellite_height_m)
            35785863

            ```
    """

    satellite: str
    band_number: int
    columns: int
    lines: int
    bits_per_pixel: int
    valid_bits: int
    compression: int
    sub_lon: float
    cfac: int
    lfac: int
    coff: float
    loff: float
    rs_km: float
    req_km: float
    rpol_km: float
    slope: float
    intercept: float
    error_count: int
    outside_count: int
    central_wavelength_m: float
    albedo_coefficient: float | None
    planck_c0: float | None
    planck_c1: float | None
    planck_c2: float | None
    light_speed: float | None
    planck_constant: float | None
    boltzmann_constant: float | None
    total_segments: int
    segment_number: int
    first_line: int

    @property
    def band(self) -> str:
        """Return the registry band key for this granule.

        Returns:
            The zero-padded ``"B<nn>"`` key for :attr:`band_number`, which is
            both the bundled registry's channel key and the band field of the
            HSD file name (e.g. ``"B13"``).
        """
        return band_name(self.band_number)

    @property
    def kind(self) -> str:
        """Return the band's radiometric kind.

        Returns:
            ``"solar"`` for bands 1-6 and ``"thermal"`` for bands 7-16 - the
            same split the specification uses to decide which variant of
            calibration block #5 a granule carries.
        """
        return "solar" if self.band_number <= _SOLAR_BAND_MAX else "thermal"

    @property
    def satellite_height_m(self) -> float:
        """Return the satellite height above the ellipsoid, in metres.

        The projection block gives the distance from the Earth's *centre*; the
        geostationary CRS wants the perspective-point height above the equator,
        which is that distance less the equatorial radius.

        Returns:
            The perspective-point height in metres, derived from the granule's
            own ``Rs`` and ``req`` rather than from a hard-coded constant.
        """
        return (self.rs_km - self.req_km) * 1000.0


def _unpack(buffer: bytes, order: ByteOrder, code: str, offset: int) -> Any:
    """Unpack one field, reporting a truncated block as a reader error.

    Args:
        buffer: The granule bytes.
        order: The `struct` byte-order prefix.
        code: A single `struct` format code (e.g. ``"H"``, ``"d"``).
        offset: Absolute byte offset of the field.

    Returns:
        The decoded value.

    Raises:
        ReaderError: When the field runs past the end of the buffer.
    """
    try:
        return struct.unpack_from(f"{order}{code}", buffer, offset)[0]
    except struct.error as exc:
        raise ReaderError(
            f"read_ahi: cannot read a {code!r} field at offset {offset} "
            f"(granule is truncated)"
        ) from exc


def parse_hsd_header(buffer: bytes) -> HsdHeader:
    """Decode the HSD header blocks a read depends on.

    Args:
        buffer: The granule bytes (already decompressed).

    Returns:
        The decoded :class:`HsdHeader`.

    Raises:
        ReaderError: When the granule is not recognisable HSD, a block is
            truncated or mislabelled, the band number is out of range, the grid
            is empty, or the count-to-radiance slope is non-physical.
    """
    order = _byte_order(buffer)
    blocks = walk_header_blocks(buffer, order)

    basic = blocks[_BASIC][0]
    raw_name = buffer[
        basic + _B1_SATELLITE : basic + _B1_SATELLITE + _B1_SATELLITE_BYTES
    ]
    satellite = raw_name.decode("ascii", "replace").strip().strip("\x00").strip()

    info = blocks[_DATA_INFO][0]
    bits_per_pixel = _unpack(buffer, order, "H", info + _B2_BITS_PER_PIXEL)
    columns = _unpack(buffer, order, "H", info + _B2_COLUMNS)
    lines = _unpack(buffer, order, "H", info + _B2_LINES)
    compression = buffer[info + _B2_COMPRESSION]
    if columns == 0 or lines == 0:
        raise ReaderError(
            f"read_ahi: granule declares an empty grid ({lines} x {columns})"
        )

    projection = blocks[_PROJECTION][0]
    calibration = blocks[_CALIBRATION][0]
    band_number = _unpack(buffer, order, "H", calibration + _B5_BAND_NUMBER)
    if not 1 <= band_number <= _BAND_MAX:
        raise ReaderError(
            f"read_ahi: granule declares band number {band_number}, outside the "
            f"AHI range 1-{_BAND_MAX} — unexpected HSD layout"
        )

    slope = _unpack(buffer, order, "d", calibration + _B5_SLOPE)
    intercept = _unpack(buffer, order, "d", calibration + _B5_INTERCEPT)
    albedo_coefficient: float | None = None
    planck: tuple[float | None, ...] = (None, None, None, None, None, None)
    if band_number <= _SOLAR_BAND_MAX:
        albedo_coefficient = _unpack(
            buffer, order, "d", calibration + _B5_VIS_ALBEDO_COEFF
        )
        # Fields 12 / 13 restate the slope / intercept after an in-flight
        # recalibration. They are the values in force when present, but a
        # granule that has never been recalibrated leaves them zero, so fall
        # back to fields 8 / 9 rather than reading a zero gain.
        updated_slope = _unpack(buffer, order, "d", calibration + _B5_VIS_SLOPE)
        updated_intercept = _unpack(buffer, order, "d", calibration + _B5_VIS_INTERCEPT)
        if math.isfinite(updated_slope) and not math.isclose(
            updated_slope, 0.0, abs_tol=_ZERO_TOLERANCE
        ):
            slope, intercept = updated_slope, updated_intercept
    else:
        planck = tuple(
            _unpack(buffer, order, "d", calibration + field)
            for field in (
                _B5_IR_C0,
                _B5_IR_C1,
                _B5_IR_C2,
                _B5_IR_LIGHT_SPEED,
                _B5_IR_PLANCK,
                _B5_IR_BOLTZMANN,
            )
        )
    if not math.isfinite(slope) or math.isclose(slope, 0.0, abs_tol=_ZERO_TOLERANCE):
        raise ReaderError(
            f"read_ahi: granule declares a non-physical count-to-radiance slope "
            f"({slope!r}) — unexpected HSD layout"
        )

    segment = blocks[_SEGMENT][0]
    return HsdHeader(
        satellite=satellite,
        band_number=int(band_number),
        columns=int(columns),
        lines=int(lines),
        bits_per_pixel=int(bits_per_pixel),
        valid_bits=int(_unpack(buffer, order, "H", calibration + _B5_VALID_BITS)),
        compression=int(compression),
        sub_lon=float(_unpack(buffer, order, "d", projection + _B3_SUB_LON)),
        cfac=int(_unpack(buffer, order, "I", projection + _B3_CFAC)),
        lfac=int(_unpack(buffer, order, "I", projection + _B3_LFAC)),
        coff=float(_unpack(buffer, order, "f", projection + _B3_COFF)),
        loff=float(_unpack(buffer, order, "f", projection + _B3_LOFF)),
        rs_km=float(_unpack(buffer, order, "d", projection + _B3_RS_KM)),
        req_km=float(_unpack(buffer, order, "d", projection + _B3_REQ_KM)),
        rpol_km=float(_unpack(buffer, order, "d", projection + _B3_RPOL_KM)),
        slope=float(slope),
        intercept=float(intercept),
        error_count=int(_unpack(buffer, order, "H", calibration + _B5_ERROR_COUNT)),
        outside_count=int(_unpack(buffer, order, "H", calibration + _B5_OUTSIDE_COUNT)),
        central_wavelength_m=float(
            _unpack(buffer, order, "d", calibration + _B5_CENTRAL_WAVELENGTH_M)
        ),
        albedo_coefficient=albedo_coefficient,
        planck_c0=planck[0],
        planck_c1=planck[1],
        planck_c2=planck[2],
        light_speed=planck[3],
        planck_constant=planck[4],
        boltzmann_constant=planck[5],
        total_segments=int(buffer[segment + _B7_TOTAL_SEGMENTS]),
        segment_number=int(buffer[segment + _B7_SEGMENT_NUMBER]),
        first_line=int(_unpack(buffer, order, "H", segment + _B7_FIRST_LINE)),
    )


def pixel_size_m(header: HsdHeader) -> tuple[float, float]:
    """Return the ``(column, line)`` ground sampling distance in metres.

    The normalised geostationary projection of LRIT/HRIT 4.4 relates a column to
    the satellite scanning angle by ``c = COFF + nint(x * 2**-16 * CFAC)``. HSD
    expresses that angle in **degrees** (unlike MSG, which uses radians), so one
    column step spans ``2**16 / CFAC`` degrees, and the ground distance is that
    angle times the satellite height. The identity is exact: the published AHI
    ``CFAC`` values reproduce 2000 / 1000 / 500 m to the metre.

    Args:
        header: The decoded granule header.

    Returns:
        The ``(x, y)`` pixel size in metres.

    Raises:
        ReaderError: When a scaling factor is zero, or the resulting pixel size
            is outside the plausible 100 m - 20 km window — which is how a
            misread projection block (wrong byte order or offsets) surfaces,
            rather than as a silently wrong geotransform.

    Examples:
        - Himawari's published 2 km scaling factor reproduces a 2 km grid.
          ``CFAC`` is an integer, so it cannot encode the step exactly - the
          residual here is well under a millimetre:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import HsdHeader, pixel_size_m
            >>> header = HsdHeader(
            ...     satellite="H08", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_m=10.4e-06, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ... )
            >>> [round(step, 4) for step in pixel_size_m(header)]
            [2000.0, 2000.0]

            ```
        - Band 3's 0.5 km grid quadruples the factor, and the step follows, so
          the grid comes from the granule rather than from a built-in default:
            ```python
            >>> import dataclasses
            >>> from pyramids_eo.sensors.readers.ahi import HsdHeader, pixel_size_m
            >>> header = HsdHeader(
            ...     satellite="H08", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_m=10.4e-06, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ... )
            >>> fine = dataclasses.replace(header, cfac=81865100, lfac=81865100)
            >>> [round(step, 4) for step in pixel_size_m(fine)]
            [500.0, 500.0]

            ```
    """
    if header.cfac == 0 or header.lfac == 0:
        raise ReaderError(
            "read_ahi: granule declares a zero CFAC / LFAC scaling factor "
            "— unexpected HSD layout"
        )
    height = header.satellite_height_m
    if not math.isfinite(height) or height <= 0.0:
        raise ReaderError(
            f"read_ahi: granule implies a non-physical satellite height "
            f"({height!r} m) — unexpected HSD layout"
        )
    x = height * math.radians(_PROJECTION_SCALE / header.cfac)
    y = height * math.radians(_PROJECTION_SCALE / header.lfac)
    for value, axis in ((x, "column"), (y, "line")):
        if not _MIN_PIXEL_M <= value <= _MAX_PIXEL_M:
            raise ReaderError(
                f"read_ahi: the {axis} scaling factor implies a {value:.1f} m "
                f"pixel, outside the plausible {_MIN_PIXEL_M:.0f}-"
                f"{_MAX_PIXEL_M:.0f} m range — unexpected HSD layout"
            )
    return x, y


def geotransform(header: HsdHeader, first_line: int) -> tuple[float, ...]:
    """Build the metre geotransform of the geostationary grid.

    ``COFF`` / ``LOFF`` give the sub-satellite point's column and line, 1-based,
    and the projection's ``y`` runs **southward** (LRIT/HRIT 4.4), so a positive
    ``LFAC`` means line 1 is the northernmost — HSD is stored north-up and needs
    no row flip. The transform places the north-west corner of `first_line`.

    Args:
        header: The decoded granule header.
        first_line: 1-based line, within the full image, of the first row of the
            array being registered. For a stitched full disk this is 1; for a
            lone segment it is the segment's own first line.

    Returns:
        A GDAL geotransform ``(x_min, dx, 0, y_max, 0, -dy)`` in projection
        metres.

    Examples:
        - A full disk registers symmetrically about the sub-satellite point at
          the grid centre, and runs north-up, so ``dy`` is negative:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import HsdHeader, geotransform
            >>> header = HsdHeader(
            ...     satellite="H08", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_m=10.4e-06, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ... )
            >>> x_min, dx, _, y_max, _, dy = geotransform(header, 1)
            >>> round(x_min), round(y_max)
            (-5500000, 5500000)
            >>> dy < 0
            True
            >>> round(x_min + 5500 * dx)
            5500000

            ```
        - A segment starting further down the disk shifts the north edge south
          by exactly its line offset, so one segment lands where it belongs:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import HsdHeader, geotransform
            >>> header = HsdHeader(
            ...     satellite="H08", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_m=10.4e-06, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ... )
            >>> whole = geotransform(header, 1)
            >>> segment = geotransform(header, 1651)
            >>> round((whole[3] - segment[3]) / abs(whole[5]))
            1650

            ```
    """
    dx, dy = pixel_size_m(header)
    # Column / line centres sit at (c - COFF) and -(l - LOFF) steps from the
    # sub-satellite point; the raster origin is the outer corner of the first
    # pixel, half a pixel further out.
    x_min = (1.0 - header.coff - 0.5) * dx
    y_max = (header.loff - first_line + 0.5) * dy
    return (x_min, dx, 0.0, y_max, 0.0, -dy)


def _geostationary_wkt(header: HsdHeader) -> str:
    """Build the AHI geostationary CRS as WKT from the granule's own ellipsoid.

    Args:
        header: The decoded granule header, which carries the sub-satellite
            longitude and the WGS84 semi-axes the product was navigated with.

    Returns:
        The geostationary CRS as a WKT string (its PROJ form carries ``+h=``).
    """
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromProj4(
        f"+proj=geos +h={header.satellite_height_m} "
        f"+a={header.req_km * 1000.0} +b={header.rpol_km * 1000.0} "
        f"+lon_0={header.sub_lon} +sweep=y +units=m +no_defs"
    )
    return str(srs.ExportToWkt())


def _counts(
    buffer: bytes, header: HsdHeader, blocks: dict[int, tuple[int, int]]
) -> Any:
    """Read the data block's ``uint16`` counts as a ``(lines, columns)`` array.

    Args:
        buffer: The granule bytes.
        header: The decoded granule header.
        blocks: The block table from :func:`walk_header_blocks`.

    Returns:
        The raw counts, shaped ``(header.lines, header.columns)``.

    Raises:
        ReaderError: When the data block is compressed (HSD compresses whole
            files, so a compressed *block* is not something this reader can
            decode), the stored bit depth is not 16, or the block is too short
            for the declared grid.
    """
    if header.compression != 0:
        raise ReaderError(
            f"read_ahi: the data block declares compression flag "
            f"{header.compression} (1 gzip, 2 bzip2); only an uncompressed data "
            "block is supported — a whole-file .bz2 granule is decompressed "
            "transparently and is not affected"
        )
    if header.bits_per_pixel != 16:
        raise ReaderError(
            f"read_ahi: the data block declares {header.bits_per_pixel} bits per "
            "pixel; HSD stores 16-bit counts — unexpected HSD layout"
        )
    offset, available = blocks[_DATA_BLOCK]
    expected = header.lines * header.columns * 2
    if available < expected:
        raise ReaderError(
            f"read_ahi: the data block holds {available} bytes but the declared "
            f"{header.lines} x {header.columns} grid needs {expected} "
            "(truncated granule or an incomplete segment)"
        )
    dtype = np.dtype(np.uint16).newbyteorder(_byte_order(buffer))
    flat = np.frombuffer(
        buffer, dtype=dtype, count=header.lines * header.columns, offset=offset
    )
    return flat.reshape(header.lines, header.columns)


def _radiance(counts: Any, header: HsdHeader) -> Any:
    """Convert raw counts to spectral radiance, masking the sentinel values.

    Args:
        counts: The raw ``uint16`` counts.
        header: The decoded granule header, carrying the slope / intercept in
            force and the two sentinel count values.

    Returns:
        Radiance in ``W m-2 sr-1 m-1``, with error and outside-scan pixels NaN.
    """
    radiance = counts.astype(float) * header.slope + header.intercept
    sentinel = (counts == header.error_count) | (counts == header.outside_count)
    radiance[sentinel] = np.nan
    return radiance


def _brightness_temperature(radiance: Any, header: HsdHeader) -> Any:
    """Convert thermal radiance to brightness temperature the HSD way.

    Inverts Planck's law in its per-wavelength form using the constants the
    granule itself was calibrated with, then applies the granule's quadratic
    effective-to-brightness correction ``Tb = c0 + c1 * Te + c2 * Te**2``. The
    registry's linear ``alpha`` / ``beta`` form cannot express that quadratic, so
    a thermal AHI band is calibrated here rather than through
    :func:`~pyramids_eo.sensors.readers._common.calibrate_channel`.

    Args:
        radiance: Spectral radiance in ``W m-2 sr-1 m-1``.
        header: The decoded granule header.

    Returns:
        Brightness temperature in kelvin, NaN where the radiance is not positive
        (off-earth and error pixels).

    Raises:
        ReaderError: When the granule omits a constant the inversion needs, or
            carries a non-physical central wavelength.
    """
    missing = [
        name
        for name, value in (
            ("c0", header.planck_c0),
            ("c1", header.planck_c1),
            ("c2", header.planck_c2),
            ("speed of light", header.light_speed),
            ("Planck constant", header.planck_constant),
            ("Boltzmann constant", header.boltzmann_constant),
        )
        if value is None or not math.isfinite(value)
    ]
    if missing:
        raise ReaderError(
            f"read_ahi: thermal band {header.band} is missing the calibration "
            f"constant(s) {', '.join(missing)} — unexpected HSD layout"
        )
    wavelength = header.central_wavelength_m
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise ReaderError(
            f"read_ahi: thermal band {header.band} declares a non-physical "
            f"central wavelength ({wavelength!r} m) — unexpected HSD layout"
        )
    # The `missing` check above proves each of these is a finite float, which the
    # declared `float | None` cannot express; converting once keeps the arithmetic
    # below plainly typed.
    c0 = float(header.planck_c0 or 0.0)
    c1 = float(header.planck_c1 or 0.0)
    c2 = float(header.planck_c2 or 0.0)
    light = float(header.light_speed or 0.0)
    planck = float(header.planck_constant or 0.0)
    boltzmann = float(header.boltzmann_constant or 0.0)

    values = np.asarray(radiance, dtype=float)
    # Planck inverse: Te = hc / (lambda k ln(1 + 2hc^2 / (lambda^5 I))). Only a
    # positive radiance has a temperature; the rest (off-earth, error) stay NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (2.0 * planck * light**2) / (wavelength**5 * values)
        effective = (planck * light) / (
            wavelength * boltzmann * np.log1p(np.where(values > 0.0, ratio, np.nan))
        )
    return c0 + c1 * effective + c2 * effective**2


def _dataset(array: Any, header: HsdHeader, first_line: int) -> Any:
    """Wrap an array on the granule's geostationary grid as a `Dataset`.

    Args:
        array: The ``(lines, columns)`` values to register.
        header: The decoded granule header.
        first_line: 1-based first line of `array` within the full image.

    Returns:
        A pyramids `Dataset` carrying the geostationary CRS, the metre
        geotransform, and NaN nodata.
    """
    from pyramids.dataset import Dataset, GeoReference

    dataset = Dataset.from_array(
        np.ascontiguousarray(array, dtype=float),
        geo_ref=GeoReference(geo=geotransform(header, first_line), epsg=None),
        no_data_value=np.nan,
    )
    dataset.crs = _geostationary_wkt(header)
    return dataset


def parse_ahi_hsd(path: Any) -> Any:
    """Decode one HSD granule to a geolocated radiance `Dataset`.

    Reads the header, decodes the band's 16-bit counts, applies the granule's
    own count-to-radiance slope / intercept, masks the error and outside-scan
    sentinel counts to NaN, and registers the geostationary grid. A segment is
    georeferenced at its true place in the full image, so one segment reads on
    its own without being mistaken for the whole disk.

    Args:
        path: Path to a Himawari Standard Data ``.DAT`` (or ``.DAT.bz2``) granule.

    Returns:
        A pyramids `Dataset` of spectral radiance (``W m-2 sr-1 m-1``) on the
        geostationary grid, with the geostationary CRS and NaN nodata.

    Raises:
        ReaderError: When the file is not a readable HSD granule (see
            :func:`parse_hsd_header` and :func:`_counts`).
    """
    buffer = _read_bytes(path)
    header = parse_hsd_header(buffer)
    blocks = walk_header_blocks(buffer, _byte_order(buffer))
    radiance = _radiance(_counts(buffer, header, blocks), header)
    return _dataset(radiance, header, header.first_line)


def _stitch(paths: Any) -> tuple[Any, HsdHeader]:
    """Assemble the segments of one band into a single radiance array.

    Args:
        paths: An iterable of HSD segment paths of the **same** band and
            timestamp.

    Returns:
        A ``(radiance, header)`` pair: the stitched array covering every line the
        segments carry, and the header of the first (northernmost) segment.

    Raises:
        ReaderError: When no path is given, the segments disagree on band, grid
            width or projection, a segment repeats, or the segments leave a gap
            (which would otherwise yield a silently part-blank disk).
    """
    decoded: list[tuple[HsdHeader, Any]] = []
    for path in paths:
        buffer = _read_bytes(path)
        header = parse_hsd_header(buffer)
        blocks = walk_header_blocks(buffer, _byte_order(buffer))
        decoded.append((header, _radiance(_counts(buffer, header, blocks), header)))
    if not decoded:
        raise ReaderError("read_ahi: no HSD segment paths were given")
    decoded.sort(key=lambda item: item[0].first_line)

    first = decoded[0][0]
    for header, _ in decoded[1:]:
        if header.band_number != first.band_number:
            raise ReaderError(
                f"read_ahi: segments mix bands ({first.band} and {header.band}); "
                "stitch one band at a time"
            )
        if header.columns != first.columns:
            raise ReaderError(
                f"read_ahi: segments disagree on width ({first.columns} and "
                f"{header.columns} columns)"
            )
        if (header.cfac, header.lfac, header.coff, header.loff) != (
            first.cfac,
            first.lfac,
            first.coff,
            first.loff,
        ):
            raise ReaderError(
                "read_ahi: segments disagree on the projection block; they are "
                "not segments of one image"
            )

    expected_line = decoded[0][0].first_line
    for header, _ in decoded:
        if header.first_line != expected_line:
            raise ReaderError(
                f"read_ahi: segment {header.segment_number} starts at line "
                f"{header.first_line}, expected {expected_line} — the segments "
                "are not contiguous (a missing or duplicated segment)"
            )
        expected_line += header.lines

    total_lines = expected_line - decoded[0][0].first_line
    stitched = np.empty((total_lines, first.columns), dtype=float)
    row = 0
    for header, radiance in decoded:
        stitched[row : row + header.lines] = radiance
        row += header.lines
    return stitched, first


def read_ahi(
    source: Any,
    band: Any = None,
    *,
    sensor: str = "ahi",
    calibrate: bool = True,
    sun_earth_distance: float = 1.0,
    cos_sza: Any = None,
    coeffs: dict[str, Any] | None = None,
) -> Any:
    """Read one AHI band from Himawari Standard Data into a `Dataset`.

    Decodes the band's counts to radiance with the granule's own calibration,
    then — when `calibrate` is set — converts to a physical quantity: reflectance
    for a solar band (1-6) and brightness temperature for a thermal one (7-16).
    Both conversions prefer the coefficients the granule carries over the
    registry's nominal table, so the result is granule-accurate.

    Pass the segment paths of one band together to read a stitched full disk;
    pass one path to read that segment alone, georeferenced where it belongs in
    the disk.

    Args:
        source: Path to an HSD ``.DAT`` / ``.DAT.bz2`` granule, an iterable of
            segment paths of one band and timestamp, or a pyramids `Dataset`
            already holding the band radiance.
        band: The band to read (``13``, ``"B13"``, ``"band13"``). Optional: a
            granule names its own band, so this only has to be given to assert
            the expectation, and a mismatch is an error. Required when `source`
            is a `Dataset`, which carries no band metadata.
        sensor: Registry sensor name (default ``"ahi"``).
        calibrate: When `True` (default), calibrate to reflectance / brightness
            temperature; when `False`, return the raw radiance.
        sun_earth_distance: Sun-earth distance (AU) for solar-band reflectance.
            The default ``1.0`` leaves reflectance up to ~3.4% off (``d`` ranges
            ~0.983-1.017 AU over the year); pass the granule's ``d`` for absolute
            accuracy.
        cos_sza: Cosine of the solar zenith angle for the reflectance sun-angle
            correction, or `None`.
        coeffs: Per-granule calibration coefficients preferred over both the
            granule's own values and the registry (see
            :func:`~pyramids_eo.sensors.readers._common.calibrate_channel`), or
            `None`.

    Returns:
        A pyramids `Dataset` of the calibrated (or raw) band on the geostationary
        grid, with the geostationary CRS, a metre geotransform and NaN nodata.

    Raises:
        ReaderError: When `source` is `None`, `band` is unknown or disagrees with
            the granule, a `Dataset` source is given without `band`, or the
            granule cannot be decoded.
        CalibrationError: When a band lacks the constants its kind needs.
        UnknownSensorError: When the sensor / band is not in the registry.
    """
    if source is None:
        raise ReaderError("read_ahi: source is required")
    requested = None if band is None else _canonical_band(band)

    if hasattr(source, "read_array"):
        if requested is None:
            raise ReaderError(
                "read_ahi: band is required when source is already a Dataset "
                "(it carries no HSD header to name the band)"
            )
        return _calibrate_dataset(
            source,
            requested,
            sensor=sensor,
            calibrate=calibrate,
            sun_earth_distance=sun_earth_distance,
            cos_sza=cos_sza,
            coeffs=coeffs,
            header=None,
        )

    paths = (
        [source]
        if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__")
        else list(source)
    )
    if len(paths) == 1:
        buffer = _read_bytes(paths[0])
        header = parse_hsd_header(buffer)
        blocks = walk_header_blocks(buffer, _byte_order(buffer))
        array = _radiance(_counts(buffer, header, blocks), header)
        first_line = header.first_line
    else:
        array, header = _stitch(paths)
        first_line = header.first_line
    if requested is not None and requested != header.band:
        raise ReaderError(
            f"read_ahi: band {requested} was requested but the granule carries "
            f"{header.band}"
        )

    dataset = _dataset(array, header, first_line)
    return _calibrate_dataset(
        dataset,
        header.band,
        sensor=sensor,
        calibrate=calibrate,
        sun_earth_distance=sun_earth_distance,
        cos_sza=cos_sza,
        coeffs=coeffs,
        header=header,
    )


def _calibrate_dataset(
    dataset: Any,
    band: str,
    *,
    sensor: str,
    calibrate: bool,
    sun_earth_distance: float,
    cos_sza: Any,
    coeffs: dict[str, Any] | None,
    header: HsdHeader | None,
) -> Any:
    """Calibrate a radiance `Dataset` and re-register it on the same grid.

    Args:
        dataset: The radiance `Dataset` to calibrate.
        band: The registry band key (e.g. ``"B13"``).
        sensor: Registry sensor name.
        calibrate: Whether to convert to a physical quantity at all.
        sun_earth_distance: Sun-earth distance (AU) for reflectance.
        cos_sza: Cosine of the solar zenith angle, or `None`.
        coeffs: Caller coefficients that take precedence over the granule's.
        header: The granule header supplying the granule's own coefficients, or
            `None` when the source was already a `Dataset`.

    Returns:
        A pyramids `Dataset` of the calibrated (or raw) band on `dataset`'s grid.
    """
    radiance = np.asarray(dataset.read_array(), dtype=float)
    if not calibrate:
        data = radiance
    elif header is not None and header.kind == "thermal":
        # The granule's quadratic Planck correction, which the registry's linear
        # alpha / beta cannot express. An explicit `coeffs` still wins, so a
        # caller can route a thermal band through the registry form instead.
        data = (
            calibrate_channel(
                radiance, band, sensor, sun_earth_distance, cos_sza, coeffs=coeffs
            )
            if coeffs
            else _brightness_temperature(radiance, header)
        )
    else:
        granule: dict[str, Any] = {}
        if header is not None and header.albedo_coefficient:
            # HSD gives c' of albedo = c' * radiance, i.e. c' = 1 / S0. The
            # registry works in the standard reflectance convention
            # (pi * L * d^2 / E0), so hand it the granule's own E0 = 1 / c'.
            granule = {
                "kind": "solar",
                "solar_irradiance": 1.0 / header.albedo_coefficient,
            }
        granule.update(coeffs or {})
        data = calibrate_channel(
            radiance,
            band,
            sensor,
            sun_earth_distance,
            cos_sza,
            coeffs=granule or None,
        )

    from pyramids.dataset import Dataset, GeoReference

    epsg = dataset.epsg if dataset.epsg else None
    result = Dataset.from_array(
        data,
        geo_ref=GeoReference(geo=dataset.geotransform, epsg=epsg),
        no_data_value=np.nan,
    )
    if epsg is None:
        # The geostationary grid is not an EPSG code, so carry the CRS across.
        result.crs = dataset.crs
    return result
