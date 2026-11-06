"""Himawari-8/9 AHI Himawari Standard Data (HSD, ``.DAT``) reader.

`read_ahi` decodes one AHI band from a real Himawari Standard Data granule and
returns a calibrated, geolocated pyramids `Dataset` (reflectance for a solar
band, brightness temperature for a thermal one).

HSD is a packed binary of **12 blocks**: 11 header blocks followed by the image
data block. Eight of those header blocks carry the fixed length the
specification states for them; #8, #9 and #10 hold per-line repeats and so vary.
Every header block opens with its own block number (`I1`) and its own block
length, so the header is *self-describing* and is walked rather than assumed — a
fixed offset table would not reach the data. The length field is an `I2` for
every block **except #10**, where the specification makes it an `I4`; reading
that one as an `I2` happens to work on a little-endian granule and reads 0 on a
big-endian one, so the width is per-block rather than assumed. All integer
fields are **unsigned**, and block #1 field 4 declares the file's byte order.

`parse_ahi_hsd` walks the header, reads the grid from block #2, the normalised
geostationary projection from block #3, the per-granule count-to-radiance
calibration from block #5 and the segment placement from block #7, then decodes
the `uint16` counts of block #12, masks the two sentinel count values, and
registers the metre geotransform of the geostationary grid. `read_ahi` is not
layered on it: it repeats that decode itself and calibrates the values *before*
registering them, preferring the granule's own coefficients over the registry,
so a full disk is wrapped as a `Dataset` once rather than built as radiance and
immediately rebuilt.

A full disk is distributed as segment files (`Skkll`) — ten of them in JMA's
operational product, though Table 4 allows up to 99 and nothing here assumes the
count; pass the segment paths together and they are stitched into the full image
by their block #7 first-line numbers. A single segment reads on its own,
georeferenced at its true position in the disk.

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
#: Block numbers whose ``Block length`` field the specification declares as an
#: ``I4`` rather than the usual ``I2``. Only #10 (error information) is one. A
#: little-endian ``I4`` holding a value under 65536 reads identically to an
#: ``I2``, so getting this wrong is invisible until a big-endian granule arrives.
_LONG_LENGTH_BLOCKS = frozenset({10})

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
#: ``Satellite name`` (``C`` x 16), which the specification spells
#: ``Himawari-8`` / ``Himawari-9``. The ``H08`` / ``H09`` form belongs to the
#: file name's ``aaa`` token (Guide Table 4), not to this field.
_B1_SATELLITE = 6
_B1_SATELLITE_BYTES = 16
#: ``Observation area`` (``C`` x 4), the same token as the file name's ``cccc``
#: field — ``FLDK`` for a full disk, ``JP01`` for a Japan-area scan.
_B1_AREA = 38
_B1_AREA_BYTES = 4
#: ``Observation timeline`` (``I2``), as ``hhmm``.
_B1_TIMELINE = 44
#: ``Observation start time`` (``R8``), Modified Julian Date.
_B1_START_MJD = 46
#: ``Total header length`` (``I4``) and ``Total data length`` (``I4``). The
#: granule states both, which makes them an independent check on the header
#: walk: if the walk ends somewhere other than the declared header length, it
#: went wrong, whatever the individual blocks claimed.
_B1_TOTAL_HEADER_LENGTH = 70
_B1_TOTAL_DATA_LENGTH = 74

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
_B5_CENTRAL_WAVELENGTH_UM = 5  # R8, micrometres (see HsdHeader)
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

#: Converts HSD's spectral radiance from its own per-micrometre-of-wavelength
#: basis to the per-wavenumber one the registry's inverse Planck documents.
#:
#: HSD block #5 gives radiance per micrometre of WAVELENGTH,
#: ``W / (m2 sr um)``, while `radiance_to_brightness_temperature` takes
#: ``mW m-2 sr-1 (cm-1)-1`` - per inverse centimetre of WAVENUMBER. Since
#: ``|dlambda / dnu| = lambda**2``, rebasing onto wavenumber multiplies by
#: ``lambda**2``; carrying the units through with ``lambda`` in micrometres,
#: ``W`` to ``mW`` and ``um-1`` to ``cm-1`` leaves
#: ``L * lambda_um**2 * 1e-1``. Verified against the forward Planck function at
#: 3.9-13.3 um and 200-300 K: the round trip lands within 2 mK, the residual
#: being the registry's rounded ``C1`` / ``C2`` rather than the conversion.
#: Skipping it is not a small error - a 280 K scene reads as ~9.96e6 K.
_WAVENUMBER_RADIANCE_SCALE = 1e-1

#: Micrometres per metre, for the one place the Planck inversion needs SI.
_UM_PER_M = 1e6

#: The only two radiometric kinds this reader dispatches on. The spelling has to
#: match exactly, because this reader branches on ``"thermal"`` while the shared
#: `calibrate_channel` branches on ``"solar"`` — so a third spelling would take
#: the solar path here and the thermal path there, silently returning a
#: brightness temperature computed from radiance that was never rebased.
_KINDS = frozenset({"solar", "thermal"})

#: Smallest count-to-radiance slope MAGNITUDE treated as a real gain. HSD leaves
#: an unused calibration field as literal zero bytes, so this separates "the
#: field is unset" from any real gain; real AHI slopes are of order 1e-2 to 1e2
#: in the granule's own units, so a magnitude below this is a misparse rather
#: than a very small gain.
#:
#: The SIGN is deliberately not constrained. An earlier version refused a
#: negative slope on the grounds that counts rise with radiance "by
#: construction" -- but Table 6 states no such thing, it is simply an affine
#: conversion, and no granule was available to check the claim against. Refusing
#: a negative gain would reject every granule of any band JMA chose to ship that
#: way, for the sake of an assumption the specification does not make. A
#: negative slope therefore passes through and produces decreasing radiance,
#: which is what the granule asked for.
_MIN_SLOPE = 1e-12

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


#: Ceiling on the header length a granule may declare before the header-only
#: read gives up and falls back to reading the whole file. The 11 blocks are a
#: few kilobytes unless block #10 lists an extreme number of error lines, so a
#: larger figure means the field is not a length.
_MAX_HEADER_BYTES = 1 << 20


def _read_header_bytes(path: Any) -> bytes:
    """Read a granule's header blocks without its pixel payload.

    Block #1 states the total header length, so the header can be read in two
    small reads instead of pulling a whole granule into memory. A full disk's
    segment is tens of megabytes and a stitch needs every segment's header
    before it can size the output, so reading headers this way keeps that first
    pass nearly free.

    Args:
        path: Path to a ``.DAT`` or ``.DAT.bz2`` granule.

    Returns:
        The header bytes — enough for :func:`walk_header_blocks` and
        :func:`parse_hsd_header`, with no pixels — so the block table
        :func:`walk_header_blocks` builds from it reports block #12 with a length
        of 0, and :func:`_counts` cannot be run against it. Falls back to the
        whole file when the granule does not declare a usable header length.

    Raises:
        ReaderError: When the file cannot be opened, or is too short to hold
            block #1.
    """
    name = str(path)
    try:
        opener = bz2.open if name.lower().endswith(".bz2") else open
        with opener(name, "rb") as handle:  # type: ignore[operator]
            head = handle.read(_FIXED_BLOCK_LENGTHS[_BASIC])
            order = _byte_order(head)
            declared = struct.unpack_from(f"{order}I", head, _B1_TOTAL_HEADER_LENGTH)[0]
            if not 0 < declared <= _MAX_HEADER_BYTES:
                # No usable declaration: fall back rather than guess a size.
                return head + handle.read()
            if declared <= len(head):
                return head
            return head + handle.read(declared - len(head))
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


def _block_length(buffer: bytes, order: ByteOrder, expected: int, offset: int) -> int:
    """Read and validate one header block's declared length.

    Every check here concerns a single block's self-description, so they live
    together rather than inside the walk: the walk's own job is just to follow
    the lengths.

    Args:
        buffer: The granule bytes.
        order: The `struct` byte-order prefix.
        expected: The block number this position must declare.
        offset: Byte offset of the block.

    Returns:
        The block's length in bytes.

    Raises:
        ReaderError: When the block is truncated, declares a number other than
            its position, reports a zero length, contradicts a fixed length the
            specification gives it, or runs past the end of the granule.
    """
    # 1 byte of block number plus the length field itself, which is wider
    # for the one block the specification gives an I4.
    preamble = 5 if expected in _LONG_LENGTH_BLOCKS else 3
    if offset + preamble > len(buffer):
        raise ReaderError(
            f"read_ahi: granule is truncated inside header block #{expected}"
        )
    number = buffer[offset]
    width = "I" if expected in _LONG_LENGTH_BLOCKS else "H"
    length = _unpack(buffer, order, width, offset + 1)
    if number != expected:
        raise ReaderError(
            f"read_ahi: header block at offset {offset} declares number "
            f"{number}, expected #{expected} — unexpected HSD layout"
        )
    if length == 0:
        raise ReaderError(f"read_ahi: header block #{expected} declares a zero length")
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
    return int(length)


def walk_header_blocks(buffer: bytes, order: ByteOrder) -> dict[int, tuple[int, int]]:
    """Walk the 11 self-describing HSD header blocks.

    Each block opens with its block number (``I1``) and its length, so the header
    is traversed by following those lengths. This is what makes the
    variable-length blocks (#8 navigation correction, #9 observation time, #10
    error information — each carrying per-line repeats) safe to step over: their
    sizes are not knowable in advance, so a fixed offset table would not find
    the data block.

    The length field is an ``I2`` for every block **except #10**, whose length
    the specification declares an ``I4`` — see :func:`_block_length`, which reads
    the per-block width. On a little-endian granule the two spellings coincide
    for any value under 65536; on a big-endian one an ``I2`` read of block #10's
    length yields 0, which is why the width is not assumed.

    Args:
        buffer: The granule bytes.
        order: The `struct` byte-order prefix from :func:`_byte_order`.

    Returns:
        A mapping of block number to its ``(offset, length)`` in bytes. The data
        block (#12) is included, with the remaining byte count as its length —
        which is 0 for a header-only buffer from :func:`_read_header_bytes`.

    Raises:
        ReaderError: When a block is truncated, reports a zero length, declares a
            number other than its position, or a fixed-size block's length
            disagrees with the specification.
    """
    blocks: dict[int, tuple[int, int]] = {}
    offset = 0
    for expected in range(1, _HEADER_BLOCK_COUNT + 1):
        blocks[expected] = (offset, _block_length(buffer, order, expected, offset))
        offset += blocks[expected][1]
    declared = _unpack(buffer, order, "I", _B1_TOTAL_HEADER_LENGTH)
    if declared and declared != offset:
        # Block #1 states the whole header's length, so this catches a walk that
        # drifted even though every individual block looked self-consistent.
        raise ReaderError(
            f"read_ahi: the header walk ended at byte {offset} but block #1 "
            f"declares a total header length of {declared} — unexpected HSD layout"
        )
    blocks[_DATA_BLOCK] = (offset, len(buffer) - offset)
    return blocks


@dataclass(frozen=True)
class HsdHeader:
    """The HSD header fields a read needs, decoded from one granule.

    Attributes:
        satellite: Satellite name from block #1, as the granule spells it --
            ``"Himawari-8"`` or ``"Himawari-9"``. This is a different vocabulary
            from the file name's ``H08`` / ``H09`` token, which
            :func:`band_from_filename` parses instead.
        band_number: AHI band number, 1-16 (block #5).
        columns: Pixels per line in this granule / segment (block #2).
        lines: Lines in this granule / segment (block #2).
        bits_per_pixel: Stored bit depth, always 16 for HSD (block #2).
        valid_bits: Valid bits per pixel for the band — 11, 12 or 14 depending on
            the band (block #5). Decoded and exposed, but deliberately **not**
            applied: the reader does not clip or mask counts that exceed
            ``2**valid_bits``. The two sentinel counts already identify every
            pixel the granule declares invalid, so a count above the valid range
            is a product defect rather than a flag, and silently discarding it
            would hide that defect instead of surfacing it. A caller that wants
            to treat the range as hard can read this field and mask.
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
        central_wavelength_um: Band central wavelength in **micrometres**
            (block #5 field 4). HSD is micrometre-based throughout: the field
            carries 10.4 for band 13, not 1.04e-05, and the slope / intercept
            of the same block give radiance per micrometre. Every text
            extraction of the specification PDF loses its micro sign -- the
            whole document decodes to ASCII but for one multiplication sign --
            so the unit renders as ``[m]`` there; Table 1 listing the bands as
            0.47 to 13.3 is what settles it. The name carries the unit so the
            reading cannot drift again.
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
        observation_area: Observation-area token from block #1 (``"FLDK"`` for a
            full disk), the same vocabulary as the file name's ``cccc`` field.
        observation_timeline: Observation timeline as ``hhmm`` (block #1).
        observation_start_mjd: Observation start time in Modified Julian Date
            (block #1), which is what identifies segments as one acquisition.
        total_header_length: Header length the granule declares (block #1), used
            to check the header walk rather than trusted blindly.
        total_data_length: Pixel-payload length the granule declares (block #1).

    Examples:
        - A decoded band-13 full-disk header reports its band, its radiometric
          kind, and the satellite height the projection is built from:
            ```python
            >>> from pyramids_eo.sensors.readers.ahi import HsdHeader
            >>> header = HsdHeader(
            ...     satellite="Himawari-8", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_um=10.4, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ...     observation_area="FLDK", observation_timeline=230,
            ...     observation_start_mjd=57127.1041666667,
            ...     total_header_length=1598, total_data_length=60500000,
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
    central_wavelength_um: float
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
    observation_area: str
    observation_timeline: int
    observation_start_mjd: float
    total_header_length: int
    total_data_length: int

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

        Examples:
            - The split falls between bands 6 and 7, which is where the
              specification switches block #5 from its visible / near-infrared
              variant to its infrared one:
                ```python
                >>> import dataclasses
                >>> from pyramids_eo.sensors.readers.ahi import HsdHeader
                >>> header = HsdHeader(
                ...     satellite="Himawari-8", band_number=13, columns=5500, lines=5500,
                ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
                ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
                ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
                ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
                ...     central_wavelength_um=10.4, albedo_coefficient=None,
                ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
                ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
                ...     boltzmann_constant=1.3806488e-23,
                ...     total_segments=10, segment_number=1, first_line=1,
                ...     observation_area="FLDK", observation_timeline=230,
                ...     observation_start_mjd=57127.1041666667,
                ...     total_header_length=1598, total_data_length=60500000,
                ... )
                >>> [dataclasses.replace(header, band_number=n).kind for n in (1, 6, 7, 16)]
                ['solar', 'solar', 'thermal', 'thermal']

                ```
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
    return _parse_header(buffer, order, walk_header_blocks(buffer, order))


def _parse_header(
    buffer: bytes, order: ByteOrder, blocks: dict[int, tuple[int, int]]
) -> HsdHeader:
    """Decode the header from an already-walked granule.

    Split out so a caller that has already determined the byte order and walked
    the blocks does not pay for both again: the single-granule read used to
    derive the order three times and walk the header twice.

    Args:
        buffer: The granule bytes.
        order: The byte-order prefix from :func:`_byte_order`.
        blocks: The block table from :func:`walk_header_blocks`.

    Returns:
        The decoded :class:`HsdHeader`.

    Raises:
        ReaderError: As :func:`parse_hsd_header`.
    """

    basic = blocks[_BASIC][0]
    raw_name = buffer[
        basic + _B1_SATELLITE : basic + _B1_SATELLITE + _B1_SATELLITE_BYTES
    ]
    # A fixed-width C field is NUL-padded, so the name ends at the first NUL;
    # stripping only the ends would keep an interior one in the decoded string.
    satellite = raw_name.split(b"\x00", 1)[0].decode("ascii", "replace").strip()

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
        # The thermal branch validates every constant it decodes; this one was
        # taken on trust, so a NaN here produced an all-NaN reflectance image
        # with no error at all. A granule that has no albedo coefficient leaves
        # the field zero, which is legitimate and handled at calibration time;
        # a non-finite value means the block is not what we think it is.
        if not math.isfinite(albedo_coefficient):
            raise ReaderError(
                f"read_ahi: solar band {band_name(band_number)} declares a "
                f"non-finite albedo coefficient ({albedo_coefficient!r}) "
                "— unexpected HSD layout"
            )
        # Fields 12 / 13 restate the slope / intercept after an in-flight
        # recalibration. They are the values in force when present, but a
        # granule that has never been recalibrated leaves them zero, so fall
        # back to fields 8 / 9 rather than reading a zero gain.
        updated_slope = _unpack(buffer, order, "d", calibration + _B5_VIS_SLOPE)
        updated_intercept = _unpack(buffer, order, "d", calibration + _B5_VIS_INTERCEPT)
        if math.isfinite(updated_slope) and abs(updated_slope) >= _MIN_SLOPE:
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
    if not math.isfinite(slope) or abs(slope) < _MIN_SLOPE:
        raise ReaderError(
            f"read_ahi: granule declares an unusable count-to-radiance slope "
            f"({slope!r}); it must be finite with a magnitude of at least "
            f"{_MIN_SLOPE}, below which the field is unset rather than a very "
            "small gain — unexpected HSD layout"
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
        central_wavelength_um=float(
            _unpack(buffer, order, "d", calibration + _B5_CENTRAL_WAVELENGTH_UM)
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
        observation_area=buffer[basic + _B1_AREA : basic + _B1_AREA + _B1_AREA_BYTES]
        .decode("ascii", "replace")
        .strip()
        .strip("\x00")
        .strip(),
        observation_timeline=int(_unpack(buffer, order, "H", basic + _B1_TIMELINE)),
        observation_start_mjd=float(_unpack(buffer, order, "d", basic + _B1_START_MJD)),
        total_header_length=int(
            _unpack(buffer, order, "I", basic + _B1_TOTAL_HEADER_LENGTH)
        ),
        total_data_length=int(
            _unpack(buffer, order, "I", basic + _B1_TOTAL_DATA_LENGTH)
        ),
    )


def pixel_size_m(header: HsdHeader) -> tuple[float, float]:
    """Return the ``(column, line)`` ground sampling distance in metres.

    The normalised geostationary projection of LRIT/HRIT 4.4 relates a column to
    the satellite scanning angle by ``c = COFF + nint(x * 2**-16 * CFAC)``. HSD
    expresses that angle in **degrees** (unlike MSG, which uses radians), so one
    column step spans ``2**16 / CFAC`` degrees, and the ground distance is that
    angle times the satellite height. The published AHI ``CFAC`` values reproduce
    2000 / 1000 / 500 m to five significant figures — not exactly, since ``CFAC``
    is an integer and cannot encode the step precisely; the residual is about
    36 micrometres at 2 km (see the doctest below). A degrees/radians mix-up, by
    contrast, would be out by a factor of 57.

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
            ...     satellite="Himawari-8", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_um=10.4, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ...     observation_area="FLDK", observation_timeline=230,
            ...     observation_start_mjd=57127.1041666667,
            ...     total_header_length=1598, total_data_length=60500000,
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
            ...     satellite="Himawari-8", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_um=10.4, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ...     observation_area="FLDK", observation_timeline=230,
            ...     observation_start_mjd=57127.1041666667,
            ...     total_header_length=1598, total_data_length=60500000,
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
    and the projection's ``y`` runs **southward** (LRIT/HRIT 4.4), so an
    increasing line number moves south: HSD is stored north-up and needs no row
    flip. ``LFAC`` is an *unsigned* ``I4`` in HSD (Guide Table 6, block #3 field
    5), so that holds for every granule rather than being a case to branch on.
    The sibling SEVIRI reader does flip, for a different reason entirely: MSG's
    native 15-data line records are stored south-to-north with their columns
    running east-to-west, so `read_seviri` reverses both axes of the array it
    read. Nothing about HSD's own line order is in question here. The transform
    places the north-west corner of `first_line`.

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
            ...     satellite="Himawari-8", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_um=10.4, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ...     observation_area="FLDK", observation_timeline=230,
            ...     observation_start_mjd=57127.1041666667,
            ...     total_header_length=1598, total_data_length=60500000,
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
            ...     satellite="Himawari-8", band_number=13, columns=5500, lines=5500,
            ...     bits_per_pixel=16, valid_bits=14, compression=0, sub_lon=140.7,
            ...     cfac=20466275, lfac=20466275, coff=2750.5, loff=2750.5,
            ...     rs_km=42164.0, req_km=6378.1370, rpol_km=6356.7523,
            ...     slope=0.5, intercept=-10.0, error_count=65535, outside_count=65534,
            ...     central_wavelength_um=10.4, albedo_coefficient=None,
            ...     planck_c0=0.0, planck_c1=1.0, planck_c2=0.0,
            ...     light_speed=299792458.0, planck_constant=6.62606957e-34,
            ...     boltzmann_constant=1.3806488e-23,
            ...     total_segments=10, segment_number=1, first_line=1,
            ...     observation_area="FLDK", observation_timeline=230,
            ...     observation_start_mjd=57127.1041666667,
            ...     total_header_length=1598, total_data_length=60500000,
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
    buffer: bytes,
    header: HsdHeader,
    blocks: dict[int, tuple[int, int]],
    order: ByteOrder,
) -> Any:
    """Read the data block's ``uint16`` counts as a ``(lines, columns)`` array.

    Args:
        buffer: The granule bytes.
        header: The decoded granule header.
        blocks: The block table from :func:`walk_header_blocks`.
        order: The byte-order prefix from :func:`_byte_order`.

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
    dtype = np.dtype(np.uint16).newbyteorder(order)
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


def _to_wavenumber_radiance(radiance: Any, wavelength_um: float) -> Any:
    """Rebase HSD radiance from per wavelength to per wavenumber.

    Args:
        radiance: Spectral radiance per micrometre of wavelength,
            ``W m-2 sr-1 um-1``, as HSD block #5's slope / intercept produce it.
        wavelength_um: The band's central wavelength in micrometres.

    Returns:
        The same radiance expressed as ``mW m-2 sr-1 (cm-1)-1``, which is what
        :func:`~pyramids_eo.sensors.registry.calibration.radiance_to_brightness_temperature`
        documents (see :data:`_WAVENUMBER_RADIANCE_SCALE` for the derivation).

    Raises:
        ReaderError: When `wavelength_um` is not a positive, finite length, since
            the conversion scales by its square.
    """
    if not math.isfinite(wavelength_um) or wavelength_um <= 0.0:
        raise ReaderError(
            f"read_ahi: cannot convert radiance units with a non-physical central "
            f"wavelength ({wavelength_um!r} um)"
        )
    return (
        np.asarray(radiance, dtype=float)
        * wavelength_um**2
        * _WAVENUMBER_RADIANCE_SCALE
    )


def _registry_wavelength_um(band: str, sensor: str) -> float:
    """Return a band's central wavelength from the bundled registry, in micrometres.

    Used when the source is already a `Dataset` and so carries no HSD header to
    take the granule's own exact wavelength from.

    Args:
        band: The registry band key (e.g. ``"B13"``).
        sensor: Registry sensor name.

    Returns:
        The nominal central wavelength in micrometres -- the registry's own unit,
        so no conversion is involved.

    Raises:
        UnknownSensorError: When the sensor / band is not in the registry.
    """
    from pyramids_eo.sensors.registry.sensors import get_sensor

    return float(get_sensor(sensor).get_channel(band).wavelength_um)


def _brightness_temperature(radiance: Any, header: HsdHeader) -> Any:
    """Convert thermal radiance to brightness temperature the HSD way.

    Inverts Planck's law in its per-wavelength form using the constants the
    granule itself was calibrated with, then applies the granule's quadratic
    effective-to-brightness correction ``Tb = c0 + c1 * Te + c2 * Te**2``. The
    registry's linear ``alpha`` / ``beta`` form cannot express that quadratic, so
    a thermal AHI band is calibrated here rather than through
    :func:`~pyramids_eo.sensors.readers._common.calibrate_channel`.

    Args:
        radiance: Spectral radiance in ``W m-2 sr-1 um-1``, as the granule's
            slope and intercept produce it.
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
    wavelength_um = header.central_wavelength_um
    if not math.isfinite(wavelength_um) or wavelength_um <= 0.0:
        raise ReaderError(
            f"read_ahi: thermal band {header.band} declares a non-physical "
            f"central wavelength ({wavelength_um!r} um) — unexpected HSD layout"
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
    # Planck's law is expressed in SI, while HSD gives both the wavelength and
    # the radiance per micrometre, so both are converted here rather than at the
    # call sites: a wavelength of 10.4 read as metres rather than micrometres
    # calibrates a 280 K scene to about 1e19 K.
    wavelength = wavelength_um / _UM_PER_M
    #
    # Evaluated in place. The readable spelling of this allocates about seven
    # image-sized temporaries, which on a 22000x22000 band-3 disk is tens of
    # gigabytes; this holds two plus a boolean mask. Only a positive radiance
    # has a temperature, so the rest are seeded NaN and propagate through.
    scale = 2.0 * planck * light**2 / wavelength**5
    inverse_c2 = wavelength * boltzmann / (planck * light)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        effective = np.where(values > 0.0, values * _UM_PER_M, np.nan)
        np.reciprocal(effective, out=effective)
        effective *= scale
        np.log1p(effective, out=effective)
        effective *= inverse_c2
        np.reciprocal(effective, out=effective)  # now the effective temperature
        # Horner: Tb = c0 + Te * (c1 + c2 * Te), one scratch array.
        brightness = effective * c2
        brightness += c1
        brightness *= effective
        brightness += c0
    return brightness


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
    order = _byte_order(buffer)
    blocks = walk_header_blocks(buffer, order)
    header = _parse_header(buffer, order, blocks)
    radiance = _radiance(_counts(buffer, header, blocks, order), header)
    return _dataset(radiance, header, header.first_line)


def _decode_into(path: Any, header: HsdHeader, out: Any) -> None:
    """Decode one granule's radiance straight into `out`.

    Writing into a caller-owned slice is what keeps a stitched full disk to one
    copy of the image: the alternative holds every segment's array *and* the
    assembled one at the same time.

    Args:
        path: Path to the granule.
        header: That granule's already-decoded header.
        out: A ``(header.lines, header.columns)`` float view to fill.

    Raises:
        ReaderError: When the granule cannot be decoded.
    """
    buffer = _read_bytes(path)
    order = _byte_order(buffer)
    blocks = walk_header_blocks(buffer, order)
    counts = _counts(buffer, header, blocks, order)
    np.multiply(counts, header.slope, out=out, casting="unsafe")
    out += header.intercept
    out[(counts == header.error_count) | (counts == header.outside_count)] = np.nan


def _assert_same_image(first: HsdHeader, header: HsdHeader) -> None:
    """Check that two segment headers describe one image, or raise.

    Lifted out of the stitch as a single predicate: the question is not six
    separate ones but whether these two granules are segments of the same
    acquisition of the same band on the same grid.

    Args:
        first: The header of the northernmost segment, used as the reference.
        header: The header being compared against it.

    Raises:
        ReaderError: When the two disagree on band, grid width, projection,
            satellite, observation (time or area), or how many segments the
            image has.
    """
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
    # Contiguous line numbers do NOT establish that segments belong
    # together: every timeline numbers its segments the same way, so two
    # acquisitions ten minutes apart stitch into a seamless-looking image
    # made of two different scenes. These are the fields that actually
    # identify one acquisition.
    if header.satellite != first.satellite:
        raise ReaderError(
            f"read_ahi: segments come from different satellites "
            f"({first.satellite!r} and {header.satellite!r})"
        )
    if (header.observation_start_mjd, header.observation_area) != (
        first.observation_start_mjd,
        first.observation_area,
    ):
        raise ReaderError(
            f"read_ahi: segments come from different observations "
            f"({first.observation_area!r} at MJD "
            f"{first.observation_start_mjd!r} and "
            f"{header.observation_area!r} at MJD "
            f"{header.observation_start_mjd!r}); stitch one acquisition at a time"
        )
    if header.total_segments != first.total_segments:
        raise ReaderError(
            f"read_ahi: segments disagree on how many segments the image has "
            f"({first.total_segments} and {header.total_segments})"
        )
    if (header.sub_lon, header.req_km, header.rpol_km) != (
        first.sub_lon,
        first.req_km,
        first.rpol_km,
    ):
        raise ReaderError(
            "read_ahi: segments disagree on the projection (sub-satellite "
            "longitude or ellipsoid); they are not segments of one image"
        )


def _stitch(paths: Any) -> tuple[Any, HsdHeader]:
    """Assemble the segments of one band into a single radiance array.

    A **contiguous subset** is accepted and returns only the lines it covers,
    georeferenced where those lines belong in the disk — reading one segment is
    already supported, so a run of several is not a different kind of request.
    What is rejected is a set with a gap or a duplicate, and a set whose members
    are not from one acquisition.

    Args:
        paths: An iterable of HSD segment paths of the **same** band and
            timestamp.

    Returns:
        A ``(radiance, header)`` pair: the stitched array covering every line the
        segments carry, and the header of the first (northernmost) segment.

    Raises:
        ReaderError: When no path is given; the segments disagree on band, grid
            width, projection, satellite, observation (time or area), or how many
            segments the image has; a segment repeats; or the segments leave a
            gap, which would otherwise yield a silently part-blank disk.
    """
    # Pass one reads only the headers, so the output can be sized and the set
    # validated before any pixels are held.
    decoded: list[tuple[HsdHeader, Any]] = [
        (parse_hsd_header(_read_header_bytes(path)), path) for path in paths
    ]
    if not decoded:
        raise ReaderError("read_ahi: no HSD segment paths were given")
    decoded.sort(key=lambda item: item[0].first_line)

    first = decoded[0][0]
    for header, _ in decoded[1:]:
        _assert_same_image(first, header)

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
    # Pass two decodes each segment directly into its rows, so only one copy of
    # the image exists plus the segment currently being read.
    row = 0
    for header, path in decoded:
        _decode_into(path, header, stitched[row : row + header.lines])
        row += header.lines
    return stitched, first


def _resolve_paths(source: Any) -> list[Any]:
    """Normalise a non-`Dataset` `source` to a non-empty list of granule paths.

    Args:
        source: A single granule path, or an iterable of segment paths.

    Returns:
        The paths as a list.

    Raises:
        ReaderError: When `source` is neither a path nor an iterable of them, or
            is an empty iterable.
    """
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        return [source]
    try:
        paths = list(source)
    except TypeError as exc:
        # A bare non-path, non-iterable would otherwise surface as a TypeError
        # from list(), which is not the documented failure.
        raise ReaderError(
            f"read_ahi: source must be a granule path, an iterable of segment "
            f"paths, or a Dataset, not {type(source).__name__}"
        ) from exc
    if not paths:
        raise ReaderError("read_ahi: no HSD segment paths were given")
    return paths


def _assert_requested_band(path: Any, requested: str | None) -> None:
    """Check a caller's expected band against the granule's own, or raise.

    Settled from a header-only read, so a mismatch costs one small read rather
    than a whole decoded disk.

    Args:
        path: Path to the first granule of the set.
        requested: The canonical band the caller asked for, or `None` to accept
            whatever the granule carries.

    Raises:
        ReaderError: When the granule carries a different band.
    """
    if requested is None:
        return
    declared = parse_hsd_header(_read_header_bytes(path)).band
    if requested != declared:
        raise ReaderError(
            f"read_ahi: band {requested} was requested but the granule "
            f"carries {declared}"
        )


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
    the disk. A contiguous subset of the segments is also accepted and returns
    just those lines, so a partial download reads as the band of the disk it
    actually covers rather than being padded out or refused. The segments must
    come from one acquisition — same band, satellite, observation time and area
    — since line numbers alone repeat across timelines.

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
        A pyramids `Dataset` of the calibrated (or raw) band. Read from a granule
        it carries the granule's own geostationary grid — the geostationary CRS,
        a metre geotransform and NaN nodata. Read from a `Dataset` source it
        keeps **that** dataset's geotransform, CRS and EPSG instead, since no HSD
        header is available to rebuild a grid from; only the values change.

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
        return _wrap_like(
            source,
            _calibrate_array(
                source.read_array(),
                requested,
                sensor=sensor,
                calibrate=calibrate,
                sun_earth_distance=sun_earth_distance,
                cos_sza=cos_sza,
                coeffs=coeffs,
                header=None,
            ),
        )

    paths = _resolve_paths(source)
    _assert_requested_band(paths[0], requested)

    if len(paths) == 1:
        buffer = _read_bytes(paths[0])
        order = _byte_order(buffer)
        blocks = walk_header_blocks(buffer, order)
        header = _parse_header(buffer, order, blocks)
        array = _radiance(_counts(buffer, header, blocks, order), header)
        first_line = header.first_line
    else:
        array, header = _stitch(paths)
        first_line = header.first_line

    # Calibrate the values before registering them, so a full disk is wrapped
    # once rather than built as radiance and immediately rebuilt as brightness
    # temperature.
    return _dataset(
        _calibrate_array(
            array,
            header.band,
            sensor=sensor,
            calibrate=calibrate,
            sun_earth_distance=sun_earth_distance,
            cos_sza=cos_sza,
            coeffs=coeffs,
            header=header,
        ),
        header,
        first_line,
    )


def _effective_kind(
    band: str, sensor: str, header: HsdHeader | None, overrides: dict[str, Any]
) -> str:
    """Return the radiometric kind actually in force for this calibration.

    The caller's override wins, then the granule, then the registry — which is
    the only source when `source` was already a `Dataset`. A caller declaring
    ``kind`` takes that path even when it disagrees with the band's usual one.

    Args:
        band: The registry band key.
        sensor: Registry sensor name.
        header: The granule header, or `None`.
        overrides: The caller's coefficient overrides.

    Returns:
        ``"solar"`` or ``"thermal"``, validated — never any other string.

    Raises:
        ReaderError: When a caller's ``coeffs["kind"]`` is not exactly
            ``"solar"`` or ``"thermal"``. It is rejected rather than coerced or
            passed through, because this reader branches on ``"thermal"`` while
            the shared `calibrate_channel` branches on ``"solar"``: a third
            spelling satisfies neither as intended and used to take the solar
            path here and the thermal path there, returning a brightness
            temperature from radiance that had not been rebased.
        UnknownSensorError: When the registry has to be consulted and does not
            know the sensor / band.
    """
    if "kind" in overrides:
        kind = overrides["kind"]
        if kind not in _KINDS:
            raise ReaderError(
                f"read_ahi: coeffs['kind'] must be one of "
                f"{sorted(_KINDS)}, not {kind!r}; the spelling has to match "
                "exactly, since a third value would be calibrated as solar here "
                "and as thermal by the registry"
            )
        return str(kind)
    if header is not None:
        return header.kind
    from pyramids_eo.sensors.registry.sensors import get_sensor

    return str(get_sensor(sensor).get_channel(band).kind)


def _calibrate_thermal(
    radiance: Any,
    band: str,
    *,
    sensor: str,
    sun_earth_distance: float,
    cos_sza: Any,
    overrides: dict[str, Any],
    header: HsdHeader | None,
) -> Any:
    """Convert thermal radiance to brightness temperature.

    Prefers the granule's own quadratic Planck model, which the registry's
    linear ``alpha`` / ``beta`` cannot express. When the caller overrides the
    coefficients, or there is no granule, the registry form is used instead —
    and the radiance is rebased from HSD's per-wavelength basis onto the
    per-wavenumber one that form documents, without which a 280 K scene
    calibrates to roughly 921,613 K.

    Args:
        radiance: Spectral radiance per wavelength, ``W m-2 sr-1 m-1``.
        band: The registry band key.
        sensor: Registry sensor name.
        sun_earth_distance: Passed through to the registry.
        cos_sza: Passed through to the registry.
        overrides: The caller's coefficient overrides; any entry routes this
            through the registry rather than the granule's model.
        header: The granule header, or `None`.

    Returns:
        Brightness temperature in kelvin.

    Raises:
        ReaderError: When the granule's model is used but a constant it needs is
            missing, or the wavelength is non-physical.
        CalibrationError: When the registry form lacks a constant it needs.
    """
    if header is not None and not overrides:
        return _brightness_temperature(radiance, header)
    wavelength = (
        header.central_wavelength_um
        if header is not None
        else _registry_wavelength_um(band, sensor)
    )
    return calibrate_channel(
        _to_wavenumber_radiance(radiance, wavelength),
        band,
        sensor,
        sun_earth_distance,
        cos_sza,
        coeffs=overrides or None,
    )


def _calibrate_array(
    radiance: Any,
    band: str,
    *,
    sensor: str,
    calibrate: bool,
    sun_earth_distance: float,
    cos_sza: Any,
    coeffs: dict[str, Any] | None,
    header: HsdHeader | None,
) -> Any:
    """Calibrate a radiance array to its physical quantity.

    Deliberately array-in, array-out. Wrapping the radiance in a `Dataset` only
    to read it straight back out doubled the peak footprint of a full disk for
    no benefit, so the caller wraps the finished values once.

    Args:
        radiance: The radiance array to calibrate.
        band: The registry band key (e.g. ``"B13"``).
        sensor: Registry sensor name.
        calibrate: Whether to convert to a physical quantity at all.
        sun_earth_distance: Sun-earth distance (AU) for reflectance.
        cos_sza: Cosine of the solar zenith angle, or `None`.
        coeffs: Caller coefficients that take precedence over the granule's.
        header: The granule header supplying the granule's own coefficients, or
            `None` when the source was already a `Dataset`.

    Returns:
        The calibrated (or, with `calibrate` false, the unchanged) values.
    """
    radiance = np.asarray(radiance, dtype=float)
    overrides: dict[str, Any] = dict(coeffs or {})
    kind = _effective_kind(band, sensor, header, overrides)

    if not calibrate:
        data = radiance
    elif kind == "thermal":
        data = _calibrate_thermal(
            radiance,
            band,
            sensor=sensor,
            sun_earth_distance=sun_earth_distance,
            cos_sza=cos_sza,
            overrides=overrides,
            header=header,
        )
    else:
        granule: dict[str, Any] = {}
        if (
            header is not None
            and header.albedo_coefficient is not None
            and header.albedo_coefficient > 0.0
        ):
            # HSD gives c' of albedo = c' * radiance, i.e. c' = 1 / S0. The
            # registry works in the standard reflectance convention
            # (pi * L * d^2 / E0), so hand it the granule's own E0 = 1 / c'.
            # Both are per wavelength, so reflectance needs no rebasing.
            #
            # Two consequences worth being explicit about:
            #
            # 1. What comes back is pi times HSD's own "albedo" (the guide
            #    defines A = I / S0, with no pi). That is deliberate: it is the
            #    standard bidirectional reflectance factor this package returns
            #    for every other solar band, so AHI stays comparable with FCI,
            #    SEVIRI and the Sentinel readers rather than matching JMA's
            #    quantity. Divide by pi to recover HSD's albedo.
            # 2. The d^2 term is applied on top of S0 and does NOT double-count.
            #    The guide gives no sun-earth distance anywhere, and c' is a
            #    static per-band constant -- block #5 field 11's update time
            #    covers fields 12/13 (the recalibrated slope and intercept)
            #    only, never c'. A static constant cannot carry a distance that
            #    changes daily, so S0 is at the conventional 1 AU reference and
            #    the d^2 correction belongs on top of it.
            granule = {
                "kind": "solar",
                "solar_irradiance": 1.0 / header.albedo_coefficient,
            }
        granule.update(overrides)
        data = calibrate_channel(
            radiance,
            band,
            sensor,
            sun_earth_distance,
            cos_sza,
            coeffs=granule or None,
        )

    return data


def _wrap_like(dataset: Any, data: Any) -> Any:
    """Register `data` on the same grid and CRS as `dataset`.

    Args:
        dataset: The `Dataset` whose geotransform and CRS to reuse.
        data: The values to register.

    Returns:
        A pyramids `Dataset` carrying `data` on `dataset`'s grid, with NaN nodata.
    """
    from pyramids.dataset import Dataset, GeoReference

    epsg = dataset.epsg if dataset.epsg else None
    result = Dataset.from_array(
        data,
        geo_ref=GeoReference(geo=dataset.geotransform, epsg=epsg),
        no_data_value=np.nan,
    )
    # Carried across unconditionally. The geostationary CRS has no EPSG code
    # today, so the `epsg is None` branch happened to cover it -- but relying on
    # that means the day pyramids resolves this WKT to a code, `+h=` and
    # `+sweep=y` would be dropped silently and every pixel would land in the
    # wrong place. Setting it either way costs nothing and cannot regress.
    if dataset.crs:
        result.crs = dataset.crs
    return result
