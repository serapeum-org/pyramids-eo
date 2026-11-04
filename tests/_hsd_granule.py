"""Build synthetic Himawari Standard Data granules for the AHI reader tests.

The HSD layout is reproduced here **field by field in the specification's own
declaration order** (Himawari Standard Data User's Guide v1.3, Table 6), with
each block padded to its declared length. Nothing imports the reader's offset
constants: the writer encodes the layout as a *sequence*, the reader seeks by
*offset*, so a wrong offset in either makes the two disagree instead of
cancelling out. The block lengths are asserted on write, which is the same
checksum the specification's own "= N [bytes]" rows give.

Blocks #8, #9 and #10 are written at deliberately different, non-default lengths
so the tests exercise the self-describing header walk: a reader using a fixed
offset table would not find the data block in these fixtures.

The published Himawari full-disk navigation values are used by default (``CFAC``
= ``LFAC`` = 20466275 with ``COFF`` = ``LOFF`` = 2750.5 for the 2 km grid), so a
decoded fixture is checked against the real 2000 m / 1000 m / 500 m grids rather
than against numbers invented here.
"""

from __future__ import annotations

import bz2
import struct
from pathlib import Path
from typing import Any

import numpy as np

#: Declared lengths of the fixed-size header blocks (Guide Table 6).
FIXED_BLOCK_LENGTHS = {1: 282, 2: 50, 3: 127, 4: 139, 5: 147, 6: 259, 7: 47, 11: 259}

#: Lengths used for the three variable-length blocks. Chosen to differ from one
#: another and from every fixed length, so the header walk is genuinely tested.
VARIABLE_BLOCK_LENGTHS = {8: 61, 9: 93, 10: 134}

#: Block #1 field 5 carries a 16-byte ASCII satellite name, which the
#: specification gives as ``Himawari-8`` / ``Himawari-9``. That is NOT the
#: ``aaa`` token of the file name (Table 4), which is ``H08`` / ``H09``; the two
#: vocabularies are different and a fixture that uses one for both hides the
#: difference. This maps the header name onto the file-name token.
SATELLITE_FILE_TOKENS = {"Himawari-8": "H08", "Himawari-9": "H09"}

#: Published Himawari full-disk navigation: the column/line scaling factor for
#: each native grid, with the sub-satellite point at the grid centre.
CFAC_BY_RESOLUTION_M = {2000: 20466275, 1000: 40932550, 500: 81865100}

#: Sentinel count values (Guide block #5 fields 6 and 7).
ERROR_COUNT = 65535
OUTSIDE_COUNT = 65534

#: Physical constants as a granule carries them (block #5 fields 16-18).
LIGHT_SPEED = 299792458.0
PLANCK_CONSTANT = 6.62606957e-34
BOLTZMANN_CONSTANT = 1.3806488e-23


class _BlockWriter:
    """Accumulate a block's fields in declaration order, then pad to length."""

    def __init__(
        self, number: int, length: int, order: str, *, long_length: bool = False
    ) -> None:
        """Start a block.

        Args:
            number: The block number, written as the block's first field.
            length: The block's declared length in bytes, written as the second.
            order: A `struct` byte-order prefix, ``"<"`` or ``">"``.
            long_length: Write the length as ``I4`` instead of ``I2``. Block #10
                (error information) is the one block the specification gives a
                4-byte length; every other block uses 2. Writing it the
                specification's way is what makes a byte-order mistake in a
                reader visible, since a little-endian ``I4`` whose value fits in
                16 bits reads the same as an ``I2`` by coincidence.
        """
        self.length = length
        self.order = order
        self.parts: list[bytes] = []
        self.i1(number)
        if long_length:
            self.i4(length)
        else:
            self.i2(length)

    def i1(self, value: int) -> None:
        """Append an unsigned 1-byte integer."""
        self.parts.append(struct.pack(f"{self.order}B", value))

    def i2(self, value: int) -> None:
        """Append an unsigned 2-byte integer."""
        self.parts.append(struct.pack(f"{self.order}H", value))

    def i4(self, value: int) -> None:
        """Append an unsigned 4-byte integer."""
        self.parts.append(struct.pack(f"{self.order}I", value))

    def r4(self, value: float) -> None:
        """Append a single-precision float."""
        self.parts.append(struct.pack(f"{self.order}f", value))

    def r8(self, value: float) -> None:
        """Append a double-precision float."""
        self.parts.append(struct.pack(f"{self.order}d", value))

    def text(self, value: str, size: int) -> None:
        """Append a fixed-width ASCII field, NUL-padded."""
        self.parts.append(value.encode("ascii").ljust(size, b"\x00")[:size])

    def spare(self, size: int) -> None:
        """Append the block's trailing spare bytes.

        Writing the spare explicitly is what makes :meth:`done`'s checksum mean
        something: with implicit padding, a field written at the wrong width
        just shifted the padding and the block still came out the declared
        length.

        Args:
            size: Number of spare bytes the specification gives the block.
        """
        self.parts.append(b"\x00" * size)

    def done(self) -> bytes:
        """Return the finished block, checking its length exactly.

        Returns:
            The block bytes, exactly `length` long.

        Raises:
            AssertionError: When the fields written do not total the declared
                length. This is the specification's own "= N [bytes]" row used as
                a checksum, and it has to be an equality: the earlier version
                asserted only that the fields did not *overflow* and then
                zero-padded the remainder, so a field written at the wrong width
                — exactly the mistake these fixtures exist to catch — produced a
                block of the right length with everything after it displaced.
        """
        body = b"".join(self.parts)
        assert len(body) == self.length, (
            f"block fields total {len(body)} bytes, but the specification "
            f"declares {self.length}; a field is missing or the wrong width"
        )
        return body


def _basic_block(
    order: str,
    satellite: str,
    *,
    area: str,
    timeline: int,
    start_mjd: float,
    header_length: int,
    data_length: int,
) -> bytes:
    """Build block #1, basic information (282 bytes).

    Writes fields 7-14 as well as 1-6, so a reader can cross-check its header
    walk against the granule's own declared total header and data lengths. A
    fixture that left those zero could not exercise that check.

    Args:
        order: A `struct` byte-order prefix.
        satellite: Satellite name as the spec spells it (``"Himawari-8"``).
        area: Observation area token (field 7, e.g. ``"FLDK"``).
        timeline: Observation timeline as ``hhmm`` (field 9).
        start_mjd: Observation start time in Modified Julian Date (field 10).
        header_length: Total header length in bytes (field 13).
        data_length: Total data length in bytes (field 14).

    Returns:
        The 282-byte block.
    """
    block = _BlockWriter(1, FIXED_BLOCK_LENGTHS[1], order)
    block.i2(11)  # 3 total number of header blocks
    block.i1(0 if order == "<" else 1)  # 4 byte order
    block.text(satellite, 16)  # 5 satellite name
    block.text("MSC", 16)  # 6 processing center name
    block.text(area, 4)  # 7 observation area
    block.text("  ", 2)  # 8 other observation information
    block.i2(timeline)  # 9 observation timeline, hhmm
    block.r8(start_mjd)  # 10 observation start time [MJD]
    block.r8(start_mjd + 600.0 / 86400.0)  # 11 observation end time [MJD]
    block.r8(start_mjd + 0.01)  # 12 file creation time [MJD]
    block.i4(header_length)  # 13 total header length
    block.i4(data_length)  # 14 total data length
    # Fields 15 onward (quality flags, file version, spare) are not decoded by
    # the reader, so they are left zeroed -- but the width is still declared, so
    # the block has to add up.
    block.spare(FIXED_BLOCK_LENGTHS[1] - 78)
    return block.done()


def _data_info_block(order: str, columns: int, lines: int, compression: int) -> bytes:
    """Build block #2, data information (50 bytes)."""
    block = _BlockWriter(2, FIXED_BLOCK_LENGTHS[2], order)
    block.i2(16)  # 3 number of bits per pixel
    block.i2(columns)  # 4 number of columns
    block.i2(lines)  # 5 number of lines
    block.i1(compression)  # 6 compression flag for data block
    block.spare(40)  # 7 spare
    return block.done()


def _projection_block(
    order: str, sub_lon: float, cfac: int, lfac: int, coff: float, loff: float
) -> bytes:
    """Build block #3, projection information (127 bytes)."""
    equatorial, polar = 6378.1370, 6356.7523
    block = _BlockWriter(3, FIXED_BLOCK_LENGTHS[3], order)
    block.r8(sub_lon)  # 3 sub_lon
    block.i4(cfac)  # 4 CFAC
    block.i4(lfac)  # 5 LFAC
    block.r4(coff)  # 6 COFF
    block.r4(loff)  # 7 LOFF
    block.r8(42164.0)  # 8 distance to virtual satellite (Rs)
    block.r8(equatorial)  # 9 equatorial radius (req)
    block.r8(polar)  # 10 polar radius (rpol)
    block.r8(0.00669438444)  # 11 (req^2 - rpol^2) / req^2
    block.r8(0.993305616)  # 12 rpol^2 / req^2
    block.r8(1.006739501)  # 13 req^2 / rpol^2
    block.r8(1737122264.0)  # 14 coefficient for Sd
    block.i2(0)  # 15 resampling types
    block.i2(4)  # 16 resampling size
    block.spare(40)  # 17 spare
    return block.done()


def _calibration_block(
    order: str,
    band: int,
    *,
    slope: float,
    intercept: float,
    wavelength_m: float,
    albedo_coefficient: float,
    planck: tuple[float, float, float],
    updated: tuple[float, float] | None,
    error_count: int,
    outside_count: int,
) -> bytes:
    """Build block #5, calibration information (147 bytes), in the band's variant."""
    block = _BlockWriter(5, FIXED_BLOCK_LENGTHS[5], order)
    block.i2(band)  # 3 band number
    block.r8(wavelength_m)  # 4 central wave length
    # Guide Table 1 gives a per-band valid bit depth (11, 12 or 14), but the
    # column is misaligned in the text extraction available here, so the true
    # per-band mapping is NOT reproduced rather than guessed. These two stand-in
    # values only have to be a legal depth: the reader exposes the field and
    # deliberately does not clip counts to it, so nothing under test depends on
    # which band gets which.
    block.i2(11 if band <= 6 else 14)  # 5 valid number of bits per pixel
    block.i2(error_count)  # 6 count value of error pixels
    block.i2(outside_count)  # 7 count value of pixels outside scan area
    block.r8(slope)  # 8 slope for count-radiance conversion
    block.r8(intercept)  # 9 intercept for count-radiance conversion
    if band <= 6:  # visible / near-infrared variant
        block.r8(albedo_coefficient)  # 10 coefficient c' (radiance -> albedo)
        block.r8(0.0)  # 11 update time of fields 12 and 13 [MJD]
        block.r8(0.0 if updated is None else updated[0])  # 12 calibrated slope
        block.r8(0.0 if updated is None else updated[1])  # 13 calibrated intercept
        block.spare(80)  # 14 spare
    else:  # infrared variant
        block.r8(planck[0])  # 10 c0
        block.r8(planck[1])  # 11 c1
        block.r8(planck[2])  # 12 c2
        block.r8(0.0)  # 13 C0
        block.r8(1.0)  # 14 C1
        block.r8(0.0)  # 15 C2
        block.r8(LIGHT_SPEED)  # 16 speed of light
        block.r8(PLANCK_CONSTANT)  # 17 Planck constant
        block.r8(BOLTZMANN_CONSTANT)  # 18 Boltzmann constant
        block.spare(40)  # 19 spare
    return block.done()


def _segment_block(order: str, total: int, number: int, first_line: int) -> bytes:
    """Build block #7, segment information (47 bytes)."""
    block = _BlockWriter(7, FIXED_BLOCK_LENGTHS[7], order)
    block.i1(total)  # 3 total number of segments
    block.i1(number)  # 4 segment sequence number
    block.i2(first_line)  # 5 first line number of image segment
    block.spare(40)  # 6 spare
    return block.done()


#: Block #10 (error information) is the only block whose length field the
#: specification declares as ``I4``; every other block uses ``I2``.
LONG_LENGTH_BLOCKS = frozenset({10})


def _filler_block(order: str, number: int, length: int) -> bytes:
    """Build a block carrying nothing but its number, length and spare bytes.

    The reader skips these blocks entirely, so only their number and length
    matter -- but the spare is still declared explicitly so the block satisfies
    the same exact-length checksum as every other.
    """
    long_length = number in LONG_LENGTH_BLOCKS
    block = _BlockWriter(number, length, order, long_length=long_length)
    block.spare(length - (5 if long_length else 3))
    return block.done()


def granule_bytes(
    *,
    band: int = 13,
    columns: int = 8,
    lines: int = 4,
    resolution_m: int = 2000,
    satellite: str = "Himawari-8",
    sub_lon: float = 140.7,
    coff: float | None = None,
    loff: float | None = None,
    slope: float = 0.5,
    intercept: float = -10.0,
    wavelength_m: float = 10.4e-6,
    albedo_coefficient: float = 0.0025,
    planck: tuple[float, float, float] = (0.0, 1.0, 0.0),
    updated: tuple[float, float] | None = None,
    counts: Any = None,
    total_segments: int = 1,
    segment_number: int = 1,
    first_line: int = 1,
    compression: int = 0,
    error_count: int = ERROR_COUNT,
    outside_count: int = OUTSIDE_COUNT,
    area: str = "FLDK",
    timeline: int = 230,
    start_mjd: float = 57127.1041666667,
    header_length: int | None = None,
    data_length: int | None = None,
    order: str = "<",
) -> bytes:
    """Build one synthetic HSD granule.

    Args:
        band: AHI band number, 1-16; decides which block #5 variant is written.
        columns: Pixels per line.
        lines: Lines in this granule / segment.
        resolution_m: Native grid whose published ``CFAC`` / ``LFAC`` to write
            (2000, 1000 or 500).
        satellite: Satellite name for block #1, as the specification spells it
            (``"Himawari-8"`` / ``"Himawari-9"``) -- not the file name's ``H08``
            token.
        sub_lon: Sub-satellite longitude, degrees east.
        coff: Column offset; defaults to the grid centre for a full disk of this
            resolution, which is what a real granule carries.
        loff: Line offset; defaults likewise.
        slope: Count-to-radiance slope.
        intercept: Count-to-radiance intercept.
        wavelength_m: Band central wavelength in metres.
        albedo_coefficient: ``c'`` of ``albedo = c' * radiance`` (solar bands).
        planck: The ``(c0, c1, c2)`` brightness-temperature correction (thermal).
        updated: An optional ``(slope, intercept)`` pair for the solar variant's
            recalibrated fields 12 / 13; `None` leaves them zero, as a granule
            that has never been recalibrated does.
        counts: The ``(lines, columns)`` counts to write; defaults to a ramp.
        total_segments: Number of segments the full image is split into.
        segment_number: This granule's segment sequence number.
        first_line: 1-based first line of this segment in the full image.
        compression: Block #2's data-block compression flag.
        error_count: The count value block #5 declares for an error pixel. The
            default is the specification's 65535; override it to prove a reader
            honours the granule's own declaration rather than the constant.
        outside_count: The count value block #5 declares for a pixel outside the
            scan area (default 65534).
        area: Observation area token for block #1 field 7.
        timeline: Observation timeline (``hhmm``) for block #1 field 9.
        start_mjd: Observation start time (MJD) for block #1 field 10. Segments
            of one image share it; change it to build a granule from another
            timeline.
        header_length: Override block #1 field 13. Defaults to the real summed
            header length, so a reader's walk can be checked against it.
        data_length: Override block #1 field 14. Defaults to the real pixel
            payload size.
        order: Byte order to write — ``"<"`` little-endian or ``">"`` big-endian.

    Returns:
        The granule bytes.
    """
    full = {2000: 5500, 1000: 11000, 500: 22000}[resolution_m]
    centre = full / 2.0 + 0.5
    cfac = CFAC_BY_RESOLUTION_M[resolution_m]
    if counts is None:
        counts = np.arange(lines * columns, dtype=np.uint16).reshape(lines, columns)
    counts = np.asarray(counts, dtype=np.uint16)

    # Blocks #2-#11 are built first so block #1 can declare the true total
    # header length, which is how a reader can verify its own walk.
    tail = [
        _data_info_block(order, columns, lines, compression),
        _projection_block(
            order,
            sub_lon,
            cfac,
            cfac,
            centre if coff is None else coff,
            centre if loff is None else loff,
        ),
        _filler_block(order, 4, FIXED_BLOCK_LENGTHS[4]),
        _calibration_block(
            order,
            band,
            slope=slope,
            intercept=intercept,
            wavelength_m=wavelength_m,
            albedo_coefficient=albedo_coefficient,
            planck=planck,
            updated=updated,
            error_count=error_count,
            outside_count=outside_count,
        ),
        _filler_block(order, 6, FIXED_BLOCK_LENGTHS[6]),
        _segment_block(order, total_segments, segment_number, first_line),
        _filler_block(order, 8, VARIABLE_BLOCK_LENGTHS[8]),
        _filler_block(order, 9, VARIABLE_BLOCK_LENGTHS[9]),
        _filler_block(order, 10, VARIABLE_BLOCK_LENGTHS[10]),
        _filler_block(order, 11, FIXED_BLOCK_LENGTHS[11]),
    ]
    payload = counts.astype(np.dtype(np.uint16).newbyteorder(order)).tobytes()
    header = _basic_block(
        order,
        satellite,
        area=area,
        timeline=timeline,
        start_mjd=start_mjd,
        header_length=(
            FIXED_BLOCK_LENGTHS[1] + sum(len(b) for b in tail)
            if header_length is None
            else header_length
        ),
        data_length=len(payload) if data_length is None else data_length,
    )
    return b"".join([header, *tail]) + payload


def hsd_filename(
    *,
    band: int = 13,
    satellite: str = "H08",
    date: str = "20150415",
    time: str = "0230",
    area: str = "FLDK",
    resolution_m: int = 2000,
    segment_number: int = 1,
    total_segments: int = 1,
) -> str:
    """Build a file name following the HSD convention (Guide Table 4).

    Args:
        band: Band number, 1-16.
        satellite: ``"H08"`` or ``"H09"``.
        date: Observation start date, ``yyyymmdd``.
        time: Observation start time, ``hhnn``.
        area: Observation area and number, e.g. ``"FLDK"``.
        resolution_m: Native grid, used for the ``Rjj`` field (20 / 10 / 05).
        segment_number: The ``kk`` segment sequence number.
        total_segments: The ``ll`` total segment count.

    Returns:
        The ``HS_….DAT`` file name.
    """
    resolution = {2000: "20", 1000: "10", 500: "05"}[resolution_m]
    return (
        f"HS_{satellite}_{date}_{time}_B{band:02d}_{area}_R{resolution}"
        f"_S{segment_number:02d}{total_segments:02d}.DAT"
    )


def write_granule(directory: Any, *, compress: bool = False, **kwargs: Any) -> Path:
    """Write one synthetic HSD granule under a conventional file name.

    Args:
        directory: Directory to write into.
        compress: When True, write a bzip2-compressed ``.DAT.bz2`` granule, as
            JMA distributes them.
        **kwargs: Passed to :func:`granule_bytes`; the band, satellite,
            resolution and segment numbers also shape the file name.

    Returns:
        The path written.
    """
    satellite = kwargs.get("satellite", "Himawari-8")
    name = hsd_filename(
        band=kwargs.get("band", 13),
        satellite=SATELLITE_FILE_TOKENS.get(satellite, "H08"),
        resolution_m=kwargs.get("resolution_m", 2000),
        segment_number=kwargs.get("segment_number", 1),
        total_segments=kwargs.get("total_segments", 1),
    )
    payload = granule_bytes(**kwargs)
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (f"{name}.bz2" if compress else name)
    path.write_bytes(bz2.compress(payload) if compress else payload)
    return path


def segment_set(
    directory: Any,
    *,
    segments: int = 3,
    lines_per_segment: int = 4,
    columns: int = 8,
    band: int = 13,
    **kwargs: Any,
) -> list[Path]:
    """Write a contiguous set of segment granules for one band.

    Each segment carries a distinct count block, so a stitch can be checked to
    place every segment at the right rows rather than merely producing an array
    of the right shape.

    Args:
        directory: Directory to write into.
        segments: Number of segments to write.
        lines_per_segment: Lines in each segment.
        columns: Pixels per line.
        band: Band number for every segment.
        **kwargs: Passed through to :func:`granule_bytes`.

    Returns:
        The segment paths, in segment order.
    """
    paths = []
    for index in range(segments):
        counts = np.full((lines_per_segment, columns), index + 1, dtype=np.uint16)
        paths.append(
            write_granule(
                directory,
                band=band,
                columns=columns,
                lines=lines_per_segment,
                counts=counts,
                total_segments=segments,
                segment_number=index + 1,
                first_line=index * lines_per_segment + 1,
                **kwargs,
            )
        )
    return paths
