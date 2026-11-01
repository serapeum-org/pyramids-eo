"""Shared decoding helpers for the Sentinel-3 `.SEN3` NetCDF readers.

OLCI and SLSTR ship the same way: a `.SEN3` *directory* holding one NetCDF file
per measurement variable plus annotation files (geolocation, geometry, quality)
and an `xfdumanifest.xml`. Both are read here through pyramids' bundled GDAL
NetCDF driver -- no `netCDF4` dependency, and nothing from the satpy stack.

Unlike the geostationary SEVIRI / FCI readers, these are **swath** instruments: a
pushbroom sensor sweeps a strip of ground, so the pixels land on a curved,
irregular footprint that no affine geotransform can describe. The product
therefore carries a latitude and a longitude for *every* pixel, in its
geolocation annotation file. The readers express that the way GDAL does -- a
`GEOLOCATION` metadata domain pointing at those arrays (see
`geolocation_domain`) -- so a caller can warp on demand while these readers
resample nothing (resampling is deliberately out of scope).

Consequently a swath `Dataset` from here carries a **placeholder** pixel/line
geotransform, not a geographic one; `attach_geolocation` documents that contract.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from pyramids_eo.errors import ReaderError
from pyramids_eo.sensors.readers._common import finite_float, invalid_mask

#: The identity pixel/line geotransform a swath `Dataset` carries in place of a
#: geographic one. A swath has no affine mapping to the ground, so this is an
#: explicit placeholder: real locations come from the `GEOLOCATION` domain.
PLACEHOLDER_GEOTRANSFORM: tuple[float, float, float, float, float, float] = (
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    -1.0,
)

#: Directory suffix of a Sentinel-3 granule (SAFE-style product directory).
SEN3_SUFFIX = ".SEN3"

#: Below this magnitude a `scale_factor` / `add_offset` is treated as zero. The
#: packing is compared with a tolerance rather than `==`, because these values come
#: from a file as decimal text and a float round-trip need not land exactly on 1.0
#: or 0.0. The bound sits far below any real geolocation scale (Sentinel-3 uses
#: 1e-06), so a genuine scale is never mistaken for a degenerate one.
_ZERO_SCALE = 1e-30


def resolve_granule(source: Any, reader: str) -> Path:
    """Resolve a reader `source` to the `.SEN3` granule directory holding its files.

    Args:
        source: A `.SEN3` directory, or any file inside one (e.g. a band NetCDF).
        reader: The calling reader's name, for error messages.

    Returns:
        The granule directory as a `Path`.

    Raises:
        ReaderError: When `source` is neither a directory nor a file inside one.
    """
    path = Path(str(source))
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    raise ReaderError(
        f"{reader}: {str(source)!r} is not a Sentinel-3 granule directory (or a "
        f"file inside one); pass the {SEN3_SUFFIX} directory"
    )


def granule_file(granule: Path, filename: str, reader: str) -> Path:
    """Return `filename` inside `granule`, checking it exists.

    Args:
        granule: The `.SEN3` granule directory.
        filename: The file to locate (e.g. `"geo_coordinates.nc"`).
        reader: The calling reader's name, for error messages.

    Returns:
        The file's `Path`.

    Raises:
        ReaderError: When the file is not in the granule.
    """
    path = granule / filename
    if not path.is_file():
        raise ReaderError(
            f"{reader}: granule {granule.name!r} has no {filename!r}; the product "
            f"looks incomplete or is not the expected Sentinel-3 level/type"
        )
    return path


def subdataset(path: Any, variable: str) -> str:
    """Build the GDAL NetCDF subdataset string for `variable` inside `path`.

    Args:
        path: Path to the NetCDF file.
        variable: The variable name inside it.

    Returns:
        A `NETCDF:"<path>":<variable>` connection string.

    Examples:
        - A variable in a band file becomes a GDAL subdataset string:
            ```python
            >>> from pyramids_eo.sensors.readers._sentinel3 import subdataset
            >>> subdataset("granule.SEN3/Oa08_radiance.nc", "Oa08_radiance")
            'NETCDF:"granule.SEN3/Oa08_radiance.nc":Oa08_radiance'

            ```
    """
    return f'NETCDF:"{path}":{variable}'


def _open(path: Any, variable: str, reader: str) -> Any:
    """Open one NetCDF variable as a GDAL raster, or raise `ReaderError`."""
    from osgeo import gdal

    source = subdataset(path, variable)
    try:
        raster = gdal.Open(source)
    except RuntimeError as exc:  # GDAL exceptions enabled
        raise ReaderError(f"{reader}: cannot open {source} ({exc!r})") from exc
    if raster is None:  # GDAL exceptions disabled
        raise ReaderError(f"{reader}: cannot open {source}")
    return raster


def read_packed_variable(
    path: Any, variable: str, *, reader: str
) -> tuple[np.ndarray, dict[str, str]]:
    """Read one CF-packed NetCDF variable, unpacked and masked to NaN.

    Applies the CF unpacking the Sentinel-3 products use -- `raw * scale_factor +
    add_offset` -- and masks the `_FillValue` and anything outside the variable's
    validity limits to NaN, via the shared CF decoder in `_common` (so the
    attribute spellings are resolved the same way for every reader). A variable
    with no `scale_factor` / `add_offset` (the geolocation and already-geophysical
    variables) is read as-is, so this serves both the packed measurement
    variables and the plain annotation ones.

    Args:
        path: Path to the NetCDF file.
        variable: The variable to read.
        reader: The calling reader's name, for error messages.

    Returns:
        A `(array, metadata)` pair. `array` is float64 with invalid pixels NaN;
        `metadata` is the variable's GDAL attribute dict.

    Raises:
        ReaderError: When the variable cannot be opened, or carries a
            `scale_factor` / `add_offset` that is not a finite number.

    Note:
        A `_FillValue` the decoder cannot read as a finite number is not fatal
        here, unlike an unusable `scale_factor`: nothing can be masked from it, so
        `invalid_mask` emits a `UserWarning` and the returned array's nodata is
        incomplete. Callers that surface this (`read_olci`, `read_slstr`) say so
        too.
    """
    raster = _open(path, variable, reader)
    band = raster.GetRasterBand(1)
    raw = np.asarray(band.ReadAsArray())
    meta = dict(band.GetMetadata())
    # Release both handles before any raise below: a live band reference keeps the
    # owning dataset open, so drop them as soon as the data is in memory.
    del raster, band

    values = raw.astype(float)
    raw_scale, raw_offset = meta.get("scale_factor"), meta.get("add_offset")
    if raw_scale is not None or raw_offset is not None:
        scale = 1.0 if raw_scale is None else finite_float(raw_scale)
        offset = 0.0 if raw_offset is None else finite_float(raw_offset)
        if scale is None or offset is None:
            raise ReaderError(
                f"{reader}: {variable!r} has a scale_factor / add_offset that is "
                f"not a finite number ({raw_scale!r} / {raw_offset!r})"
            )
        values = values * scale + offset

    values[invalid_mask(raw, meta)] = np.nan
    return values, meta


def _wgs84_wkt() -> str:
    """Return WGS 84 as a WKT string, for the `GEOLOCATION` domain's `SRS` key.

    WKT is used because it is self-contained: it names the datum and ellipsoid
    outright rather than deferring to an authority lookup. Measured on GDAL 3.13,
    GDAL imports the key permissively -- this WKT and a bare `"EPSG:4326"` both
    work, while an unparseable value is rejected loudly ("Failed to import
    coordinate system") -- so the requirement is simply that it parses.

    What the key *does* buy is the warped output's coordinate system. Omitting it
    leaves the geotransform unchanged but produces a result with **no CRS at all**:

        with this SRS -> GetProjection() == 'GEOGCS["WGS 84", ...'
        without it    -> GetProjection() == ''

    so a caller who warps gets a raster that is positioned correctly but does not
    say in what. That, rather than the antimeridian handling originally supposed,
    is why the key is emitted.

    Returns:
        The WGS 84 WKT string. Note this is EPSG:4326's own axis order, i.e. the
        WKT carries `AXIS["Latitude",NORTH], AXIS["Longitude",EAST]`; the
        geolocation arrays themselves are identified as longitude and latitude by
        the domain's `X_DATASET` / `Y_DATASET` keys, not by this string.
    """
    from osgeo import osr

    reference = osr.SpatialReference()
    reference.ImportFromEPSG(4326)
    # No SetAxisMappingStrategy call here: it sets an in-memory property that
    # ExportToWkt() ignores -- measured, the emitted string is byte-identical with
    # and without it -- so it would be dead code. Which array is longitude and
    # which is latitude is carried by the domain's X_DATASET / Y_DATASET keys.
    return str(reference.ExportToWkt())


def _unscaling_vrt(
    source: str,
    variable: str,
    width: int,
    height: int,
    scale: float,
    offset: float,
    fill: float | None,
) -> str:
    """Write a `/vsimem` VRT that unpacks `source` to degrees, and return its path.

    GDAL's geolocation transformer reads geolocation arrays **raw** through
    `RasterIO` and never applies `scale_factor` / `add_offset` (its own
    `GetScale` documentation says applying them "is of the responsibility of the
    user"). Sentinel-3 stores longitude / latitude CF-packed as `int32`
    micro-degrees with `scale_factor = 1e-06`, so handing GDAL the bare
    subdataset feeds the warp micro-degrees. This wraps the variable in a
    `ComplexSource` carrying `ScaleRatio` / `ScaleOffset`, which GDAL *does*
    apply on read, so the transformer sees real degrees.

    The VRT is tiny XML and reads nothing until a warp actually pulls from it, so
    no geolocation array is loaded or copied at decode time. It lives in
    `/vsimem` under a name derived from the source, so repeated reads of one
    granule reuse a single entry rather than accumulating copies.

    That entry is **process-global and not reference-counted**: GDAL keeps it for
    the life of the process, and a returned `Dataset`'s geolocation stays valid
    only while it exists. Unlinking it by hand makes a later warp fail with
    `RuntimeError: No such file or directory`; nothing in this package removes it.
    Measured on a full-length OLCI product path, one entry is ~0.6 kB and a grid
    takes two (longitude and latitude), so ~1.2 kB per granule-grid, i.e. ~12 MB
    across 10 000 granules -- negligible for a script, worth bounding in a
    long-lived service. The exact size tracks the source path's length, since the
    path is embedded in the XML.

    The variable's fill value must be carried across too. GDAL's geolocation
    transformer consults the band's nodata to decide which samples to skip, so a
    VRT that drops it lets the fill sentinel through as a real coordinate --
    `-2147483648 * 1e-06` is -2147.48 degrees, which drags the warp's computed
    extent far off the globe while `gdal.Warp` only *warns*. The fill is declared
    twice, in the two domains GDAL reads it in: `NODATA` inside the
    `ComplexSource` is the raw, still-packed sentinel, and the band's
    `NoDataValue` is that sentinel in the scaled output units.

    Args:
        source: The GDAL subdataset string of the packed variable.
        variable: The variable name, used to build a stable `/vsimem` name.
        width: Raster width in pixels.
        height: Raster height in pixels.
        scale: The variable's `scale_factor`.
        offset: The variable's `add_offset`.
        fill: The variable's raw `_FillValue`, or `None` when it declares none.

    Returns:
        The `/vsimem` path of the written VRT.
    """
    from osgeo import gdal

    # Not a security digest -- just a short, stable name for the /vsimem entry, so
    # repeated reads of one granule reuse it instead of accumulating copies.
    digest = hashlib.sha1(
        source.encode("utf-8", "replace"), usedforsecurity=False
    ).hexdigest()[:16]
    path = f"/vsimem/pyramids_eo_geoloc/{variable}-{digest}.vrt"
    nodata = ""
    if fill is not None:
        nodata = f"<NoDataValue>{fill * scale + offset!r}</NoDataValue>"
    source_nodata = "" if fill is None else f"<NODATA>{fill!r}</NODATA>"
    xml = (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
        f'<VRTRasterBand dataType="Float64" band="1">'
        f"{nodata}"
        f"<ComplexSource>"
        f'<SourceFilename relativeToVRT="0">{escape(source)}</SourceFilename>'
        f"<SourceBand>1</SourceBand>"
        f'<SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>'
        f'<DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>'
        f"<ScaleOffset>{offset!r}</ScaleOffset>"
        f"<ScaleRatio>{scale!r}</ScaleRatio>"
        f"{source_nodata}"
        f"</ComplexSource>"
        f"</VRTRasterBand>"
        f"</VRTDataset>"
    )
    gdal.FileFromMemBuffer(path, xml.encode("utf-8"))
    return path


def geolocation_source(
    geo_path: Any, variable: str, *, reader: str, shape: tuple[int, int] | None = None
) -> str:
    """Resolve one geolocation variable to a GDAL source that yields degrees.

    Opens the variable -- which is also what validates that it *exists*, so a
    truncated or renamed annotation file fails here, in the reader, rather than
    much later inside the caller's warp -- and inspects its CF packing. A packed
    variable is wrapped in an unscaling VRT (see `_unscaling_vrt`); an already
    unpacked one is referenced directly.

    Existence alone is not enough, which is why `shape` exists: an annotation grid
    of the wrong size places every pixel wrongly, and an **oversized** one is the
    dangerous case because GDAL accepts it without a single warning -- the warp
    simply lands somewhere else. Passing the band's shape turns the open into a
    conformance check as well. The packing is checked for the same reason: a
    non-finite or zero scale builds an all-NaN VRT that reads fine and fails only
    inside the caller's warp, with GDAL's own `Int overflow : -nan(ind)`.

    Args:
        geo_path: Path to the geolocation NetCDF file.
        variable: The variable to resolve (e.g. `"longitude"`).
        reader: The calling reader's name, for error messages.
        shape: The band's `(rows, columns)` for the variable to be checked
            against. `None` (default) skips that check and validates existence and
            packing only; both Sentinel-3 readers pass the decoded band's shape.

    Returns:
        A GDAL source string for the `GEOLOCATION` domain, in degrees -- the bare
        subdataset when the variable is already unpacked, otherwise the `/vsimem`
        VRT path that unscales it.

    Raises:
        ReaderError: When the variable cannot be opened -- which includes the case
            of the file not carrying it at all -- when `shape` is given and the
            variable's own `(rows, columns)` differ from it, or when its
            `scale_factor` / `add_offset` is not finite or the scale is zero.
    """
    source = subdataset(geo_path, variable)
    raster = _open(geo_path, variable, reader)
    band = raster.GetRasterBand(1)
    raw_scale, raw_offset = band.GetScale(), band.GetOffset()
    fill = band.GetNoDataValue()
    width, height = raster.RasterXSize, raster.RasterYSize
    del raster, band

    if shape is not None and (height, width) != shape:
        raise ReaderError(
            f"{reader}: geolocation variable {variable!r} is {height}x{width} but the "
            f"band is {shape[0]}x{shape[1]}; a geolocation grid that does not match "
            f"the image would place every pixel wrongly"
        )

    scale = 1.0 if raw_scale is None else finite_float(raw_scale)
    offset = 0.0 if raw_offset is None else finite_float(raw_offset)
    if scale is None or offset is None or math.isclose(scale, 0.0, abs_tol=_ZERO_SCALE):
        raise ReaderError(
            f"{reader}: geolocation variable {variable!r} has an unusable "
            f"scale_factor / add_offset ({raw_scale!r} / {raw_offset!r}); it must be "
            f"finite and the scale non-zero"
        )
    if math.isclose(scale, 1.0) and math.isclose(offset, 0.0, abs_tol=_ZERO_SCALE):
        return source
    return _unscaling_vrt(source, variable, width, height, scale, offset, fill)


def geolocation_domain(
    geo_path: Any,
    *,
    reader: str,
    longitude_variable: str = "longitude",
    latitude_variable: str = "latitude",
    pixel_step: int = 1,
    line_step: int = 1,
    shape: tuple[int, int] | None = None,
) -> dict[str, str]:
    """Build the GDAL `GEOLOCATION` metadata domain for a swath grid.

    This is how a swath product states where its pixels are without an affine
    geotransform: the domain names the NetCDF subdatasets holding a longitude and
    a latitude for every pixel, and GDAL reads them when a caller warps. It
    describes the geolocation rather than applying it, so no resampling happens
    here.

    Args:
        geo_path: Path to the geolocation NetCDF file (e.g.
            `geo_coordinates.nc` for OLCI, `geodetic_an.nc` for SLSTR).
        reader: The calling reader's name, for error messages.
        longitude_variable: Longitude variable name inside it.
        latitude_variable: Latitude variable name inside it.
        pixel_step: Column stride between geolocation samples (`1` for a
            full-resolution per-pixel grid). Both readers pass the default,
            because they use the full-resolution annotation; the parameters exist
            for the tie-point grids, where OLCI FR subsamples 64 across-track and
            not at all along-track. A tie-point path should read the granule's own
            `ac_subsampling_factor` / `al_subsampling_factor` rather than
            hard-coding either number -- they differ for ERR.
        line_step: Row stride between geolocation samples.
        shape: The band's `(rows, columns)` that both geolocation variables must
            match, forwarded to `geolocation_source`. `None` (default) builds the
            domain without that check; both readers pass the decoded band's shape,
            which is why they decode before building the domain.

    Returns:
        The `GEOLOCATION` domain as a dict of GDAL metadata items: the longitude /
        latitude sources and bands, the offsets and steps, plus the `SRS` (WGS 84,
        which the arrays are on) and a `GEOREFERENCING_CONVENTION` of
        `PIXEL_CENTER`.

        The keys are not equally required, measured against GDAL 3.13 rather than
        assumed: `X_BAND`, `Y_BAND`, `PIXEL_OFFSET`, `LINE_OFFSET`, `PIXEL_STEP`
        and `LINE_STEP` are hard-required, and a warp missing any one of them fails
        **loudly** with `RuntimeError: Missing some geolocation fields in
        GDALCreateGeoLocTransformer()`. A missing `X_DATASET` / `Y_DATASET` is the
        one silent failure: `gdal.Warp` simply returns `None`. `SRS` and
        `GEOREFERENCING_CONVENTION` are both optional -- but leaving the convention
        out shifts the result by half a pixel, which is why it is always stated.

    Raises:
        ReaderError: Propagated from `geolocation_source` -- the file does not
            carry one of the variables, a variable's grid does not match `shape`,
            or its `scale_factor` / `add_offset` is unusable.

    Notes:
        This opens the geolocation variables, so it has no runnable example here --
        it needs a real annotation file. For an OLCI granule,
        `geolocation_domain(granule / "geo_coordinates.nc", reader="read_olci")`
        yields a domain whose `X_DATASET` / `Y_DATASET` are unscaling VRTs over the
        file's `longitude` / `latitude`, at a stride of one. SLSTR passes the
        grid-and-view-suffixed variable names (`longitude_an`, `latitude_an`) for
        the `geodetic_an.nc` of the grid it read. The behaviour is covered end to
        end -- including a real `gdal.Warp` -- in `tests/test_readers_sentinel3.py`.
    """
    return {
        "X_DATASET": geolocation_source(
            geo_path, longitude_variable, reader=reader, shape=shape
        ),
        "X_BAND": "1",
        "Y_DATASET": geolocation_source(
            geo_path, latitude_variable, reader=reader, shape=shape
        ),
        "Y_BAND": "1",
        "PIXEL_OFFSET": "0",
        "LINE_OFFSET": "0",
        "PIXEL_STEP": str(pixel_step),
        "LINE_STEP": str(line_step),
        # WKT, not an authority string: GDAL reads this key with importFromWkt.
        "SRS": _wgs84_wkt(),
        # Per-pixel lon/lat sample pixel centres, not corners. GDAL defaults to
        # TOP_LEFT_CORNER, so this must be stated to avoid a half-pixel shift.
        "GEOREFERENCING_CONVENTION": "PIXEL_CENTER",
    }


def wrap_swath(data: np.ndarray, domain: dict[str, str]) -> Any:
    """Wrap a swath array in a `Dataset` carrying its `GEOLOCATION` domain.

    The returned `Dataset` has NaN nodata and a **placeholder** pixel/line
    geotransform (`PLACEHOLDER_GEOTRANSFORM`) with no CRS, because a swath has no
    affine mapping to the ground. Its real geolocation is the attached
    `GEOLOCATION` domain, which `gdal.Warp(..., geoloc=True)` consumes.

    Args:
        data: The `(row, col)` swath array, NaN where invalid.
        domain: The `GEOLOCATION` domain from `geolocation_domain`, or an empty
            dict to attach none -- which is what a reader's `geolocation=False`
            passes, leaving a result with no geolocation at all.

    Returns:
        A pyramids `Dataset` of `data`, with `domain` set on its GDAL handle. The
        domain is copied, so the caller's dict is not adopted.

        The set happens unconditionally, so `"GEOLOCATION"` is listed by
        `GetMetadataDomainList()` either way -- for an empty `domain` it is simply
        an empty domain. Code deciding whether a result is geolocated must
        therefore test the domain's *contents*, not its presence in that list.
    """
    from pyramids.dataset import Dataset, GeoReference

    dataset = Dataset.from_array(
        np.asarray(data, dtype=float),
        geo_ref=GeoReference(geo=PLACEHOLDER_GEOTRANSFORM, epsg=None),
        no_data_value=np.nan,
    )
    dataset.raster.SetMetadata(dict(domain), "GEOLOCATION")
    return dataset
