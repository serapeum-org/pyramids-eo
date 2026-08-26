"""Scene-Classification (SCL) masking for Sentinel-2 Level-2A.

L2A carries a Scene Classification band whose integer codes label each pixel
(cloud, shadow, water, vegetation, …). The GDAL ``SENTINEL2`` driver attaches
the class names as band category names; :class:`SclClass` mirrors them as a
typed enum so a mask reads ``[SclClass.CLOUD_HIGH_PROBA, SclClass.CLOUD_SHADOW]``
rather than ``[9, 3]``.

:func:`scl_mask` sets the masked-out pixels to the dataset's no-data value.
Because category names survive reprojection since pyramids-gis #1024, masking
may run before or after a warp — order is a convenience, not a correctness rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import numpy as np

from pyramids_eo.errors import ProductError

#: Default no-data value stamped on a masked output when the dataset has none.
_DEFAULT_NODATA = 0.0


class SclClass(IntEnum):
    """Sentinel-2 L2A Scene-Classification codes (GDAL category names).

    Each member's value is the integer pixel code the SCL band stores, so a
    member can be used directly as a class selector or compared with a raw code.

    Examples:
        - A member carries the SCL pixel code and its name:
            ```python
            >>> from pyramids_eo.sentinel.s2.masks import SclClass
            >>> SclClass.CLOUD_HIGH_PROBA.value
            9
            >>> SclClass.WATER.name
            'WATER'

            ```
        - Build a cloud/shadow mask selection and read back the codes:
            ```python
            >>> from pyramids_eo.sentinel.s2.masks import SclClass
            >>> clouds = [SclClass.CLOUD_MEDIUM_PROBA, SclClass.CLOUD_HIGH_PROBA]
            >>> [int(c) for c in clouds]
            [8, 9]

            ```
    """

    NODATA = 0
    SATURATED_DEFECTIVE = 1
    DARK_FEATURE_SHADOW = 2
    CLOUD_SHADOW = 3
    VEGETATION = 4
    BARE_SOIL_DESERT = 5
    WATER = 6
    CLOUD_LOW_PROBA = 7
    CLOUD_MEDIUM_PROBA = 8
    CLOUD_HIGH_PROBA = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11


def _resolve_classes(classes: Sequence[SclClass | str | int]) -> set[int]:
    """Coerce a mix of enum members / names / ints to a set of class codes.

    Args:
        classes: Class selectors — :class:`SclClass`, a class name string
            (case-insensitive, matched against the enum), or a raw int code.

    Returns:
        The integer class codes.

    Raises:
        ProductError: A string does not name an ``SclClass`` member.
    """
    out: set[int] = set()
    for c in classes:
        if isinstance(c, SclClass):
            out.add(int(c))
        elif isinstance(c, int):
            out.add(int(c))
        else:
            try:
                out.add(int(SclClass[str(c).strip().upper()]))
            except KeyError as exc:
                raise ProductError(
                    f"unknown SCL class {c!r}; known: {[m.name for m in SclClass]}"
                ) from exc
    return out


def _find_scl_band(dataset: Any) -> int | None:
    """Return the 0-based index of a band named ``SCL`` in ``dataset``, else None."""
    for i, meta in enumerate(dataset.band_meta_data):
        if (meta.get("BANDNAME") or "").strip().upper() == "SCL":
            return i
    for i, name in enumerate(dataset.band_names):
        if name.strip().upper().startswith("SCL"):
            return i
    return None


def scl_mask(
    dataset: Any,
    classes: Sequence[SclClass | str | int],
    *,
    scl: Any = None,
) -> Any:
    """Mask ``dataset`` pixels whose SCL class is in ``classes``.

    Masked pixels are set to the dataset's no-data value across every band. The
    SCL source is either an explicit ``scl`` (a single-band pyramids ``Dataset``
    or a 2-D NumPy array on the same grid), or — when ``scl`` is ``None`` — an
    ``SCL`` band found within ``dataset`` itself.

    Args:
        dataset: The pyramids ``Dataset`` to mask.
        classes: Classes to mask out (see :func:`_resolve_classes`), e.g.
            ``[SclClass.CLOUD_HIGH_PROBA, SclClass.CLOUD_SHADOW]``.
        scl: Optional explicit SCL source aligned to ``dataset``'s grid. When
            ``None``, an ``SCL`` band inside ``dataset`` is used.

    Returns:
        A new pyramids ``Dataset`` with the masked pixels set to no-data; band
        names, no-data value, and scale/offset tags are preserved.

    Raises:
        ProductError: No SCL source is available, or its shape does not match
            ``dataset``.
    """
    from pyramids.dataset import Dataset

    codes = _resolve_classes(classes)
    scl_array = _scl_array(dataset, scl)

    # read_array returns (rows, cols) for one band and (bands, rows, cols) for
    # many; normalise a single band to (1, rows, cols) so the mask indexing and
    # the shape check below are uniform. (np.atleast_3d would wrongly append the
    # band axis, giving (rows, cols, 1).)
    data = np.asarray(dataset.read_array())
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if scl_array.shape != data.shape[-2:]:
        raise ProductError(
            f"SCL grid {scl_array.shape} does not match data grid {data.shape[-2:]}"
        )

    nodata = _nodata_of(dataset)
    mask = np.isin(scl_array, list(codes))
    masked = data.astype("float64", copy=True)
    masked[:, mask] = nodata

    out = Dataset.create_from_array(
        arr=masked.astype(data.dtype, copy=False),
        geo=dataset.raster.GetGeoTransform(),
        epsg=dataset.epsg,
    )
    out.no_data_value = [nodata] * out.band_count
    _carry_band_state(dataset, out)
    return out


def _scl_array(dataset: Any, scl: Any) -> np.ndarray:
    """Resolve the SCL source to a 2-D array on ``dataset``'s grid."""
    if scl is None:
        idx = _find_scl_band(dataset)
        if idx is None:
            raise ProductError(
                "no SCL band in the dataset and no `scl=` given; "
                "read a subdataset that contains SCL (20 m / 60 m), or pass scl="
            )
        return np.asarray(dataset.read_array(band=idx))
    if isinstance(scl, np.ndarray):
        return scl
    # Assume a pyramids Dataset (single band).
    return np.asarray(scl.read_array(band=0))


def _nodata_of(dataset: Any) -> float:
    """First defined no-data value on ``dataset``, else the S2 default (0)."""
    for value in dataset.no_data_value:
        if value is not None:
            return float(value)
    return _DEFAULT_NODATA


def _carry_band_state(source: Any, dest: Any) -> None:
    """Copy band names and scale/offset from ``source`` to ``dest``.

    Band names are display-only, so a failure to copy them is swallowed. The
    scale/offset tags are the reflectance calibration, so their copy is **not**
    swallowed — losing them would silently return raw DN from a scaled read
    (a caller who masks an already-reflectance-tagged dataset relies on this).
    """
    try:
        dest.band_names = list(source.band_names)
    except Exception:  # noqa: BLE001 - display metadata only
        pass
    dest.scale = list(source.scale)
    dest.offset = list(source.offset)
