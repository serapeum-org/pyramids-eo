"""Offline tests for `_crop_to_bbox` — grid-stable bbox windowing (#81).

`from_sentinel2` used `Dataset.crop(bbox=)`, which trims all-no-data border
rows/cols — and for a multi-band array a row is trimmed only when *every* band
is no-data there, so the output grid shrank by an amount that depended on how
many bands were read. `_crop_to_bbox` reads exactly the bbox window without
trimming, so the grid is a deterministic function of the bbox and resolution.
These build small in-memory rasters with a no-data border and assert the window
is kept whole and is identical across band counts.
"""

from __future__ import annotations

import numpy as np
import pyramids  # noqa: F401  (activates the bundled osgeo)
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.errors import ProductError
from pyramids_eo.sentinel.s2 import reader as _reader

_GEO = (0.0, 1.0, 0.0, 10.0, 0.0, -1.0)  # 10x10 grid, origin (0, 10), 1-unit pixels


def _band(nodata_last_col: bool) -> np.ndarray:
    """A 10x10 band of ones, optionally with the last column set to no-data (0)."""
    arr = np.ones((10, 10), dtype="uint16")
    if nodata_last_col:
        arr[:, -1] = 0
    return arr


def _dataset(arr: np.ndarray) -> Dataset:
    """Build a MEM dataset from a (bands, rows, cols) array with no-data 0."""
    return Dataset.create_from_array(arr=arr, geo=_GEO, epsg=4326, no_data_value=0)


class TestCropToBbox:
    """Tests for `_reader._crop_to_bbox`."""

    def test_keeps_the_full_window_despite_a_no_data_border(self):
        """A window whose edge column is all-no-data keeps its full width.

        Test scenario:
            The last column is all no-data; `crop(bbox=)` would trim it, but
            `_crop_to_bbox` returns the full 10x10 window.
        """
        one = _dataset(_band(nodata_last_col=True)[np.newaxis, :, :])
        out = _reader._crop_to_bbox(one, (0.0, 0.0, 10.0, 10.0))
        assert out.shape == (1, 10, 10), f"window was trimmed: {out.shape}"

    def test_grid_is_stable_across_band_count(self):
        """A single-band and a multi-band read of one bbox get the same grid.

        Test scenario:
            Band 0 has a no-data last column; band 1 does not. A single-band and
            a two-band crop of the same bbox must return the same rows x cols —
            the invariant #81 broke.
        """
        single = _dataset(_band(nodata_last_col=True)[np.newaxis, :, :])
        multi = _dataset(
            np.stack([_band(nodata_last_col=True), _band(nodata_last_col=False)])
        )
        one = _reader._crop_to_bbox(single, (0.0, 0.0, 10.0, 10.0))
        two = _reader._crop_to_bbox(multi, (0.0, 0.0, 10.0, 10.0))
        assert one.shape[1:] == two.shape[1:] == (10, 10), (
            f"grid differs by band count: {one.shape} vs {two.shape}"
        )

    def test_sub_window_size_is_deterministic(self):
        """A 4-unit sub-window yields exactly a 4x4 grid.

        Test scenario:
            Cropping (2, 2, 6, 6) of the 10x10 grid returns a 4x4 window at the
            expected origin.
        """
        ds = _dataset(_band(nodata_last_col=False)[np.newaxis, :, :])
        out = _reader._crop_to_bbox(ds, (2.0, 2.0, 6.0, 6.0))
        assert out.shape == (1, 4, 4), f"unexpected window: {out.shape}"
        assert out.raster.GetGeoTransform()[0] == pytest.approx(2.0)

    def test_bbox_outside_extent_raises(self):
        """A bbox that does not overlap the raster raises ProductError.

        Test scenario:
            A window far outside the 10x10 extent has no pixels and is rejected.
        """
        ds = _dataset(_band(nodata_last_col=False)[np.newaxis, :, :])
        with pytest.raises(ProductError, match="does not overlap"):
            _reader._crop_to_bbox(ds, (100.0, 100.0, 110.0, 110.0))
