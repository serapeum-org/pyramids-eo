"""Unit tests for `pyramids_eo.resample.to_area` (offline, GDAL warp)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset
from pyramids_eo.resample import _scalar_nodata, to_area


def _gradient(bands: int = 1, rows: int = 8, cols: int = 8) -> Dataset:
    """A 4326 raster with a horizontal 0..1 gradient, top-left at (0, rows)."""
    row = np.linspace(0.0, 1.0, cols, dtype="float64")
    base = np.tile(row, (rows, 1))
    arr = base if bands == 1 else np.stack([base * (i + 1) for i in range(bands)])
    return Dataset.create_from_array(
        arr, top_left_corner=(0.0, float(rows)), cell_size=1.0, epsg=4326
    )


def _shape(dataset: Dataset) -> tuple[int, ...]:
    """The read_array shape of a Dataset."""
    return np.asarray(dataset.read_array()).shape


class TestToAreaGrid:
    """`to_area` lands the exact requested extent and pixel dimensions."""

    def test_exact_width_height(self):
        """The output has exactly the requested rows/cols."""
        out = to_area(_gradient(), 4326, (0.0, 0.0, 8.0, 8.0), 16, 16)
        assert _shape(out)[-2:] == (16, 16), f"grid not pinned: {_shape(out)}"

    def test_asymmetric_size(self):
        """Non-square width/height are honoured exactly."""
        out = to_area(_gradient(), 4326, (0.0, 0.0, 8.0, 8.0), 20, 10)
        assert _shape(out)[-2:] == (10, 20), f"expected (10, 20): {_shape(out)}"

    def test_bilinear_differs_from_nearest(self):
        """On a gradient, bilinear upsampling differs from nearest."""
        args = (_gradient(), 4326, (0.0, 0.0, 8.0, 8.0), 16, 16)
        near = np.asarray(to_area(*args, method="nearest").read_array())
        bilin = np.asarray(to_area(*args, method="bilinear").read_array())
        assert not np.allclose(near, bilin), "bilinear should smooth the gradient"


class TestToAreaBandsAndTypes:
    """Band count, dtype and nodata survive the warp."""

    @pytest.mark.parametrize("bands", [1, 3, 4])
    def test_band_count_preserved(self, bands):
        """1-, 3- and 4-band inputs keep their band count."""
        out = to_area(_gradient(bands=bands), 4326, (0.0, 0.0, 8.0, 8.0), 8, 8)
        shape = _shape(out)
        got = 1 if len(shape) == 2 else shape[0]
        assert got == bands, f"expected {bands} bands, got {got}"

    def test_dtype_preserved(self):
        """The output keeps the source dtype (float64)."""
        out = to_area(_gradient(), 4326, (0.0, 0.0, 8.0, 8.0), 8, 8)
        assert np.asarray(out.read_array()).dtype == np.float64, "dtype changed"

    def test_source_without_nodata_warps(self):
        """A source with no nodata warps without passing srcNodata/dstNodata."""
        ds = Dataset.create_from_array(
            np.ones((8, 8)),
            top_left_corner=(0.0, 8.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=None,
        )
        out = to_area(ds, 4326, (0.0, 0.0, 8.0, 8.0), 8, 8)
        assert isinstance(out, Dataset), f"expected Dataset, got {type(out)}"
        assert _shape(out)[-2:] == (8, 8), f"grid not pinned: {_shape(out)}"

    def test_nodata_region_survives(self):
        """A NaN nodata cell is not interpolated across (nearest keeps NaN)."""
        arr = np.tile(np.linspace(0.0, 1.0, 8), (8, 1))
        arr[0, 0] = np.nan
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 8.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=np.nan,
        )
        out = to_area(ds, 4326, (0.0, 0.0, 8.0, 8.0), 8, 8, method="nearest")
        assert np.isnan(np.asarray(out.read_array())).any(), "nodata NaN lost"


class TestToAreaCrs:
    """CRS handling, including an EPSG-less PROJ4 target."""

    def test_proj4_without_epsg(self):
        """A PROJ4 string with no EPSG code is accepted."""
        proj4 = "+proj=longlat +datum=WGS84 +no_defs"
        out = to_area(_gradient(), proj4, (0.0, 0.0, 8.0, 8.0), 8, 8)
        assert isinstance(out, Dataset), f"expected Dataset, got {type(out)}"
        assert _shape(out)[-2:] == (8, 8), f"grid not pinned: {_shape(out)}"


class TestToAreaValidation:
    """Argument validation."""

    def test_ndarray_input_rejected(self):
        """A plain ndarray has no grid to place and is rejected."""
        with pytest.raises(ValueError, match="Dataset"):
            to_area(np.ones((8, 8)), 4326, (0.0, 0.0, 8.0, 8.0), 8, 8)

    def test_unknown_method_rejected(self):
        """An unknown resampling method is rejected."""
        with pytest.raises(ValueError, match="method"):
            to_area(_gradient(), 4326, (0.0, 0.0, 8.0, 8.0), 8, 8, method="nope")

    def test_bad_extent_rejected(self):
        """An extent with min >= max is rejected."""
        with pytest.raises(ValueError, match="extent"):
            to_area(_gradient(), 4326, (8.0, 0.0, 0.0, 8.0), 8, 8)

    def test_non_positive_size_rejected(self):
        """Zero / negative width or height is rejected."""
        with pytest.raises(ValueError, match="positive"):
            to_area(_gradient(), 4326, (0.0, 0.0, 8.0, 8.0), 0, 8)

    def test_wrong_length_extent_rejected(self):
        """An extent that is not a 4-tuple is rejected."""
        with pytest.raises(ValueError, match="extent"):
            to_area(_gradient(), 4326, (0.0, 0.0, 8.0), 8, 8)


class TestScalarNodata:
    """`_scalar_nodata` reduces a scalar or per-band nodata to one value."""

    def test_scalar_passes_through(self):
        """A scalar nodata is returned unchanged."""
        assert _scalar_nodata(-9999) == -9999, "scalar nodata should pass through"

    def test_list_returns_first_band(self):
        """A per-band list returns the first band's nodata."""
        assert _scalar_nodata([5.0, 6.0, 7.0]) == 5.0, "should take the first band"

    def test_empty_list_returns_none(self):
        """An empty per-band list yields None (nothing to carry)."""
        assert _scalar_nodata([]) is None, "empty list should give None"

    def test_none_returns_none(self):
        """A missing nodata (None) yields None."""
        assert _scalar_nodata(None) is None, "None should give None"
