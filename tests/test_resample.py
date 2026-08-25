"""Unit tests for `pyramids_eo.resample.to_area` (offline, GDAL warp)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.resample import _warp_nodata, to_area


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

    def test_numpy_integer_crs(self):
        """A NumPy-integer EPSG code (not a Python int) is accepted."""
        out = to_area(_gradient(), np.int64(4326), (0.0, 0.0, 8.0, 8.0), 8, 8)
        assert isinstance(out, Dataset), f"expected Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS not set from numpy int, got {out.epsg}"

    def test_output_geotransform_matches_request(self):
        """The output extent and pixel size equal the requested grid exactly."""
        out = to_area(_gradient(), 4326, (0.0, 0.0, 8.0, 8.0), 8, 4)
        gt = out.geotransform
        assert gt[0] == pytest.approx(0.0), f"min_x wrong: {gt}"
        assert gt[3] == pytest.approx(8.0), f"max_y wrong: {gt}"
        assert gt[1] == pytest.approx(1.0), f"x pixel size wrong: {gt}"
        assert gt[5] == pytest.approx(-2.0), f"y pixel size wrong: {gt}"

    def test_reprojection_lands_requested_grid(self):
        """A true reprojection (4326 -> 3857) lands the exact requested grid."""
        out = to_area(_gradient(), 3857, (-1.0e6, -1.0e6, 1.0e6, 1.0e6), 10, 10)
        assert out.epsg == 3857, f"target CRS not applied, got {out.epsg}"
        gt = out.geotransform
        assert gt[0] == pytest.approx(-1.0e6), f"min_x wrong: {gt}"
        assert gt[3] == pytest.approx(1.0e6), f"max_y wrong: {gt}"
        assert gt[1] == pytest.approx(2.0e5), f"x pixel size wrong: {gt}"


class TestToAreaValidation:
    """Argument validation."""

    def test_ndarray_input_rejected(self):
        """A plain ndarray has no grid to place and is rejected."""
        arr = np.ones((8, 8))
        with pytest.raises(ValueError, match="Dataset"):
            to_area(arr, 4326, (0.0, 0.0, 8.0, 8.0), 8, 8)

    def test_unknown_method_rejected(self):
        """An unknown resampling method is rejected."""
        src = _gradient()
        with pytest.raises(ValueError, match="method"):
            to_area(src, 4326, (0.0, 0.0, 8.0, 8.0), 8, 8, method="nope")

    def test_bad_extent_rejected(self):
        """An extent with min >= max is rejected."""
        src = _gradient()
        with pytest.raises(ValueError, match="extent"):
            to_area(src, 4326, (8.0, 0.0, 0.0, 8.0), 8, 8)

    def test_non_positive_size_rejected(self):
        """Zero / negative width or height is rejected."""
        src = _gradient()
        with pytest.raises(ValueError, match="positive"):
            to_area(src, 4326, (0.0, 0.0, 8.0, 8.0), 0, 8)

    def test_wrong_length_extent_rejected(self):
        """An extent that is not a 4-tuple is rejected."""
        src = _gradient()
        with pytest.raises(ValueError, match="extent"):
            to_area(src, 4326, (0.0, 0.0, 8.0), 8, 8)

    def test_non_integer_size_rejected(self):
        """A fractional width / height is rejected rather than truncated."""
        src = _gradient()
        with pytest.raises(ValueError, match="whole"):
            to_area(src, 4326, (0.0, 0.0, 8.0, 8.0), 8.5, 8)

    def test_bool_crs_rejected(self):
        """A bool CRS (an Integral) is rejected rather than mapped to EPSG:1."""
        src = _gradient()
        with pytest.raises(ValueError, match="bool"):
            to_area(src, True, (0.0, 0.0, 8.0, 8.0), 8, 8)


class TestWarpNodata:
    """`_warp_nodata` maps a scalar or per-band nodata to a GDAL nodata arg."""

    def test_scalar_passes_through(self):
        """A scalar nodata is returned unchanged."""
        assert _warp_nodata(-9999) == -9999, "scalar nodata should pass through"

    def test_single_band_list_returns_scalar(self):
        """A one-band list unwraps to that band's scalar value."""
        assert _warp_nodata([5.0]) == 5.0, "single-band list should unwrap"

    def test_per_band_returns_space_joined(self):
        """Distinct per-band values are preserved as a space-separated string."""
        assert _warp_nodata([1.0, 2.0, 3.0]) == "1.0 2.0 3.0", "per-band not preserved"

    def test_all_none_list_returns_none(self):
        """A list of only None yields None (no nodata to carry)."""
        assert _warp_nodata([None]) is None, "all-None list should give None"

    def test_empty_list_returns_none(self):
        """An empty per-band list yields None."""
        assert _warp_nodata([]) is None, "empty list should give None"

    def test_none_returns_none(self):
        """A missing nodata (None) yields None."""
        assert _warp_nodata(None) is None, "None should give None"
