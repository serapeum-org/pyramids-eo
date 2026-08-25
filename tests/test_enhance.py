"""Unit tests for `pyramids_eo.enhance.stretch` (offline, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset
from pyramids_eo.enhance import _CIRA_DENOM, _CIRA_LOG_ROOT, stretch


class TestStretchCrude:
    """`kind="crude"` is a fixed min/max linear rescale."""

    def test_given_bounds_rescale_to_unit(self):
        """(x - lo) / (hi - lo) with explicit bounds maps lo->0, hi->1."""
        x = np.array([[2.0, 6.0, 10.0]])
        out = stretch(
            x, kind="crude", min_stretch=2.0, max_stretch=10.0, dtype="float64"
        )
        assert np.allclose(out, [[0.0, 0.5, 1.0]]), f"crude rescale wrong: {out}"

    def test_auto_min_max_spans_unit(self):
        """With no bounds the data min/max map to 0 and 1."""
        out = stretch(np.array([[2.0, 6.0, 10.0]]), kind="crude", dtype="float64")
        assert out.min() == pytest.approx(0.0), "min should map to 0"
        assert out.max() == pytest.approx(1.0), "max should map to 1"

    def test_values_below_min_clip_to_zero(self):
        """Values under min_stretch clamp to 0, not negative."""
        out = stretch(
            np.array([[-1.0, 0.0, 1.0]]),
            kind="crude",
            min_stretch=0.0,
            max_stretch=1.0,
            dtype="float64",
        )
        assert out.min() >= 0.0, f"not clipped low: {out}"


class TestStretchLinear:
    """`kind="linear"` clips at percentile cutoffs then rescales."""

    def test_outlier_clipped_by_cutoff(self):
        """A high outlier is clipped by the right cutoff, so max stays 1."""
        x = np.concatenate([np.linspace(0.0, 1.0, 99), [1000.0]]).reshape(1, -1)
        out = stretch(x, kind="linear", cutoffs=(0.0, 0.02), dtype="float64")
        assert out.max() <= 1.0, "outlier should be clipped to 1"
        assert out[0, -1] == pytest.approx(1.0), "the outlier saturates high"

    def test_default_is_linear(self):
        """The default kind is linear (percentile stretch)."""
        x = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        out = stretch(x, dtype="float64")
        assert out.min() == pytest.approx(0.0), "linear default should hit 0"
        assert out.max() == pytest.approx(1.0), "linear default should hit 1"


class TestStretchCira:
    """`kind="cira"` is the logarithmic true-colour stretch."""

    def test_reference_zero_point(self):
        """Reflectance 0.0223 maps to 0 (log10 == the CIRA log root)."""
        out = stretch(np.array([[0.0223]]), kind="cira", dtype="float64")
        assert float(out[0, 0]) == pytest.approx(0.0, abs=1e-9), "0.0223 -> 0"

    def test_full_reflectance_point(self):
        """Reflectance 1.0 maps to (-log_root)/denom."""
        out = stretch(np.array([[1.0]]), kind="cira", dtype="float64")
        expected = (0.0 - _CIRA_LOG_ROOT) / _CIRA_DENOM
        assert float(out[0, 0]) == pytest.approx(expected), f"cira(1.0) wrong: {out}"

    def test_monotone_increasing(self):
        """The curve is monotone increasing in reflectance."""
        out = stretch(np.array([[0.05, 0.2, 0.5, 1.0]]), kind="cira", dtype="float64")
        assert np.all(np.diff(out[0]) > 0), f"cira not monotone: {out}"

    def test_ignores_min_max_and_cutoffs(self):
        """cira is a fixed curve — bounds/cutoffs do not change it."""
        x = np.array([[0.1, 0.5, 0.9]])
        a = stretch(x, kind="cira", dtype="float64")
        b = stretch(
            x,
            kind="cira",
            min_stretch=0.0,
            max_stretch=2.0,
            cutoffs=(0.2, 0.2),
            dtype="float64",
        )
        assert np.allclose(a, b), "cira should ignore min/max/cutoffs"


class TestStretchHistogram:
    """`kind="histogram"` equalises to [0, 1]."""

    def test_output_within_unit_range(self):
        """Equalised output lies in [0, 1]."""
        rng = np.random.default_rng(0)
        out = stretch(rng.random((16, 16)), kind="histogram", dtype="float64")
        assert out.min() >= 0.0 and out.max() <= 1.0, "equalised out of range"

    def test_monotone_in_value(self):
        """Equalisation is order-preserving (a sorted input stays sorted)."""
        x = np.linspace(0.0, 1.0, 64).reshape(1, -1)
        out = stretch(x, kind="histogram", dtype="float64")
        assert np.all(np.diff(out[0]) >= 0), "equalisation should preserve order"


class TestGamma:
    """`gamma` applies x ** (1 / gamma) after the stretch."""

    def test_gamma_brightens_midtones(self):
        """gamma > 1 raises a mid grey (0.25 -> 0.5 at gamma 2)."""
        out = stretch(
            np.array([[0.25]]),
            kind="crude",
            min_stretch=0.0,
            max_stretch=1.0,
            gamma=2.0,
            dtype="float64",
        )
        assert float(out[0, 0]) == pytest.approx(0.5), f"gamma not applied: {out}"

    def test_non_positive_gamma_raises(self):
        """A non-positive gamma is rejected."""
        with pytest.raises(ValueError, match="gamma"):
            stretch(np.array([[0.5]]), gamma=0.0)


class TestDtypeAndNaN:
    """Dtype scaling and NaN handling."""

    def test_uint8_scales_to_full_range(self):
        """A [0, 1] stretch scales onto [0, 255] uint8."""
        out = stretch(
            np.array([[0.0, 1.0]]), kind="crude", min_stretch=0.0, max_stretch=1.0
        )
        assert out.dtype == np.uint8, f"expected uint8, got {out.dtype}"
        assert out.tolist() == [[0, 255]], f"uint8 scaling wrong: {out}"

    def test_nan_excluded_from_stats(self):
        """A NaN pixel does not skew the auto min/max."""
        out = stretch(np.array([[0.0, np.nan, 10.0]]), kind="crude", dtype="float64")
        assert out[0, 0] == pytest.approx(0.0), "min should still map to 0"
        assert out[0, 2] == pytest.approx(1.0), "max should still map to 1"

    def test_nan_preserved_for_float_output(self):
        """Float output keeps NaN where the input was NaN."""
        out = stretch(np.array([[0.5, np.nan]]), kind="cira", dtype="float64")
        assert np.isnan(out[0, 1]), "NaN should survive to float output"

    def test_nan_filled_zero_for_integer_output(self):
        """Integer output fills NaN with 0 (no valid integer NaN)."""
        out = stretch(np.array([[0.5, np.nan]]), kind="cira", dtype="uint8")
        assert out[0, 1] == 0, f"NaN should fill to 0, got {out[0, 1]}"

    def test_all_nan_crude_stays_nan(self):
        """An all-NaN input has empty stats and stays NaN (float output)."""
        out = stretch(np.full((2, 2), np.nan), kind="crude", dtype="float64")
        assert np.isnan(out).all(), f"all-NaN crude should stay NaN, got {out}"

    def test_all_nan_histogram_stays_nan(self):
        """Histogram equalisation of an all-NaN input returns all-NaN."""
        out = stretch(np.full((2, 2), np.nan), kind="histogram", dtype="float64")
        assert np.isnan(out).all(), f"all-NaN histogram should stay NaN, got {out}"


class TestReturnType:
    """Datasets round-trip to Datasets; arrays to arrays."""

    def test_dataset_in_dataset_out(self):
        """A Dataset input yields a georeferenced Dataset with the dtype."""
        ds = Dataset.create_from_array(
            np.full((2, 2), 0.5), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
        )
        out = stretch(ds, kind="crude", min_stretch=0.0, max_stretch=1.0)
        assert isinstance(out, Dataset), f"expected Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS not preserved: {out.epsg}"

    def test_array_in_array_out(self):
        """A plain array returns an ndarray."""
        out = stretch(np.full((2, 2), 0.5))
        assert isinstance(out, np.ndarray), f"expected ndarray, got {type(out)}"

    def test_unknown_kind_raises(self):
        """An unknown stretch kind is rejected."""
        with pytest.raises(ValueError, match="kind"):
            stretch(np.array([[0.5]]), kind="nope")
