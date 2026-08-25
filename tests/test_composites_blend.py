"""Unit tests for `pyramids_eo.composites.day_night_blend` / `day_weight` (offline)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.composites import day_night_blend, day_weight
from pyramids_eo.composites.blend import _as_array


class TestDayWeight:
    """The cos-SZA day weight ramps 1 -> 0 through the twilight band."""

    def test_full_day_below_lim_low(self):
        """An overhead Sun (SZA=0) is fully day (weight 1)."""
        assert float(day_weight(0.0)) == pytest.approx(1.0), "SZA 0 should be weight 1"

    def test_full_night_above_lim_high(self):
        """The antisolar point (SZA=180) is fully night (weight 0)."""
        assert float(day_weight(180.0)) == pytest.approx(0.0), "SZA 180 -> weight 0"

    def test_exactly_lim_low_is_one(self):
        """At SZA == lim_low the weight is exactly 1."""
        assert float(day_weight(78.0)) == pytest.approx(1.0), "SZA=lim_low -> 1"

    def test_exactly_lim_high_is_zero(self):
        """At SZA == lim_high the weight is exactly 0."""
        assert float(day_weight(88.0)) == pytest.approx(0.0, abs=1e-9), "lim_high -> 0"

    def test_twilight_midpoint_is_about_half(self):
        """Midway through the 78..88 band the weight is ~0.5."""
        assert float(day_weight(83.0)) == pytest.approx(0.5, abs=0.05), "midpoint ~0.5"

    def test_monotonic_decreasing_with_sza(self):
        """Weight decreases as SZA increases across the twilight band."""
        weights = day_weight(np.array([70.0, 78.0, 83.0, 88.0, 95.0]))
        assert np.all(np.diff(weights) <= 0), f"weights not monotone: {weights}"

    def test_custom_limits(self):
        """Custom lim_low/lim_high shift the ramp endpoints."""
        assert float(day_weight(85.0, lim_low=85.0, lim_high=90.0)) == pytest.approx(
            1.0
        ), "SZA at the custom lim_low should be full day"

    def test_lim_low_not_less_than_lim_high_raises(self):
        """lim_low >= lim_high is rejected."""
        with pytest.raises(ValueError, match="lim_low"):
            day_weight(80.0, lim_low=88.0, lim_high=78.0)

    def test_array_shape_preserved(self):
        """The weight keeps the shape of the SZA input."""
        out = day_weight(np.zeros((3, 4)))
        assert out.shape == (3, 4), f"expected (3, 4), got {out.shape}"

    def test_scalar_returns_zero_dim_array(self):
        """A scalar SZA yields a 0-d weight array."""
        out = day_weight(0.0)
        assert out.shape == (), f"expected scalar (0-d), got shape {out.shape}"

    def test_nan_sza_propagates_to_nan_weight(self):
        """A NaN SZA yields a NaN weight (no silent clipping to 0/1)."""
        assert np.isnan(day_weight(np.nan)), "NaN SZA should give NaN weight"

    def test_does_not_mutate_input(self):
        """Computing the weight leaves the SZA array unchanged."""
        sza = np.array([70.0, 83.0, 95.0])
        day_weight(sza)
        assert np.array_equal(sza, [70.0, 83.0, 95.0]), "input SZA was mutated"


class TestDayNightBlend:
    """`day_night_blend` mixes day and night by the SZA day weight."""

    def test_all_day_returns_day(self):
        """With the Sun overhead everywhere the result is the day image."""
        day = np.full((2, 2), 5.0)
        night = np.full((2, 2), 9.0)
        out = day_night_blend(day, night, np.zeros((2, 2)))
        assert np.allclose(out, day), f"all-day blend should equal day, got {out}"

    def test_all_night_returns_night(self):
        """With the Sun below the horizon everywhere the result is the night image."""
        day = np.full((2, 2), 5.0)
        night = np.full((2, 2), 9.0)
        out = day_night_blend(day, night, np.full((2, 2), 180.0))
        assert np.allclose(out, night), f"all-night blend should equal night: {out}"

    def test_twilight_is_convex_mix(self):
        """In the twilight band the output is the weighted average of day and night."""
        day = np.array([[1.0]])
        night = np.array([[0.0]])
        sza = np.array([[83.0]])
        expected = float(day_weight(83.0))
        assert float(day_night_blend(day, night, sza)[0, 0]) == pytest.approx(expected)

    def test_matches_explicit_formula(self):
        """The blend equals day*weight + night*(1-weight) computed directly."""
        rng = np.random.default_rng(1337)
        day = rng.random((3, 5))
        night = rng.random((3, 5))
        sza = rng.uniform(0, 180, (3, 5))
        weight = day_weight(sza)
        assert np.allclose(
            day_night_blend(day, night, sza), day * weight + night * (1 - weight)
        )

    def test_nan_in_weighted_out_region_is_suppressed(self):
        """A NaN day pixel on the fully-night side does not leak into the output."""
        day = np.array([[1.0, np.nan]])  # night pixel (col 1) is NaN
        night = np.array([[0.0, 7.0]])
        sza = np.array([[0.0, 180.0]])  # col 0 full day, col 1 full night
        out = day_night_blend(day, night, sza)
        assert out[0, 0] == pytest.approx(1.0), "day side should be the day image"
        assert out[0, 1] == pytest.approx(7.0), "night side should be the night image"

    def test_day_only_suppresses_nan_night_region(self):
        """day_only zeros the night side even when day is NaN there."""
        out = day_night_blend(
            np.array([[1.0, np.nan]]),
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 180.0]]),
            mode="day_only",
        )
        assert out[0, 1] == pytest.approx(0.0), "weighted-out NaN should not leak"

    def test_nan_sza_pixel_stays_nan(self):
        """A NaN SZA (undefined geometry) keeps the pixel masked, not black."""
        out = day_night_blend(
            np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]]), np.array([[0.0, np.nan]])
        )
        assert out[0, 0] == pytest.approx(1.0), "defined pixel should blend normally"
        assert np.isnan(out[0, 1]), "NaN SZA pixel should stay NaN, not 0"

    def test_night_only_suppresses_nan_day_region(self):
        """night_only zeros the day side even when night is NaN there."""
        out = day_night_blend(
            np.array([[0.0, 0.0]]),
            np.array([[np.nan, 5.0]]),
            np.array([[0.0, 180.0]]),
            mode="night_only",
        )
        assert out[0, 0] == pytest.approx(0.0), "weighted-out NaN should not leak"
        assert out[0, 1] == pytest.approx(5.0), "night side should be the night image"

    def test_multiband_weight_broadcasts_over_bands(self):
        """A 2-D SZA weight applies to every band of a (band, H, W) image."""
        day = np.ones((3, 2, 2))
        night = np.zeros((3, 2, 2))
        sza = np.array([[0.0, 180.0], [0.0, 180.0]])
        out = day_night_blend(day, night, sza)
        assert out.shape == (3, 2, 2), f"expected (3, 2, 2), got {out.shape}"
        assert np.allclose(out[:, :, 0], 1.0), "day column should be the day image"
        assert np.allclose(out[:, :, 1], 0.0), "night column should be the night image"

    def test_day_only_mode_ignores_night(self):
        """day_only returns day * weight, so the night side goes to zero."""
        day = np.full((1, 2), 4.0)
        night = np.full((1, 2), 9.0)
        sza = np.array([[0.0, 180.0]])
        out = day_night_blend(day, night, sza, mode="day_only")
        assert out[0, 0] == pytest.approx(4.0), "day side should keep the day value"
        assert out[0, 1] == pytest.approx(0.0), "night side should be zeroed"

    def test_night_only_mode_ignores_day(self):
        """night_only returns night * (1 - weight), so the day side goes to zero."""
        day = np.full((1, 2), 4.0)
        night = np.full((1, 2), 9.0)
        sza = np.array([[0.0, 180.0]])
        out = day_night_blend(day, night, sza, mode="night_only")
        assert out[0, 0] == pytest.approx(0.0), "day side should be zeroed"
        assert out[0, 1] == pytest.approx(9.0), "night side should keep the night value"

    def test_unknown_mode_raises(self):
        """An unknown blend mode is rejected."""
        with pytest.raises(ValueError, match="mode"):
            day_night_blend(
                np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1)), mode="x"
            )

    def test_invalid_limits_raise(self):
        """lim_low >= lim_high propagates from day_weight."""
        with pytest.raises(ValueError, match="lim_low"):
            day_night_blend(
                np.ones((1, 1)),
                np.ones((1, 1)),
                np.zeros((1, 1)),
                lim_low=90,
                lim_high=80,
            )

    def test_does_not_mutate_inputs(self):
        """The blend is pure — day and night arrays are left unchanged."""
        day = np.ones((2, 2))
        night = np.zeros((2, 2))
        day_night_blend(day, night, np.full((2, 2), 83.0))
        assert np.array_equal(day, np.ones((2, 2))), "day was mutated"
        assert np.array_equal(night, np.zeros((2, 2))), "night was mutated"

    def test_mismatched_day_night_shapes_raise(self):
        """Incompatible day/night shapes raise a broadcasting ValueError."""
        with pytest.raises(ValueError):
            day_night_blend(np.ones((2, 2)), np.ones((3, 3)), np.full((2, 2), 83.0))


class TestDayNightBlendDataset:
    """When given pyramids Datasets the blend returns a georeferenced Dataset."""

    @staticmethod
    def _ds(fill: float) -> Dataset:
        return Dataset.create_from_array(
            np.full((2, 2), fill), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
        )

    def test_dataset_inputs_return_dataset_with_geo(self):
        """Dataset day/night yield a Dataset carrying the same geotransform + CRS."""
        day = self._ds(5.0)
        night = self._ds(9.0)
        out = day_night_blend(day, night, np.zeros((2, 2)))
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS should be preserved, got {out.epsg}"
        assert out.geotransform == day.geotransform, "geotransform should be preserved"
        assert np.allclose(out.read_array(), 5.0), "all-day blend should equal day"

    def test_array_inputs_return_ndarray(self):
        """Plain arrays return an ndarray, not a Dataset."""
        out = day_night_blend(np.ones((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
        assert isinstance(out, np.ndarray), f"expected ndarray, got {type(out)}"

    def test_night_dataset_makes_result_a_dataset(self):
        """A Dataset supplied only as `night` still georeferences the result."""
        night = self._ds(9.0)
        out = day_night_blend(np.ones((2, 2)), night, np.full((2, 2), 180.0))
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS should be preserved, got {out.epsg}"
        assert np.allclose(out.read_array(), 9.0), "all-night blend should equal night"


class TestDayNightBlendKeepAlpha:
    """`keep_alpha=True` appends a validity-based coverage band."""

    def test_adds_alpha_band_to_rgb(self):
        """A 3-band blend gains a 4th (alpha) band."""
        out = day_night_blend(
            np.ones((3, 2, 2)), np.zeros((3, 2, 2)), np.zeros((2, 2)), keep_alpha=True
        )
        assert out.shape == (4, 2, 2), f"expected (4, 2, 2), got {out.shape}"

    def test_default_keeps_three_bands(self):
        """Without keep_alpha the output is unchanged (3-band)."""
        out = day_night_blend(np.ones((3, 2, 2)), np.zeros((3, 2, 2)), np.zeros((2, 2)))
        assert out.shape == (3, 2, 2), f"expected (3, 2, 2), got {out.shape}"

    def test_alpha_zero_only_when_all_inputs_nan(self):
        """Alpha is 0 where both inputs are NaN, 1 where either is finite."""
        day = np.array([[np.nan, 1.0]])
        night = np.array([[np.nan, 0.0]])
        out = day_night_blend(day, night, np.zeros((1, 2)), keep_alpha=True)
        assert out[-1][0, 0] == 0.0, "both-NaN pixel should be uncovered"
        assert out[-1][0, 1] == 1.0, "finite pixel should be covered"

    def test_dark_but_valid_pixel_is_covered(self):
        """A dark night pixel (day NaN, night finite) stays covered."""
        out = day_night_blend(
            np.array([[np.nan]]),
            np.array([[0.02]]),
            np.array([[180.0]]),
            keep_alpha=True,
        )
        assert out[-1][0, 0] == 1.0, "dark-but-valid pixel should be covered"

    def test_undefined_geometry_is_uncovered(self):
        """A NaN-SZA pixel (undefined geometry) is marked uncovered."""
        out = day_night_blend(
            np.array([[1.0]]),
            np.array([[0.0]]),
            np.array([[np.nan]]),
            keep_alpha=True,
        )
        assert out[-1][0, 0] == 0.0, "undefined-geometry pixel should be uncovered"

    def test_day_only_coverage_from_day_input(self):
        """In day_only mode the coverage band comes from the day input alone."""
        out = day_night_blend(
            np.array([[np.nan, 1.0]]),
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            mode="day_only",
            keep_alpha=True,
        )
        assert out[-1][0, 0] == 0.0, "NaN day pixel should be uncovered"
        assert out[-1][0, 1] == 1.0, "finite day pixel should be covered"

    def test_night_only_coverage_from_night_input(self):
        """In night_only mode the coverage band comes from the night input alone."""
        out = day_night_blend(
            np.array([[0.0, 0.0]]),
            np.array([[np.nan, 5.0]]),
            np.array([[180.0, 180.0]]),
            mode="night_only",
            keep_alpha=True,
        )
        assert out[-1][0, 0] == 0.0, "NaN night pixel should be uncovered"
        assert out[-1][0, 1] == 1.0, "finite night pixel should be covered"


class TestAsArray:
    """`_as_array` normalises arrays and Datasets to float ndarrays."""

    def test_ndarray_passthrough_as_float(self):
        """An integer array is returned as float."""
        out = _as_array(np.array([[1, 2], [3, 4]], dtype="int32"))
        assert out.dtype == float, f"expected float dtype, got {out.dtype}"

    def test_dataset_read_via_read_array(self):
        """A Dataset is read through read_array()."""
        ds = Dataset.create_from_array(
            np.full((2, 2), 7.0), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
        )
        assert np.allclose(_as_array(ds), 7.0), "Dataset values should be read"
