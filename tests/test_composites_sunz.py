"""Unit tests for `sunz_correct` / `sunz_reduce` (offline, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.composites import sunz_correct, sunz_reduce


class TestSunzCorrect:
    """`sunz_correct` divides a band by cos(SZA), capped near the terminator."""

    def test_overhead_sun_unchanged(self):
        """SZA 0 gives cos=1, so the band passes through unchanged."""
        out = sunz_correct(np.array([1.0]), np.array([0.0]))
        assert out[0] == pytest.approx(1.0), f"overhead sun changed: {out}"

    def test_sixty_degrees_doubles(self):
        """SZA 60 gives cos=0.5, so the correction factor is 1/0.5 = 2."""
        out = sunz_correct(np.array([1.0]), np.array([60.0]))
        assert out[0] == pytest.approx(2.0), f"1/cos(60) should be 2: {out}"

    def test_correction_increases_with_sza(self):
        """Below the limit the correction grows monotonically with SZA."""
        out = sunz_correct(np.array([1.0, 1.0, 1.0]), np.array([0.0, 45.0, 80.0]))
        assert np.all(np.diff(out) > 0), f"correction not increasing: {out}"

    def test_taper_decreases_from_limit_to_max_sza(self):
        """Past the limit the factor falls from its peak to 0 at max_sza."""
        out = sunz_correct(np.ones(4), np.array([88.0, 90.0, 92.0, 95.0]))
        assert np.all(np.diff(out) < 0), f"taper not decreasing: {out}"

    def test_interior_taper_matches_reference(self):
        """At SZA 91 the tapered factor matches the reference formula."""
        out = sunz_correct(np.array([1.0]), np.array([91.0]))
        ramp = (91.0 - 88.0) / (95.0 - 88.0)
        expected = (1.0 - np.log2(ramp + 1.0)) / np.cos(np.deg2rad(88.0))
        assert out[0] == pytest.approx(expected, rel=1e-6), f"taper parity off: {out}"

    def test_broadcasts_over_a_2d_grid(self):
        """A 2-D band and matching 2-D SZA grid preserve shape per-pixel."""
        band = np.ones((2, 2))
        sza = np.array([[0.0, 60.0], [90.0, np.nan]])
        out = sunz_correct(band, sza)
        assert out.shape == (2, 2), f"shape not preserved: {out.shape}"
        assert out[0, 0] == pytest.approx(1.0), "overhead pixel wrong"
        assert out[0, 1] == pytest.approx(2.0), "60deg pixel wrong"
        assert out[1, 1] == pytest.approx(0.0), "NaN pixel should be 0"

    def test_finite_at_terminator(self):
        """At SZA 90 the capped factor stays finite (no 1/0 blow-up)."""
        out = sunz_correct(np.array([1.0]), np.array([90.0]))
        assert np.isfinite(out).all(), f"correction not finite at 90deg: {out}"

    def test_continuous_at_limit(self):
        """At SZA == correction_limit the factor equals 1/cos(limit)."""
        out = sunz_correct(np.array([1.0]), np.array([88.0]))
        assert out[0] == pytest.approx(1.0 / np.cos(np.deg2rad(88.0)), rel=1e-6), (
            f"factor at the limit should be 1/cos(88): {out}"
        )

    def test_zero_at_max_sza(self):
        """At SZA == max_sza the taper has fallen to 0."""
        out = sunz_correct(np.array([1.0]), np.array([95.0]))
        assert out[0] == pytest.approx(0.0, abs=1e-9), f"should be 0 at max_sza: {out}"

    def test_nan_sza_gives_zero(self):
        """A NaN SZA (night / off-disk) yields a 0 correction."""
        out = sunz_correct(np.array([1.0]), np.array([np.nan]))
        assert out[0] == pytest.approx(0.0), f"NaN SZA should give 0: {out}"

    def test_max_sza_none_holds_constant(self):
        """With max_sza=None the factor is capped constant beyond the limit."""
        out = sunz_correct(np.array([1.0, 1.0]), np.array([90.0, 93.0]), max_sza=None)
        cap = 1.0 / np.cos(np.deg2rad(88.0))
        assert out[0] == pytest.approx(cap, rel=1e-6), f"not capped constant: {out}"
        assert out[1] == pytest.approx(cap, rel=1e-6), f"not capped constant: {out}"

    def test_bad_max_sza_raises(self):
        """max_sza not greater than correction_limit is rejected."""
        with pytest.raises(ValueError, match="max_sza"):
            sunz_correct(
                np.array([1.0]), np.array([10.0]), correction_limit=85.0, max_sza=80.0
            )

    def test_correction_limit_at_or_beyond_90_raises(self):
        """correction_limit >= 90 (cos <= 0, cap explodes) is rejected."""
        with pytest.raises(ValueError, match="correction_limit"):
            sunz_correct(np.array([1.0]), np.array([80.0]), correction_limit=90.0)

    def test_negative_correction_limit_raises(self):
        """A negative correction_limit is rejected."""
        with pytest.raises(ValueError, match="correction_limit"):
            sunz_correct(np.array([1.0]), np.array([0.0]), correction_limit=-1.0)

    def test_nan_band_stays_nan(self):
        """A NaN band value is preserved as NaN (0 * NaN), not blacked to 0."""
        out = sunz_correct(np.array([np.nan]), np.array([120.0]))
        assert np.isnan(out[0]), f"NaN band should stay NaN (nodata): {out}"

    def test_multiband_broadcasts_over_bands(self):
        """A (band, H, W) band applies the 2-D SZA factor to every band."""
        band = np.ones((3, 1, 2))
        sza = np.array([[0.0, 60.0]])
        out = sunz_correct(band, sza)
        assert out.shape == (3, 1, 2), f"band shape not preserved: {out.shape}"
        assert np.allclose(out[:, 0, 0], 1.0), "overhead column wrong per band"
        assert np.allclose(out[:, 0, 1], 2.0), "60deg column wrong per band"

    def test_interior_taper_independent_parity(self):
        """At SZA 91 the factor matches a log-base-change independent of np.log2."""
        out = sunz_correct(np.array([1.0]), np.array([91.0]))
        ramp = (91.0 - 88.0) / (95.0 - 88.0)
        grad = 1.0 - np.log(ramp + 1.0) / np.log(2.0)
        expected = grad / np.cos(np.deg2rad(88.0))
        assert out[0] == pytest.approx(expected, rel=1e-9), (
            f"independent parity off: {out}"
        )

    def test_taper_hardcoded_oracle_at_91(self):
        """At SZA 91 the default-parameter factor is ~13.909 (fixed oracle)."""
        out = sunz_correct(np.array([1.0]), np.array([91.0]))
        assert out[0] == pytest.approx(13.909, abs=0.01), f"taper oracle off: {out}"

    def test_scalar_input_returns_0d_array(self):
        """A scalar band/sza yields a 0-d array (the documented contract)."""
        out = sunz_correct(1.0, 60.0)
        assert out.shape == (), f"scalar input should give a 0-d array: {out.shape}"
        assert float(out) == pytest.approx(2.0), f"scalar value wrong: {out}"

    def test_max_sza_none_with_nan_sza_is_zero(self):
        """With max_sza=None a NaN SZA still yields a 0 factor (NaN override wins)."""
        out = sunz_correct(np.array([1.0]), np.array([np.nan]), max_sza=None)
        assert out[0] == pytest.approx(0.0), (
            f"NaN under max_sza=None should be 0: {out}"
        )

    def test_does_not_mutate_inputs(self):
        """The correction is pure — band and sza arrays are left unchanged."""
        band = np.array([1.0, 1.0])
        sza = np.array([0.0, 60.0])
        sunz_correct(band, sza)
        assert np.array_equal(band, [1.0, 1.0]), "band was mutated"
        assert np.array_equal(sza, [0.0, 60.0]), "sza was mutated"

    def test_dataset_in_dataset_out(self):
        """A Dataset band yields a georeferenced Dataset."""
        ds = Dataset.create_from_array(
            np.full((2, 2), 1.0), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
        )
        out = sunz_correct(ds, np.zeros((2, 2)))
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS not preserved, got {out.epsg}"
        assert out.geotransform == ds.geotransform, "geotransform not preserved"


class TestSunzReduce:
    """`sunz_reduce` tapers a band's signal toward the terminator."""

    def test_unchanged_below_limit(self):
        """Below correction_limit the signal is unchanged (factor 1)."""
        out = sunz_reduce(np.array([1.0, 1.0]), np.array([0.0, 70.0]))
        assert np.allclose(out, 1.0), f"signal below the limit changed: {out}"

    def test_zero_at_max_sza(self):
        """At SZA == max_sza the signal is fully reduced to 0."""
        out = sunz_reduce(np.array([1.0]), np.array([90.0]))
        assert out[0] == pytest.approx(0.0, abs=1e-9), f"not fully reduced: {out}"

    def test_dimmed_inside_band(self):
        """Inside [limit, max_sza] the factor is strictly between 0 and 1."""
        out = sunz_reduce(np.array([1.0]), np.array([85.0]))
        assert 0.0 < out[0] < 1.0, f"factor not in (0, 1) inside the band: {out}"

    def test_broadcasts_over_a_2d_grid(self):
        """A 2-D band and matching 2-D SZA grid reduce per-pixel by region."""
        band = np.ones((2, 2))
        sza = np.array([[70.0, 85.0], [90.0, np.nan]])
        out = sunz_reduce(band, sza)
        assert out.shape == (2, 2), f"shape not preserved: {out.shape}"
        assert out[0, 0] == pytest.approx(1.0), "below-limit pixel should be unchanged"
        assert 0.0 < out[0, 1] < 1.0, "in-band pixel should be dimmed"
        assert out[1, 0] == pytest.approx(0.0, abs=1e-9), "max_sza pixel should be 0"
        assert out[1, 1] == pytest.approx(0.0), "NaN pixel should be 0"

    def test_matches_reference_value_at_85(self):
        """The default-parameter factor at SZA 85 matches the reference formula."""
        out = sunz_reduce(np.array([1.0]), np.array([85.0]))
        assert out[0] == pytest.approx(0.390, abs=0.005), f"parity value off: {out}"

    def test_monotonic_decreasing_through_band(self):
        """The factor decreases monotonically from the limit to max_sza."""
        out = sunz_reduce(
            np.array([1.0, 1.0, 1.0, 1.0]), np.array([80.0, 84.0, 87.0, 90.0])
        )
        assert np.all(np.diff(out) <= 0), f"reduction not monotone: {out}"

    def test_strength_one_has_no_sigmoid(self):
        """strength=1.0 leaves the plain inverted-log2 ramp (sigmoid is identity)."""
        out = sunz_reduce(np.array([1.0]), np.array([85.0]), strength=1.0)
        expected = 1.0 - np.log2(0.5 + 1.0)  # ramp=0.5 at sza=85
        assert out[0] == pytest.approx(expected, rel=1e-6), f"strength=1 off: {out}"

    def test_nan_sza_gives_zero(self):
        """A NaN SZA yields a 0 factor (fully masked)."""
        out = sunz_reduce(np.array([1.0]), np.array([np.nan]))
        assert out[0] == pytest.approx(0.0), f"NaN SZA should give 0: {out}"

    def test_bad_max_sza_raises(self):
        """max_sza not greater than correction_limit is rejected."""
        with pytest.raises(ValueError, match="max_sza"):
            sunz_reduce(
                np.array([1.0]), np.array([10.0]), correction_limit=90.0, max_sza=80.0
            )

    def test_non_positive_strength_raises(self):
        """A non-positive strength is rejected."""
        with pytest.raises(ValueError, match="strength"):
            sunz_reduce(np.array([1.0]), np.array([85.0]), strength=0.0)

    def test_max_sza_none_raises(self):
        """max_sza=None is rejected with a clear message (unlike sunz_correct)."""
        with pytest.raises(ValueError, match="max_sza"):
            sunz_reduce(np.array([1.0]), np.array([85.0]), max_sza=None)

    def test_negative_correction_limit_raises(self):
        """A negative correction_limit is rejected."""
        with pytest.raises(ValueError, match="correction_limit"):
            sunz_reduce(np.array([1.0]), np.array([85.0]), correction_limit=-1.0)

    def test_nan_band_stays_nan(self):
        """A NaN band value is preserved as NaN (0 * NaN), not blacked to 0."""
        out = sunz_reduce(np.array([np.nan]), np.array([90.0]))
        assert np.isnan(out[0]), f"NaN band should stay NaN (nodata): {out}"

    def test_fractional_strength_softens_the_ramp(self):
        """A strength < 1 keeps the factor in (0, 1) inside the band (softer curve)."""
        out = sunz_reduce(np.array([1.0]), np.array([85.0]), strength=0.5)
        assert 0.0 < out[0] < 1.0, f"fractional strength out of range: {out}"

    def test_does_not_mutate_inputs(self):
        """The reduction is pure — band and sza arrays are left unchanged."""
        band = np.array([1.0, 1.0])
        sza = np.array([85.0, 90.0])
        sunz_reduce(band, sza)
        assert np.array_equal(band, [1.0, 1.0]), "band was mutated"
        assert np.array_equal(sza, [85.0, 90.0]), "sza was mutated"

    def test_dataset_in_dataset_out(self):
        """A Dataset band yields a georeferenced Dataset."""
        ds = Dataset.create_from_array(
            np.full((2, 2), 1.0), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
        )
        out = sunz_reduce(ds, np.full((2, 2), 70.0))
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS not preserved, got {out.epsg}"
        assert out.geotransform == ds.geotransform, "geotransform not preserved"


class TestSunzCompose:
    """`sunz_correct` then `sunz_reduce` is the reference solar-band pipeline."""

    def test_reduce_pulls_near_terminator_toward_zero(self):
        """After correcting, reducing drives a pixel near max_sza toward 0."""
        sza = np.array([89.9])
        corrected = sunz_correct(np.array([1.0]), sza)
        reduced = sunz_reduce(corrected, sza)
        assert reduced[0] < 0.1 * corrected[0], (
            f"reduction should dim the corrected near-terminator value: "
            f"corrected={corrected}, reduced={reduced}"
        )
