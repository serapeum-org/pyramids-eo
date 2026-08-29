"""Unit tests for `pyramids_eo.composites.true_color` (offline, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.composites import true_color
from pyramids_eo.composites.true_color import _ndvi_hybrid_green, _rayleigh_wants_role


class TestTrueColor:
    """`true_color` stacks red / synthetic-green / blue into an RGB image."""

    def test_shape_is_three_band(self):
        """The output has three bands (R, synthetic G, B)."""
        r = np.ones((2, 2))
        out = true_color(r, r, r)
        assert out.shape == (3, 2, 2), f"expected (3, 2, 2), got {out.shape}"

    def test_red_and_blue_pass_through(self):
        """Bands 0 and 2 are the untouched red and blue inputs."""
        red = np.full((1, 1), 0.3)
        blue = np.full((1, 1), 0.7)
        out = true_color(red, blue, np.zeros((1, 1)))
        assert out[0].item() == pytest.approx(0.3), "red band changed"
        assert out[2].item() == pytest.approx(0.7), "blue band changed"

    def test_synthetic_green_uses_cimss_weights(self):
        """Green = 0.45*red + 0.10*nir + 0.45*blue by default."""
        red, blue, nir = 0.2, 0.6, 0.9
        out = true_color(
            np.full((1, 1), red), np.full((1, 1), blue), np.full((1, 1), nir)
        )
        expected = 0.45 * red + 0.10 * nir + 0.45 * blue
        assert out[1].item() == pytest.approx(expected), f"green mismatch: {out[1]}"

    def test_custom_green_weights(self):
        """Custom (red, nir, blue) weights are applied to the green channel."""
        out = true_color(
            np.full((1, 1), 1.0),
            np.full((1, 1), 1.0),
            np.full((1, 1), 1.0),
            green_weights=(0.2, 0.5, 0.3),
        )
        assert out[1].item() == pytest.approx(1.0), "weights summing to 1 -> 1.0"

    def test_gamma_applied(self):
        """A gamma applies value ** (1 / gamma) to the RGB."""
        out = true_color(
            np.full((1, 1), 0.25),
            np.full((1, 1), 0.25),
            np.full((1, 1), 0.25),
            gamma=2.0,
        )
        assert out[0].item() == pytest.approx(0.25**0.5), "gamma not applied"

    def test_gamma_guards_negatives(self):
        """Gamma clamps negative inputs to 0 before the power (no NaN)."""
        out = true_color(
            np.full((1, 1), -0.1),
            np.full((1, 1), -0.1),
            np.full((1, 1), -0.1),
            gamma=2.0,
        )
        assert np.all(np.isfinite(out)), "gamma of a negative should stay finite"

    def test_clip_to_unit_range(self):
        """clip=True clamps the output to [0, 1]."""
        out = true_color(
            np.full((1, 1), 2.0),
            np.full((1, 1), -1.0),
            np.full((1, 1), 0.5),
            clip=True,
        )
        assert out.max() <= 1.0, f"not clipped high: {out.ravel()}"
        assert out.min() >= 0.0, f"not clipped low: {out.ravel()}"

    def test_dataset_inputs_return_dataset(self):
        """Dataset inputs yield a georeferenced Dataset."""

        def ds(v):
            return Dataset.create_from_array(
                np.full((2, 2), v), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
            )

        out = true_color(ds(0.3), ds(0.7), ds(0.5))
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS not preserved, got {out.epsg}"

    def test_array_inputs_return_ndarray(self):
        """Plain arrays return an ndarray."""
        out = true_color(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        assert isinstance(out, np.ndarray), f"expected ndarray, got {type(out)}"

    def test_default_mode_is_synthetic(self):
        """The default green_mode reproduces the CIMSS synthetic green."""
        r, b, n = 0.2, 0.6, 0.9
        out = true_color(np.full((1, 1), r), np.full((1, 1), b), np.full((1, 1), n))
        expected = 0.45 * r + 0.10 * n + 0.45 * b
        assert out[1].item() == pytest.approx(expected), "default should be synthetic"


class TestTrueColorGreenModes:
    """`green_mode` selects native / NDVI-hybrid green from a real band."""

    def test_native_green_passes_through(self):
        """green_mode='native' uses the supplied green band as the G channel."""
        out = true_color(
            np.full((1, 1), 0.2),
            np.full((1, 1), 0.6),
            np.full((1, 1), 0.9),
            green=np.full((1, 1), 0.44),
            green_mode="native",
        )
        assert out[1].item() == pytest.approx(0.44), f"native green changed: {out[1]}"

    def test_ndvi_hybrid_matches_helper(self):
        """green_mode='ndvi_hybrid' band equals the NDVI-blend helper."""
        r, b, n, g = 0.2, 0.6, 0.8, 0.5
        out = true_color(
            np.full((1, 1), r),
            np.full((1, 1), b),
            np.full((1, 1), n),
            green=np.full((1, 1), g),
            green_mode="ndvi_hybrid",
        )
        expected = _ndvi_hybrid_green(
            np.full((1, 1), g), np.full((1, 1), r), np.full((1, 1), n)
        )
        assert out[1].item() == pytest.approx(float(expected.item())), "hybrid mismatch"

    def test_ndvi_hybrid_between_green_and_nir(self):
        """The blended green lies between the native green and the NIR."""
        g, n = 0.5, 0.8
        out = _ndvi_hybrid_green(
            np.full((1, 1), g), np.full((1, 1), 0.2), np.full((1, 1), n)
        )
        assert g <= out.item() <= n, f"blend {out.item()} not between {g} and {n}"

    def test_native_without_green_raises(self):
        """green_mode='native' requires a green band."""
        with pytest.raises(ValueError, match="native"):
            true_color(
                np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), green_mode="native"
            )

    def test_ndvi_hybrid_without_green_raises(self):
        """green_mode='ndvi_hybrid' requires a green band."""
        with pytest.raises(ValueError, match="ndvi_hybrid"):
            true_color(
                np.ones((1, 1)),
                np.ones((1, 1)),
                np.ones((1, 1)),
                green_mode="ndvi_hybrid",
            )

    def test_unknown_green_mode_raises(self):
        """An unknown green_mode is rejected."""
        with pytest.raises(ValueError, match="green_mode"):
            true_color(
                np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), green_mode="nope"
            )

    def test_wrong_length_green_weights_raises(self):
        """A green_weights that is not a 3-tuple is rejected by name."""
        with pytest.raises(ValueError, match="green_weights"):
            true_color(
                np.ones((1, 1)),
                np.ones((1, 1)),
                np.ones((1, 1)),
                green_weights=(0.5, 0.5),
            )

    def test_ndvi_hybrid_strength_must_be_positive(self):
        """A non-positive NDVI strength is rejected."""
        with pytest.raises(ValueError, match="strength"):
            _ndvi_hybrid_green(
                np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), strength=0.0
            )

    def test_ndvi_hybrid_linear_strength_skips_sharpening(self):
        """strength=1.0 uses the raw NDVI (no non-linear sharpening)."""
        out = _ndvi_hybrid_green(
            np.full((1, 1), 0.5),
            np.full((1, 1), 0.2),
            np.full((1, 1), 0.8),
            strength=1.0,
        )
        # NDVI=0.6 -> fraction=0.6*(0.05-0.15)+0.15=0.09 -> 0.91*0.5 + 0.09*0.8
        assert out.item() == pytest.approx(0.527), f"linear-strength blend wrong: {out}"

    def test_ndvi_hybrid_accepts_integer_inputs(self):
        """Integer-dtype inputs do not crash (a float accumulator is forced)."""
        out = _ndvi_hybrid_green(np.array([[1]]), np.array([[2]]), np.array([[4]]))
        assert np.isfinite(out).all(), f"integer inputs should give finite green: {out}"

    def test_ndvi_limits_wrong_length_raises(self):
        """A limits that is not a 2-tuple is rejected by name."""
        with pytest.raises(ValueError, match="limits"):
            _ndvi_hybrid_green(
                np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), limits=(0.1,)
            )

    def test_true_color_bad_ndvi_limits_raises(self):
        """A wrong-length ndvi_limits propagates a named error from true_color."""
        with pytest.raises(ValueError, match="limits"):
            true_color(
                np.ones((1, 1)),
                np.ones((1, 1)),
                np.ones((1, 1)),
                green=np.ones((1, 1)),
                green_mode="ndvi_hybrid",
                ndvi_limits=(0.1,),
            )


class TestTrueColorRayleigh:
    """The `rayleigh` hook corrects each solar band before green synthesis."""

    def test_rayleigh_applied_to_bands(self):
        """A supplied callable is applied to red/blue before stacking."""
        out = true_color(
            np.full((1, 1), 0.5),
            np.full((1, 1), 0.5),
            np.full((1, 1), 0.5),
            rayleigh=lambda a: a - 0.1,
        )
        assert out[0].item() == pytest.approx(0.4), "rayleigh not applied to red"
        assert out[2].item() == pytest.approx(0.4), "rayleigh not applied to blue"

    def test_default_none_is_byte_identical(self):
        """rayleigh=None leaves the composite exactly as the plain call."""
        r = np.full((2, 2), 0.3)
        base = true_color(r, r, r)
        same = true_color(r, r, r, rayleigh=None)
        assert np.array_equal(base, same), "rayleigh=None changed the output"

    def test_rayleigh_corrects_native_green(self):
        """The hook also corrects a native green band before it is used."""
        out = true_color(
            np.full((1, 1), 0.5),
            np.full((1, 1), 0.5),
            np.full((1, 1), 0.5),
            green=np.full((1, 1), 0.6),
            green_mode="native",
            rayleigh=lambda a: a - 0.1,
        )
        assert out[1].item() == pytest.approx(0.5), (
            "native green not rayleigh-corrected"
        )

    def test_role_is_passed_to_a_role_aware_callable(self):
        """A role-aware callable receives a distinct role for each solar band."""
        seen = []

        def correct(band, *, role):
            seen.append(role)
            return band

        true_color(
            np.full((1, 1), 0.5),
            np.full((1, 1), 0.5),
            np.full((1, 1), 0.5),
            green=np.full((1, 1), 0.6),
            green_mode="native",
            rayleigh=correct,
        )
        assert seen == ["red", "blue", "nir", "green"], f"roles seen: {seen}"

    def test_role_aware_callable_can_decline_a_band(self):
        """Returning the band unchanged for a role declines that band's correction."""

        def correct(band, *, role):
            return band if role == "nir" else band - 0.1

        out = true_color(
            np.full((1, 1), 0.5),
            np.full((1, 1), 0.5),
            np.full((1, 1), 0.8),
            rayleigh=correct,
        )
        assert out[0].item() == pytest.approx(0.4), "red should be corrected"
        # synthetic green uses the (declined, so uncorrected) nir=0.8:
        # 0.45*0.4 + 0.10*0.8 + 0.45*0.4 = 0.44
        assert out[1].item() == pytest.approx(0.44), "declined nir should reach green"


class TestRayleighWantsRole:
    """`_rayleigh_wants_role` probes the callable once for the `role` keyword."""

    def test_role_keyword_callable_is_rich(self):
        """A callable declaring `role` is detected as role-aware."""

        def f(band, *, role):
            return band

        assert _rayleigh_wants_role(f) is True, "role-kwarg callable not detected"

    def test_var_keyword_callable_is_rich(self):
        """A callable taking `**kwargs` is treated as role-aware."""

        def f(band, **kwargs):
            return band

        assert _rayleigh_wants_role(f) is True, "**kwargs callable not detected"

    def test_plain_band_callable_is_legacy(self):
        """A `(band)`-only callable is the legacy form."""
        assert _rayleigh_wants_role(lambda a: a) is False, "plain callable mis-detected"

    def test_uninspectable_callable_is_legacy(self):
        """An un-introspectable callable (a numpy ufunc) falls back to legacy."""
        assert _rayleigh_wants_role(np.negative) is False, "ufunc should be legacy"
