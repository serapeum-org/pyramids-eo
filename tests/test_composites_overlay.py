"""Unit tests for `pyramids_eo.composites.alpha_overlay` (offline, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from pyramids_eo.composites import alpha_overlay


def _rgba(r, g, b, a, shape=(1, 1)):
    """Build a (4, H, W) RGBA image with constant channels."""
    return np.stack([np.full(shape, v, dtype=float) for v in (r, g, b, a)])


def _rgb(r, g, b, shape=(1, 1)):
    """Build a (3, H, W) RGB image with constant channels."""
    return np.stack([np.full(shape, v, dtype=float) for v in (r, g, b)])


class TestAlphaOverlay:
    """`alpha_overlay` composites an RGBA foreground over an RGB(A) background."""

    def test_opaque_foreground_hides_background(self):
        """A fully opaque foreground (alpha=1) replaces the background entirely."""
        out = alpha_overlay(_rgba(1.0, 0.0, 0.0, 1.0), _rgb(0.0, 0.0, 1.0))
        assert np.allclose(out.ravel(), [1.0, 0.0, 0.0]), (
            f"expected red, got {out.ravel()}"
        )

    def test_transparent_foreground_shows_background(self):
        """A fully transparent foreground (alpha=0) leaves the background visible."""
        out = alpha_overlay(_rgba(1.0, 0.0, 0.0, 0.0), _rgb(0.0, 0.0, 1.0))
        assert np.allclose(out.ravel(), [0.0, 0.0, 1.0]), (
            f"expected blue, got {out.ravel()}"
        )

    def test_half_alpha_blends_50_50(self):
        """Alpha=0.5 averages foreground and background per channel."""
        out = alpha_overlay(_rgba(1.0, 0.0, 0.0, 0.5), _rgb(0.0, 0.0, 1.0))
        assert np.allclose(out.ravel(), [0.5, 0.0, 0.5]), (
            f"expected 50/50, got {out.ravel()}"
        )

    def test_rgb_background_returns_three_bands(self):
        """An RGB background yields a 3-band RGB result."""
        out = alpha_overlay(_rgba(1, 1, 1, 0.5, (2, 3)), _rgb(0, 0, 0, (2, 3)))
        assert out.shape == (3, 2, 3), f"expected (3, 2, 3), got {out.shape}"

    def test_matches_explicit_over_formula(self):
        """The RGB result equals fg_rgb*fg_a + bg_rgb*(1 - fg_a)."""
        rng = np.random.default_rng(1337)
        fg = rng.random((4, 4, 5))
        bg = rng.random((3, 4, 5))
        expected = fg[:3] * fg[3] + bg * (1 - fg[3])
        assert np.allclose(alpha_overlay(fg, bg), expected), "over formula mismatch"

    def test_rgba_background_returns_four_bands(self):
        """An RGBA background yields a 4-band RGBA result."""
        out = alpha_overlay(_rgba(1, 0, 0, 0.5), _rgba(0, 0, 1, 1.0))
        assert out.shape[0] == 4, f"expected 4 bands, got {out.shape[0]}"

    def test_rgba_over_opaque_background_alpha_is_one(self):
        """Compositing over a fully opaque RGBA background gives out alpha 1."""
        out = alpha_overlay(_rgba(1, 0, 0, 0.3), _rgba(0, 0, 1, 1.0))
        assert out[3].item() == pytest.approx(1.0), (
            f"out alpha should be 1, got {out[3]}"
        )

    def test_rgba_over_rgba_over_operator(self):
        """RGBA-over-RGBA follows premultiplied 'over': a_out = fa + ba*(1-fa)."""
        fg = _rgba(1.0, 0.0, 0.0, 0.5)
        bg = _rgba(0.0, 0.0, 1.0, 0.5)
        out = alpha_overlay(fg, bg)
        a_out = 0.5 + 0.5 * (1 - 0.5)
        assert out[3].item() == pytest.approx(a_out), f"alpha mismatch: {out[3]}"
        expected_r = (1.0 * 0.5 + 0.0 * 0.5 * (1 - 0.5)) / a_out
        assert out[0].item() == pytest.approx(expected_r), f"red mismatch: {out[0]}"

    def test_fully_transparent_stack_yields_zero_rgb(self):
        """When both alphas are 0 the out alpha is 0 and RGB is left at 0 (no divide-by-zero)."""
        out = alpha_overlay(_rgba(1, 1, 1, 0.0), _rgba(1, 1, 1, 0.0))
        assert np.all(np.isfinite(out)), "result must stay finite when out alpha is 0"
        assert out[3].item() == pytest.approx(0.0), "out alpha should be 0"

    def test_does_not_mutate_inputs(self):
        """The overlay is pure — foreground and background are unchanged."""
        fg = _rgba(1, 0, 0, 0.5)
        bg = _rgb(0, 0, 1)
        fg_copy, bg_copy = fg.copy(), bg.copy()
        alpha_overlay(fg, bg)
        assert np.array_equal(fg, fg_copy), "foreground was mutated"
        assert np.array_equal(bg, bg_copy), "background was mutated"

    def test_foreground_without_alpha_raises(self):
        """A 3-band (no alpha) foreground is rejected."""
        foreground, background = _rgb(1, 0, 0), _rgb(0, 0, 1)
        with pytest.raises(ValueError, match="foreground"):
            alpha_overlay(foreground, background)

    def test_foreground_not_3d_raises(self):
        """A 2-D foreground is rejected."""
        foreground, background = np.ones((4, 4)), _rgb(0, 0, 1)
        with pytest.raises(ValueError, match="foreground"):
            alpha_overlay(foreground, background)

    def test_background_wrong_band_count_raises(self):
        """A background with an unsupported band count is rejected."""
        foreground, background = _rgba(1, 0, 0, 0.5), np.ones((2, 1, 1))
        with pytest.raises(ValueError, match="background"):
            alpha_overlay(foreground, background)


class TestAlphaOverlayDataset:
    """Dataset inputs yield a georeferenced Dataset result."""

    @staticmethod
    def _ds(arr: np.ndarray) -> Dataset:
        return Dataset.from_array(
            arr,
            geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
        )

    def test_dataset_inputs_return_dataset_with_geo(self):
        """A Dataset foreground/background returns a Dataset carrying the geo + CRS."""
        fg = self._ds(_rgba(1.0, 0.0, 0.0, 0.5, (2, 2)))
        bg = self._ds(_rgb(0.0, 0.0, 1.0, (2, 2)))
        out = alpha_overlay(fg, bg)
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS should be preserved, got {out.epsg}"
        assert out.geotransform == fg.geotransform, "geotransform should be preserved"
        assert np.allclose(out.read_array().reshape(3, -1)[:, 0], [0.5, 0.0, 0.5])

    def test_array_inputs_return_ndarray(self):
        """Plain arrays return an ndarray, not a Dataset."""
        out = alpha_overlay(_rgba(1, 0, 0, 0.5), _rgb(0, 0, 1))
        assert isinstance(out, np.ndarray), f"expected ndarray, got {type(out)}"
