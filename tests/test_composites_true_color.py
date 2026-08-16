"""Unit tests for `pyramids_eo.composites.true_color` (offline, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.composites import true_color


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
        assert out.max() <= 1.0 and out.min() >= 0.0, f"not clipped: {out.ravel()}"

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
