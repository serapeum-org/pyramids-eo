"""Unit tests for `night_ir` / `true_color_with_night_ir` (offline, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.composites import (
    alpha_overlay,
    day_night_blend,
    night_ir,
    true_color_with_night_ir,
)


class TestNightIr:
    """`night_ir` stacks three IR bands into an RGBA image."""

    def test_rgba_shape(self):
        """Three IR bands produce a 4-band RGBA image."""
        band = np.ones((2, 2))
        out = night_ir(band, band, band)
        assert out.shape == (4, 2, 2), f"expected (4, 2, 2), got {out.shape}"

    def test_default_alpha_is_opaque(self):
        """Without an explicit alpha the alpha channel is all ones."""
        out = night_ir(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        assert np.allclose(out[3], 1.0), "default alpha should be opaque"

    def test_bands_map_to_rgb_in_order(self):
        """red/green/blue land on channels 0/1/2."""
        out = night_ir(np.full((1, 1), 0.1), np.full((1, 1), 0.2), np.full((1, 1), 0.3))
        assert out[0].item() == pytest.approx(0.1), "red channel wrong"
        assert out[1].item() == pytest.approx(0.2), "green channel wrong"
        assert out[2].item() == pytest.approx(0.3), "blue channel wrong"

    def test_custom_alpha_used(self):
        """An explicit alpha array becomes the alpha channel."""
        alpha = np.array([[0.25, 0.5]])
        out = night_ir(np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)), alpha=alpha)
        assert np.allclose(out[3], alpha), "custom alpha not applied"

    def test_dataset_inputs_return_dataset(self):
        """Dataset inputs yield a georeferenced Dataset."""

        def ds(v):
            return Dataset.create_from_array(
                np.full((2, 2), v), top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
            )

        out = night_ir(ds(0.1), ds(0.2), ds(0.3))
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"


class TestTrueColorWithNightIr:
    """`true_color_with_night_ir` overlays IR clouds then blends by SZA."""

    def test_all_day_returns_day_image(self):
        """With the Sun overhead the result is the day true-colour image."""
        day = np.stack([np.full((2, 2), c) for c in (0.2, 0.4, 0.6)])
        clouds = night_ir(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
        bg = np.ones((3, 2, 2))
        out = true_color_with_night_ir(day, clouds, bg, np.zeros((2, 2)))
        assert np.allclose(out, day), f"all-day result should equal day, got {out}"

    def test_all_night_returns_overlaid_night(self):
        """At full night the result equals the IR-clouds-over-background overlay."""
        day = np.ones((3, 2, 2))
        clouds = night_ir(
            np.full((2, 2), 0.3),
            np.full((2, 2), 0.4),
            np.full((2, 2), 0.5),
            alpha=np.full((2, 2), 0.5),
        )
        bg = np.zeros((3, 2, 2))
        out = true_color_with_night_ir(day, clouds, bg, np.full((2, 2), 180.0))
        expected = alpha_overlay(clouds, bg)
        assert np.allclose(out, expected), "night result should equal the overlay"

    def test_matches_manual_composition(self):
        """The helper equals alpha_overlay + day_night_blend done by hand."""
        rng = np.random.default_rng(1337)
        day = rng.random((3, 4, 4))
        clouds = night_ir(
            rng.random((4, 4)),
            rng.random((4, 4)),
            rng.random((4, 4)),
            alpha=rng.random((4, 4)),
        )
        bg = rng.random((3, 4, 4))
        sza = rng.uniform(0, 180, (4, 4))
        manual = day_night_blend(day, alpha_overlay(clouds, bg), sza)
        assert np.allclose(true_color_with_night_ir(day, clouds, bg, sza), manual)

    def test_dataset_inputs_return_dataset(self):
        """Dataset inputs flow through to a georeferenced Dataset result."""

        def ds(arr):
            return Dataset.create_from_array(
                arr, top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
            )

        day = ds(np.ones((3, 2, 2)))
        clouds = ds(night_ir(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))))
        bg = ds(np.zeros((3, 2, 2)))
        out = true_color_with_night_ir(day, clouds, bg, np.full((2, 2), 40.0))
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS not preserved, got {out.epsg}"
