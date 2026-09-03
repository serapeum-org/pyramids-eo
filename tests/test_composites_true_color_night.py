"""Unit tests for `night_ir` / `true_color_with_night_ir` (offline, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

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
            return Dataset.from_array(
                np.full((2, 2), v),
                geo_ref=GeoReference(
                    top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
                ),
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
            return Dataset.from_array(
                arr,
                geo_ref=GeoReference(
                    top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
                ),
            )

        day = ds(np.ones((3, 2, 2)))
        clouds = ds(night_ir(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))))
        bg = ds(np.zeros((3, 2, 2)))
        out = true_color_with_night_ir(day, clouds, bg, np.full((2, 2), 40.0))
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS not preserved, got {out.epsg}"


class TestTrueColorWithNightIrKeepAlpha:
    """`keep_alpha=True` adds a whole-disk coverage band from the FCI inputs."""

    def test_adds_alpha_band(self):
        """The 3-band composite gains a 4th coverage band."""
        day = np.ones((3, 2, 2))
        clouds = night_ir(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        bg = np.zeros((3, 2, 2))
        out = true_color_with_night_ir(
            day, clouds, bg, np.zeros((2, 2)), keep_alpha=True
        )
        assert out.shape == (4, 2, 2), f"expected (4, 2, 2), got {out.shape}"

    def test_default_stays_three_band(self):
        """Without keep_alpha the result is the plain 3-band composite."""
        day = np.ones((3, 2, 2))
        clouds = night_ir(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        out = true_color_with_night_ir(
            day, clouds, np.zeros((3, 2, 2)), np.zeros((2, 2))
        )
        assert out.shape == (3, 2, 2), f"expected (3, 2, 2), got {out.shape}"

    def test_offdisk_uncovered_despite_background(self):
        """Off-disk (both FCI inputs NaN) is uncovered even though bg fills RGB."""
        # Pixel 0: on the day disk. Pixel 1: off-disk (day + clouds both NaN),
        # but the global background is finite there.
        day = np.stack([np.array([[1.0, np.nan]])] * 3)
        clouds = night_ir(
            np.array([[0.3, np.nan]]),
            np.array([[0.3, np.nan]]),
            np.array([[0.3, np.nan]]),
        )
        bg = np.ones((3, 1, 2))
        out = true_color_with_night_ir(
            day, clouds, bg, np.zeros((1, 2)), keep_alpha=True
        )
        assert out[3][0, 0] == 1.0, "on-disk pixel should be covered"
        assert out[3][0, 1] == 0.0, "off-disk pixel should be uncovered"

    def test_night_side_covered_by_clouds(self):
        """On the night side (day true-colour NaN) the clouds carry coverage."""
        day = np.stack([np.full((1, 1), np.nan)] * 3)  # night side: no day reflectance
        clouds = night_ir(
            np.full((1, 1), 250.0), np.full((1, 1), 250.0), np.full((1, 1), 250.0)
        )
        bg = np.ones((3, 1, 1))
        out = true_color_with_night_ir(
            day, clouds, bg, np.full((1, 1), 180.0), keep_alpha=True
        )
        assert out[3][0, 0] == 1.0, "night-side disk pixel should be covered"

    def test_keep_alpha_dataset_returns_dataset(self):
        """keep_alpha with Dataset inputs still returns a georeferenced Dataset."""

        def ds(arr):
            return Dataset.from_array(
                arr,
                geo_ref=GeoReference(
                    top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326
                ),
            )

        day = ds(np.ones((3, 2, 2)))
        clouds = ds(night_ir(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))))
        bg = ds(np.zeros((3, 2, 2)))
        out = true_color_with_night_ir(
            day, clouds, bg, np.zeros((2, 2)), keep_alpha=True
        )
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert out.epsg == 4326, f"CRS not preserved, got {out.epsg}"
