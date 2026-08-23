"""Live end-to-end composite tests: real EUMETSAT granules -> composites.

These validate the acceptance path of #40 — decode a real granule with the
native byte decoders, calibrate (reflectance / brightness temperature), and run
the pyramids-eo composites, without hand-writing a decoder. Gated by the `live`
marker; each skips unless its fixtures directory is provided
(`SEVIRI_FIXTURES_DIR` for the `.nat`, `FCI_FIXTURES_DIR` for the FDHSI chunks).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyramids_eo.composites import night_ir, true_color, true_color_with_night_ir
from pyramids_eo.sensors.readers import read_fci_l1c, read_seviri


def _seviri_granule() -> str:
    granules = sorted(
        Path(os.environ.get("SEVIRI_FIXTURES_DIR", "tests/data/seviri")).glob("*.nat")
    )
    if not granules:
        pytest.skip("real SEVIRI fixtures not available (set SEVIRI_FIXTURES_DIR)")
    return str(granules[0])


def _fci_chunks() -> list[str]:
    chunks = sorted(
        Path(os.environ.get("FCI_FIXTURES_DIR", "tests/data/fci_l1c")).glob("*BODY*.nc")
    )
    if len(chunks) < 1:
        pytest.skip("real FCI fixtures not available (set FCI_FIXTURES_DIR)")
    return [str(p) for p in chunks]


def _arr(dataset) -> np.ndarray:
    return np.asarray(dataset.read_array(), dtype=float)


@pytest.mark.live
def test_seviri_reflectance_is_physical():
    """The SEVIRI reflectance calibration lands in a physical BRF range.

    Validates the per-wavenumber solar-irradiance fix on real data: a daylit
    sub-satellite scene reflects a few % over ocean up to ~1 on bright cloud.
    """
    granule = _seviri_granule()
    refl = _arr(read_seviri(granule, "VIS006"))
    finite = refl[np.isfinite(refl)]
    assert 0.02 < np.median(finite) < 0.6, (
        f"VIS006 median reflectance off: {np.median(finite)}"
    )
    assert finite.max() < 1.15, f"VIS006 max reflectance unphysical: {finite.max()}"
    assert np.percentile(finite, 99) > 0.3, (
        "bright cloud should approach high reflectance"
    )


@pytest.mark.live
def test_seviri_true_color_from_real_granule():
    """A real `.nat` decodes + calibrates + composites into a true-colour RGB."""
    granule = _seviri_granule()
    red = _arr(read_seviri(granule, "VIS006"))
    veg = _arr(read_seviri(granule, "VIS008"))
    rgb = np.asarray(true_color(red, red, veg, gamma=2.2, clip=True))
    assert rgb.shape == (3, *red.shape), f"expected (3,H,W), got {rgb.shape}"
    assert np.isfinite(rgb).all(), "RGB should be finite after clip"
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0, "clipped RGB must be in [0, 1]"
    assert rgb.mean() > 0.05, "a daylit scene should carry real signal, not be black"


@pytest.mark.live
def test_fci_true_color_from_real_chunks():
    """Real FDHSI chunks decode + calibrate + composite into a true-colour RGB."""
    chunks = _fci_chunks()
    red = _arr(read_fci_l1c(chunks, "vis_06"))
    blue = _arr(read_fci_l1c(chunks, "vis_04"))
    veg = _arr(read_fci_l1c(chunks, "vis_08"))
    assert red.shape == blue.shape == veg.shape, "the RGB bands must share one grid"
    rgb = np.asarray(true_color(red, blue, veg, gamma=1.8, clip=True))
    assert rgb.shape == (3, *red.shape), f"expected (3,H,W), got {rgb.shape}"
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0, "clipped RGB must be in [0, 1]"
    assert np.isfinite(rgb).all() and rgb.max() > 0.0, "the strip should carry signal"


@pytest.mark.live
def test_seviri_true_color_with_night_ir_chain():
    """The full day/night chain runs on real day + night-IR imagery.

    Real true-colour day image and real night-IR cloud stack from the granule,
    blended by solar zenith angle over a background. Uses a synthetic background /
    SZA (both separately unit-tested) to stay network-free while proving the
    real calibrated bands flow through `true_color_with_night_ir`.
    """
    granule = _seviri_granule()
    red = _arr(read_seviri(granule, "VIS006"))
    height, width = red.shape

    day = true_color(
        red, red, _arr(read_seviri(granule, "VIS008")), gamma=2.2, clip=True
    )

    def _bt_norm(bt, lo=200.0, hi=300.0):
        return np.clip((hi - bt) / (hi - lo), 0.0, 1.0)

    night = night_ir(
        _bt_norm(_arr(read_seviri(granule, "IR_039"))),
        _bt_norm(_arr(read_seviri(granule, "IR_108"))),
        _bt_norm(_arr(read_seviri(granule, "IR_120"))),
    )
    background = np.full((3, height, width), 0.02)
    # SZA ramps west->east across the strip so one edge is full day, the other night.
    sza = np.broadcast_to(np.linspace(20.0, 95.0, width), (height, width))

    out = np.asarray(
        true_color_with_night_ir(np.asarray(day), np.asarray(night), background, sza)
    )
    assert out.shape == (3, height, width), f"expected (3,H,W), got {out.shape}"
    # Off-earth space corners are NaN in both day and night; the on-disk majority
    # must be finite and in range.
    assert np.isfinite(out).mean() > 0.5, "most (on-disk) pixels should be finite"
    finite = out[np.isfinite(out)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0, (
        "blended values must be in [0, 1]"
    )
    day_side = np.nanmean(out[:, :, : width // 5])
    night_side = np.nanmean(out[:, :, -width // 5 :])
    assert day_side > night_side, "the day edge should be brighter than the night edge"
