"""Unit tests for `pyramids_eo.readers.harmonise` (offline; synthetic grids)."""

from __future__ import annotations

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.errors import ReaderError
from pyramids_eo.readers import harmonise


def _grid(shape, cell=1.0, tlc=(0.0, 4.0)) -> Dataset:
    """A pyramids Dataset on a chosen grid."""
    return Dataset.create_from_array(
        np.ones(shape, dtype=float), top_left_corner=tlc, cell_size=cell, epsg=4326
    )


class TestHarmonise:
    """`harmonise` aligns every band to the reference grid."""

    def test_dict_bands_aligned_to_reference(self):
        """A mapping of bands returns a mapping aligned to the reference grid."""
        reference = _grid((2, 2), cell=2.0)
        bands = {"hi": _grid((4, 4), cell=1.0), "lo": _grid((2, 2), cell=2.0)}
        out = harmonise(bands, reference)
        assert set(out) == {"hi", "lo"}, "band keys should be preserved"
        for name, ds in out.items():
            assert (ds.rows, ds.columns) == (2, 2), f"{name} not aligned: {ds.shape}"
            assert ds.geotransform == reference.geotransform, f"{name} geo mismatch"

    def test_list_bands_return_list(self):
        """An iterable of bands returns a list aligned to the reference grid."""
        reference = _grid((2, 2), cell=2.0)
        out = harmonise([_grid((4, 4), cell=1.0), _grid((2, 2), cell=2.0)], reference)
        assert isinstance(out, list) and len(out) == 2, "expected a list of two"
        assert all((d.rows, d.columns) == (2, 2) for d in out), "bands not aligned"

    def test_none_reference_raises(self):
        """A missing reference grid is a ReaderError."""
        with pytest.raises(ReaderError, match="reference"):
            harmonise([_grid((2, 2))], None)

    def test_empty_dict_raises(self):
        """An empty mapping of bands is a ReaderError."""
        with pytest.raises(ReaderError, match="no bands"):
            harmonise({}, _grid((2, 2)))

    def test_empty_list_raises(self):
        """An empty iterable of bands is a ReaderError."""
        with pytest.raises(ReaderError, match="no bands"):
            harmonise([], _grid((2, 2)))
