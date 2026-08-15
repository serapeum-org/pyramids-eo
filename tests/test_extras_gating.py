"""Tests for the extras-gated skip machinery.

Covers ``tests/_marks.py`` (``_has``, ``EXTRA_MARKERS``) and
``tests/conftest.py`` (``pytest_collection_modifyitems``), proving that a
marked test is collected then skipped when its extra's headline dependency is
unavailable — without needing the real dependency to be absent.
"""

from __future__ import annotations

from tests import _marks
from tests._marks import EXTRA_MARKERS, _has
from tests.conftest import pytest_collection_modifyitems


class _FakeItem:
    """Minimal pytest item double recording markers added to it."""

    def __init__(self, keywords):
        self.keywords = set(keywords)
        self.added = []

    def add_marker(self, marker):
        """Record the marker and mirror pytest by exposing it as a keyword."""
        self.added.append(marker)
        self.keywords.add(marker.name)


class TestHas:
    """Tests for the ``_has`` optional-dependency probe."""

    def test_returns_true_for_importable_module(self):
        """A stdlib module that is always importable probes as available."""
        assert _has("os") is True, "os must be importable"

    def test_returns_true_only_when_all_present(self):
        """``_has`` is a conjunction: every named module must resolve."""
        assert _has("os", "sys") is True, "both stdlib modules resolve"
        assert _has("os", "totally_missing_pkg_xyz") is False, (
            "one missing fails the conjunction"
        )

    def test_returns_false_for_missing_module(self):
        """An unimportable top-level name probes as unavailable."""
        assert _has("totally_missing_pkg_xyz") is False, (
            "unknown module must be unavailable"
        )

    def test_returns_false_when_find_spec_raises(self, monkeypatch):
        """A ``ModuleNotFoundError`` from ``find_spec`` is swallowed to False."""

        def _boom(name):
            raise ModuleNotFoundError(name)

        monkeypatch.setattr(_marks.importlib.util, "find_spec", _boom)
        assert _has("anything") is False, "raised ModuleNotFoundError must map to False"


class TestExtraMarkers:
    """Tests for the ``EXTRA_MARKERS`` registry shape."""

    def test_plot_marker_registered(self):
        """``plot`` maps to a (bool, reason) pair mentioning the viz extra."""
        available, reason = EXTRA_MARKERS["plot"]
        assert isinstance(available, bool), (
            f"availability must be bool, got {type(available)}"
        )
        assert "viz" in reason, f"skip reason should name the extra, got: {reason}"


class TestPytestCollectionModifyItems:
    """Tests for the extras-gated collection hook."""

    def test_skips_marked_item_when_extra_unavailable(self, monkeypatch):
        """A ``plot`` item is skipped (with the registry reason) when unavailable."""
        monkeypatch.setattr(
            "tests.conftest.EXTRA_MARKERS",
            {"plot": (False, "requires the [viz] extra")},
        )
        item = _FakeItem({"plot"})
        pytest_collection_modifyitems(config=None, items=[item])
        assert len(item.added) == 1, (
            f"exactly one skip marker expected, got {item.added}"
        )
        assert item.added[0].name == "skip", (
            f"marker must be skip, got {item.added[0].name}"
        )
        assert item.added[0].kwargs["reason"] == "requires the [viz] extra", (
            "skip reason must propagate"
        )

    def test_does_not_skip_when_extra_available(self, monkeypatch):
        """A ``plot`` item is left untouched when the extra is available."""
        monkeypatch.setattr("tests.conftest.EXTRA_MARKERS", {"plot": (True, "unused")})
        item = _FakeItem({"plot"})
        pytest_collection_modifyitems(config=None, items=[item])
        assert item.added == [], (
            f"available extra must not add a skip, got {item.added}"
        )

    def test_ignores_unmarked_item(self, monkeypatch):
        """An item without the extra marker is never skipped, even when unavailable."""
        monkeypatch.setattr(
            "tests.conftest.EXTRA_MARKERS",
            {"plot": (False, "requires the [viz] extra")},
        )
        item = _FakeItem({"live"})
        pytest_collection_modifyitems(config=None, items=[item])
        assert item.added == [], f"unmarked item must not be skipped, got {item.added}"

    def test_handles_empty_item_list(self, monkeypatch):
        """The hook is a no-op on an empty collection."""
        monkeypatch.setattr(
            "tests.conftest.EXTRA_MARKERS",
            {"plot": (False, "requires the [viz] extra")},
        )
        pytest_collection_modifyitems(config=None, items=[])
