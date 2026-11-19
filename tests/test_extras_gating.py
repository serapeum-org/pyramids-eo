"""Tests for the extras-gated skip machinery.

Covers ``tests/_marks.py`` (``_has``, ``EXTRA_MARKERS``) and
``tests/conftest.py`` (``pytest_collection_modifyitems``), proving that a
marked test is collected then skipped when its extra's headline dependency is
unavailable — without needing the real dependency to be absent.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import requires as _dist_requires

import pytest

from tests import _marks
from tests._marks import EXTRA_MARKERS, HAS_DASK, _has
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


def _lazy_extra_requirements():
    """This package's own ``[lazy]`` extra requirements, from installed metadata.

    Returns:
        The ``Requires-Dist`` entries of ``pyramids-eo`` that are gated on
        ``extra == "lazy"``, or ``None`` when the distribution is not installed
        (an editable/source tree without metadata).
    """
    try:
        reqs = _dist_requires("pyramids-eo") or []
    except PackageNotFoundError:  # pragma: no cover - always installed under pytest
        return None
    return [r for r in reqs if 'extra == "lazy"' in r]


class TestLazyExtraContract:
    """The ``[lazy]`` extra must deliver the dask stack, not a dead core extra.

    This is the regression guard for the break this extra was repaired from:
    ``pyramids-core`` 0.80.0 deleted the ``[lazy]`` extra that
    ``pyramids-eo[lazy]`` used to forward to, so the old
    ``pyramids-core[lazy]`` requirement resolved to a pip warning and no dask —
    ``pip install pyramids-eo[lazy]`` silently installed nothing a
    ``DatasetCollection`` reduction (#131) needs. These assertions read declared
    metadata only, so they fail the moment the extra regresses to a form that
    cannot bring dask, without needing a full resolve.
    """

    def test_lazy_extra_targets_the_pyramids_lazy_plugin(self):
        """``[lazy]`` requires ``pyramids-lazy`` and never the dead core extra."""
        reqs = _lazy_extra_requirements()
        assert reqs, "pyramids-eo must declare a non-empty [lazy] extra"
        names = [r.split(";")[0].split()[0].lower().replace("_", "-") for r in reqs]
        assert any(n.startswith("pyramids-lazy") for n in names), (
            f"[lazy] must forward to the pyramids-lazy plugin, got {reqs}"
        )
        assert not any("pyramids-core[lazy]" in r.replace(" ", "") for r in reqs), (
            "[lazy] must not name pyramids-core[lazy]: 0.80.0 deleted that extra, so "
            f"it resolves to no backend (the bug this guards). Got {reqs}"
        )

    @pytest.mark.lazy
    def test_lazy_plugin_brings_the_dask_array_stack(self):
        """With the extra installed, ``dask.array`` — what #131 reductions read — resolves.

        Marked ``lazy`` so it runs only where the extra is present; the collection
        hook skips it otherwise, keeping this free of an in-body availability branch.
        """
        assert HAS_DASK, (
            "the lazy extra is installed in this environment, so dask.array must import; "
            "if this fails, pyramids-lazy stopped delivering the dask stack"
        )
        import dask.array as da  # noqa: F401

        pl = [
            r
            for r in (_dist_requires("pyramids-lazy") or [])
            if "extra ==" not in r and r.split()[0].lower().startswith("dask")
        ]
        assert pl, f"pyramids-lazy must hard-depend on dask, got {pl!r}"
