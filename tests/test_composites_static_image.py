"""Unit tests for `pyramids_eo.composites.background` (offline; downloads mocked)."""

from __future__ import annotations

import io
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from pyramids.dataset import Dataset

from pyramids_eo.composites import static_image
from pyramids_eo.composites.background import _cache_path, _download, _resolve_source
from pyramids_eo.errors import EOError


def _make_tif(path: Path, shape=(4, 4), epsg=4326, cell=1.0, tlc=(0.0, 4.0)) -> Path:
    """Write a small georeferenced GeoTIFF fixture and return its path."""
    arr = np.arange(shape[0] * shape[1], dtype=float).reshape(shape)
    ds = Dataset.create_from_array(arr, top_left_corner=tlc, cell_size=cell, epsg=epsg)
    ds.to_file(str(path))
    return path


class _FakeUrlopen:
    """A urlopen() stand-in whose context manager yields fixed bytes."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    def __enter__(self) -> io.BytesIO:
        return self._buffer

    def __exit__(self, *exc: object) -> bool:
        return False


class TestDownload:
    """`_download` streams a URL to disk atomically."""

    def test_writes_bytes_and_removes_part_file(self, tmp_path, monkeypatch):
        """The payload lands at the target and the .part temp file is gone.

        Test scenario:
            A mocked urlopen returns fixed bytes; _download writes them to the
            target and renames the .part file away.
        """
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda req, timeout=None: _FakeUrlopen(b"blob")
        )
        target = tmp_path / "img.tif"
        _download("https://example.com/img.tif", target, 5.0)
        assert target.read_bytes() == b"blob", "downloaded bytes mismatch"
        assert not target.with_name("img.tif.part").exists(), (
            ".part file should be gone"
        )

    def test_size_cap_aborts_and_cleans_part(self, tmp_path, monkeypatch):
        """A download over the byte cap raises and leaves no .part file."""
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda req, timeout=None: _FakeUrlopen(b"x" * 100),
        )
        target = tmp_path / "img.tif"
        with pytest.raises(EOError, match="download cap"):
            _download("https://example.com/img.tif", target, 5.0, max_bytes=10)
        assert not target.exists(), "oversized download should not be kept"
        assert not target.with_name("img.tif.part").exists(), ".part not cleaned"

    def test_part_removed_on_stream_error(self, tmp_path, monkeypatch):
        """A mid-stream read error leaves no orphan .part file."""

        class _Boom:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, size=-1):
                raise OSError("boom")

        monkeypatch.setattr(
            urllib.request, "urlopen", lambda req, timeout=None: _Boom()
        )
        target = tmp_path / "img.tif"
        with pytest.raises(OSError, match="boom"):
            _download("https://example.com/img.tif", target, 5.0)
        assert not target.with_name("img.tif.part").exists(), ".part not cleaned"


class TestResolveSource:
    """`_resolve_source` returns a local path, downloading a URL on demand."""

    def test_local_existing_path_returned(self, tmp_path):
        """An existing local path is returned unchanged."""
        p = tmp_path / "x.tif"
        p.write_bytes(b"data")
        assert _resolve_source(p, None, 5.0) == p, "local path should pass through"

    def test_missing_local_path_raises(self, tmp_path):
        """A missing local path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            _resolve_source(tmp_path / "nope.tif", None, 5.0)

    def test_url_downloaded_into_cache(self, tmp_path, monkeypatch):
        """A URL is downloaded into cache_dir under its basename."""
        calls = {"n": 0}

        def _fake_dl(url, target, timeout, max_bytes=None):
            calls["n"] += 1
            Path(target).write_bytes(b"downloaded")

        monkeypatch.setattr("pyramids_eo.composites.background._download", _fake_dl)
        out = _resolve_source("https://host/dir/bm.tif", tmp_path / "cache", 5.0)
        assert out == _cache_path(tmp_path / "cache", "https://host/dir/bm.tif")
        assert out.name.endswith("bm.tif"), f"basename lost in cache key: {out.name}"
        assert out.read_bytes() == b"downloaded", "cache file not written"
        assert calls["n"] == 1, "download should run once"

    def test_same_basename_different_urls_do_not_collide(self, tmp_path):
        """Two URLs sharing a basename map to distinct cache files."""
        cache = tmp_path / "cache"
        a = _cache_path(cache, "https://host/2016/BlackMarble.tif")
        b = _cache_path(cache, "https://host/2020/BlackMarble.tif")
        assert a != b, "same-basename URLs must not collide"

    def test_empty_url_path_gets_a_basename(self, tmp_path):
        """A URL with an empty path still yields a file (not the cache dir itself)."""
        out = _cache_path(tmp_path / "cache", "https://host")
        assert out != tmp_path / "cache", "empty basename must not be the cache dir"
        assert out.name.endswith("download"), f"unexpected fallback name {out.name}"

    def test_url_cached_skips_download(self, tmp_path, monkeypatch):
        """A non-empty cached file is reused without re-downloading."""
        cache = tmp_path / "cache"
        cache.mkdir()
        _cache_path(cache, "https://host/dir/bm.tif").write_bytes(b"cached")

        def _boom(url, target, timeout, max_bytes=None):
            raise AssertionError("download should not run for a cached file")

        monkeypatch.setattr("pyramids_eo.composites.background._download", _boom)
        out = _resolve_source("https://host/dir/bm.tif", cache, 5.0)
        assert out.read_bytes() == b"cached", "cached file should be reused"

    def test_refresh_forces_redownload(self, tmp_path, monkeypatch):
        """refresh=True re-downloads even when a cached file exists."""
        cache = tmp_path / "cache"
        cache.mkdir()
        _cache_path(cache, "https://host/bm.tif").write_bytes(b"stale")
        calls = {"n": 0}

        def _fake_dl(url, target, timeout, max_bytes=None):
            calls["n"] += 1
            Path(target).write_bytes(b"fresh")

        monkeypatch.setattr("pyramids_eo.composites.background._download", _fake_dl)
        out = _resolve_source("https://host/bm.tif", cache, 5.0, refresh=True)
        assert calls["n"] == 1, "refresh should force a re-download"
        assert out.read_bytes() == b"fresh", "cache should hold the fresh bytes"

    def test_default_cache_used_when_cache_dir_none(self, tmp_path, monkeypatch):
        """With cache_dir=None the module's default cache directory is used."""
        default = tmp_path / "default-cache"
        monkeypatch.setattr("pyramids_eo.composites.background._DEFAULT_CACHE", default)
        monkeypatch.setattr(
            "pyramids_eo.composites.background._download",
            lambda url, target, timeout, max_bytes=None: Path(target).write_bytes(b"x"),
        )
        out = _resolve_source("https://host/bm.tif", None, 5.0)
        assert out == _cache_path(default, "https://host/bm.tif"), (
            f"expected default cache path, got {out}"
        )

    def test_relative_local_path_accepted(self, tmp_path, monkeypatch):
        """A plain relative path is accepted (no satpy absolute-path trap)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "bm.tif").write_bytes(b"data")
        assert _resolve_source("bm.tif", None, 5.0) == Path("bm.tif"), (
            "a relative local path should resolve without requiring an absolute path"
        )


class TestStaticImage:
    """`static_image` loads a georeferenced image and optionally aligns it."""

    def test_local_image_native_grid(self, tmp_path):
        """A local image with no `like` is returned at its native grid."""
        src = _make_tif(tmp_path / "bm.tif", shape=(4, 4))
        out = static_image(src)
        assert isinstance(out, Dataset), f"expected a Dataset, got {type(out)}"
        assert (out.rows, out.columns) == (4, 4), f"native grid changed: {out.shape}"

    def test_like_aligns_to_target_grid(self, tmp_path):
        """With `like`, the image is warped/cropped onto the target's grid."""
        src = _make_tif(tmp_path / "bm.tif", shape=(4, 4), cell=1.0, tlc=(0.0, 4.0))
        like = Dataset.create_from_array(
            np.zeros((2, 2)), top_left_corner=(1.0, 3.0), cell_size=1.0, epsg=4326
        )
        out = static_image(src, like=like)
        assert (out.rows, out.columns) == (like.rows, like.columns), "not aligned"
        assert out.epsg == like.epsg, f"CRS not aligned: {out.epsg}"
        assert out.geotransform == like.geotransform, "geotransform not aligned"

    def test_missing_local_source_raises(self, tmp_path):
        """A missing local source raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            static_image(tmp_path / "absent.tif")

    def test_url_source_downloaded_then_cached(self, tmp_path, monkeypatch):
        """A URL source downloads once, then subsequent calls reuse the cache."""
        fixture = _make_tif(tmp_path / "fixture.tif", shape=(3, 3))
        calls = {"n": 0}

        def _fake_dl(url, target, timeout, max_bytes=None):
            calls["n"] += 1
            shutil.copyfile(fixture, target)

        monkeypatch.setattr("pyramids_eo.composites.background._download", _fake_dl)
        cache = tmp_path / "cache"
        first = static_image("https://host/bm.tif", cache_dir=cache)
        second = static_image("https://host/bm.tif", cache_dir=cache)
        assert isinstance(first, Dataset) and isinstance(second, Dataset)
        assert calls["n"] == 1, f"download should run once, ran {calls['n']} times"
        assert _cache_path(cache, "https://host/bm.tif").exists(), (
            "cache should persist"
        )
