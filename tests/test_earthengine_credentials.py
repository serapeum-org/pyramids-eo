"""Unit tests for :class:`EarthEngineCredentials` (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids_eo.earthengine import EarthEngineCredentials
from pyramids_eo.earthengine.credentials import GOOGLE_APPLICATION_CREDENTIALS
from pyramids_eo.errors import AuthenticationError


def _make_key(tmp_path: Path) -> Path:
    key = tmp_path / "service-account.json"
    key.write_text('{"type": "service_account"}', encoding="utf-8")
    return key


def test_from_service_account_sets_gdal_env(tmp_path: Path) -> None:
    key = _make_key(tmp_path)
    creds = EarthEngineCredentials.from_service_account(key)
    assert creds.service_account_path == key
    assert creds.gdal_env() == {GOOGLE_APPLICATION_CREDENTIALS: str(key)}


def test_missing_service_account_raises(tmp_path: Path) -> None:
    with pytest.raises(AuthenticationError, match="not found"):
        EarthEngineCredentials.from_service_account(tmp_path / "nope.json")


def test_application_default_has_empty_env() -> None:
    creds = EarthEngineCredentials.application_default()
    assert creds.service_account_path is None
    assert creds.gdal_env() == {}


def test_coerce_passthrough_and_conversion(tmp_path: Path) -> None:
    key = _make_key(tmp_path)
    instance = EarthEngineCredentials.application_default()
    assert EarthEngineCredentials.coerce(instance) is instance
    assert EarthEngineCredentials.coerce(None).service_account_path is None
    assert EarthEngineCredentials.coerce(str(key)).service_account_path == key


def test_activate_sets_and_restores_config(tmp_path: Path) -> None:
    from osgeo import gdal

    key = _make_key(tmp_path)
    creds = EarthEngineCredentials.from_service_account(key)
    before = gdal.GetConfigOption(GOOGLE_APPLICATION_CREDENTIALS, None)
    with creds.activate():
        assert gdal.GetConfigOption(GOOGLE_APPLICATION_CREDENTIALS, None) == str(key)
    assert gdal.GetConfigOption(GOOGLE_APPLICATION_CREDENTIALS, None) == before
