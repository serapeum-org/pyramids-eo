"""Unit tests for :mod:`pyramids_eo.earthengine.credentials` (no network).

Covers :class:`EarthEngineCredentials` (constructor, named constructors,
``coerce``, ``gdal_env``, the ``activate`` context manager, ``__repr__``, and
inline-JSON credentials). The only GDAL touch is ``activate``'s config get/set,
which is exercised without opening any dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids_eo.earthengine import EarthEngineCredentials
from pyramids_eo.earthengine.credentials import GOOGLE_APPLICATION_CREDENTIALS
from pyramids_eo.errors import AuthenticationError


@pytest.fixture
def key_file(tmp_path: Path) -> Path:
    """Create a dummy service-account JSON key file.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Path to a readable JSON file standing in for a real key (its contents
        are never parsed by the code under test).
    """
    key = tmp_path / "service-account.json"
    key.write_text('{"type": "service_account"}', encoding="utf-8")
    return key


class TestEarthEngineCredentials:
    """Tests for :class:`EarthEngineCredentials`."""

    def test_init_service_account_stores_path(self, key_file: Path) -> None:
        """Constructing with an existing key path stores the expanded path.

        Test scenario:
            A real file path is accepted and exposed via ``service_account_path``.
        """
        creds = EarthEngineCredentials(key_file)
        assert creds.service_account_path == key_file, (
            f"Expected stored path {key_file}, got {creds.service_account_path}"
        )

    def test_init_none_is_application_default(self) -> None:
        """Constructing with ``None`` yields application-default mode.

        Test scenario:
            No path → ``service_account_path`` is ``None`` (ambient ADC).
        """
        creds = EarthEngineCredentials(None)
        assert creds.service_account_path is None, (
            f"Expected ADC (None) path, got {creds.service_account_path}"
        )

    def test_init_expands_user(self, monkeypatch, tmp_path: Path) -> None:
        """A ``~``-prefixed path is expanded against the home directory.

        Test scenario:
            ``~/key.json`` resolves under the patched home and is accepted.
        """
        home = tmp_path / "home"
        home.mkdir()
        (home / "key.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.setenv("HOME", str(home))
        creds = EarthEngineCredentials("~/key.json")
        assert creds.service_account_path == home / "key.json", (
            f"'~' was not expanded correctly: {creds.service_account_path}"
        )

    def test_from_service_account_missing_raises(self, tmp_path: Path) -> None:
        """A non-existent key path fails fast with ``AuthenticationError``.

        Test scenario:
            Pointing at a missing file raises with a 'not found' message.
        """
        missing = tmp_path / "nope.json"
        with pytest.raises(AuthenticationError, match="not found") as exc_info:
            EarthEngineCredentials.from_service_account(missing)
        assert "not found" in str(exc_info.value), (
            f"Message should mention 'not found', got: {exc_info.value}"
        )

    def test_from_service_account_directory_raises(self, tmp_path: Path) -> None:
        """A directory path (not a file) is rejected.

        Test scenario:
            ``is_file()`` is false for a directory, so construction raises.
        """
        with pytest.raises(AuthenticationError, match="not found"):
            EarthEngineCredentials.from_service_account(tmp_path)

    def test_application_default_has_empty_env(self) -> None:
        """Application-default credentials expose no explicit GDAL config.

        Test scenario:
            ``application_default().gdal_env()`` is empty; path is ``None``.
        """
        creds = EarthEngineCredentials.application_default()
        assert creds.service_account_path is None, "ADC path should be None"
        assert creds.gdal_env() == {}, (
            f"ADC gdal_env should be empty, got {creds.gdal_env()}"
        )

    def test_gdal_env_service_account(self, key_file: Path) -> None:
        """A service account surfaces its key path as GDAL config.

        Test scenario:
            ``gdal_env`` maps ``GOOGLE_APPLICATION_CREDENTIALS`` → key path.
        """
        creds = EarthEngineCredentials.from_service_account(key_file)
        assert creds.gdal_env() == {GOOGLE_APPLICATION_CREDENTIALS: str(key_file)}, (
            f"Unexpected gdal_env mapping: {creds.gdal_env()}"
        )

    @pytest.mark.parametrize(
        "value_kind",
        ["instance", "none", "str", "path"],
    )
    def test_coerce(self, value_kind: str, key_file: Path) -> None:
        """``coerce`` normalises instance / ``None`` / path-like inputs.

        Args:
            value_kind: Which input flavour to feed ``coerce``.
            key_file: Dummy key path fixture.

        Test scenario:
            - instance → returned unchanged (identity)
            - None → application-default (path is None)
            - str / Path → service account with that key path
        """
        if value_kind == "instance":
            original = EarthEngineCredentials.application_default()
            assert EarthEngineCredentials.coerce(original) is original, (
                "coerce must pass an EarthEngineCredentials through unchanged"
            )
        elif value_kind == "none":
            assert EarthEngineCredentials.coerce(None).service_account_path is None, (
                "coerce(None) must be application-default"
            )
        elif value_kind == "str":
            assert (
                EarthEngineCredentials.coerce(str(key_file)).service_account_path
                == key_file
            ), "coerce(str) must build a service-account credential"
        else:
            assert (
                EarthEngineCredentials.coerce(key_file).service_account_path == key_file
            ), "coerce(Path) must build a service-account credential"

    def test_activate_sets_and_restores_config(self, key_file: Path) -> None:
        """``activate`` sets its config option and restores the prior value.

        Test scenario:
            Inside the ``with`` block the key path is set as a GDAL config
            option; on exit the previous value is restored.
        """
        from osgeo import gdal

        creds = EarthEngineCredentials.from_service_account(key_file)
        before = gdal.GetConfigOption(GOOGLE_APPLICATION_CREDENTIALS, None)
        with creds.activate():
            assert gdal.GetConfigOption(GOOGLE_APPLICATION_CREDENTIALS, None) == str(
                key_file
            ), "activate() should set GOOGLE_APPLICATION_CREDENTIALS inside the block"
        assert gdal.GetConfigOption(GOOGLE_APPLICATION_CREDENTIALS, None) == before, (
            "activate() should restore the previous config on exit"
        )

    def test_activate_restores_on_exception(self, key_file: Path) -> None:
        """``activate`` restores config even when the block raises.

        Test scenario:
            An exception inside the ``with`` block still triggers the finally
            restore, leaving the config option as it was.
        """
        from osgeo import gdal

        creds = EarthEngineCredentials.from_service_account(key_file)
        before = gdal.GetConfigOption(GOOGLE_APPLICATION_CREDENTIALS, None)
        with pytest.raises(RuntimeError, match="boom"), creds.activate():
            raise RuntimeError("boom")
        assert gdal.GetConfigOption(GOOGLE_APPLICATION_CREDENTIALS, None) == before, (
            "activate() must restore config on the exception path too"
        )

    def test_activate_application_default_is_noop(self) -> None:
        """``activate`` on ADC credentials sets nothing (empty env).

        Test scenario:
            With no service account, entering the block yields the instance and
            touches no config option.
        """
        creds = EarthEngineCredentials.application_default()
        with creds.activate() as yielded:
            assert yielded is creds, "activate() should yield the credentials instance"

    def test_repr_service_account(self, key_file: Path) -> None:
        """``__repr__`` shows the key path for a service account.

        Test scenario:
            The repr includes the class name and the quoted key path.
        """
        text = repr(EarthEngineCredentials.from_service_account(key_file))
        assert "EarthEngineCredentials" in text, f"repr missing class name: {text}"
        assert "service_account_json=" in text, f"repr missing key path: {text}"

    def test_repr_application_default(self) -> None:
        """``__repr__`` marks application-default mode explicitly.

        Test scenario:
            The ADC repr reads ``EarthEngineCredentials(application_default)``.
        """
        text = repr(EarthEngineCredentials.application_default())
        assert text == "EarthEngineCredentials(application_default)", (
            f"Unexpected ADC repr: {text}"
        )


class TestFromServiceAccountInfo:
    """Tests for inline-JSON credentials (:meth:`from_service_account_info`)."""

    def test_from_dict_materialises_key_file(self) -> None:
        """A mapping is written to a temp key file exposed via ``gdal_env``.

        Test scenario:
            The temp file exists, holds the JSON, and drives ``gdal_env``.
        """
        creds = EarthEngineCredentials.from_service_account_info(
            {"type": "service_account"}
        )
        path = creds.service_account_path
        assert path is not None, "Key file path should be set"
        assert path.is_file(), f"Key file not materialised: {path}"
        assert creds.gdal_env() == {GOOGLE_APPLICATION_CREDENTIALS: str(path)}, (
            f"Unexpected gdal_env: {creds.gdal_env()}"
        )
        import json

        assert json.loads(path.read_text(encoding="utf-8")) == {
            "type": "service_account"
        }, "Temp key file does not hold the supplied JSON"

    def test_from_json_string(self) -> None:
        """A JSON string is accepted and materialised.

        Test scenario:
            A valid JSON string produces a readable temp key file.
        """
        creds = EarthEngineCredentials.from_service_account_info(
            '{"type": "service_account"}'
        )
        assert creds.service_account_path.is_file(), (
            "Key file not materialised from JSON string"
        )

    def test_invalid_json_raises(self) -> None:
        """An invalid JSON string is rejected.

        Test scenario:
            Malformed JSON raises ``AuthenticationError``.
        """
        with pytest.raises(AuthenticationError, match="not valid JSON"):
            EarthEngineCredentials.from_service_account_info("{not json")

    def test_non_str_or_mapping_raises(self) -> None:
        """A non-string/non-mapping payload is rejected.

        Test scenario:
            An integer payload raises ``AuthenticationError``.
        """
        with pytest.raises(AuthenticationError, match="JSON string or mapping"):
            EarthEngineCredentials.from_service_account_info(1234)  # type: ignore[arg-type]

    def test_temp_file_cleaned_up_on_gc(self) -> None:
        """The temp key file is removed when the credentials are collected.

        Test scenario:
            After dropping the last reference and forcing GC, the file is gone.
        """
        import gc

        creds = EarthEngineCredentials.from_service_account_info(
            {"type": "service_account"}
        )
        path = creds.service_account_path
        assert path.is_file(), "precondition: key file exists"
        del creds
        gc.collect()
        assert not path.exists(), f"Temp key file was not cleaned up: {path}"

    def test_repr_redacts_inline_info(self) -> None:
        """``__repr__`` never leaks an inline key path.

        Test scenario:
            The inline repr is redacted rather than showing the temp path.
        """
        creds = EarthEngineCredentials.from_service_account_info(
            {"type": "service_account"}
        )
        assert (
            repr(creds) == "EarthEngineCredentials(service_account_info=<redacted>)"
        ), f"Inline repr should be redacted, got: {creds!r}"

    def test_coerce_dict_builds_inline(self) -> None:
        """``coerce`` routes a mapping to inline credentials.

        Test scenario:
            ``coerce({...})`` materialises a temp key file.
        """
        creds = EarthEngineCredentials.coerce({"type": "service_account"})
        assert creds.service_account_path.is_file(), (
            "coerce(dict) should build inline credentials"
        )

    def test_non_serialisable_dict_raises(self) -> None:
        """A dict with a non-JSON value is wrapped as AuthenticationError.

        Test scenario:
            ``{"k": {1, 2}}`` (a set value) raises AuthenticationError, not a raw
            TypeError.
        """
        with pytest.raises(AuthenticationError, match="not JSON-serialisable"):
            EarthEngineCredentials.from_service_account_info({"k": {1, 2}})
