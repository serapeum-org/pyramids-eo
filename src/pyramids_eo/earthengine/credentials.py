"""Earth Engine provider credentials for the GDAL EEDAI/EEDA backend.

The bundled GDAL ``EEDAI`` (raster) and ``EEDA`` (catalog) drivers reach the
Earth Engine Data API with **Application Default Credentials (ADC)** — either a
Google *service-account* JSON key pointed at by ``GOOGLE_APPLICATION_CREDENTIALS``
or ambient ``gcloud`` ADC. That is a different mechanism from the STAC bearer /
S3 signers in :mod:`pyramids_eo.stac`, so it lives in its own small helper rather
than reusing ``_BearerProviderSigner``; it sits alongside them as the same
"EO-provider auth" concern.

:class:`EarthEngineCredentials` is deliberately thin: it records *which* ADC
source to use and exposes it as a GDAL-config mapping
(:meth:`EarthEngineCredentials.gdal_env`) — the same shape the STAC signers
return from ``gdal_env`` — so the reader can hand it straight to
:class:`pyramids.dataset.Dataset`. It never reads or parses the key file (that is
a secret); it only checks that the path exists so a typo fails early with a clear
:class:`~pyramids_eo.errors.AuthenticationError` instead of an opaque GDAL error.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import tempfile
import weakref
from pathlib import Path
from typing import Union

from pyramids_eo.errors import AuthenticationError

#: The GDAL config option / environment variable that carries the service-account
#: key path for the EEDAI/EEDA drivers' ADC handshake.
GOOGLE_APPLICATION_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"

CredentialsLike = Union["EarthEngineCredentials", str, Path, dict, None]


def _write_secret_json(text: str) -> Path:
    """Write service-account JSON to a private temp file and return its path.

    On POSIX the file is created with owner-only permissions (0600). On Windows
    POSIX mode bits are largely ignored, so the protection there is the per-user
    temporary directory rather than the mode. Contents are never logged.

    Args:
        text: The service-account JSON payload.

    Returns:
        Path to the newly written temp file.
    """
    fd, name = tempfile.mkstemp(prefix="ee-sa-", suffix=".json")
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)
    with contextlib.suppress(OSError):
        os.chmod(name, stat.S_IRUSR | stat.S_IWUSR)
    return Path(name)


def _unlink_quiet(path: Path) -> None:
    """Delete ``path`` if it exists, ignoring errors (used as a GC finalizer)."""
    with contextlib.suppress(OSError):
        Path(path).unlink()


class EarthEngineCredentials:
    """Application Default Credentials for the Earth Engine GDAL drivers.

    Three modes:

    * **service-account file** — an explicit Google service-account JSON key
      file, surfaced to GDAL as ``GOOGLE_APPLICATION_CREDENTIALS``
      (:meth:`from_service_account`).
    * **service-account info (inline JSON)** — the key material as a JSON string
      or mapping; it is written to a private temp file (owner-only, cleaned up
      when the credentials are garbage-collected) and used as the key path
      (:meth:`from_service_account_info`). Useful when the key comes from a secret
      store rather than a file on disk.
    * **application default** — no explicit key; GDAL resolves ambient ADC
      (an already-exported ``GOOGLE_APPLICATION_CREDENTIALS`` or ``gcloud``
      login). :meth:`gdal_env` is empty in this mode.

    Prefer the named constructors :meth:`from_service_account` /
    :meth:`from_service_account_info` / :meth:`application_default` over the raw
    initializer.

    Args:
        service_account_json: Path to a service-account JSON key, or ``None`` for
            application-default resolution. ``~`` is expanded; the file must
            exist when given.

    Raises:
        AuthenticationError: The given ``service_account_json`` path does not
            exist.

    Examples:
        Application-default credentials carry no explicit GDAL config — GDAL
        resolves ambient ADC on its own:

            >>> from pyramids_eo.earthengine import EarthEngineCredentials
            >>> EarthEngineCredentials.application_default().gdal_env()
            {}

        A missing service-account file is rejected early:

            >>> EarthEngineCredentials.from_service_account(  # doctest: +ELLIPSIS
            ...     "/no/such/key.json"
            ... )
            Traceback (most recent call last):
                ...
            pyramids_eo.errors.AuthenticationError: ...not found...
    """

    def __init__(self, service_account_json: str | Path | None = None) -> None:
        self._finalizer: weakref.finalize | None = None
        self._owns_path = False
        if service_account_json is None:
            self._path: Path | None = None
            return
        path = Path(service_account_json).expanduser()
        if not path.is_file():
            raise AuthenticationError(
                f"Earth Engine service-account key file not found: {path}"
            )
        self._path = path

    @classmethod
    def from_service_account_info(cls, info: str | dict) -> EarthEngineCredentials:
        """Build credentials from inline service-account JSON (string or mapping).

        The key material is written to a private, owner-only temp file that is
        removed when the returned credentials are garbage-collected. The contents
        are never logged.

        Args:
            info: The service-account key as a JSON string or a mapping.

        Returns:
            Credentials backed by a temp key file; :meth:`gdal_env` points GDAL at
            it.

        Raises:
            AuthenticationError: ``info`` is not valid JSON, or not a string /
                mapping.

        Examples:
            - Build from a mapping and confirm a key file is materialised:
                ```python
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> creds = EarthEngineCredentials.from_service_account_info(
                ...     {"type": "service_account"}
                ... )
                >>> creds.service_account_path.is_file()
                True
                >>> creds.gdal_env()["GOOGLE_APPLICATION_CREDENTIALS"] == str(
                ...     creds.service_account_path
                ... )
                True

                ```
            - Invalid JSON is rejected:
                ```python
                >>> from pyramids_eo.errors import AuthenticationError
                >>> try:
                ...     EarthEngineCredentials.from_service_account_info("{not json")
                ... except AuthenticationError as exc:
                ...     print("not valid JSON" in str(exc))
                True

                ```
        """
        if isinstance(info, dict):
            try:
                text = json.dumps(info)
            except (TypeError, ValueError) as exc:
                raise AuthenticationError(
                    "Earth Engine service-account info is not JSON-serialisable."
                ) from exc
        elif isinstance(info, str):
            try:
                json.loads(info)
            except (ValueError, TypeError) as exc:
                raise AuthenticationError(
                    "Earth Engine service-account info is not valid JSON."
                ) from exc
            text = info
        else:
            raise AuthenticationError(
                "Earth Engine service-account info must be a JSON string or mapping."
            )
        path = _write_secret_json(text)
        creds = cls(path)
        creds._owns_path = True
        creds._finalizer = weakref.finalize(creds, _unlink_quiet, path)
        return creds

    @classmethod
    def from_service_account(cls, path: str | Path) -> EarthEngineCredentials:
        """Build credentials from a service-account JSON key path.

        Args:
            path: Path to the Google service-account JSON key. ``~`` is
                expanded; the file must exist.

        Returns:
            The credentials, whose :meth:`gdal_env` points GDAL at ``path``.

        Examples:
            - Build from an existing key file and read the stored path back:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> key = Path(tempfile.mkdtemp()) / "sa.json"
                >>> _ = key.write_text("{}")
                >>> creds = EarthEngineCredentials.from_service_account(key)
                >>> creds.service_account_path == key
                True

                ```
            - A missing key path is rejected early:
                ```python
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> EarthEngineCredentials.from_service_account(  # doctest: +ELLIPSIS
                ...     "/no/such/key.json"
                ... )
                Traceback (most recent call last):
                    ...
                pyramids_eo.errors.AuthenticationError: ...not found...

                ```
        """
        return cls(service_account_json=path)

    @classmethod
    def application_default(cls) -> EarthEngineCredentials:
        """Build credentials that defer to ambient ADC (no explicit key).

        Returns:
            Credentials with an empty :meth:`gdal_env`; GDAL resolves an
            already-exported ``GOOGLE_APPLICATION_CREDENTIALS`` or ``gcloud``
            login at read time.

        Examples:
            - Application-default credentials hold no explicit key:
                ```python
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> creds = EarthEngineCredentials.application_default()
                >>> creds.service_account_path is None
                True
                >>> creds.gdal_env()
                {}

                ```
            - They report their mode in ``repr``:
                ```python
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> repr(EarthEngineCredentials.application_default())
                'EarthEngineCredentials(application_default)'

                ```
        """
        return cls(service_account_json=None)

    @classmethod
    def coerce(cls, credentials: CredentialsLike) -> EarthEngineCredentials:
        """Normalise a caller-supplied credentials value.

        Args:
            credentials: An :class:`EarthEngineCredentials`, a path-like pointing
                at a service-account key, a mapping of inline service-account JSON,
                or ``None`` for application-default.

        Returns:
            An :class:`EarthEngineCredentials` instance.

        Examples:
            - ``None`` becomes application-default:
                ```python
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> EarthEngineCredentials.coerce(None).service_account_path is None
                True

                ```
            - An existing instance passes through unchanged:
                ```python
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> creds = EarthEngineCredentials.application_default()
                >>> EarthEngineCredentials.coerce(creds) is creds
                True

                ```
        """
        if isinstance(credentials, EarthEngineCredentials):
            return credentials
        if credentials is None:
            return cls.application_default()
        if isinstance(credentials, dict):
            return cls.from_service_account_info(credentials)
        return cls.from_service_account(credentials)

    @property
    def service_account_path(self) -> Path | None:
        """The service-account key path, or ``None`` in application-default mode."""
        return self._path

    def gdal_env(self) -> dict[str, str]:
        """Return the GDAL-config mapping for this credential.

        Returns:
            ``{"GOOGLE_APPLICATION_CREDENTIALS": <path>}`` for a service account,
            or an empty mapping for application-default (ambient) resolution.

        Examples:
            - Application-default carries no explicit config:
                ```python
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> EarthEngineCredentials.application_default().gdal_env()
                {}

                ```
            - A service account surfaces its key path for GDAL:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> key = Path(tempfile.mkdtemp()) / "sa.json"
                >>> _ = key.write_text("{}")
                >>> EarthEngineCredentials.from_service_account(key).gdal_env() == {
                ...     "GOOGLE_APPLICATION_CREDENTIALS": str(key)
                ... }
                True

                ```
        """
        if self._path is None:
            return {}
        return {GOOGLE_APPLICATION_CREDENTIALS: str(self._path)}

    @contextlib.contextmanager
    def activate(self):
        """Apply this credential's GDAL config for the duration of a ``with`` block.

        Sets each :meth:`gdal_env` key as a GDAL config option and restores the
        previous value (or clears it) on exit, so the process-wide GDAL config is
        left untouched afterwards.

        Note:
            GDAL config options are **process-global**, so this save/restore is not
            thread-safe: two threads activating different service-account
            credentials concurrently would clobber each other's
            ``GOOGLE_APPLICATION_CREDENTIALS``. The reader is intended for
            single-threaded, single-credential use per process.

        Yields:
            This :class:`EarthEngineCredentials` instance.

        Examples:
            - Application-default activation is a no-op that yields the credentials:
                ```python
                >>> from pyramids_eo.earthengine import EarthEngineCredentials
                >>> creds = EarthEngineCredentials.application_default()
                >>> with creds.activate() as active:
                ...     active is creds
                True

                ```
        """
        from osgeo import gdal

        env = self.gdal_env()
        previous: dict[str, str | None] = {}
        try:
            for key, value in env.items():
                previous[key] = gdal.GetConfigOption(key, None)
                gdal.SetConfigOption(key, value)
            yield self
        finally:
            for key, old in previous.items():
                gdal.SetConfigOption(key, old)

    def __repr__(self) -> str:
        if self._path is None:
            return f"{type(self).__name__}(application_default)"
        if self._owns_path:
            return f"{type(self).__name__}(service_account_info=<redacted>)"
        return f"{type(self).__name__}(service_account_json={str(self._path)!r})"
