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
import os
from pathlib import Path
from typing import Union

from pyramids_eo.errors import AuthenticationError

#: The GDAL config option / environment variable that carries the service-account
#: key path for the EEDAI/EEDA drivers' ADC handshake.
GOOGLE_APPLICATION_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"

CredentialsLike = Union["EarthEngineCredentials", str, Path, None]


class EarthEngineCredentials:
    """Application Default Credentials for the Earth Engine GDAL drivers.

    Two modes:

    * **service account** — an explicit Google service-account JSON key file,
      surfaced to GDAL as ``GOOGLE_APPLICATION_CREDENTIALS``.
    * **application default** — no explicit key; GDAL resolves ambient ADC
      (an already-exported ``GOOGLE_APPLICATION_CREDENTIALS`` or ``gcloud``
      login). :meth:`gdal_env` is empty in this mode.

    Prefer the named constructors :meth:`from_service_account` /
    :meth:`application_default` over the raw initializer.

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
    def from_service_account(cls, path: str | Path) -> EarthEngineCredentials:
        """Build credentials from a service-account JSON key path.

        Args:
            path: Path to the Google service-account JSON key. ``~`` is
                expanded; the file must exist.

        Returns:
            The credentials, whose :meth:`gdal_env` points GDAL at ``path``.
        """
        return cls(service_account_json=path)

    @classmethod
    def application_default(cls) -> EarthEngineCredentials:
        """Build credentials that defer to ambient ADC (no explicit key).

        Returns:
            Credentials with an empty :meth:`gdal_env`; GDAL resolves an
            already-exported ``GOOGLE_APPLICATION_CREDENTIALS`` or ``gcloud``
            login at read time.
        """
        return cls(service_account_json=None)

    @classmethod
    def coerce(cls, credentials: CredentialsLike) -> EarthEngineCredentials:
        """Normalise a caller-supplied credentials value.

        Args:
            credentials: An :class:`EarthEngineCredentials`, a path-like pointing
                at a service-account key, or ``None`` for application-default.

        Returns:
            An :class:`EarthEngineCredentials` instance.
        """
        if isinstance(credentials, EarthEngineCredentials):
            return credentials
        if credentials is None:
            return cls.application_default()
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

        Yields:
            This :class:`EarthEngineCredentials` instance.
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
        return f"{type(self).__name__}(service_account_json={str(self._path)!r})"


def _ambient_service_account() -> str | None:
    """Return the ambient ``GOOGLE_APPLICATION_CREDENTIALS`` value, if any."""
    value = os.environ.get(GOOGLE_APPLICATION_CREDENTIALS)
    return value or None
