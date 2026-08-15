"""Exception hierarchy for pyramids-eo.

Every error raised by this package derives from :class:`EOError`, so callers can
catch the whole family with a single ``except EOError``.
"""

from __future__ import annotations


class EOError(Exception):
    """Base class for all pyramids-eo errors."""


class ReaderError(EOError):
    """Raised when an instrument reader cannot decode the input."""


class UnknownSensorError(EOError):
    """Raised when a sensor / channel is not present in the registry."""


class CalibrationError(EOError):
    """Raised when a requested calibration is unavailable for the input."""


class AuthenticationError(EOError):
    """Raised when provider credentials for a signed EO asset are missing or invalid."""
