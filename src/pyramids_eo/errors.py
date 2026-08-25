"""Exception hierarchy for pyramids-eo.

Every error raised by this package derives from `EOError`, so callers can
catch the whole family with a single `except EOError`.
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


class ProductError(EOError):
    """Raised when a satellite product cannot be parsed or is malformed.

    Covers a container whose subdatasets / metadata cannot be read, a requested
    band / resolution / tile that the product does not contain, and any other
    structural problem discovered while modelling the product.
    """


class UnsupportedProductError(ProductError):
    """Raised when a path is opened as a product type this package cannot model.

    Distinct from `ProductError`: the input is a valid raster, but its GDAL
    driver / product family is not one the Sentinel readers understand.
    """
