"""Sentinel-2 product model — parses one MSI product into a typed catalog.

The GDAL ``SENTINEL2`` driver exposes a product as subdatasets, one per
resolution × UTM zone (10 m / 20 m / 60 m) plus preview / true-colour views.
:class:`S2Product` reads that layout plus the product metadata block into a
queryable model: which bands live at which resolution, the processing baseline
and quantification value needed to turn DN into reflectance, cloud cover, and
the footprint.

It holds no pixels — :meth:`S2Product.subdataset_for` hands back a pyramids
``SubDataset`` you open on demand, and the reader
(:func:`pyramids_eo.sentinel.s2.reader.from_sentinel2`) drives it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pyramids_eo.errors import ProductError
from pyramids_eo.sentinel import _connection
from pyramids_eo.sentinel.product import SentinelProduct, open_connection

#: Resolution tokens that are not spectral image subdatasets.
_NON_SPECTRAL = frozenset({"PREVIEW", "TCI"})

#: Default DN scale when the product metadata carries no quantification value.
#: Sentinel-2's canonical value; used only as a last-resort fallback.
_DEFAULT_QUANTIFICATION = 10000.0


class S2Level(StrEnum):
    """Sentinel-2 processing level."""

    L1B = "L1B"
    L1C = "L1C"
    L2A = "L2A"


class S2Subdataset:
    """One resolution × UTM-zone image subdataset of an S2 product.

    A thin typed wrapper over the pyramids ``SubDataset`` value object, adding
    the parsed resolution / EPSG / band list. Open it with :meth:`open`.

    Attributes:
        resolution_m: Native ground sampling distance in metres (10 / 20 / 60).
        epsg: UTM EPSG code of the subdataset.
        bands: Band names the subdataset carries, in driver order (spectral
            ``B*`` plus any derived layers such as ``AOT`` / ``SCL``).
    """

    def __init__(
        self, subdataset: Any, resolution_m: int, epsg: int | None, bands: list[str]
    ) -> None:
        self._sd = subdataset
        self.resolution_m = resolution_m
        self.epsg = epsg
        self.bands = bands

    @property
    def connection(self) -> str:
        """The GDAL connection string for this subdataset."""
        return str(self._sd.name)

    def open(self) -> Any:
        """Open the subdataset as a pyramids ``Dataset``."""
        return open_connection(self._sd.name)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"S2Subdataset({self.resolution_m}m, EPSG:{self.epsg}, {len(self.bands)} bands)"


class S2Product(SentinelProduct):
    """A parsed Sentinel-2 MSI product (L1C / L2A; L1B partially).

    Built by :func:`pyramids_eo.sentinel.open_product`. Exposes the product's
    structure and the metadata a reader needs to calibrate it.

    Attributes:
        level: The :class:`S2Level`.
        baseline: Processing baseline string (e.g. ``"05.09"``); ``""`` if
            absent. Baselines ``>= 04.00`` carry a radiometric offset.
        quantification: DN → reflectance divisor (``L2A_BOA`` / ``L1C_TOA``
            quantification value).
        cloud_cover: Scene cloud-cover percentage, or ``None``.
        footprint: Product footprint as a WKT polygon string, or ``""``.
        resolutions: Sorted native resolutions present (metres).
        epsg_codes: Sorted UTM EPSG codes present.
    """

    def __init__(self, path: str, container: Any) -> None:
        super().__init__(path, container)
        self._image_subdatasets: list[S2Subdataset] = []
        self._preview: list[Any] = []
        self._parse_subdatasets()
        if not self._image_subdatasets:
            raise ProductError(
                f"no Sentinel-2 image subdatasets found in {path!r}; "
                "is this a valid MSI product?"
            )
        self.level = self._resolve_level()
        self.baseline = self.metadata.get("PROCESSING_BASELINE", "")
        self.quantification = self._resolve_quantification()
        self.cloud_cover = _as_float(self.metadata.get("CLOUD_COVERAGE_ASSESSMENT"))
        self.footprint = self.metadata.get("FOOTPRINT", "")

    # -- construction ------------------------------------------------------

    def _parse_subdatasets(self) -> None:
        """Split the raw subdatasets into image vs preview, parsing each."""
        for sd in self.subdatasets:
            conn = _connection.parse_s2(sd.name)
            token = conn.resolution.upper()
            if token in _NON_SPECTRAL:
                self._preview.append(sd)
                continue
            resolution_m = _resolution_metres(conn.resolution)
            if resolution_m is None:
                self._preview.append(sd)
                continue
            self._image_subdatasets.append(
                S2Subdataset(sd, resolution_m, conn.epsg, _bands_of(sd.description))
            )

    def _resolve_level(self) -> S2Level:
        """Derive the processing level from the subdataset connection prefix."""
        level_token = _connection.parse_s2(self.subdatasets[0].name).level
        try:
            return S2Level(level_token)
        except ValueError as exc:
            raise ProductError(f"unknown Sentinel-2 level {level_token!r}") from exc

    def _resolve_quantification(self) -> float:
        """Pick the level-appropriate quantification value from metadata."""
        keys = ("L2A_BOA_QUANTIFICATION_VALUE", "L1C_TOA_QUANTIFICATION_VALUE")
        for key in keys:
            value = _as_float(self.metadata.get(key))
            if value:
                return value
        return _DEFAULT_QUANTIFICATION

    # -- public surface ----------------------------------------------------

    @property
    def mission(self) -> str:
        """Mission identifier."""
        return "sentinel-2"

    @property
    def product_type(self) -> str:
        """Product type token (e.g. ``"S2MSI2A"``), or the level as a fallback."""
        return self.metadata.get("PRODUCT_TYPE", f"S2MSI{self.level.value}")

    @property
    def resolutions(self) -> list[int]:
        """Sorted native resolutions present in the product (metres)."""
        return sorted({sd.resolution_m for sd in self._image_subdatasets})

    @property
    def epsg_codes(self) -> list[int]:
        """Sorted UTM EPSG codes present in the product."""
        return sorted(
            {sd.epsg for sd in self._image_subdatasets if sd.epsg is not None}
        )

    @property
    def image_subdatasets(self) -> list[S2Subdataset]:
        """The spectral / derived image subdatasets (excludes preview / TCI)."""
        return list(self._image_subdatasets)

    def resolution_of(self, band: str) -> int:
        """Return the finest native resolution (m) that carries ``band``.

        Args:
            band: Band name, e.g. ``"B04"`` (matched case-insensitively; a bare
                ``"B4"`` also matches ``"B04"``).

        Returns:
            The finest resolution in metres offering the band.

        Raises:
            ProductError: The band is not present in the product.
        """
        candidates = [
            sd.resolution_m
            for sd in self._image_subdatasets
            if _has_band(sd.bands, band)
        ]
        if not candidates:
            raise ProductError(
                f"band {band!r} not in product; available: {sorted(self.available_bands)}"
            )
        return min(candidates)

    @property
    def available_bands(self) -> set[str]:
        """Every band name present across the product's image subdatasets."""
        return {b for sd in self._image_subdatasets for b in sd.bands}

    def subdataset_for(
        self, resolution_m: int, epsg: int | None = None
    ) -> S2Subdataset:
        """Return the image subdataset at ``resolution_m`` (and ``epsg``).

        Args:
            resolution_m: Native resolution in metres (10 / 20 / 60).
            epsg: UTM EPSG code to disambiguate a multi-zone product. ``None``
                is allowed only when the product has a single zone.

        Returns:
            The matching :class:`S2Subdataset`.

        Raises:
            ProductError: No unique subdataset matches.
        """
        matches = [
            sd
            for sd in self._image_subdatasets
            if sd.resolution_m == resolution_m and (epsg is None or sd.epsg == epsg)
        ]
        if not matches:
            raise ProductError(
                f"no {resolution_m}m subdataset"
                + (f" for EPSG:{epsg}" if epsg else "")
                + f"; resolutions present: {self.resolutions}"
            )
        if len(matches) > 1:
            raise ProductError(
                f"{resolution_m}m is ambiguous across EPSG {self.epsg_codes}; "
                "pass epsg=."
            )
        return matches[0]


# -- parsing helpers -------------------------------------------------------


def _resolution_metres(token: str) -> int | None:
    """Return metres from a ``"60m"`` token, or ``None`` if not ``<int>m``."""
    token = token.strip().lower()
    if token.endswith("m") and token[:-1].isdigit():
        return int(token[:-1])
    return None


def _bands_of(description: str) -> list[str]:
    """Extract the band list from a SENTINEL2 subdataset description.

    The driver formats it as ``"Bands B1, B2, ... with 60m resolution, UTM
    32N"``. Returns the names between ``"Bands "`` and ``" with "``.

    Args:
        description: The ``SubDataset.description`` string.

    Returns:
        The band names in driver order; empty when the shape is unexpected.
    """
    if "Bands " not in description:
        return []
    head = description.split("Bands ", 1)[1]
    head = head.split(" with ", 1)[0]
    return [b.strip() for b in head.split(",") if b.strip()]


def is_spectral_band(name: str) -> bool:
    """True for a spectral MSI band (``B01``…``B12``, ``B8A``); False for
    auxiliary layers (``AOT`` / ``CLD`` / ``SCL`` / ``SNW`` / ``WVP`` / ``TCI``).

    A spectral band is ``B`` followed by a digit (so ``B8A`` counts, ``BAND``
    does not).
    """
    text = name.strip().upper()
    return text.startswith("B") and len(text) > 1 and text[1].isdigit()


def _has_band(bands: list[str], wanted: str) -> bool:
    """Case-insensitive band match tolerating ``B4`` vs ``B04`` zero-padding."""
    wanted_norm = _normalise_band(wanted)
    return any(_normalise_band(b) == wanted_norm for b in bands)


def _normalise_band(name: str) -> str:
    """Normalise a band name so ``B4`` and ``B04`` compare equal."""
    text = name.strip().upper()
    if text.startswith("B") and text[1:].isdigit():
        return f"B{int(text[1:]):02d}"
    return text


def _as_float(value: str | None) -> float | None:
    """Parse a metadata string to float, or ``None`` when absent/malformed."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
