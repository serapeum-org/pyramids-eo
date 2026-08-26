"""DN → reflectance as lazy scale/offset tags, not an eager float array.

Sentinel-2 stores reflectance as scaled integers: ``reflectance = (DN + offset)
/ quantification``, where ``quantification`` is the product's
``L2A_BOA`` / ``L1C_TOA`` quantification value and ``offset`` is the per-band
radiometric offset that processing baselines ``>= 04.00`` introduced
(``BOA_ADD_OFFSET`` / ``RADIO_ADD_OFFSET``; ``0`` before that).

The GDAL ``SENTINEL2`` driver surfaces the offset in **per-band** metadata
(``Dataset.band_meta_data[i]``), which is where :func:`_band_offset` reads it —
verified end-to-end against a real baseline-05.09 product (offset ``-1000``
flows into ``read_array(scaled=True)``; see ``test_baseline_509_*``), not just
the offset parser.

Rather than materialise a float array, :func:`tag_reflectance` records the
conversion on the dataset's GDAL scale/offset so it is applied lazily by
``Dataset.read_array(scaled=True)`` and rides through ``crop`` / ``to_crs``
(pyramids-gis #1031). In GDAL's ``real = DN * scale + offset`` terms:

    scale  = 1 / quantification
    offset = band_offset / quantification

Only spectral ``B*`` bands are tagged; classification / auxiliary bands
(``SCL`` / ``CLD`` / ``AOT`` / …) keep ``scale=1, offset=0`` — they are not
reflectance.
"""

from __future__ import annotations

from typing import Any

from pyramids_eo.sentinel.s2.product import S2Product, _normalise_band


def _is_spectral(band_name: str) -> bool:
    """True for a spectral ``B*`` band (``B01`` … ``B12`` / ``B8A``)."""
    text = band_name.strip().upper()
    return text.startswith("B") and any(c.isdigit() for c in text)


def _band_offset(band_meta: dict[str, str]) -> float:
    """Radiometric offset for one band from its metadata, ``0.0`` if absent.

    Reads ``BOA_ADD_OFFSET`` (L2A) or ``RADIO_ADD_OFFSET`` (L1C); both are the
    baseline-≥04.00 shift and are absent on older products.
    """
    for key in ("BOA_ADD_OFFSET", "RADIO_ADD_OFFSET"):
        raw = band_meta.get(key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def tag_reflectance(
    dataset: Any, product: S2Product, *, offsets: list[float] | None = None
) -> Any:
    """Tag ``dataset``'s spectral bands so a scaled read yields reflectance.

    Sets per-band GDAL scale/offset on ``dataset`` in place and returns it.
    Spectral bands get ``scale = 1/quantification`` and
    ``offset = band_offset/quantification``; every other band is left at
    ``scale=1, offset=0``. The reflectance is then obtained lazily with
    ``dataset.read_array(scaled=True)``.

    Args:
        dataset: A pyramids ``Dataset`` of Sentinel-2 bands (must be writable —
            e.g. the result of ``Dataset.bands.select`` or a MEM copy).
        product: The :class:`S2Product` supplying the quantification value.
        offsets: Optional per-band radiometric offsets, in band order. Passed by
            the cross-resolution read path, which stacks bands into a fresh
            dataset that no longer carries the driver's ``BOA_ADD_OFFSET``
            metadata. ``None`` reads the offsets from each band's metadata.

    Returns:
        The same ``dataset``, tagged.

    Note:
        The tags do not touch the ``no_data_value``, which stays in the DN
        domain. A scaled read therefore returns ``DN * scale + offset`` for the
        no-data pixels too — compare against the raw no-data value, not a scaled
        one (see :func:`~pyramids_eo.sentinel.from_sentinel2`).

    Raises:
        ProductError: The product's quantification value is zero / unusable.
    """
    quant = product.quantification
    if not quant:
        from pyramids_eo.errors import ProductError

        raise ProductError("product has no usable quantification value")

    band_names = dataset.band_names
    band_meta = dataset.band_meta_data
    scales: list[float] = []
    tagged_offsets: list[float] = []
    for i, name in enumerate(band_names):
        # Prefer the driver's BANDNAME tag; fall back to the display name.
        tag = (band_meta[i].get("BANDNAME") if i < len(band_meta) else None) or name
        if _is_spectral(_normalise_band(tag)):
            band_offset = (
                offsets[i]
                if offsets is not None
                else _band_offset(band_meta[i] if i < len(band_meta) else {})
            )
            scales.append(1.0 / quant)
            tagged_offsets.append(band_offset / quant)
        else:
            scales.append(1.0)
            tagged_offsets.append(0.0)

    dataset.scale = scales
    dataset.offset = tagged_offsets
    _stamp_baseline(dataset, product)
    return dataset


def _stamp_baseline(dataset: Any, product: S2Product) -> None:
    """Record the processing baseline + scaling marker on the output metadata.

    Best-effort: a read-only dataset that refuses the metadata write is left
    untagged rather than failing the reflectance conversion.
    """
    try:
        meta = dict(dataset.meta_data or {})
        meta["PROCESSING_BASELINE"] = product.baseline
        meta["PYRAMIDS_EO_REFLECTANCE"] = "scaled"
        dataset.meta_data = meta
    except Exception:  # noqa: BLE001 - metadata stamping is non-essential
        pass
