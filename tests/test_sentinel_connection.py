"""Unit tests for the Sentinel connection-string grammar (`sentinel._connection`)."""

from __future__ import annotations

import pytest

from pyramids_eo.errors import ProductError
from pyramids_eo.sentinel import _connection


class _SD:
    """Minimal stand-in for a pyramids ``SubDataset`` (name + description)."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description


def test_parse_s2_full_with_epsg():
    conn = _connection.parse_s2(
        r"SENTINEL2_L2A:C:/data/S2A_MTD_SAFL2A.xml:60m:EPSG_32632"
    )
    assert conn.level == "L2A"
    assert conn.source == r"C:/data/S2A_MTD_SAFL2A.xml"
    assert conn.resolution == "60m"
    assert conn.epsg == 32632


def test_parse_s2_preview_without_epsg_token():
    conn = _connection.parse_s2(r"SENTINEL2_L1C:C:/x/MTD.xml:PREVIEW:EPSG_32631")
    assert conn.resolution == "PREVIEW"
    assert conn.epsg == 32631


def test_parse_s2_rejects_non_s2():
    with pytest.raises(ProductError):
        _connection.parse_s2("NETCDF:file.nc:var")


def test_parse_s1_grd_all_pols():
    conn = _connection.parse_s1(
        r"SENTINEL1_CALIB:UNCALIB:C:/x/test.SAFE/manifest.safe:IW:AMPLITUDE"
    )
    assert conn.calibration == "UNCALIB"
    assert conn.swath == "IW"
    assert conn.polarisation is None
    assert conn.unit == "AMPLITUDE"


def test_parse_s1_with_polarisation():
    conn = _connection.parse_s1(
        r"SENTINEL1_CALIB:SIGMA0:C:/x/manifest.safe:IW_VV:INTENSITY"
    )
    assert conn.polarisation == "VV"
    assert conn.calibration == "SIGMA0"


def test_select_matches_one_by_resolution_and_epsg():
    subs = [
        _SD(r"SENTINEL2_L2A:C:/x/MTD.xml:10m:EPSG_32632"),
        _SD(r"SENTINEL2_L2A:C:/x/MTD.xml:60m:EPSG_32632"),
    ]
    chosen = _connection.select(subs, resolution="60m", epsg=32632)
    assert chosen is subs[1]


def test_select_no_match_raises():
    subs = [_SD(r"SENTINEL2_L2A:C:/x/MTD.xml:10m:EPSG_32632")]
    with pytest.raises(ProductError):
        _connection.select(subs, resolution="20m")


def test_select_ambiguous_raises():
    subs = [
        _SD(r"SENTINEL2_L2A:C:/x/MTD.xml:10m:EPSG_32631"),
        _SD(r"SENTINEL2_L2A:C:/x/MTD.xml:10m:EPSG_32632"),
    ]
    with pytest.raises(ProductError):
        _connection.select(subs, resolution="10m")
