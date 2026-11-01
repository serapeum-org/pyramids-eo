"""Unit tests for `tests._netcdf3.write_netcdf`, the fixture writer itself.

Every Sentinel-3 reader fixture is built by this writer, so a bug here weakens the
whole suite silently: a flipped row order, a dropped attribute or a mis-encoded
`nc_type` would make a wrong reader just as green as a right one. These read the
written files back through the same bundled GDAL netCDF driver the readers use, so
what is asserted is what a reader would actually see.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._netcdf3 import write_netcdf

#: A row-asymmetric array: reversing its rows is detectable, unlike a symmetric one.
_ROWS = np.array([[1.0, 2.0, 3.0], [40.0, 50.0, 60.0]])


def _read(path, variable):
    """Open `variable` in `path` through GDAL and return its array and attributes."""
    from osgeo import gdal

    raster = gdal.Open(f'NETCDF:"{path}":{variable}')
    assert raster is not None, f"GDAL could not open {variable!r} in {path}"
    band = raster.GetRasterBand(1)
    array = np.asarray(band.ReadAsArray())
    meta = dict(band.GetMetadata())
    del band, raster
    return array, meta


class TestWriteNetcdfRejections:
    """Tests for `write_netcdf`'s input validation."""

    def test_no_variables_raises(self, tmp_path):
        """A file with no variables is rejected rather than written empty."""
        with pytest.raises(ValueError, match="at least one variable"):
            write_netcdf(tmp_path / "empty.nc", {})

    def test_mismatched_shapes_raise_naming_both(self, tmp_path):
        """Variables of different shapes are rejected, the message naming the shapes.

        Test scenario:
            The classic layout declares the two dimensions once for the file, so a
            second shape would be written against the wrong dimension lengths and
            read back as silently transposed or truncated data.
        """
        with pytest.raises(ValueError, match=r"share one shape") as error:
            write_netcdf(
                tmp_path / "mixed.nc",
                {
                    "a": (np.zeros((2, 3), dtype="int32"), {}),
                    "b": (np.zeros((3, 2), dtype="int32"), {}),
                },
            )
        assert "(2, 3)" in str(error.value) and "(3, 2)" in str(error.value), (
            f"the message should name both shapes: {error.value}"
        )

    @pytest.mark.parametrize("dtype", ["int8", "uint16", "uint8", "float16"])
    def test_unsupported_variable_dtype_raises(self, tmp_path, dtype):
        """A dtype the classic format has no `nc_type` for is rejected, not guessed.

        Args:
            dtype: A numpy dtype with no NetCDF-3 classic equivalent.

        Test scenario:
            NetCDF-3 classic has no unsigned and no 8-bit numeric types, so a
            fixture written from a `uint16` array (the natural spelling for OLCI
            counts) must fail loudly rather than be reinterpreted.
        """
        with pytest.raises(ValueError, match="unsupported dtype"):
            write_netcdf(
                tmp_path / "bad.nc", {"v": (np.zeros((2, 2), dtype=dtype), {})}
            )

    def test_unsupported_attribute_type_raises(self, tmp_path):
        """An attribute that is neither text nor a number is rejected.

        Test scenario:
            A `None` or an object array cannot be encoded as a CF attribute, and
            silently skipping it would drop a `_FillValue` or a `scale_factor` from
            a fixture -- which is exactly the class of mistake these fixtures exist
            to catch.
        """
        with pytest.raises(TypeError, match="unsupported attribute type"):
            write_netcdf(
                tmp_path / "bad_attr.nc",
                {"v": (np.zeros((2, 2), dtype="int32"), {"odd": None})},
            )


class TestWriteNetcdfRoundTrip:
    """Tests that what `write_netcdf` writes is what GDAL reads back."""

    @pytest.mark.parametrize("dtype", ["int16", "int32", "float32", "float64"])
    def test_every_supported_dtype_round_trips_in_row_order(self, tmp_path, dtype):
        """Each supported dtype reads back with its rows in the order written.

        Args:
            dtype: One of the four `nc_type`s the writer supports.

        Test scenario:
            GDAL reads a netCDF grid bottom-up (CF is south-to-north), so the
            writer reverses rows on write. The array here is row-asymmetric, so a
            missing or doubled reversal is visible rather than cancelling out --
            which would flip every fixture's geolocation and counts together, and
            so stay invisible in a reader assertion.
        """
        expected = _ROWS.astype(dtype)
        path = write_netcdf(tmp_path / f"{dtype}.nc", {"v": (expected, {})})
        array, _ = _read(path, "v")
        assert array.shape == expected.shape, (
            f"shape changed on round trip: {array.shape} != {expected.shape}"
        )
        assert np.allclose(array, expected), (
            f"{dtype} round trip changed the values or the row order:\n{array}"
        )

    def test_several_variables_in_one_file_keep_their_own_attributes(self, tmp_path):
        """Three variables coexist, each with its own data and its own attributes.

        Test scenario:
            This is the whole reason the writer exists: a real `geo_coordinates.nc`
            carries `longitude` *and* `latitude`, each with its own packing, and
            GDAL's netCDF writer cannot express that (it suffixes the names). A
            regression that cross-wired the per-variable attributes would make a
            packed-versus-unpacked confusion untestable.
        """
        path = write_netcdf(
            tmp_path / "three.nc",
            {
                "alpha": (_ROWS.astype("int32"), {"units": "a", "scale_factor": 0.5}),
                "beta": (_ROWS.astype("float32") * 2, {"units": "b"}),
                "gamma": (_ROWS.astype("int16") * 3, {"units": "c", "_FillValue": 9}),
            },
        )
        for name, multiplier, dtype in (
            ("alpha", 1, "int32"),
            ("beta", 2, "float32"),
            ("gamma", 3, "int16"),
        ):
            array, meta = _read(path, name)
            assert np.allclose(array, _ROWS.astype(dtype) * multiplier), (
                f"{name} read back wrong data:\n{array}"
            )
            assert meta["units"] == {"alpha": "a", "beta": "b", "gamma": "c"}[name], (
                f"{name} picked up another variable's attributes: {meta}"
            )
        assert "scale_factor" in _read(path, "alpha")[1], "alpha lost its packing"
        assert "scale_factor" not in _read(path, "beta")[1], (
            "beta must not inherit alpha's scale_factor"
        )

    def test_attribute_kinds_survive_as_cf_metadata(self, tmp_path):
        """Text, integer, float and array attributes all reach GDAL's attribute dict.

        Test scenario:
            The readers parse `scale_factor` / `_FillValue` / `valid_min` out of
            exactly this dict, and the writer encodes a `str` as NC_CHAR, an `int`
            as NC_INT, a `float` as NC_DOUBLE and a numpy array as a vector. A
            wrong `nc_type` makes the attribute unreadable, which would quietly
            disable the masking a fixture was built to exercise.
        """
        path = write_netcdf(
            tmp_path / "attrs.nc",
            {
                "v": (
                    _ROWS.astype("int32"),
                    {
                        "units": "K",
                        "_FillValue": 65535,
                        "scale_factor": 0.25,
                        "pair": np.array([1, 2, 3], dtype="int32"),
                    },
                )
            },
        )
        _, meta = _read(path, "v")
        assert meta["units"] == "K", meta
        assert float(meta["_FillValue"]) == 65535.0, meta
        assert float(meta["scale_factor"]) == 0.25, meta
        assert "1" in meta["pair"] and "3" in meta["pair"], (
            f"a vector attribute should round trip: {meta.get('pair')!r}"
        )

    def test_a_variable_without_attributes_is_written_and_read(self, tmp_path):
        """A variable with no attributes uses the ABSENT marker and still reads.

        Test scenario:
            An empty attribute list is encoded as two zero words rather than an
            NC_ATTRIBUTE block; getting that wrong corrupts the header for every
            variable after it, so the data assertion is the real check.
        """
        path = write_netcdf(tmp_path / "bare.nc", {"v": (_ROWS.astype("int32"), {})})
        array, _ = _read(path, "v")
        assert np.allclose(array, _ROWS), f"a bare variable read back wrong:\n{array}"

    def test_custom_dimension_names_do_not_change_the_grid(self, tmp_path):
        """Renaming the dimensions changes the header only, not the raster shape.

        Test scenario:
            The real products name their dimensions `rows`/`columns` for OLCI and
            per-grid names for SLSTR, and the header's name lengths are padded to a
            4-byte boundary -- so names of different lengths must not shift the data
            offsets.
        """
        path = write_netcdf(
            tmp_path / "dims.nc",
            {"v": (_ROWS.astype("int32"), {})},
            dimensions=("along_track_a", "across"),
        )
        array, _ = _read(path, "v")
        assert array.shape == _ROWS.shape, f"unexpected shape {array.shape}"
        assert np.allclose(array, _ROWS), (
            f"custom dimensions corrupted the data:\n{array}"
        )

    def test_a_one_pixel_grid_is_written_and_read(self, tmp_path):
        """A 1x1 grid is a valid file, so the degenerate-grid reader tests can use one."""
        path = write_netcdf(tmp_path / "tiny.nc", {"v": (np.array([[7]], "int32"), {})})
        array, _ = _read(path, "v")
        assert array.reshape(-1).tolist() == [7], f"1x1 grid read back as {array!r}"
