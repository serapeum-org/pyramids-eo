"""Unit tests for the shared raster-driver helpers in `pyramids_eo.drivers._base`."""

import inspect

import pytest

import pyramids_eo.sensors.readers as readers
from pyramids_eo.drivers._base import (
    RASTER_DRIVER_API_VERSION,
    coerce_booleans,
    coerce_floats,
    expand_sources,
    options_as_dict,
)
from pyramids_eo.drivers.fci import FciDriver
from pyramids_eo.drivers.olci import OlciDriver
from pyramids_eo.drivers.slstr import SlstrDriver


def test_api_version_is_one():
    """The declared contract version matches pyramids' RASTER_DRIVER_API_VERSION (1)."""
    assert RASTER_DRIVER_API_VERSION == 1


def test_options_none_is_empty_dict():
    """`None` options normalise to an empty dict."""
    assert options_as_dict(None) == {}


def test_options_dict_is_copied():
    """A dict is returned as a shallow copy, leaving the caller's object untouched."""
    src = {"channel": "IR_108"}
    out = options_as_dict(src)
    assert out == src
    assert out is not src


def test_options_key_value_list_parsed():
    """A GDAL-style ``KEY=VALUE`` sequence parses to a dict of strings."""
    assert options_as_dict(["reducer=median", "tile_size=256"]) == {
        "reducer": "median",
        "tile_size": "256",
    }


def test_options_value_with_equals_keeps_the_rest():
    """Only the first ``=`` splits a ``KEY=VALUE`` item; later ``=`` stay in the value."""
    assert options_as_dict(["expr=a=b"]) == {"expr": "a=b"}


def test_options_item_with_empty_value():
    """A ``KEY=`` item parses to a key mapped to the empty string."""
    assert options_as_dict(["channel="]) == {"channel": ""}


def test_options_accepts_a_tuple_sequence():
    """A tuple of ``KEY=VALUE`` strings parses the same as a list."""
    assert options_as_dict(("a=1", "b=2")) == {"a": "1", "b": "2"}


def test_options_item_without_equals_raises():
    """A sequence item that is not ``KEY=VALUE`` is a ValueError naming the item."""
    with pytest.raises(ValueError, match="channel"):
        options_as_dict(["channel"])


@pytest.mark.parametrize("bare", ["channel=IR_108", b"channel=IR_108"])
def test_options_bare_string_rejected(bare):
    """A bare str/bytes is rejected outright, not iterated character by character."""
    with pytest.raises(ValueError, match="bare"):
        options_as_dict(bare)


def test_options_accepts_a_non_dict_mapping():
    """Any Mapping, not only an exact dict, is coerced to a plain dict (not iterated as keys)."""
    from types import MappingProxyType

    out = options_as_dict(MappingProxyType({"channel": "IR_108"}))
    assert out == {"channel": "IR_108"}
    assert type(out) is dict


def test_expand_sources_single_file_is_a_one_element_list():
    """A plain path that is not a directory or glob is returned as a single-element list."""
    assert expand_sources("granule/chunk_0.nc") == ["granule/chunk_0.nc"]


def test_expand_sources_directory_globs_the_pattern(tmp_path):
    """A directory source expands to its sorted entries matching the pattern."""
    for name in ["c1.nc", "c0.nc", "note.txt"]:
        (tmp_path / name).write_bytes(b"")
    out = expand_sources(str(tmp_path), pattern="*.nc")
    assert out == sorted([str(tmp_path / "c0.nc"), str(tmp_path / "c1.nc")])


def test_expand_sources_glob_string_is_expanded(tmp_path):
    """A glob string expands to its sorted matches, ignoring the directory pattern."""
    for name in ["b.nc", "a.nc"]:
        (tmp_path / name).write_bytes(b"")
    out = expand_sources(str(tmp_path / "*.nc"))
    assert out == sorted([str(tmp_path / "a.nc"), str(tmp_path / "b.nc")])


def test_expand_sources_glob_matching_nothing_is_empty(tmp_path):
    """A glob string that matches no file expands to an empty list."""
    assert expand_sources(str(tmp_path / "*.zzz")) == []


def test_expand_sources_empty_directory_is_empty(tmp_path):
    """A directory with no match expands to an empty list."""
    assert expand_sources(str(tmp_path), pattern="*.nc") == []


def test_expand_sources_empty_string_is_empty():
    """An empty source expands to [], not the current working directory's contents.

    Uses the default ``*`` pattern: without the guard, ``Path("")`` is ``.`` and would list the
    CWD's files, so an empty result proves the empty-string short-circuit (not an empty CWD).
    """
    assert expand_sources("") == []


def test_expand_sources_directory_excludes_subdirectories(tmp_path):
    """The directory branch returns files only, not a subdirectory that matches the pattern."""
    (tmp_path / "c0.nc").write_bytes(b"")
    (tmp_path / "sub.nc").mkdir()
    assert expand_sources(str(tmp_path), pattern="*.nc") == [str(tmp_path / "c0.nc")]


def test_expand_sources_directory_name_with_glob_metachar(tmp_path):
    """A real directory whose name contains a glob metachar is treated literally, not as a pattern."""
    granule = tmp_path / "granule[1]"
    granule.mkdir()
    (granule / "c0.nc").write_bytes(b"")
    assert expand_sources(str(granule), pattern="*.nc") == [str(granule / "c0.nc")]


def test_expand_sources_existing_file_with_glob_metachar(tmp_path):
    """A real file whose name contains a glob metachar is returned literally, not expanded."""
    chunk = tmp_path / "a[1].nc"
    chunk.write_bytes(b"")
    assert expand_sources(str(chunk)) == [str(chunk)]


class TestCoerceBooleansSpellings:
    """`coerce_booleans` over every spelling and every value type it is handed."""

    @pytest.mark.parametrize("word", ["1", "true", "TRUE", "True", "yes", "on", " ON "])
    def test_every_true_spelling_becomes_true(self, word):
        """Each accepted true spelling, including padded and mixed case, maps to `True`.

        Args:
            word: The string spelling as it arrives from a ``KEY=VALUE`` option.
        """
        settings = {"flag": word}
        coerce_booleans(settings, ("flag",), driver="d")
        assert settings["flag"] is True, f"{word!r} should map to True"

    @pytest.mark.parametrize(
        "word", ["0", "false", "FALSE", "False", "no", "off", " off\t"]
    )
    def test_every_false_spelling_becomes_false(self, word):
        """Each accepted false spelling maps to `False`, not to a truthy string.

        Args:
            word: The string spelling as it arrives from a ``KEY=VALUE`` option.

        Test scenario:
            A non-empty string is truthy, so a missed spelling turns the option
            *on* when the caller asked for it off. The suite previously pinned
            ``"false"`` on its own, which cannot show the other four words are
            mapped at all.
        """
        settings = {"flag": word}
        coerce_booleans(settings, ("flag",), driver="d")
        assert settings["flag"] is False, f"{word!r} should map to False"

    def test_a_named_key_that_is_absent_is_skipped(self):
        """Naming an option the caller did not pass neither adds nor raises.

        Test scenario:
            Drivers name every boolean option they accept, most of which are absent
            on any one call, so an absent key must be a no-op rather than a
            ``KeyError`` or an invented ``False``.
        """
        settings = {"channel": "S5"}
        coerce_booleans(settings, ("geolocation", "calibrate"), driver="d")
        assert settings == {"channel": "S5"}, f"settings were altered: {settings}"

    @pytest.mark.parametrize("value", [0, 1, None, 1.5, [], ("yes",)])
    def test_a_non_string_value_is_left_for_the_reader(self, value):
        """Only strings are converted; every other type reaches the reader unchanged.

        Args:
            value: A non-string, non-bool option value.

        Test scenario:
            Imposing truthiness here would turn ``0`` into ``False`` and ``[]``
            into ``False`` behind the reader's back, hiding a caller's mistake
            instead of letting the reader reject it.
        """
        settings = {"flag": value}
        coerce_booleans(settings, ("flag",), driver="d")
        assert settings["flag"] is value, f"{value!r} should be untouched"

    @pytest.mark.parametrize("word", ["maybe", "", " ", "2", "truthy", "none"])
    def test_an_unrecognised_spelling_is_rejected_naming_the_option(self, word):
        """A string that is not a boolean spelling raises, naming driver and option.

        Args:
            word: A string that is not an accepted boolean spelling.
        """
        with pytest.raises(ValueError, match=r"open option 'flag' must be a boolean"):
            coerce_booleans({"flag": word}, ("flag",), driver="d")


class TestCoerceFloatsSpellings:
    """`coerce_floats` over the numeric spellings and value types it is handed."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("0.983", 0.983),
            ("1", 1.0),
            ("-2.5", -2.5),
            ("1e-06", 1e-06),
            ("  0.5 ", 0.5),
        ],
    )
    def test_a_numeric_string_becomes_a_float(self, text, expected):
        """Every numeric spelling `float()` accepts is converted, not just a decimal.

        Args:
            text: The string value as it arrives from a ``KEY=VALUE`` option.
            expected: The float it must become.

        Test scenario:
            An integer-looking string is the easy miss: ``"1"`` stays a ``str`` if
            the conversion is gated on a decimal point, and only fails later inside
            the reader's arithmetic.
        """
        settings = {"distance": text}
        coerce_floats(settings, ("distance",), driver="d")
        assert isinstance(settings["distance"], float), (
            f"{text!r} stayed a {type(settings['distance']).__name__}"
        )
        assert settings["distance"] == expected, (
            f"{text!r} became {settings['distance']}"
        )

    @pytest.mark.parametrize("value", [1, 2.5, None, True, [1]])
    def test_a_non_string_value_is_left_alone(self, value):
        """A value that is not a string is passed through for the reader to interpret.

        Args:
            value: A non-string option value.

        Test scenario:
            ``float(None)`` raises ``TypeError``, not ``ValueError``, so dropping
            the type guard would turn an absent-or-odd value into an error this
            helper does not describe. An ``int`` is already usable arithmetically
            and is deliberately not widened.
        """
        settings = {"distance": value}
        coerce_floats(settings, ("distance",), driver="d")
        assert settings["distance"] is value, f"{value!r} should be untouched"

    def test_a_named_key_that_is_absent_is_skipped(self):
        """Naming an option the caller did not pass leaves the dict alone."""
        settings = {"channel": "ir_105"}
        coerce_floats(settings, ("sun_earth_distance",), driver="d")
        assert settings == {"channel": "ir_105"}, f"settings were altered: {settings}"

    @pytest.mark.parametrize("text", ["far", "", "1,5", "0x10", "--3"])
    def test_a_non_numeric_string_is_rejected_naming_the_option(self, text):
        """A string that is not a number raises where the option is named.

        Args:
            text: A string ``float()`` cannot parse.
        """
        with pytest.raises(
            ValueError, match=r"open option 'distance' must be a finite number"
        ):
            coerce_floats({"distance": text}, ("distance",), driver="d")

    @pytest.mark.parametrize("text", ["nan", "NaN", "inf", "-inf", "Infinity", " nan "])
    def test_a_non_finite_spelling_is_rejected_like_a_non_numeric_one(self, text):
        """`"nan"` and `"inf"` parse as floats but are refused all the same.

        Args:
            text: A spelling ``float()`` accepts but that is not finite.

        Test scenario:
            These are the spellings a numeric guard misses by construction: they
            satisfy ``float()``, so a conversion that only catches ``ValueError``
            lets them through. A non-finite ``sun_earth_distance`` then voids every
            calibrated pixel, and the same message must name the option either way
            so the two rejections are indistinguishable to a caller.
        """
        with pytest.raises(
            ValueError, match=r"open option 'distance' must be a finite number"
        ):
            coerce_floats({"distance": text}, ("distance",), driver="d")

    def test_a_non_finite_float_value_is_not_rejected(self):
        """The finiteness guard applies to strings only, as the type guard does.

        Test scenario:
            A caller who passes a real ``float('nan')`` in a dict is not going
            through the ``KEY=VALUE`` path this helper exists for, so the value is
            left for the reader -- the same division of labour as the ``str``-only
            conversion. Pinning it keeps the guard from quietly widening.
        """
        value = float("nan")
        settings = {"distance": value}
        coerce_floats(settings, ("distance",), driver="d")
        assert settings["distance"] is value, (
            f"a real non-finite float should be untouched, got {settings['distance']!r}"
        )


def _annotated_keywords(reader, annotation):
    """Return the reader's keyword names annotated exactly `annotation`.

    The reader modules use ``from __future__ import annotations``, so every
    annotation is already a string and no evaluation is needed.

    Args:
        reader: The reader function to inspect.
        annotation: The annotation source text to match (e.g. ``"bool"``).

    Returns:
        The matching parameter names, in declaration order.
    """
    return [
        name
        for name, parameter in inspect.signature(reader).parameters.items()
        if parameter.annotation == annotation
    ]


_DRIVERS = [
    ("fci", FciDriver, "read_fci_l1c", "ir_105"),
    ("olci", OlciDriver, "read_olci", "Oa08"),
    ("slstr", SlstrDriver, "read_slstr", "S5"),
]


class TestEveryCoercibleReaderKeywordIsCoerced:
    """Each driver must coerce *every* boolean and numeric keyword it forwards.

    The coercion lists are hand-maintained tuples, so a reader keyword added later
    silently arrives as a ``str`` through ``read_file`` -- which is how FCI's
    ``sun_earth_distance`` came to fail inside the calibration arithmetic with a
    raw ``TypeError``. These derive the keyword set from the **reader's own
    signature** instead of restating the tuples, so a new ``bool`` or ``float``
    option fails here until its driver coerces it.
    """

    @staticmethod
    def _capture(monkeypatch, reader_name):
        """Replace the reader facade with a recorder and return the captured kwargs."""
        captured: dict = {}

        def _recorder(*args, **kwargs):
            captured.update(kwargs)
            return "dataset"

        monkeypatch.setattr(readers, reader_name, _recorder)
        return captured

    @staticmethod
    def _source(tmp_path, name):
        """Return a source string each driver accepts without decoding anything."""
        granule = tmp_path / f"{name}_granule"
        granule.mkdir()
        (granule / "chunk_FDHSI_0.nc").write_bytes(b"")
        return str(granule)

    @pytest.mark.parametrize("name, driver_class, reader_name, channel", _DRIVERS)
    def test_boolean_keywords_arrive_as_real_booleans(
        self, tmp_path, monkeypatch, name, driver_class, reader_name, channel
    ):
        """Every `bool` keyword the reader declares reaches it as `False`, not `"false"`.

        Args:
            name: The driver's registry name.
            driver_class: The driver class under test.
            reader_name: The reader facade attribute the driver calls.
            channel: A channel name the driver accepts.

        Test scenario:
            ``"false"`` is a non-empty string and therefore truthy, so an
            un-coerced boolean turns its option on when the caller asked for it
            off -- the defect this helper exists to prevent, checked here against
            the reader's declared options rather than against a chosen one.
        """
        reader = getattr(readers, reader_name)
        keywords = _annotated_keywords(reader, "bool")
        assert keywords, f"{reader_name} declares no boolean keyword to check"
        for keyword in keywords:
            captured = self._capture(monkeypatch, reader_name)
            driver_class().read(
                self._source(tmp_path, f"{name}_{keyword}"),
                options=[f"channel={channel}", f"{keyword}=false"],
            )
            assert captured.get(keyword) is False, (
                f"{name}: {keyword}='false' reached the reader as "
                f"{captured.get(keyword)!r}; add it to this driver's coerce_booleans call"
            )

    @pytest.mark.parametrize("name, driver_class, reader_name, channel", _DRIVERS)
    def test_numeric_keywords_arrive_as_real_floats(
        self, tmp_path, monkeypatch, name, driver_class, reader_name, channel
    ):
        """Every `float` keyword the reader declares reaches it as a number.

        Args:
            name: The driver's registry name.
            driver_class: The driver class under test.
            reader_name: The reader facade attribute the driver calls.
            channel: A channel name the driver accepts.

        Test scenario:
            A numeric option left as a ``str`` fails much later inside the
            reader's arithmetic with a message naming neither the option nor the
            driver. OLCI and SLSTR declare no numeric keyword, so this asserts
            nothing for them today and starts asserting the moment one is added.
        """
        reader = getattr(readers, reader_name)
        for keyword in _annotated_keywords(reader, "float"):
            captured = self._capture(monkeypatch, reader_name)
            driver_class().read(
                self._source(tmp_path, f"{name}_{keyword}"),
                options=[f"channel={channel}", f"{keyword}=0.5"],
            )
            assert captured.get(keyword) == 0.5, (
                f"{name}: {keyword}='0.5' reached the reader as "
                f"{captured.get(keyword)!r}; add it to this driver's coerce_floats call"
            )
            assert isinstance(captured[keyword], float), (
                f"{name}: {keyword} must be a float, got "
                f"{type(captured[keyword]).__name__}"
            )
