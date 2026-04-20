from pathlib import Path

import pytest

from robust_lid.constants import UNDEFINED_LANG
from robust_lid.utils import ISOConverter, normalize_language_code


@pytest.mark.unit
def test_two_letter_code_maps_to_three(small_converter: ISOConverter) -> None:
    assert small_converter.to_iso639_3("en") == "eng"
    assert small_converter.to_iso639_3("ko") == "kor"


@pytest.mark.unit
def test_three_letter_code_passes_through(small_converter: ISOConverter) -> None:
    assert small_converter.to_iso639_3("eng") == "eng"


@pytest.mark.unit
def test_locale_suffix_stripped(small_converter: ISOConverter) -> None:
    assert small_converter.to_iso639_3("en-US") == "eng"
    assert small_converter.to_iso639_3("zh_Hans") == "zho"


@pytest.mark.unit
def test_unknown_code_returns_none(small_converter: ISOConverter) -> None:
    assert small_converter.to_iso639_3("xx") is None
    assert small_converter.to_iso639_3("xyz") is None


@pytest.mark.unit
def test_case_insensitive(small_converter: ISOConverter) -> None:
    assert small_converter.to_iso639_3("EN") == "eng"
    assert small_converter.to_iso639_3("Eng") == "eng"


@pytest.mark.unit
def test_normalize_injects_converter(small_converter: ISOConverter) -> None:
    assert normalize_language_code("en", converter=small_converter) == "eng"
    assert normalize_language_code("xx", converter=small_converter) == UNDEFINED_LANG


@pytest.mark.unit
def test_loads_from_tsv_path(glotscript_small: Path) -> None:
    converter = ISOConverter(iso639_3_map={"en": "eng"}, tsv_path=glotscript_small)
    assert "eng" in converter.mapping
    assert "kor" in converter.mapping


@pytest.mark.unit
def test_missing_tsv_is_warning_not_error(tmp_path: Path) -> None:
    converter = ISOConverter(iso639_3_map={}, tsv_path=tmp_path / "missing.tsv")
    assert converter.mapping == {}


@pytest.mark.unit
def test_default_init_loads_real_pycountry_map(glotscript_small: Path) -> None:
    converter = ISOConverter(tsv_path=glotscript_small)
    assert converter.iso639_3_map.get("en") == "eng"
    assert converter.iso639_3_map.get("ko") == "kor"


@pytest.mark.unit
def test_three_letter_code_from_mapping_only() -> None:
    # "qaa" is in GlotScript aux but not in pycountry — exercises the
    # `code in self.mapping` fallback branch.
    converter = ISOConverter(mapping={"qaa": "qaa"}, iso639_3_map={})
    assert converter.to_iso639_3("qaa") == "qaa"


@pytest.mark.unit
def test_unknown_two_letter_returns_none() -> None:
    converter = ISOConverter(mapping={}, iso639_3_map={"en": "eng"})
    assert converter.to_iso639_3("zz") is None


@pytest.mark.unit
def test_get_converter_instantiates_default_on_first_call(
    glotscript_small: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robust_lid import utils

    monkeypatch.setattr(utils, "GLOTSCRIPT_TSV", glotscript_small)
    utils.set_converter(None)
    converter = utils.get_converter()
    assert isinstance(converter, ISOConverter)
    # Second call returns the cached instance
    assert utils.get_converter() is converter


@pytest.mark.unit
def test_scripts_for_known_three_letter(small_converter: ISOConverter) -> None:
    assert small_converter.scripts_for("eng") == frozenset({"Latn"})
    assert small_converter.scripts_for("kor") == frozenset({"Hang"})


@pytest.mark.unit
def test_scripts_for_two_letter(small_converter: ISOConverter) -> None:
    assert small_converter.scripts_for("en") == frozenset({"Latn"})
    assert small_converter.scripts_for("ko") == frozenset({"Hang"})


@pytest.mark.unit
def test_scripts_for_unknown_returns_empty(small_converter: ISOConverter) -> None:
    assert small_converter.scripts_for("xx") == frozenset()
    assert small_converter.scripts_for("zzz") == frozenset()


@pytest.mark.unit
def test_load_from_tsv_populates_lang_to_scripts(glotscript_small: Path) -> None:
    converter = ISOConverter(iso639_3_map={}, tsv_path=glotscript_small)
    assert "Latn" in converter.lang_to_scripts.get("eng", frozenset())
    assert "Hang" in converter.lang_to_scripts.get("kor", frozenset())
    assert "Arab" in converter.lang_to_scripts.get("ara", frozenset())


@pytest.mark.unit
def test_deprecated_iso1_aliases_normalize() -> None:
    # Need the full pycountry map (for iw→he→heb, in→id→ind, etc.)
    converter = ISOConverter(
        mapping={"heb": "heb", "ind": "ind", "yid": "yid", "jav": "jav"},
        lang_to_scripts={},
    )
    assert converter.to_iso639_3("iw") == "heb"
    assert converter.to_iso639_3("in") == "ind"
    assert converter.to_iso639_3("ji") == "yid"
    assert converter.to_iso639_3("jw") == "jav"
