from collections.abc import Iterator
from pathlib import Path

import pytest

from robust_lid.utils import ISOConverter, set_converter


@pytest.fixture
def glotscript_small() -> Path:
    return Path(__file__).parent / "fixtures" / "glotscript_small.tsv"


@pytest.fixture
def small_converter() -> ISOConverter:
    mapping = {"eng": "eng", "kor": "kor", "jpn": "jpn", "zho": "zho", "fra": "fra"}
    iso639_3_map = {"en": "eng", "ko": "kor", "ja": "jpn", "zh": "zho", "fr": "fra"}
    lang_to_scripts: dict[str, frozenset[str]] = {
        "eng": frozenset({"Latn"}),
        "kor": frozenset({"Hang"}),
        "jpn": frozenset({"Jpan"}),
        "zho": frozenset({"Hani"}),
        "fra": frozenset({"Latn"}),
    }
    return ISOConverter(
        mapping=mapping,
        iso639_3_map=iso639_3_map,
        lang_to_scripts=lang_to_scripts,
    )


@pytest.fixture(autouse=True)
def _reset_global_converter() -> Iterator[None]:
    """Ensure converter singleton state never leaks between tests."""
    yield
    set_converter(None)
