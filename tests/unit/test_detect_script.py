import pytest

from robust_lid.constants import UNDEFINED_SCRIPT
from robust_lid.utils import detect_script


@pytest.mark.unit
def test_latin_uppercase() -> None:
    assert detect_script("HELLO") == "Latn"


@pytest.mark.unit
def test_latin_lowercase() -> None:
    assert detect_script("hello") == "Latn"


@pytest.mark.unit
def test_latin_extended_diacritics() -> None:
    assert detect_script("café naïve résumé") == "Latn"


@pytest.mark.unit
def test_hangul() -> None:
    assert detect_script("안녕하세요") == "Hang"


@pytest.mark.unit
def test_han_only() -> None:
    assert detect_script("你好世界") == "Hani"


@pytest.mark.unit
def test_japanese_mixed_becomes_jpan() -> None:
    assert detect_script("こんにちは世界") == "Jpan"
    assert detect_script("カタカナ漢字") == "Jpan"


@pytest.mark.unit
def test_arabic() -> None:
    assert detect_script("مرحبا") == "Arab"


@pytest.mark.unit
def test_cyrillic() -> None:
    assert detect_script("Привет") == "Cyrl"


@pytest.mark.unit
def test_empty_returns_undefined_script() -> None:
    assert detect_script("") == UNDEFINED_SCRIPT


@pytest.mark.unit
def test_digits_only_returns_undefined_script() -> None:
    assert detect_script("12345 !!! ???") == UNDEFINED_SCRIPT


@pytest.mark.unit
def test_majority_wins_with_punctuation() -> None:
    assert detect_script("hello, world!") == "Latn"


@pytest.mark.unit
def test_devanagari() -> None:
    assert detect_script("नमस्ते दुनिया") == "Deva"


@pytest.mark.unit
def test_bengali() -> None:
    assert detect_script("হ্যালো পৃথিবী") == "Beng"


@pytest.mark.unit
def test_thai() -> None:
    assert detect_script("สวัสดีชาวโลก") == "Thai"


@pytest.mark.unit
def test_greek() -> None:
    assert detect_script("Γειά σου κόσμε") == "Grek"


@pytest.mark.unit
def test_hebrew() -> None:
    assert detect_script("שלום עולם") == "Hebr"
