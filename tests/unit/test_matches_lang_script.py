"""Unit tests for the strict lang_Script comparator used by the FLORES+ E2E
test. Kept in unit/ so they run in the fast, no-network default suite."""

import pytest

from tests.integration._common import matches_lang_script


@pytest.mark.unit
def test_exact_match() -> None:
    assert matches_lang_script("eng_Latn", "eng_Latn")


@pytest.mark.unit
def test_same_lang_different_script_is_rejected() -> None:
    """This is the whole point: a backend that predicts 'eng_Cyrl' on
    English text should be counted wrong, even though the language matches."""
    assert not matches_lang_script("eng_Cyrl", "eng_Latn")
    assert not matches_lang_script("srp_Latn", "srp_Cyrl")


@pytest.mark.unit
def test_different_lang_same_script_is_rejected() -> None:
    assert not matches_lang_script("deu_Latn", "eng_Latn")


@pytest.mark.unit
def test_macrolang_equivalence_matches() -> None:
    """GlotLID's fine-grained Arabic dialect should count as correct when
    the FLORES label is Modern Standard Arabic."""
    assert matches_lang_script("apc_Arab", "arb_Arab")
    assert matches_lang_script("arz_Arab", "arb_Arab")
    assert matches_lang_script("ary_Arab", "ara_Arab")


@pytest.mark.unit
def test_script_supercode_equivalence_matches() -> None:
    """Hans / Hani / Hant / Hanb all represent Han writing; detect_script
    returns the coarse 'Hani' but FLORES labels use 'Hans'. These should
    match."""
    assert matches_lang_script("zho_Hani", "cmn_Hans")
    assert matches_lang_script("cmn_Hant", "cmn_Hans")
    assert matches_lang_script("kor_Hang", "kor_Kore")


@pytest.mark.unit
def test_chinese_macrolang_variants_all_match() -> None:
    assert matches_lang_script("cmn_Hans", "zho_Hans")
    assert matches_lang_script("yue_Hant", "cmn_Hant")


@pytest.mark.unit
def test_malay_variants_match() -> None:
    assert matches_lang_script("zsm_Latn", "msa_Latn")
    assert matches_lang_script("ind_Latn", "msa_Latn")


@pytest.mark.unit
def test_missing_script_in_prediction_does_not_match_when_expected_has_one() -> None:
    assert not matches_lang_script("eng", "eng_Latn")


@pytest.mark.unit
def test_missing_script_in_expected_allows_any_script() -> None:
    """If the ground truth only specifies the language, any script passes
    the script layer. (Unused by FLORES but defensible default.)"""
    assert matches_lang_script("eng_Latn", "eng")
    assert matches_lang_script("eng_Cyrl", "eng")
    assert not matches_lang_script("deu_Latn", "eng")
