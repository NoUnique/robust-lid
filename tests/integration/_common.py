"""Shared constants for LID integration benchmarks."""

from __future__ import annotations

# Top 30 most widely used world languages (by combined speakers, internet
# content, and presence in NLP benchmarks). All present in WiLi-2018 labels;
# papluca/language-identification covers 16 of the 30 Latin/CJK/Arabic/Cyrl
# subset plus `bg` and `sw` that aren't in this list.
MAJOR_LANGS: list[str] = [
    # Tier 1 — Top 14 by speakers
    "eng",
    "zho",
    "spa",
    "hin",
    "ara",
    "por",
    "rus",
    "jpn",
    "deu",
    "fra",
    "kor",
    "tur",
    "ita",
    "vie",
    # Tier 2 — Regional heavyweights, diverse scripts
    "ben",
    "urd",
    "ind",
    "tha",
    "pol",
    "nld",
    "ukr",
    "ell",
    "ces",
    "swe",
    "ron",
    "hun",
    "heb",
    "fas",
    "fin",
    "msa",
]

# Macrolanguage / individual-language variants and near-equivalents that
# different backends may emit interchangeably.
_ARABIC_VARIANTS = {
    "ara",
    "arb",  # Modern Standard
    "arz",
    "apc",
    "ary",
    "acm",
    "aeb",
    "ars",
    "acq",
    "ajp",  # dialects
}
_CHINESE_VARIANTS = {"zho", "cmn", "wuu", "yue", "nan", "hak", "lzh", "gan", "hsn"}
_MALAY_VARIANTS = {"msa", "ind", "zlm", "zsm", "min", "bjn"}

LANG_EQUIVS: dict[str, set[str]] = {
    **{v: _ARABIC_VARIANTS for v in _ARABIC_VARIANTS},
    **{v: _CHINESE_VARIANTS for v in _CHINESE_VARIANTS},
    **{v: _MALAY_VARIANTS for v in _MALAY_VARIANTS},
    "hin": {"hin", "urd"},  # Hindi and Urdu share much lexicon
    "urd": {"urd", "hin"},
    "nob": {"nob", "nor", "nno"},
    "fas": {"fas", "pes", "prs"},
    "pes": {"fas", "pes", "prs"},
}


def expected_set(lang: str) -> set[str]:
    """Language-only equivalence set for datasets whose ground-truth labels
    don't include script (WiLi-2018, papluca)."""
    return LANG_EQUIVS.get(lang, {lang})


def matches_lang_script(predicted: str, expected: str) -> bool:
    """Strict `lang_Script` comparison for datasets whose labels carry script
    info (FLORES+). Both arguments are ``'xxx_XXXX'`` strings.

    Two layers of equivalence are applied:

    * **Language**: macrolanguage/individual-language variants from
      ``LANG_EQUIVS`` (e.g. ``arb ≡ ara ≡ arz ≡ apc`` for Arabic).
    * **Script**: Unicode supercode classes from ``_expand_script`` (e.g.
      ``Hans ≡ Hani ≡ Hant ≡ Hanb`` so Simplified/Traditional/Han all count
      as "Chinese script").

    A mismatch on either layer fails — so predicting ``eng_Cyrl`` on a
    ``eng_Latn`` ground truth is correctly caught as wrong.
    """
    from robust_lid.utils import _expand_script

    pred_lang, _, pred_script = predicted.partition("_")
    exp_lang, _, exp_script = expected.partition("_")

    if pred_lang not in LANG_EQUIVS.get(exp_lang, {exp_lang}):
        return False
    if not exp_script:  # ground truth has no script constraint
        return True
    pred_scripts = _expand_script(pred_script) if pred_script else set()
    return bool(pred_scripts & _expand_script(exp_script))
