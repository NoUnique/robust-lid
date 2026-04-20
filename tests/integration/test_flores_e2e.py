"""End-to-end LID accuracy benchmark on FLORES-200+ (gated).

`openlanguagedata/flores_plus` is the ungated-license successor to
`facebook/flores` but still requires HF authentication (free: accept the
dataset's terms on its Hub page, then set HF_TOKEN in `.env`).

Skips automatically when HF_TOKEN is not set.

Run with:  uv run pytest -m "slow and network" tests/integration/test_flores_e2e.py -s
"""

from __future__ import annotations

import os

import pytest

datasets = pytest.importorskip("datasets")

from robust_lid import RobustLID  # noqa: E402

from ._common import MAJOR_LANGS, matches_lang_script  # noqa: E402

# FLORES+ uses `<iso639-3>_<script>` config names, but with individual
# (not macro) language codes for some macrolanguages.
FLORES_CONFIGS: dict[str, str] = {
    "eng": "eng_Latn",
    "zho": "cmn_Hans",
    "spa": "spa_Latn",
    "hin": "hin_Deva",
    "ara": "arb_Arab",
    "por": "por_Latn",
    "rus": "rus_Cyrl",
    "jpn": "jpn_Jpan",
    "deu": "deu_Latn",
    "fra": "fra_Latn",
    "kor": "kor_Hang",
    "tur": "tur_Latn",
    "ita": "ita_Latn",
    "vie": "vie_Latn",
    "ben": "ben_Beng",
    "urd": "urd_Arab",
    "ind": "ind_Latn",
    "tha": "tha_Thai",
    "pol": "pol_Latn",
    "nld": "nld_Latn",
    "ukr": "ukr_Cyrl",
    "ell": "ell_Grek",
    "ces": "ces_Latn",
    "swe": "swe_Latn",
    "ron": "ron_Latn",
    "hun": "hun_Latn",
    "heb": "heb_Hebr",
    "fas": "pes_Arab",
    "fin": "fin_Latn",
    "msa": "zsm_Latn",
}
FLORES_LANGS = [lang for lang in MAJOR_LANGS if lang in FLORES_CONFIGS]
SAMPLES_PER_LANG: int = 30

pytestmark = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set — copy .env.example to .env and fill in to run FLORES tests",
)


@pytest.fixture(scope="module")
def lid_engine() -> RobustLID:
    return RobustLID()


@pytest.fixture(scope="module")
def flores_by_lang() -> dict[str, list[str]]:
    token = os.environ["HF_TOKEN"]
    buckets: dict[str, list[str]] = {}
    for lang in FLORES_LANGS:
        ds = datasets.load_dataset(
            "openlanguagedata/flores_plus",
            FLORES_CONFIGS[lang],
            split="dev",
            token=token,
        )
        buckets[lang] = [row["text"] for row in ds.select(range(min(SAMPLES_PER_LANG, len(ds))))]
    return buckets


@pytest.mark.slow
@pytest.mark.network
def test_flores_major_languages_accuracy_report(
    lid_engine: RobustLID, flores_by_lang: dict[str, list[str]]
) -> None:
    """Strict `lang_Script` comparison — FLORES labels carry script so we
    test both dimensions together; script mis-assignments (e.g. eng→Cyrl)
    are counted as misses here."""
    rows: list[tuple[str, int, int, list[tuple[str, str]]]] = []
    total_correct = 0
    total_count = 0
    for lang in FLORES_LANGS:
        expected_label = FLORES_CONFIGS[lang]  # e.g. 'eng_Latn'
        samples = flores_by_lang[lang]
        hits = 0
        misses: list[tuple[str, str]] = []
        for s in samples:
            predicted, _c = lid_engine.predict(s)
            if matches_lang_script(predicted, expected_label):
                hits += 1
            else:
                misses.append((predicted, s[:60]))
        rows.append((lang, hits, len(samples), misses))
        total_correct += hits
        total_count += len(samples)

    print("\nFLORES-200+ — " + "=" * 58)
    print(f"{'LANG':<8} {'ACC':<10} {'HITS':<10}")
    print("-" * 72)
    for lang, hits, n, _misses in rows:
        print(f"{FLORES_CONFIGS[lang]:<8} {hits / n:>6.1%}    {hits}/{n}")
    overall = total_correct / total_count if total_count else 0.0
    print("-" * 72)
    print(f"{'TOTAL':<8} {overall:>6.1%}    {total_correct}/{total_count}")
    print("=" * 72)

    print("\nSample misses:")
    for lang, _hits, _n, misses in rows:
        if misses:
            print(f"  {FLORES_CONFIGS[lang]}:")
            for predicted, snippet in misses[:2]:
                print(f"    → {predicted:<10}  {snippet}")

    assert overall > 0.0


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("lang", FLORES_LANGS)
def test_flores_each_language_majority_hit(
    lid_engine: RobustLID, flores_by_lang: dict[str, list[str]], lang: str
) -> None:
    expected_label = FLORES_CONFIGS[lang]
    samples = flores_by_lang[lang]
    hits = sum(1 for s in samples if matches_lang_script(lid_engine.predict(s)[0], expected_label))
    accuracy = hits / len(samples)
    assert accuracy > 0.5, f"{expected_label}: {hits}/{len(samples)} = {accuracy:.1%} below 50%"
