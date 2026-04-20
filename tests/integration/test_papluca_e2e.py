"""End-to-end LID accuracy benchmark on papluca/language-identification.

A cross-domain check against WiLi-2018 (which is Wikipedia-only). The papluca
dataset is a balanced mix of product reviews, news, and other short text in
20 languages, 500 samples each in the test split.

Run with:  uv run pytest -m "slow and network" tests/integration/test_papluca_e2e.py -s
"""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")

from robust_lid import RobustLID  # noqa: E402
from robust_lid.utils import normalize_language_code  # noqa: E402

from ._common import expected_set  # noqa: E402

# papluca uses 2-letter ISO 639-1 codes — map to 3-letter for comparison.
PAPLUCA_LANGS_ISO1: list[str] = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "nl",
    "pl",
    "pt",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]
SAMPLES_PER_LANG: int = 50


@pytest.fixture(scope="module")
def lid_engine() -> RobustLID:
    return RobustLID()


@pytest.fixture(scope="module")
def papluca_by_lang() -> dict[str, list[str]]:
    """Bucket papluca test-split samples by the 3-letter ISO 639-3 code we
    compare against. Same conversion our models apply to their own outputs."""
    ds = datasets.load_dataset("papluca/language-identification", split="test")
    buckets: dict[str, list[str]] = {
        normalize_language_code(code): [] for code in PAPLUCA_LANGS_ISO1
    }
    for row in ds:
        iso3 = normalize_language_code(row["labels"])
        if iso3 in buckets and len(buckets[iso3]) < SAMPLES_PER_LANG:
            buckets[iso3].append(row["text"])
        if all(len(v) >= SAMPLES_PER_LANG for v in buckets.values()):
            break
    return buckets


@pytest.mark.slow
@pytest.mark.network
def test_papluca_accuracy_report(
    lid_engine: RobustLID, papluca_by_lang: dict[str, list[str]]
) -> None:
    rows: list[tuple[str, int, int, list[tuple[str, str]]]] = []
    total_correct = 0
    total_count = 0

    for lang in sorted(papluca_by_lang.keys()):
        acceptable = expected_set(lang)
        samples = papluca_by_lang[lang]
        hits = 0
        misses: list[tuple[str, str]] = []
        for sample in samples:
            predicted, _conf = lid_engine.predict(sample)
            predicted_lang = predicted.split("_")[0]
            if predicted_lang in acceptable:
                hits += 1
            else:
                misses.append((predicted, sample[:60]))
        rows.append((lang, hits, len(samples), misses))
        total_correct += hits
        total_count += len(samples)

    print("\npapluca — " + "=" * 62)
    print(f"{'LANG':<6} {'ACC':<10} {'HITS':<10}")
    print("-" * 72)
    for lang, hits, n, _misses in rows:
        acc = hits / n if n else 0.0
        print(f"{lang:<6} {acc:>6.1%}    {hits}/{n}")
    overall = total_correct / total_count if total_count else 0.0
    print("-" * 72)
    print(f"{'TOTAL':<6} {overall:>6.1%}    {total_correct}/{total_count}")
    print("=" * 72)

    print("\nSample misses:")
    for lang, _hits, _n, misses in rows:
        if misses:
            print(f"  {lang}:")
            for predicted, snippet in misses[:2]:
                print(f"    → {predicted:<10}  {snippet}")

    assert overall > 0.0


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("lang_iso1", PAPLUCA_LANGS_ISO1)
def test_papluca_each_language_majority_hit(
    lid_engine: RobustLID, papluca_by_lang: dict[str, list[str]], lang_iso1: str
) -> None:
    lang = normalize_language_code(lang_iso1)
    acceptable = expected_set(lang)
    samples = papluca_by_lang[lang]
    hits = sum(1 for s in samples if lid_engine.predict(s)[0].split("_")[0] in acceptable)
    accuracy = hits / len(samples)
    assert accuracy > 0.5, f"{lang} ({lang_iso1}): {hits}/{len(samples)} = {accuracy:.1%} below 50%"
