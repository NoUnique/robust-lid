"""End-to-end LID accuracy benchmark on WiLi-2018.

WiLi-2018 (martinthoma/wili_2018) is an ungated Wikipedia-derived LID
benchmark with 235 languages labelled directly with ISO 639-3 codes.

Runtime: loads WiLi (~30s download first time), plus a one-time
~1.5 GB fastText download for the ensemble. Subsequent runs are fast.

Run with:  uv run pytest -m "slow and network" tests/integration/test_lid_benchmark.py -s

Requires the `e2e` extras:  uv sync --extra dev --extra e2e
"""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")

from robust_lid import RobustLID  # noqa: E402

from ._common import MAJOR_LANGS, expected_set  # noqa: E402

SAMPLES_PER_LANG: int = 30


@pytest.fixture(scope="module")
def lid_engine() -> RobustLID:
    return RobustLID()


@pytest.fixture(scope="module")
def wili_by_lang() -> dict[str, list[str]]:
    """Load WiLi-2018 test split and bucket sentences by ISO 639-3 code."""
    ds = datasets.load_dataset("martinthoma/wili_2018", split="test")
    names: list[str] = ds.features["label"].names
    buckets: dict[str, list[str]] = {lang: [] for lang in MAJOR_LANGS}
    wanted = set(MAJOR_LANGS)

    for row in ds:
        code = names[row["label"]]
        if code in wanted and len(buckets[code]) < SAMPLES_PER_LANG:
            buckets[code].append(row["sentence"])
        if all(len(buckets[c]) >= SAMPLES_PER_LANG for c in MAJOR_LANGS):
            break
    return buckets


@pytest.mark.slow
@pytest.mark.network
def test_major_languages_accuracy_report(
    lid_engine: RobustLID, wili_by_lang: dict[str, list[str]]
) -> None:
    """Per-language accuracy report on WiLi-2018 major languages."""
    rows: list[tuple[str, int, int, list[tuple[str, str]]]] = []
    total_correct = 0
    total_count = 0

    for lang in MAJOR_LANGS:
        acceptable = expected_set(lang)
        samples = wili_by_lang[lang]
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

    print("\nWiLi-2018 — " + "=" * 60)
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

    assert overall > 0.0  # sanity only — report is the point


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("lang", MAJOR_LANGS)
def test_each_major_language_majority_hit(
    lid_engine: RobustLID, wili_by_lang: dict[str, list[str]], lang: str
) -> None:
    """Each major language should score above 50% on WiLi-2018."""
    acceptable = expected_set(lang)
    samples = wili_by_lang[lang]
    hits = sum(1 for s in samples if lid_engine.predict(s)[0].split("_")[0] in acceptable)
    accuracy = hits / len(samples)
    assert accuracy > 0.5, f"{lang}: {hits}/{len(samples)} = {accuracy:.1%} below 50%"
