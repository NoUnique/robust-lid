"""Per-backend accuracy matrix across LID datasets.

Runs each LID backend independently on WiLi-2018 and/or
papluca/language-identification to see which languages each backend struggles
with. Useful as an input to weighted ensemble tuning.

Usage:
    uv run python scripts/per_backend_accuracy.py
    uv run python scripts/per_backend_accuracy.py --dataset wili papluca
    uv run python scripts/per_backend_accuracy.py --lang por deu tur --n 50
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

# Make tests/integration/_common / conftest importable when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robust_lid.models import (  # noqa: E402
    CLD2LID,
    CLD3LID,
    LID,
    FastText176LID,
    FastText218eLID,
    GlotLID,
    LangdetectLID,
    LangidLID,
)
from robust_lid.utils import normalize_language_code  # noqa: E402
from tests.integration._common import MAJOR_LANGS, expected_set  # noqa: E402
from tests.integration.conftest import PROJECT_ROOT, load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# FLORES+ uses `<iso639-3>_<script>` config names. Only 14 of our 30 majors
# have stable single-script configs; the rest use the primary script.
_FLORES_CONFIGS: dict[str, str] = {
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


def _bucket_wili(langs: list[str], n: int) -> dict[str, list[str]]:
    ds = load_dataset("martinthoma/wili_2018", split="test")
    names: list[str] = ds.features["label"].names
    buckets: dict[str, list[str]] = defaultdict(list)
    wanted = set(langs)
    for row in ds:
        code = names[row["label"]]
        if code in wanted and len(buckets[code]) < n:
            buckets[code].append(row["sentence"])
        if all(len(buckets[c]) >= n for c in langs):
            break
    return buckets


def _bucket_papluca(langs: list[str], n: int) -> dict[str, list[str]]:
    """papluca has 20 langs; requested langs outside that set just stay empty."""
    ds = load_dataset("papluca/language-identification", split="test")
    buckets: dict[str, list[str]] = defaultdict(list)
    wanted = set(langs)
    for row in ds:
        iso3 = normalize_language_code(row["labels"])
        if iso3 in wanted and len(buckets[iso3]) < n:
            buckets[iso3].append(row["text"])
    return buckets


def _bucket_flores(langs: list[str], n: int) -> dict[str, list[str]]:
    """FLORES+ is gated — requires HF_TOKEN in .env."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set; copy .env.example to .env first")
    buckets: dict[str, list[str]] = defaultdict(list)
    for lang in langs:
        cfg = _FLORES_CONFIGS.get(lang)
        if cfg is None:
            continue
        ds = load_dataset("openlanguagedata/flores_plus", cfg, split="dev", token=token)
        buckets[lang] = [row["text"] for row in ds.select(range(min(n, len(ds))))]
    return buckets


DATASET_LOADERS = {
    "wili": _bucket_wili,
    "papluca": _bucket_papluca,
    "flores": _bucket_flores,
}


def _accuracy(backend: LID, samples: list[str], acceptable: set[str]) -> float:
    hits = 0
    for text in samples:
        preds = backend.predict(text)
        if preds and preds[0][0] in acceptable:
            hits += 1
    return hits / len(samples) if samples else 0.0


def _print_matrix(
    title: str,
    langs: list[str],
    buckets: dict[str, list[str]],
    backends: dict[str, LID],
) -> None:
    header = f"{'LANG':<6} | " + " | ".join(f"{name:>8}" for name in backends)
    print(f"\n{title}")
    print(header)
    print("-" * len(header))

    per_backend_totals: dict[str, list[float]] = {name: [] for name in backends}
    for lang in langs:
        samples = buckets.get(lang, [])
        if not samples:
            print(f"{lang:<6} | (no samples)")
            continue
        acceptable = expected_set(lang)
        row_parts = []
        for name, backend in backends.items():
            acc = _accuracy(backend, samples, acceptable)
            row_parts.append(f"{acc:>7.1%} ")
            per_backend_totals[name].append(acc)
        print(f"{lang:<6} | " + " | ".join(row_parts))

    print("-" * len(header))
    avg_parts = []
    for accs in per_backend_totals.values():
        avg = sum(accs) / len(accs) if accs else 0.0
        avg_parts.append(f"{avg:>7.1%} ")
    print(f"{'AVG':<6} | " + " | ".join(avg_parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", nargs="+", default=MAJOR_LANGS)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["wili", "papluca", "flores"],
        choices=list(DATASET_LOADERS.keys()),
    )
    args = parser.parse_args()

    print("Instantiating backends (this downloads ~1.5 GB on first run)...")
    backends: dict[str, LID] = {
        "langid": LangidLID(),
        "langdtct": LangdetectLID(),
        "cld2": CLD2LID(),
        "cld3": CLD3LID(),
        "ft176": FastText176LID(),
        "ft218e": FastText218eLID(),
        "glotlid": GlotLID(),
    }

    for ds_name in args.dataset:
        print(f"\nLoading {ds_name} (target langs: {len(args.lang)}, n={args.n})...")
        buckets = DATASET_LOADERS[ds_name](args.lang, args.n)
        _print_matrix(
            f"{ds_name.upper()} — per-backend accuracy",
            [lang for lang in args.lang if buckets.get(lang)],
            buckets,
            backends,
        )


if __name__ == "__main__":
    main()
