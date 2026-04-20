# robust-lid

Robust language identification that ensembles multiple LID backends into a single
`(language_script, confidence)` prediction. Designed for short/noisy text where
any single classifier is unreliable.

## Quick start

### Python

```python
from robust_lid import RobustLID

lid = RobustLID()
code, confidence = lid.predict("The quick brown fox jumps over the lazy dog.")
# ('eng_Latn', 0.91)
```

### CLI

Two entry points are registered: `rlid` (short) and `robust-lid` (long alias).

```bash
rlid "The quick brown fox jumps over the lazy dog."
# eng_Latn    0.987    The quick brown fox jumps over the lazy dog.

echo "안녕하세요" | rlid --json
# {"text": "안녕하세요", "lang": "kor_Hang", "confidence": 0.94}

rlid --file input.txt --no-text          # one pred per input line, no echo
rlid --models ft176,glotlid "Hello"      # use a subset of backends
rlid --uniform "Hello"                   # disable tuned defaults
rlid --low-memory "Hello"                # load one backend at a time (peak ~1.9 GB)
rlid --no-parallel "Hello"               # sequential predict (default is threaded)
rlid --verbose "Hello"                   # stage-by-stage progress on stderr
rlid --list-backends                     # inventory and exit
rlid --help
```

First call downloads ~1.5 GB of fastText models to `~/.cache/robust_lid/`.

### Batch prediction

For multi-text workloads, call `predict_batch` instead of `predict` in a loop:

```python
from robust_lid import RobustLID

lid = RobustLID()
results = lid.predict_batch(["Hello world", "안녕하세요", "Bonjour"])
# [('eng_Latn', 0.99), ('kor_Hang', 1.0), ('fra_Latn', 0.97)]
```

The CLI automatically switches to the batch path when more than one text is
provided (via `--file` or stdin).

**Where the speedup actually comes from**:

1. **Single thread pool per batch** instead of one per text — eliminates the
   7-worker pool-construction overhead × N repetitions.
2. **Cached label normalization** in `FastTextLID.predict_batch` — the
   ISOConverter lookup runs once per distinct fasttext label across the
   batch, not k × N times.
3. **fastText `multilinePredict` (C++)** — a single C entry per backend.
   Measured benefit is small for already-fast models (lid.176: ~1× vs
   sequential) but more meaningful for the heavier fasttext-218e / GlotLID
   on large batches.

**The unfixable bottleneck** (and what `fast_mode` solves): per-backend wall
time on N=200 English sentences, grouped by backend family:

| family | backend | implementation | ms/text |
|---|---|---|---|
| pure Python | **langid** | Naive Bayes | **~6.0** |
| pure Python | **langdetect** | Naive Bayes | **~2.2** |
| CLD | cld2 | C binding | ~0.00 |
| CLD | cld3 | C++ via gcld3 | ~0.04 |
| fastText | ft176 | C++ | ~0.01 |
| fastText | ft218e | C++ | ~0.06 |
| fastText | glotlid | C++ | ~0.43 |

`langid` and `langdetect` are pure-Python, GIL-bound, and have no batch
API — they sit at ~95 % of total ensemble wall time regardless of how we
call them.

### `fast_mode` (default: on)

`RobustLID(fast_mode=True)` — which is the **default** — drops those two
backends from the ensemble, leaving 4-5 all-C/C++ backends
(`cld2`, `cld3`, `ft176`, `ft218e`, `glotlid`). This gives a large wall-time
reduction with a small accuracy cost (fastText-176 alone already covers
176 languages — most of what langid+langdetect contribute).

```python
from robust_lid import RobustLID
RobustLID()                   # fast_mode=True, 5-backend ensemble (default)
RobustLID(fast_mode=False)    # all 7 backends — maximum ensemble diversity
```

CLI equivalents:

```bash
rlid "text"                   # fast_mode default
rlid --with-slow "text"       # include langid + langdetect
```

``SLOW_BACKEND_NAMES`` in ``robust_lid.ensemble`` exposes the excluded set
(``frozenset({"langid", "langdetect"})``) for introspection.

### Execution modes and memory footprint

| Mode | How | Peak RSS | Per-call latency |
|---|---|---|---|
| Fast (default) | All backends eagerly loaded, predict calls run on a thread pool | **~3.2 GB** | ~30-100 ms |
| Sequential | `parallel=False` / `--no-parallel` — no thread pool | ~3.2 GB | ~100-300 ms |
| Low memory | `low_memory=True` / `--low-memory` — each predict re-instantiates every backend, releases when done | **~1.9 GB peak, ~250 MB between calls** | seconds (re-loads fastText from disk each call) |

Low-memory mode trades per-call latency for a much smaller resident footprint
(useful on CI runners, small VPSes, or embedded-like environments). It
disables supported-script gating — backends aren't live between calls, so
their `supported_scripts` attribute can't be inspected. Incompatible with a
custom `models=` list.

## Backends

| Backend | Bundled | Notes |
|---|---|---|
| [langid](https://github.com/saffsd/langid.py) | yes | pure Python |
| [langdetect](https://github.com/Mimino666/langdetect) | yes | pure Python |
| [pycld2](https://github.com/aboSamoor/pycld2) (CLD2) | yes | C binding |
| [gcld3](https://github.com/google/cld3) (CLD3) | **opt-in** | requires `protoc` — see below |
| [fastText-176](https://fasttext.cc/docs/en/language-identification.html) | yes (downloaded on first use) | 126 MB |
| [fastText-218e](https://huggingface.co/facebook/fasttext-language-identification) | yes (downloaded on first use) | 1.2 GB |
| [GlotLID v3](https://huggingface.co/cis-lmu/glotlid) | yes (downloaded on first use) | 172 MB, 2,100+ languages |

If `gcld3` is not installed, the ensemble runs with 6 backends and emits an
`ImportWarning` on package import.

## Installation

```bash
pip install robust-lid
```

### Optional: enable the CLD3 backend

`gcld3` depends on the Protocol Buffers compiler (`protoc`), which must be
installed at the system level **before** `pip install`:

| Platform | Command |
|---|---|
| RHEL / Fedora / Rocky | `sudo dnf install protobuf-compiler protobuf-devel` |
| Debian / Ubuntu | `sudo apt-get install protobuf-compiler libprotobuf-dev` |
| macOS | `brew install protobuf` |

Then:

```bash
pip install 'robust-lid[cld3]'
```

If you skip this, `RobustLID` will print:

> `ImportWarning: gcld3 is not installed; the CLD3 backend will be excluded
> from the RobustLID ensemble. ...`

You can check availability at runtime:

```python
from robust_lid.models import is_cld3_available
print(is_cld3_available())  # True or False
```

## Development

```bash
uv sync --extra dev              # lint, mypy, pytest, pre-commit
uv sync --extra dev --extra e2e  # + datasets (WiLi-2018, papluca) + gcld3
uv run pytest                    # unit tests only (no network)
uv run pytest -m "slow and network"  # E2E — downloads ~1.5 GB of models on first run
uv run mypy src/robust_lid
uv run ruff check src/ tests/
```

### LID benchmarks

The ensemble is evaluated on three Hub datasets with different domain
characteristics and label granularity. Each dataset is compared at its
**native label granularity**:

| Dataset | Domain | Langs | Label format | Comparison | Accuracy (tuned defaults) |
|---|---|---|---|---|---|
| [`martinthoma/wili_2018`](https://huggingface.co/datasets/martinthoma/wili_2018) | Wikipedia, 235 langs | 30 major | ISO 639-3 (`eng`) | lang only | **98.4 %** (886 / 900) |
| [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification) | reviews + news, 20 langs | 18 overlap | ISO 639-1 (`en`) | lang only | **99.6 %** (538 / 540) |
| [`openlanguagedata/flores_plus`](https://huggingface.co/datasets/openlanguagedata/flores_plus) (gated) | translated Wikipedia, 200 langs | 30 major | `lang_Script` (`eng_Latn`) | **strict `lang_Script`** | **100.0 %** (900 / 900) |

**Why different comparison granularities?** WiLi-2018 and papluca ship only
language labels (no script), so we can't verify the script dimension against
them. FLORES+ labels carry both, so `matches_lang_script` (in
[`tests/integration/_common.py`](tests/integration/_common.py)) enforces
exact language AND exact script (modulo macrolanguage and script-supercode
equivalence classes — so `cmn_Hant ≡ zho_Hans` and `arb_Arab ≡ arz_Arab`).
This catches the class of bugs where a backend nails the language but
mis-detects the script.

The 30 major languages tracked (`tests/integration/_common.py`) cover all
commonly used scripts (Latn / Hang / Jpan / Hani / Hira / Kana / Arab / Cyrl /
Deva / Beng / Thai / Grek / Hebr). Per-backend numbers are available via
`scripts/per_backend_accuracy.py --dataset wili papluca`.

### Running gated datasets (FLORES-200)

1. Copy the template: `cp .env.example .env`
2. Fill in a read-only Hugging Face token from
   <https://huggingface.co/settings/tokens> and accept the dataset's terms on
   its page.
3. Run: `uv run pytest -m "slow and network" tests/integration/test_flores_e2e.py -s`

`.env` is already in `.gitignore`. The token is loaded by
`tests/integration/conftest.py` (stdlib-only parser — no extra dep). When
`HF_TOKEN` is not set, FLORES tests skip automatically; the ungated
benchmarks keep running.

Why project root? Every Python/Node tool (`python-dotenv`, `pytest-env`,
Docker Compose, VS Code's `python.envFile`, Vercel/Netlify, etc.) searches
for `.env` from the project root. A gitignored `.env` at root never shows
up in `git status`, so the "clutter" cost is zero in practice.

### Injecting fake backends (for tests)

```python
from robust_lid import RobustLID
from robust_lid.models import LID

class FakeLID(LID):
    def predict(self, text: str) -> list[tuple[str, float]]:
        return [("eng", 0.99)]

lid = RobustLID(models=[FakeLID(), FakeLID()])  # no network, no models
```

`FastTextLID(cache_dir=..., download_fn=..., model_loader=...)` and
`ISOConverter(mapping=..., iso639_3_map=..., tsv_path=...)` accept injected
dependencies for testing.

### Weighted voting

`RobustLID` combines three multiplicative knobs per backend:

| Knob | Shape | Applies when | Default source |
|---|---|---|---|
| `weights` | `list[float]` | always (per-model scalar) | `default_weights()` |
| `script_weights` | `list[dict[script → float]]` | when `detect_script(text)` hits a key | `default_script_weights()` |
| `lang_weights` | `list[dict[predicted_lang → float]]` | when the backend's top-1 matches a key | `default_lang_weights()` |

Effective contribution of backend *i*:
```
weights[i] * script_weights[i].get(script, 1.0) * lang_weights[i].get(pred_lang, 1.0) * prob
```

### Script-based backend gating

Each backend exposes `supported_langs` (ISO 639-3 codes it can ever emit) and
`supported_scripts` (derived ISO 15924 codes). `RobustLID` uses the latter to
**auto-silence any backend whose supported-script set doesn't cover the
input's detected script** — preventing a backend from dragging the ensemble
down with a confidently-wrong guess on text outside its coverage.

```python
from robust_lid.models import LangdetectLID, FastText176LID

LangdetectLID().supported_langs       # frozenset of 55 ISO-639-3 codes
LangdetectLID().supported_scripts     # 41 ISO-15924 codes (Latn, Cyrl, …)
# langdetect has no Khmer, Ethiopic, Tibetan coverage;
# fastText-176 does, so on Amharic text only fastText's vote counts.
```

Custom backends (`models=[MyLID(), ...]`) default to `frozenset()` → gating
is disabled. Override `supported_langs` to opt in.

`RobustLID` also applies two upstream-bug fixes at import time:
- **langid** returns raw log-probabilities by default (large negative values).
  `LangidLID` constructs the identifier with `norm_probs=True` so it yields
  `[0, 1]` probabilities; otherwise the negative vote totals flip sign during
  normalization and hand wins to whatever language langid disagreed with.
- **fasttext-wheel 0.9.2** uses `np.array(copy=False)` which breaks on NumPy 2.
  `_patch_fasttext_for_numpy2()` monkey-patches `_FastText.predict` to use
  `np.asarray` instead.

`compute_ensemble_vote` also skips any vote with `prob ≤ 0` defensively.

**Calling `RobustLID()` without args auto-applies all three defaults.** They
were tuned on WiLi-2018 across 14 major languages to patch the known
per-backend weak spots:
- `langdetect` × `Hani` → 0.3 (73 % recall on Chinese)
- `langdetect` × `Jpan` → 0.5
- `cld2` × `Hani` → 0.8
- `glotlid` × `Deva` → 0.5 (confuses Hindi with Marathi)
- `langid` × `{ltz, kir}` → 0.5 / 0.3 (rare mis-labels of German/Turkish)
- `glotlid` × `mar` → 0.7
- `ft176` scalar → 1.3, `ft218e` scalar → 1.2 (the two strongest backends)

Override or disable selectively:
```python
from robust_lid import RobustLID
from robust_lid.ensemble import default_backend_order

order = default_backend_order()
# ['langid', 'langdetect', 'cld2', 'cld3', 'ft176', 'ft218e', 'glotlid']
# (cld3 is omitted if gcld3 isn't installed)

# Uniform (disable all tuning)
lid = RobustLID(
    weights=[1.0] * len(order),
    script_weights=[{}] * len(order),
    lang_weights=[{}] * len(order),
)

# Custom scalar weights by name
weights_by_name = {
    "langid": 1.0, "langdetect": 0.5, "cld2": 1.0, "cld3": 1.0,
    "ft176": 2.0, "ft218e": 2.0, "glotlid": 1.5,
}
lid = RobustLID(weights=[weights_by_name[name] for name in order])
```

For custom models (`RobustLID(models=[...])`) defaults are **not** applied
because the tuning is keyed by backend name.

To measure per-backend accuracy on your own data and re-tune:
```bash
uv run python scripts/per_backend_accuracy.py --lang por deu tur --n 50
```
