import gc
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .constants import UNDEFINED_LANG, UNDEFINED_SCRIPT
from .models import (
    CLD2LID,
    CLD3LID,
    LID,
    FastText176LID,
    FastText218eLID,
    GlotLID,
    LangdetectLID,
    LangidLID,
    is_cld3_available,
)
from .utils import detect_script


def compute_ensemble_vote(
    predictions: list[list[tuple[str, float]]],
    weights: list[float] | None = None,
) -> tuple[str, float]:
    """Aggregate top-1 predictions from each model into a single decision.

    Pure function. Each model i contributes ``weights[i] * prob`` to its
    predicted language. The winner is the language with the highest normalized
    total. ``weights=None`` means uniform (every model contributes 1.0 * prob).
    """
    if weights is None:
        weights = [1.0] * len(predictions)
    if len(weights) != len(predictions):
        raise ValueError(
            f"weights length {len(weights)} does not match predictions length {len(predictions)}"
        )

    votes: dict[str, float] = defaultdict(float)
    for preds, weight in zip(predictions, weights, strict=True):
        if not preds:
            continue
        lang, prob = preds[0]
        # Defensive: skip non-positive probabilities. A backend that returns
        # log-probs or garbage would otherwise flip the sign of the normalized
        # total and hand victory to the wrong language.
        if lang == UNDEFINED_LANG or prob <= 0:
            continue
        votes[lang] += weight * prob

    if not votes:
        return UNDEFINED_LANG, 0.0

    total = sum(votes.values())
    normalized = {k: v / total for k, v in votes.items()}
    best = max(normalized, key=lambda k: normalized[k])
    return best, normalized[best]


# ---------------------------------------------------------------------------
# Default backend / weight tables
# ---------------------------------------------------------------------------

DEFAULT_BACKEND_ORDER: tuple[str, ...] = (
    "langid",
    "langdetect",
    "cld2",
    "cld3",  # only present when gcld3 is importable
    "ft176",
    "ft218e",
    "glotlid",
)

# Per-backend scalar multiplier. Tuned from WiLi-2018 + papluca overall
# accuracy across 30 languages (WiLi / papluca):
#   ft176     98.8 / 99.1
#   glotlid   98.0 / 98.7
#   ft218e    98.3 / 98.1
#   langdetect 97.2 / 98.3
#   langid    97.6 / 97.6
#   cld2      93.8 / 93.5
#   cld3      92.9 / 96.3
# Boost the three consistently strongest backends slightly.
DEFAULT_WEIGHTS_BY_NAME: dict[str, float] = {
    "langid": 1.0,
    "langdetect": 1.0,
    "cld2": 1.0,
    "cld3": 1.0,
    "ft176": 1.3,
    "ft218e": 1.2,
    "glotlid": 1.1,
}

# Script-conditional multipliers applied when the input text's detected script
# matches. Addresses recall weaknesses that are structural (specific to a
# script family):
#   - langdetect: 73% on Hani (Chinese), weaker on Jpan
#   - cld2: 96.7% on Hani; 0% on Hebr (complete Hebrew failure)
#   - cld3: 0% on Hebr (complete Hebrew failure)
#   - glotlid: 86.7% on Deva (Hindi / Marathi confusion)
DEFAULT_SCRIPT_WEIGHTS_BY_NAME: dict[str, dict[str, float]] = {
    "langdetect": {"Hani": 0.3, "Jpan": 0.5},
    "cld2": {"Hani": 0.8, "Hebr": 0.0},
    "cld3": {"Hebr": 0.0},
    "glotlid": {"Deva": 0.5},
}

# Per-predicted-language multipliers. Addresses the miss patterns observed on
# WiLi-2018: when a given backend predicts one of these specific languages, it
# is statistically more likely to be wrong, so downweight its vote.
#   - langid often mis-labels German as Luxembourgish; Turkish as Kyrgyz
#   - langdetect confuses Italian text with Luxembourgish
#   - glotlid confuses Hindi with Marathi
DEFAULT_LANG_WEIGHTS_BY_NAME: dict[str, dict[str, float]] = {
    "langid": {"ltz": 0.5, "kir": 0.3},
    "langdetect": {"ltz": 0.6},
    "glotlid": {"mar": 0.7},
}


def default_backend_order() -> list[str]:
    """Names of the models returned by `_default_models()`, in order."""
    order = ["langid", "langdetect", "cld2"]
    if is_cld3_available():
        order.append("cld3")
    order.extend(["ft176", "ft218e", "glotlid"])
    return order


def default_weights() -> list[float]:
    """Per-model scalar defaults aligned to ``default_backend_order()``."""
    return [DEFAULT_WEIGHTS_BY_NAME.get(name, 1.0) for name in default_backend_order()]


def default_script_weights() -> list[dict[str, float]]:
    """Per-model script-conditional multiplier tables."""
    return [dict(DEFAULT_SCRIPT_WEIGHTS_BY_NAME.get(name, {})) for name in default_backend_order()]


def default_lang_weights() -> list[dict[str, float]]:
    """Per-model predicted-language multiplier tables."""
    return [dict(DEFAULT_LANG_WEIGHTS_BY_NAME.get(name, {})) for name in default_backend_order()]


def _default_models() -> list[LID]:
    items: list[LID] = [LangidLID(), LangdetectLID(), CLD2LID()]
    if is_cld3_available():
        items.append(CLD3LID())
    items.extend([FastText176LID(), FastText218eLID(), GlotLID()])
    return items


def _default_factories() -> list[Callable[[], LID]]:
    """Zero-argument backend constructors, in the same order as ``default_backend_order()``.
    Used by low-memory mode: each predict call instantiates backends one at a time
    and releases them afterwards, so peak RAM ≈ size of the largest single model."""
    items: list[Callable[[], LID]] = [LangidLID, LangdetectLID, CLD2LID]
    if is_cld3_available():
        items.append(CLD3LID)
    items.extend([FastText176LID, FastText218eLID, GlotLID])
    return items


# ---------------------------------------------------------------------------
# RobustLID
# ---------------------------------------------------------------------------


class RobustLID:
    models: list[LID]
    weights: list[float] | None
    script_weights: list[dict[str, float]] | None
    lang_weights: list[dict[str, float]] | None
    parallel: bool
    low_memory: bool

    def __init__(
        self,
        models: list[LID] | None = None,
        weights: list[float] | None = None,
        script_weights: list[dict[str, float]] | None = None,
        lang_weights: list[dict[str, float]] | None = None,
        parallel: bool = True,
        low_memory: bool = False,
    ) -> None:
        """Build a weighted LID ensemble.

        Execution modes
        ---------------
        * Default (``parallel=True``, ``low_memory=False``) — eager-load every
          backend at construction time, run per-text predict calls concurrently
          on a thread pool. Fastest; peak RAM ≈ sum of all loaded models
          (~2 GB with the default 7 backends).
        * ``low_memory=True`` — do NOT eager-load; each ``predict`` instantiates
          one backend at a time, collects its vote, and drops the reference
          before moving on. Peak RAM ≈ size of the largest single model
          (~1.2 GB). Much slower per call (each predict re-loads fastText
          models from disk). Incompatible with a custom ``models=`` list and
          bypasses script gating (which requires live model instances).
        * ``parallel=False`` — sequential predict calls in a plain Python loop.

        When ``models`` is None the default 6/7-backend ensemble is built and
        tuned defaults for the three weight parameters are applied.
        """
        if low_memory and models is not None:
            raise ValueError(
                "low_memory=True requires the default factory ensemble; "
                "pass models=None or drop low_memory."
            )

        self.parallel = parallel
        self.low_memory = low_memory
        using_default_models = models is None

        if low_memory:
            self._factories: list[Callable[[], LID]] | None = _default_factories()
            self.models = []
            effective_n = len(self._factories)
        else:
            self._factories = None
            self.models = models if models is not None else _default_models()
            effective_n = len(self.models)

        if using_default_models:
            if weights is None:
                weights = default_weights()
            if script_weights is None:
                script_weights = default_script_weights()
            if lang_weights is None:
                lang_weights = default_lang_weights()

        for name, value in (
            ("weights", weights),
            ("script_weights", script_weights),
            ("lang_weights", lang_weights),
        ):
            if value is not None and len(value) != effective_n:
                raise ValueError(
                    f"{name} length {len(value)} does not match backend count {effective_n}"
                )

        self.weights = weights
        self.script_weights = script_weights
        self.lang_weights = lang_weights

    def _effective_weights(
        self, predictions: list[list[tuple[str, float]]], script: str
    ) -> list[float]:
        effective: list[float] = []
        for i, preds in enumerate(predictions):
            w = self.weights[i] if self.weights is not None else 1.0

            # Auto-gate: if the backend declared a non-empty supported-scripts
            # set and the input's script isn't in it, zero its vote so it can't
            # poison the ensemble with a confidently-wrong guess. Only runs
            # when we have a live model instance (not in low_memory mode).
            if self.models:
                supported = self.models[i].supported_scripts
                if supported and script != UNDEFINED_SCRIPT and script not in supported:
                    w = 0.0

            if self.script_weights is not None:
                w *= self.script_weights[i].get(script, 1.0)
            if self.lang_weights is not None and preds:
                top_lang = preds[0][0]
                w *= self.lang_weights[i].get(top_lang, 1.0)
            effective.append(w)
        return effective

    def _collect_predictions(self, text: str) -> list[list[tuple[str, float]]]:
        if self.low_memory:
            return self._predict_streaming(text)
        if self.parallel and len(self.models) > 1:
            return self._predict_parallel(text)
        return [m.predict(text) for m in self.models]

    def _predict_parallel(self, text: str) -> list[list[tuple[str, float]]]:
        """Run each backend on its own thread. fastText / pycld2 / gcld3 are
        C extensions that release the GIL, so real parallelism happens for
        the slow ones. Pool is created per call — small overhead vs predict
        latency on the heavy backends."""
        with ThreadPoolExecutor(max_workers=len(self.models)) as ex:
            return list(ex.map(lambda m: m.predict(text), self.models))

    def _predict_streaming(self, text: str) -> list[list[tuple[str, float]]]:
        """Load→predict→unload per backend. Peak RAM tracks the largest
        single model, at the cost of per-call disk I/O."""
        assert self._factories is not None
        results: list[list[tuple[str, float]]] = []
        for factory in self._factories:
            model = factory()
            try:
                results.append(model.predict(text))
            finally:
                del model
                gc.collect()
        return results

    def predict(self, text: str) -> tuple[str, float]:
        """Predicts the language and script of the text using ensemble voting.

        Returns (language_script_code, confidence). Example: ('eng_Latn', 0.9).
        Returns ('und_Zyyy', 0.0) if no model produces a usable prediction.
        """
        predictions = self._collect_predictions(text)
        script = detect_script(text)
        effective = self._effective_weights(predictions, script)
        best_lang, confidence = compute_ensemble_vote(predictions, effective)
        if best_lang == UNDEFINED_LANG:
            return f"{UNDEFINED_LANG}_{UNDEFINED_SCRIPT}", 0.0
        return f"{best_lang}_{script}", confidence

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Batch-optimised predict.

        For each backend, all N texts are processed in one call to its
        ``predict_batch`` method — fastText goes through ``multilinePredict``
        (C++) instead of N round-trips, and we build/reuse a single
        ThreadPoolExecutor for the whole batch instead of one per text.

        Returns a list of ``(language_script, confidence)`` tuples, one per
        input text in order.
        """
        if not texts:
            return []

        preds_by_backend = self._collect_predictions_batch(texts)

        results: list[tuple[str, float]] = []
        for j, text in enumerate(texts):
            per_text_preds = [preds_by_backend[i][j] for i in range(len(preds_by_backend))]
            script = detect_script(text)
            effective = self._effective_weights(per_text_preds, script)
            best_lang, confidence = compute_ensemble_vote(per_text_preds, effective)
            if best_lang == UNDEFINED_LANG:
                results.append((f"{UNDEFINED_LANG}_{UNDEFINED_SCRIPT}", 0.0))
            else:
                results.append((f"{best_lang}_{script}", confidence))
        return results

    def _collect_predictions_batch(self, texts: list[str]) -> list[list[list[tuple[str, float]]]]:
        """Returns predictions shaped as ``[backend_i][text_j]``."""
        if self.low_memory:
            assert self._factories is not None
            results: list[list[list[tuple[str, float]]]] = []
            for factory in self._factories:
                model = factory()
                try:
                    results.append(model.predict_batch(texts))
                finally:
                    del model
                    gc.collect()
            return results
        if self.parallel and len(self.models) > 1:
            # One pool for the whole batch — reused across all texts.
            with ThreadPoolExecutor(max_workers=len(self.models)) as ex:
                return list(ex.map(lambda m: m.predict_batch(texts), self.models))
        return [m.predict_batch(texts) for m in self.models]
