from collections import defaultdict

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


# ---------------------------------------------------------------------------
# RobustLID
# ---------------------------------------------------------------------------


class RobustLID:
    models: list[LID]
    weights: list[float] | None
    script_weights: list[dict[str, float]] | None
    lang_weights: list[dict[str, float]] | None

    def __init__(
        self,
        models: list[LID] | None = None,
        weights: list[float] | None = None,
        script_weights: list[dict[str, float]] | None = None,
        lang_weights: list[dict[str, float]] | None = None,
    ) -> None:
        """Build a weighted LID ensemble.

        When ``models`` is None, the default 6/7-backend ensemble is built and,
        for any of the three weight parameters that are also None, tuned
        defaults are applied (see ``default_weights``,
        ``default_script_weights``, ``default_lang_weights``).

        When ``models`` is provided, unspecified weight parameters stay None
        (uniform 1.0) — defaults are tuned for the specific backend names and
        cannot be applied to custom ensembles.

        Effective weight for model *i* with top prediction ``lang`` on text of
        script ``script`` is::

            weights[i]
              * script_weights[i].get(script, 1.0)
              * lang_weights[i].get(lang, 1.0)
        """
        using_default_models = models is None
        self.models = models if models is not None else _default_models()

        if using_default_models:
            if weights is None:
                weights = default_weights()
            if script_weights is None:
                script_weights = default_script_weights()
            if lang_weights is None:
                lang_weights = default_lang_weights()

        n = len(self.models)
        for name, value in (
            ("weights", weights),
            ("script_weights", script_weights),
            ("lang_weights", lang_weights),
        ):
            if value is not None and len(value) != n:
                raise ValueError(f"{name} length {len(value)} does not match models length {n}")

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
            # set and the input's script isn't in it, the backend cannot
            # meaningfully predict — zero its vote so it can't poison the
            # ensemble with a confidently-wrong guess.
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

    def predict(self, text: str) -> tuple[str, float]:
        """Predicts the language and script of the text using ensemble voting.

        Returns (language_script_code, confidence). Example: ('eng_Latn', 0.9).
        Returns ('und_Zyyy', 0.0) if no model produces a usable prediction.
        """
        predictions = [model.predict(text) for model in self.models]
        script = detect_script(text)
        effective = self._effective_weights(predictions, script)
        best_lang, confidence = compute_ensemble_vote(predictions, effective)
        if best_lang == UNDEFINED_LANG:
            return f"{UNDEFINED_LANG}_{UNDEFINED_SCRIPT}", 0.0
        return f"{best_lang}_{script}", confidence
