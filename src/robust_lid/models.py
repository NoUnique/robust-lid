import contextlib
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any

import fasttext
import langdetect
import langid.langid as _langid_mod
import numpy as np
import pycld2 as cld2
import requests

from .constants import (
    CACHE_DIR,
    DOWNLOAD_TIMEOUT_SEC,
    FASTTEXT_176_FILENAME,
    FASTTEXT_176_URL,
    FASTTEXT_218E_FILENAME,
    FASTTEXT_218E_URL,
    GLOTLID_V3_FILENAME,
    GLOTLID_V3_URL,
)
from .utils import _expand_script, get_converter, normalize_language_code

logger = logging.getLogger(__name__)

DownloadFn = Callable[[str, Path], None]


def _patch_fasttext_for_numpy2() -> None:
    """fasttext-wheel 0.9.2 uses `np.array(copy=False)` which raises on NumPy 2.

    Replace the offending line with `np.asarray` on the single-text predict
    path. Safe to call multiple times (idempotent).
    """
    cls = fasttext.FastText._FastText
    if getattr(cls, "_robust_lid_patched", False):
        return

    original_predict = cls.predict

    def patched_predict(
        self: Any,
        text: Any,
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
    ) -> Any:
        def check(entry: str) -> str:
            if entry.find("\n") != -1:
                raise ValueError("predict processes one line at a time (remove '\\n')")
            return entry + "\n"

        if isinstance(text, list):
            return original_predict(self, text, k, threshold, on_unicode_error)
        text = check(text)
        predictions = self.f.predict(text, k, threshold, on_unicode_error)
        labels: tuple[str, ...]
        probs: tuple[float, ...]
        if predictions:
            probs, labels = zip(*predictions, strict=False)
        else:
            probs, labels = ((), ())
        return labels, np.asarray(probs)

    cls.predict = patched_predict
    cls._robust_lid_patched = True


_patch_fasttext_for_numpy2()

CLD3_UNAVAILABLE_MSG = (
    "gcld3 is not installed; the CLD3 backend will be excluded from the "
    "RobustLID ensemble. To enable it, install the protobuf compiler first "
    "(e.g. `sudo dnf install protobuf-compiler protobuf-devel` on RHEL/Fedora, "
    "`sudo apt-get install protobuf-compiler libprotobuf-dev` on Debian/Ubuntu, "
    "or `brew install protobuf` on macOS) and then "
    "`pip install robust-lid[cld3]` or `pip install gcld3`."
)


def _emit_cld3_import_warning() -> None:
    warnings.warn(CLD3_UNAVAILABLE_MSG, category=ImportWarning, stacklevel=2)


try:
    import gcld3
except ImportError:  # pragma: no cover — helper body covered separately
    gcld3 = None
    _emit_cld3_import_warning()


def is_cld3_available() -> bool:
    """Whether the gcld3 backend is importable (protoc + gcld3 installed)."""
    return gcld3 is not None


def _normalize_lang_set(codes: list[str] | tuple[str, ...]) -> frozenset[str]:
    """Normalize a collection of language codes to ISO 639-3, dropping
    anything the converter rejects (e.g. UNDEFINED_LANG)."""
    from .constants import UNDEFINED_LANG

    result: set[str] = set()
    for code in codes:
        normalized = normalize_language_code(code)
        if normalized != UNDEFINED_LANG:
            result.add(normalized)
    return frozenset(result)


# gcld3 supports 107 languages. No runtime API exposes this list; values
# copied from https://github.com/google/cld3 (README). "iw" is an alias for
# Hebrew and is kept alongside "he" for safety.
_GCLD3_SUPPORTED: tuple[str, ...] = (
    "af",
    "am",
    "ar",
    "bg",
    "bn",
    "bs",
    "ca",
    "ceb",
    "co",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fil",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hmn",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "iw",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "ny",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sd",
    "si",
    "sk",
    "sl",
    "sm",
    "sn",
    "so",
    "sq",
    "sr",
    "st",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
)


class LID(ABC):
    @abstractmethod
    def predict(self, text: str) -> list[tuple[str, float]]:
        """Predicts the language of the text.

        Returns a list of (language_code, probability) tuples.
        Language code should be normalized to ISO 639-3 if possible.
        """
        ...

    @cached_property
    def supported_langs(self) -> frozenset[str]:
        """ISO 639-3 codes this backend can ever emit.

        Default: empty — means 'unknown, don't filter'. Concrete backends
        override this to enable script-based ensemble gating: if the input
        text's script is not in ``supported_scripts``, RobustLID zeroes out
        this backend's vote.
        """
        return frozenset()

    @cached_property
    def supported_scripts(self) -> frozenset[str]:
        """ISO 15924 script codes derived from supported_langs.

        Equivalence classes are expanded: a language that GlotScript lists
        under ``Hans`` yields ``{Hans, Hani, Hant, Hanb}`` so that
        ``detect_script``'s coarse-grained codes still hit the right bucket.
        """
        if not self.supported_langs:
            return frozenset()
        conv = get_converter()
        scripts: set[str] = set()
        for lang in self.supported_langs:
            for s in conv.scripts_for(lang):
                scripts |= _expand_script(s)
        return frozenset(scripts)


class LangidLID(LID):
    """Wraps langid.py. The default `langid.classify()` returns raw
    log-probabilities (negative, unnormalized) which would corrupt the
    ensemble arithmetic. We construct an identifier with `norm_probs=True`
    so the confidence is a [0, 1] probability."""

    def __init__(self) -> None:
        self._identifier = _langid_mod.LanguageIdentifier.from_modelstring(
            _langid_mod.model, norm_probs=True
        )

    def predict(self, text: str) -> list[tuple[str, float]]:
        try:
            lang, prob = self._identifier.classify(text)
            return [(normalize_language_code(lang), float(prob))]
        except Exception:
            logger.debug("LangidLID.predict failed", exc_info=True)
            return []

    @cached_property
    def supported_langs(self) -> frozenset[str]:
        return _normalize_lang_set(self._identifier.nb_classes)


class LangdetectLID(LID):
    DEFAULT_SEED: int = 42

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed

    def predict(self, text: str) -> list[tuple[str, float]]:
        langdetect.DetectorFactory.seed = self.seed
        try:
            results = langdetect.detect_langs(text)
            return [(normalize_language_code(res.lang), res.prob) for res in results]
        except Exception:
            logger.debug("LangdetectLID.predict failed", exc_info=True)
            return []

    @cached_property
    def supported_langs(self) -> frozenset[str]:
        # langdetect only instantiates its factory on the first call to
        # detect_langs; trigger it so the profile list is loaded.
        langdetect.DetectorFactory.seed = self.seed
        with contextlib.suppress(Exception):  # pragma: no cover — paranoia
            langdetect.detect_langs("x")
        from langdetect.detector_factory import _factory

        langlist = _factory.langlist if _factory is not None else []
        return _normalize_lang_set(langlist)


class CLD2LID(LID):
    def predict(self, text: str) -> list[tuple[str, float]]:
        try:
            _is_reliable, _bytes_found, details = cld2.detect(text)
            results: list[tuple[str, float]] = []
            for _name, code, percent, _score in details:
                if code != "un":
                    results.append((normalize_language_code(code), percent / 100.0))
            return results
        except Exception:
            logger.debug("CLD2LID.predict failed", exc_info=True)
            return []

    @cached_property
    def supported_langs(self) -> frozenset[str]:
        return _normalize_lang_set([code for _name, code in cld2.LANGUAGES])


class CLD3LID(LID):
    detector: Any | None

    def __init__(self) -> None:
        if gcld3 is None:
            self.detector = None
            logger.debug("gcld3 not importable; CLD3LID disabled")
            return
        try:
            self.detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
        except Exception:
            logger.debug("CLD3LID init failed", exc_info=True)
            self.detector = None

    def predict(self, text: str) -> list[tuple[str, float]]:
        if not self.detector:
            return []
        try:
            res = self.detector.FindLanguage(text)
            if res and res.is_reliable:
                return [(normalize_language_code(res.language), res.probability)]
            return []
        except Exception:
            logger.debug("CLD3LID.predict failed", exc_info=True)
            return []

    @cached_property
    def supported_langs(self) -> frozenset[str]:
        if self.detector is None:
            return frozenset()
        return _normalize_lang_set(list(_GCLD3_SUPPORTED))


def _default_download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model from %s to %s", url, path)
    response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info("Download complete: %s", path)


class FastTextLID(LID):
    model: Any
    model_path: Path

    def __init__(
        self,
        model_url: str,
        model_filename: str,
        cache_dir: Path | None = None,
        download_fn: DownloadFn | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
        download = download_fn if download_fn is not None else _default_download
        loader = model_loader if model_loader is not None else fasttext.load_model

        self.model_path = cache_dir / model_filename
        download(model_url, self.model_path)
        self.model = loader(str(self.model_path))

    def predict(self, text: str) -> list[tuple[str, float]]:
        try:
            text = text.replace("\n", " ")
            labels, scores = self.model.predict(text, k=5)
            results: list[tuple[str, float]] = []
            for label, score in zip(labels, scores, strict=False):
                code = label.replace("__label__", "")
                results.append((normalize_language_code(code), float(score)))
            return results
        except Exception:
            logger.debug("FastTextLID.predict failed", exc_info=True)
            return []

    @cached_property
    def supported_langs(self) -> frozenset[str]:
        raw_labels = [label.replace("__label__", "") for label in self.model.get_labels()]
        return _normalize_lang_set(raw_labels)


class FastText176LID(FastTextLID):
    def __init__(
        self,
        cache_dir: Path | None = None,
        download_fn: DownloadFn | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(
            FASTTEXT_176_URL, FASTTEXT_176_FILENAME, cache_dir, download_fn, model_loader
        )


class FastText218eLID(FastTextLID):
    def __init__(
        self,
        cache_dir: Path | None = None,
        download_fn: DownloadFn | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(
            FASTTEXT_218E_URL, FASTTEXT_218E_FILENAME, cache_dir, download_fn, model_loader
        )


class GlotLID(FastTextLID):
    def __init__(
        self,
        cache_dir: Path | None = None,
        download_fn: DownloadFn | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(GLOTLID_V3_URL, GLOTLID_V3_FILENAME, cache_dir, download_fn, model_loader)
