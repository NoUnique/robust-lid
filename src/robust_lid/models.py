import os
import requests
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import fasttext
import langid
import langdetect
import pycld2 as cld2
try:
    import gcld3
except ImportError:
    gcld3 = None
from .constants import (
    CACHE_DIR, 
    FASTTEXT_176_URL, FASTTEXT_176_FILENAME,
    FASTTEXT_218E_URL, FASTTEXT_218E_FILENAME,
    GLOTLID_V3_URL, GLOTLID_V3_FILENAME
)
from .utils import normalize_language_code

class LID(ABC):
    @abstractmethod
    def predict(self, text: str) -> List[Tuple[str, float]]:
        """
        Predicts the language of the text.
        Returns a list of (language_code, probability) tuples.
        Language code should be normalized to ISO 639-3 if possible.
        """
        pass

class LangidLID(LID):
    def __init__(self):
        # Initialize langid model if needed
        pass

    def predict(self, text: str) -> List[Tuple[str, float]]:
        try:
            lang, prob = langid.classify(text)
            return [(normalize_language_code(lang), float(prob))]
        except Exception:
            return []

class LangdetectLID(LID):
    def __init__(self):
        langdetect.DetectorFactory.seed = 42

    def predict(self, text: str) -> List[Tuple[str, float]]:
        try:
            results = langdetect.detect_langs(text)
            return [(normalize_language_code(res.lang), res.prob) for res in results]
        except Exception:
            return []

class CLD2LID(LID):
    def predict(self, text: str) -> List[Tuple[str, float]]:
        try:
            is_reliable, text_bytes_found, details = cld2.detect(text)
            results = []
            for name, code, percent, score in details:
                if code != 'un':
                    results.append((normalize_language_code(code), percent / 100.0))
            return results
        except Exception:
            return []

class CLD3LID(LID):
    def __init__(self):
        try:
            self.detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
        except Exception:
            self.detector = None

    def predict(self, text: str) -> List[Tuple[str, float]]:
        if not self.detector:
            return []
        try:
            res = self.detector.FindLanguage(text)
            if res and res.is_reliable:
                return [(normalize_language_code(res.language), res.probability)]
            return []
        except Exception:
            return []

class FastTextLID(LID):
    def __init__(self, model_url: str, model_filename: str):
        self.model_path = CACHE_DIR / model_filename
        self._download_model(model_url, self.model_path)
        self.model = fasttext.load_model(str(self.model_path))

    def _download_model(self, url: str, path: str):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading model from {url} to {path}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")

    def predict(self, text: str) -> List[Tuple[str, float]]:
        try:
            # FastText expects single line
            text = text.replace('\n', ' ')
            labels, scores = self.model.predict(text, k=5)
            results = []
            for label, score in zip(labels, scores):
                # Label format usually __label__<code>
                code = label.replace('__label__', '')
                # GlotLID might have different format, usually it's also __label__
                # But GlotLID v3 returns 3 letter codes, FT176 returns 2 letter.
                results.append((normalize_language_code(code), float(score)))
            return results
        except Exception:
            return []

class FastText176LID(FastTextLID):
    def __init__(self):
        super().__init__(FASTTEXT_176_URL, FASTTEXT_176_FILENAME)

class FastText218eLID(FastTextLID):
    def __init__(self):
        super().__init__(FASTTEXT_218E_URL, FASTTEXT_218E_FILENAME)

class GlotLID(FastTextLID):
    def __init__(self):
        super().__init__(GLOTLID_V3_URL, GLOTLID_V3_FILENAME)
