import csv
import logging
from pathlib import Path

import pycountry

from .constants import GLOTSCRIPT_TSV, UNDEFINED_LANG, UNDEFINED_SCRIPT

logger = logging.getLogger(__name__)

# Deprecated ISO 639-1 aliases that older LID backends still emit.
# Map them to the modern code before any pycountry lookup.
_ISO639_1_ALIASES: dict[str, str] = {
    "iw": "he",  # Hebrew (old code)
    "in": "id",  # Indonesian (old code)
    "ji": "yi",  # Yiddish (old code)
    "jw": "jv",  # Javanese (old code)
}

# Script equivalence classes. `detect_script` returns coarse codes (Hani, Hang)
# while GlotScript records the precise primary (Hans, Kore). Treating these as
# a single "does this backend cover script X" check requires unifying them.
_SCRIPT_EQUIV_CLASSES: tuple[frozenset[str], ...] = (
    frozenset({"Hani", "Hans", "Hant", "Hanb"}),
    frozenset({"Kore", "Hang"}),
)


def _expand_script(code: str) -> set[str]:
    """Return the equivalence class (singleton for unpaired codes)."""
    for cls in _SCRIPT_EQUIV_CLASSES:
        if code in cls:
            return set(cls)
    return {code}


class ISOConverter:
    mapping: dict[str, str]
    iso639_3_map: dict[str, str]
    lang_to_scripts: dict[str, frozenset[str]]

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        iso639_3_map: dict[str, str] | None = None,
        lang_to_scripts: dict[str, frozenset[str]] | None = None,
        tsv_path: Path | None = None,
    ) -> None:
        loaded_mapping: dict[str, str] = {}
        loaded_l2s: dict[str, frozenset[str]] = {}
        if mapping is None or lang_to_scripts is None:
            loaded_mapping, loaded_l2s = self._load_from_tsv(tsv_path or GLOTSCRIPT_TSV)
        self.mapping = mapping if mapping is not None else loaded_mapping
        self.lang_to_scripts = lang_to_scripts if lang_to_scripts is not None else loaded_l2s
        self.iso639_3_map = iso639_3_map if iso639_3_map is not None else self._load_pycountry_map()

    @staticmethod
    def _load_from_tsv(
        tsv_path: Path,
    ) -> tuple[dict[str, str], dict[str, frozenset[str]]]:
        mapping: dict[str, str] = {}
        lang_to_scripts: dict[str, frozenset[str]] = {}
        if not tsv_path.exists():
            logger.warning(
                "GlotScript TSV not found at %s; ISO639-3 validation will be limited", tsv_path
            )
            return mapping, lang_to_scripts
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                iso639_3 = row["ISO639-3"]
                mapping[iso639_3] = iso639_3
                # GlotScript's `ISO15924-Main` is a comma-separated list of
                # every ISO 15924 code the language is written in. The order
                # is alphabetical, not by prevalence, so we keep them all and
                # let the caller decide which ones matter.
                raw = (row.get("ISO15924-Main") or "").strip()
                if raw:
                    scripts = frozenset(s.strip() for s in raw.split(",") if s.strip())
                    if scripts:
                        lang_to_scripts[iso639_3] = scripts
        return mapping, lang_to_scripts

    @staticmethod
    def _load_pycountry_map() -> dict[str, str]:
        mapping: dict[str, str] = {}
        for language in pycountry.languages:
            if hasattr(language, "alpha_2") and hasattr(language, "alpha_3"):
                mapping[language.alpha_2] = language.alpha_3
        return mapping

    def to_iso639_3(self, code: str) -> str | None:
        code = code.lower().replace("_", "-")
        if "-" in code:
            code = code.split("-")[0]

        # Canonicalise deprecated ISO 639-1 aliases (iw→he, in→id, …).
        code = _ISO639_1_ALIASES.get(code, code)

        if len(code) == 3:
            if pycountry.languages.get(alpha_3=code):
                return code
            if code in self.mapping:
                return code

        if len(code) == 2:
            return self.iso639_3_map.get(code)

        return None

    def scripts_for(self, code: str) -> frozenset[str]:
        """Return every ISO 15924 script a language can be written in.

        Accepts any code format to_iso639_3() handles (2-letter, 3-letter,
        locale-subtagged). Empty set if the language is unknown or no
        scripts are recorded in GlotScript.
        """
        iso3 = self.to_iso639_3(code)
        if iso3 is None:
            return frozenset()
        return self.lang_to_scripts.get(iso3, frozenset())


_converter: ISOConverter | None = None


def get_converter() -> ISOConverter:
    global _converter
    if _converter is None:
        _converter = ISOConverter()
    return _converter


def set_converter(converter: ISOConverter | None) -> None:
    """Override the global converter. Pass None to reset (useful in tests)."""
    global _converter
    _converter = converter


def normalize_language_code(code: str, converter: ISOConverter | None = None) -> str:
    """Normalize language code to ISO 639-3; returns UNDEFINED_LANG if unknown."""
    if converter is None:
        converter = get_converter()
    iso3 = converter.to_iso639_3(code)
    return iso3 if iso3 else UNDEFINED_LANG


_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "Hang": (0xAC00, 0xD7A3),
    "Hani": (0x4E00, 0x9FFF),
    "Hira": (0x3040, 0x309F),
    "Kana": (0x30A0, 0x30FF),
    "Arab": (0x0600, 0x06FF),
    "Cyrl": (0x0400, 0x04FF),
    "Deva": (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, Sanskrit, Nepali)
    "Beng": (0x0980, 0x09FF),  # Bengali / Assamese
    "Thai": (0x0E00, 0x0E7F),
    "Grek": (0x0370, 0x03FF),
    "Hebr": (0x0590, 0x05FF),
}
_LATIN_RANGES: tuple[tuple[int, int], ...] = (
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
    (0x00C0, 0x024F),  # Latin-1 Supplement + Extended-A/B
)


def _classify_char(cp: int) -> str | None:
    for start, end in _LATIN_RANGES:
        if start <= cp <= end:
            return "Latn"
    for script, (start, end) in _SCRIPT_RANGES.items():
        if start <= cp <= end:
            return script
    return None


def detect_script(text: str) -> str:
    """Detects the ISO 15924 script code of the text.

    Returns UNDEFINED_SCRIPT (Zyyy) if no recognizable script is found.
    Returns 'Jpan' when Han ideographs mix with Hiragana or Katakana.
    """
    counts: dict[str, int] = {}
    for char in text:
        script = _classify_char(ord(char))
        if script is not None:
            counts[script] = counts.get(script, 0) + 1

    if not counts:
        return UNDEFINED_SCRIPT

    if counts.get("Hani", 0) > 0 and (counts.get("Hira", 0) > 0 or counts.get("Kana", 0) > 0):
        return "Jpan"

    return max(counts, key=lambda k: counts[k])
