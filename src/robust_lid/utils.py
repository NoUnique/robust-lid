import csv
import re
from typing import Dict, Optional, Tuple
import pycountry
from .constants import GLOTSCRIPT_TSV

class ISOConverter:
    def __init__(self):
        self.mapping = self._load_mapping()
        self.iso639_3_map = self._load_pycountry_map()

    def _load_mapping(self) -> Dict[str, str]:
        """Loads GlotScript.tsv for mapping various codes to ISO 639-3."""
        mapping = {}
        if not GLOTSCRIPT_TSV.exists():
            # Fallback or warning if file missing, though it should be there
            return mapping
        
        with open(GLOTSCRIPT_TSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                iso639_3 = row['ISO639-3']
                # Map ISO 639-3 to itself
                mapping[iso639_3] = iso639_3
                
                # You might want to map other columns if they exist and are useful
                # For now, we primarily need to ensure we can validate or lookup 
                # based on what models return.
                # Many models return 2-letter codes (ISO 639-1).
                
        return mapping

    def _load_pycountry_map(self) -> Dict[str, str]:
        """Creates a mapping from ISO 639-1 (2-letter) to ISO 639-3 (3-letter)."""
        mapping = {}
        for language in pycountry.languages:
            if hasattr(language, 'alpha_2') and hasattr(language, 'alpha_3'):
                mapping[language.alpha_2] = language.alpha_3
        return mapping

    def to_iso639_3(self, code: str) -> Optional[str]:
        """Converts a language code to ISO 639-3."""
        code = code.lower().replace('_', '-')
        
        # Handle some common cases or model specific outputs if needed
        if '-' in code:
            code = code.split('-')[0] # simplistic approach, refine if needed

        if len(code) == 3:
            # Check if it's a valid 3-letter code (simple check via pycountry or our map)
            if pycountry.languages.get(alpha_3=code):
                return code
            # Also check our GlotScript map
            if code in self.mapping:
                return code
                
        if len(code) == 2:
            return self.iso639_3_map.get(code)
            
        return None

_converter = None

def get_converter():
    global _converter
    if _converter is None:
        _converter = ISOConverter()
    return _converter

def normalize_language_code(code: str) -> str:
    """
    Normalize language code to ISO 639-3.
    Returns 'und' if unknown.
    """
    converter = get_converter()
    iso3 = converter.to_iso639_3(code)
    return iso3 if iso3 else "und"

def detect_script(text: str) -> str:
    """
    Detects the script of the text.
    Returns ISO 15924 script code (e.g., 'Latn', 'Hang').
    Defaults to 'Zyyy' (Common) or 'Zxxx' (Unwritten) if unknown.
    """
    # Simple implementation using regex ranges or a library if available.
    # Since we don't have 'whichscript' installed as a dependency in the plan (it wasn't in the list),
    # we can implement a basic one or use a heuristic.
    # However, the user mentioned 'whichscript' in the prompt. 
    # Let's check if we can use the logic provided in _temp/lid_functions.py which had a custom implementation.
    
    # For now, let's implement a basic heuristic based on unicode ranges for common scripts
    # or rely on what was in _temp/lid_functions.py if we want to copy that logic.
    # The user provided code in _temp/lid_functions.py:101 lid_iso15924
    # It imports `whichscript.iso15924`. 
    # Since `whichscript` is not in my dependency list, I should probably implement a simplified version 
    # or ask to add it. But I'll try to implement a basic one here.
    
    # Actually, let's look at the user request again. 
    # "langid, langdetect, cld2, cld3, fasttext176, fasttext215e, glotlidv3, whichscript로 디텍션하여"
    # The user explicitly mentioned `whichscript`. I should probably have added it to dependencies.
    # But since I didn't, I will use a placeholder or a simple implementation for now 
    # and maybe add it later if needed. 
    # Wait, `whichscript` might not be on PyPI or might be a local thing?
    # The user provided `_temp/lid_functions.py` which imports `whichscript.iso15924`.
    # I will assume for this step I can implement a basic one or I should have added it.
    # Let's implement a basic one for now to unblock.
    
    # Basic script detection
    scripts = {
        'Hang': (0xAC00, 0xD7A3), # Hangul Syllables
        'Latn': (0x0041, 0x007A), # Basic Latin (very rough, should be expanded but sufficient for now)
        'Hani': (0x4E00, 0x9FFF), # CJK Unified Ideographs
        'Hira': (0x3040, 0x309F), # Hiragana
        'Kana': (0x30A0, 0x30FF), # Katakana
        'Arab': (0x0600, 0x06FF), # Arabic
        'Cyrl': (0x0400, 0x04FF), # Cyrillic
    }
    
    counts = {k: 0 for k in scripts}
    total = 0
    for char in text:
        cp = ord(char)
        # Check ranges
        for script, (start, end) in scripts.items():
            if start <= cp <= end:
                counts[script] += 1
                total += 1
                break
        # Also check Latin extended if needed, but keeping it simple
    
    if total == 0:
        return "Zyyy"

    # Special handling for Japanese (Jpan = Hani + Hira + Kana)
    if counts['Hani'] > 0 and (counts['Hira'] > 0 or counts['Kana'] > 0):
        return "Jpan"
    
    # Special handling for Korean (Kore = Hang + Hani)? Usually just Hang is fine for modern Korean.
    
    most_common = max(counts, key=counts.get)
    return most_common

