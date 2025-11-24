from typing import List, Tuple, Dict
from collections import defaultdict
from .models import (
    LangidLID, LangdetectLID, CLD2LID, CLD3LID,
    FastText176LID, FastText218eLID, GlotLID
)
from .utils import detect_script

class RobustLID:
    def __init__(self):
        self.models = [
            LangidLID(),
            LangdetectLID(),
            CLD2LID(),
            CLD3LID(),
            FastText176LID(),
            FastText218eLID(),
            GlotLID()
        ]

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predicts the language and script of the text using ensemble voting.
        Returns (language_script_code, confidence).
        Example: ('eng_Latn', 0.9)
        """
        votes = defaultdict(float)
        
        # Collect predictions from all models
        for model in self.models:
            predictions = model.predict(text)
            if predictions:
                # Top 1 prediction from each model
                lang, prob = predictions[0]
                if lang != 'und':
                    votes[lang] += prob

        if not votes:
            return "und_Zyyy", 0.0

        # Normalize votes
        total_score = sum(votes.values())
        normalized_votes = {k: v / total_score for k, v in votes.items()}
        
        # Get best language
        best_lang = max(normalized_votes, key=normalized_votes.get)
        confidence = normalized_votes[best_lang]
        
        # Detect script
        script = detect_script(text)
        
        return f"{best_lang}_{script}", confidence
