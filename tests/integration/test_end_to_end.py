"""End-to-end smoke test. Skipped by default because it downloads ~1.5GB of models.

Run with: uv run pytest -m "slow and network" tests/integration
"""

import pytest

from robust_lid import RobustLID


@pytest.mark.slow
@pytest.mark.network
def test_english_text_end_to_end() -> None:
    lid = RobustLID()
    code, confidence = lid.predict("The quick brown fox jumps over the lazy dog.")
    assert code.startswith("eng_")
    assert 0.0 < confidence <= 1.0


@pytest.mark.slow
@pytest.mark.network
def test_korean_text_end_to_end() -> None:
    lid = RobustLID()
    code, confidence = lid.predict("안녕하세요. 오늘 날씨가 정말 좋네요.")
    assert code.startswith("kor_")
    assert 0.0 < confidence <= 1.0
