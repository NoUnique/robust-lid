import pytest

from robust_lid.constants import UNDEFINED_LANG
from robust_lid.ensemble import compute_ensemble_vote


@pytest.mark.unit
def test_unanimous_winner() -> None:
    lang, conf = compute_ensemble_vote([[("eng", 0.9)], [("eng", 0.8)], [("eng", 0.7)]])
    assert lang == "eng"
    assert conf == pytest.approx(1.0)


@pytest.mark.unit
def test_weighted_majority() -> None:
    lang, conf = compute_ensemble_vote([[("eng", 0.9)], [("eng", 0.8)], [("fra", 0.7)]])
    assert lang == "eng"
    assert conf == pytest.approx((0.9 + 0.8) / (0.9 + 0.8 + 0.7))


@pytest.mark.unit
def test_empty_predictions_yield_undefined() -> None:
    assert compute_ensemble_vote([]) == (UNDEFINED_LANG, 0.0)


@pytest.mark.unit
def test_all_models_empty_yield_undefined() -> None:
    assert compute_ensemble_vote([[], [], []]) == (UNDEFINED_LANG, 0.0)


@pytest.mark.unit
def test_undefined_predictions_are_ignored() -> None:
    lang, conf = compute_ensemble_vote([[(UNDEFINED_LANG, 0.99)], [("eng", 0.3)]])
    assert lang == "eng"
    assert conf == pytest.approx(1.0)


@pytest.mark.unit
def test_only_top1_counts() -> None:
    # Second-choice 'fra' in model 0 must NOT be counted.
    lang, _conf = compute_ensemble_vote([[("eng", 0.6), ("fra", 0.4)], [("fra", 0.5)]])
    assert lang == "eng"


@pytest.mark.unit
def test_weights_override_majority() -> None:
    """With weights the minority can win if its model is trusted enough."""
    # Without weights: eng would win (two votes at 0.6 each vs one fra at 0.9).
    lang_equal, _ = compute_ensemble_vote([[("eng", 0.6)], [("eng", 0.6)], [("fra", 0.9)]])
    assert lang_equal == "eng"

    # With weights that trust model 2 much more, fra wins.
    lang_weighted, _ = compute_ensemble_vote(
        [[("eng", 0.6)], [("eng", 0.6)], [("fra", 0.9)]],
        weights=[0.1, 0.1, 5.0],
    )
    assert lang_weighted == "fra"


@pytest.mark.unit
def test_zero_weight_silences_a_model() -> None:
    lang, _ = compute_ensemble_vote(
        [[("fra", 0.99)], [("eng", 0.5)]],
        weights=[0.0, 1.0],
    )
    assert lang == "eng"


@pytest.mark.unit
def test_weights_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="weights length"):
        compute_ensemble_vote([[("eng", 0.9)]], weights=[1.0, 1.0])


@pytest.mark.unit
def test_non_positive_prob_is_ignored() -> None:
    """Regression: langid used to return raw log-probabilities (e.g. -2042.7)
    which flipped the sign of the vote total and handed wins to wrong
    languages. The ensemble must now ignore non-positive confidences."""
    lang, conf = compute_ensemble_vote([[("mar", -2042.7)], [("hin", 0.99)]])
    assert lang == "hin"
    assert conf == pytest.approx(1.0)


@pytest.mark.unit
def test_all_non_positive_returns_undefined() -> None:
    lang, conf = compute_ensemble_vote([[("mar", -1.0)], [("hin", 0.0)]])
    assert lang == UNDEFINED_LANG
    assert conf == 0.0
