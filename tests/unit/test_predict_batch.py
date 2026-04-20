"""Tests for the ``predict_batch`` path across LID base, FastTextLID, and
RobustLID. We verify: (1) results match per-text ``predict`` for correctness,
(2) the fasttext path actually calls ``multilinePredict`` (single list call),
(3) RobustLID routes through ``predict_batch`` on every backend exactly once
per batch instead of once per text.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from robust_lid.ensemble import RobustLID
from robust_lid.models import FastTextLID
from robust_lid.utils import ISOConverter, set_converter
from tests.fixtures.fake_lid import FakeLID


@pytest.fixture(autouse=True)
def _small_converter(small_converter: ISOConverter) -> None:
    set_converter(small_converter)


# ---------------------------------------------------------------------------
# LID base class default
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_base_predict_batch_is_default_per_text_loop() -> None:
    """Backends that don't override predict_batch get a naive loop."""
    fake = FakeLID([("eng", 0.9)])
    results = fake.predict_batch(["a", "b", "c"])
    assert results == [[("eng", 0.9)], [("eng", 0.9)], [("eng", 0.9)]]
    assert fake.calls == ["a", "b", "c"]


@pytest.mark.unit
def test_base_predict_batch_empty_returns_empty() -> None:
    assert FakeLID([("eng", 0.9)]).predict_batch([]) == []


# ---------------------------------------------------------------------------
# FastTextLID native batch
# ---------------------------------------------------------------------------


class _BatchCountingFakeModel:
    """Mimics fasttext's native list predict path.

    Returns `(list_of_label_tuples, list_of_score_arrays)` like the real thing
    and records how many times .predict was called — a single batch call must
    hit us exactly once, not N times."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_inputs: list[str] | str | None = None

    def predict(
        self, texts: list[str] | str, k: int = 5
    ) -> tuple[list[tuple[str, ...]], list[list[float]]]:
        self.call_count += 1
        self.last_inputs = texts
        assert isinstance(texts, list), "batch path must be a single list call"
        labels = [("__label__en", "__label__de")] * len(texts)
        scores = [[0.9, 0.1]] * len(texts)
        return labels, scores


@pytest.mark.unit
def test_fasttext_predict_batch_uses_single_c_call(tmp_path: Path) -> None:
    fake_model = _BatchCountingFakeModel()
    model = FastTextLID(
        model_url="unused",
        model_filename="m.bin",
        cache_dir=tmp_path,
        download_fn=lambda _u, _p: None,
        model_loader=lambda _p: fake_model,
    )
    texts = ["one", "two", "three", "four"]
    results = model.predict_batch(texts)

    assert fake_model.call_count == 1  # single C-level batch call, not 4
    assert fake_model.last_inputs == texts
    assert len(results) == 4
    for per_text in results:
        assert per_text[0] == ("eng", pytest.approx(0.9))


@pytest.mark.unit
def test_fasttext_predict_batch_empty_input_returns_empty(tmp_path: Path) -> None:
    fake_model = _BatchCountingFakeModel()
    model = FastTextLID(
        model_url="unused",
        model_filename="m.bin",
        cache_dir=tmp_path,
        download_fn=lambda _u, _p: None,
        model_loader=lambda _p: fake_model,
    )
    assert model.predict_batch([]) == []
    assert fake_model.call_count == 0  # no C call for empty input


@pytest.mark.unit
def test_fasttext_predict_batch_swallows_errors(tmp_path: Path) -> None:
    fake_model = MagicMock()
    fake_model.predict.side_effect = RuntimeError("boom")
    model = FastTextLID(
        model_url="unused",
        model_filename="m.bin",
        cache_dir=tmp_path,
        download_fn=lambda _u, _p: None,
        model_loader=lambda _p: fake_model,
    )
    results = model.predict_batch(["a", "b", "c"])
    assert results == [[], [], []]  # preserves input length


@pytest.mark.unit
def test_fasttext_predict_batch_caches_label_normalisation(tmp_path: Path) -> None:
    """The normalized-label cache means we don't re-resolve the same code
    N times. We verify by counting normalize_language_code calls."""
    from robust_lid import models as models_mod

    fake_model = MagicMock()
    fake_model.predict.return_value = (
        [("__label__en", "__label__en")] * 100,  # same labels, 100 texts
        [[0.9, 0.1]] * 100,
    )
    model = FastTextLID(
        model_url="unused",
        model_filename="m.bin",
        cache_dir=tmp_path,
        download_fn=lambda _u, _p: None,
        model_loader=lambda _p: fake_model,
    )

    call_count = 0

    real_normalize = models_mod.normalize_language_code

    def counting_normalize(code: str) -> str:
        nonlocal call_count
        call_count += 1
        return real_normalize(code)

    models_mod.normalize_language_code = counting_normalize
    try:
        model.predict_batch(["x"] * 100)
    finally:
        models_mod.normalize_language_code = real_normalize

    # "en" appears 200 times in raw labels but should only be normalized once
    # thanks to the cache. Allow at most one per unique label.
    assert call_count == 1, f"expected 1 normalize call (cached), got {call_count}"


# ---------------------------------------------------------------------------
# RobustLID.predict_batch
# ---------------------------------------------------------------------------


class _CountingBackend(FakeLID):
    """Counts how many times predict / predict_batch were invoked."""

    def __init__(self, response: list[tuple[str, float]]) -> None:
        super().__init__(response)
        self.single_calls = 0
        self.batch_calls = 0

    def predict(self, text: str) -> list[tuple[str, float]]:
        self.single_calls += 1
        return super().predict(text)

    def predict_batch(self, texts: list[str]) -> list[list[tuple[str, float]]]:
        self.batch_calls += 1
        return [super(_CountingBackend, self).predict(t) for t in texts]


@pytest.mark.unit
def test_robustlid_predict_batch_calls_each_backend_once() -> None:
    backends = [
        _CountingBackend([("eng", 0.9)]),
        _CountingBackend([("eng", 0.7)]),
        _CountingBackend([("fra", 0.5)]),
    ]
    lid = RobustLID(models=list(backends))
    results = lid.predict_batch(["Hello", "world", "there"])

    assert len(results) == 3
    for b in backends:
        assert b.batch_calls == 1  # exactly one batch call per backend
        assert b.single_calls == 0  # NOT called per text


@pytest.mark.unit
def test_robustlid_predict_batch_matches_predict_loop() -> None:
    """Same inputs, same outputs whether we call predict per text or
    predict_batch once."""
    backends = [
        FakeLID([("eng", 0.9)]),
        FakeLID([("kor", 0.8)]),
    ]
    lid_a = RobustLID(models=list(backends))
    lid_b = RobustLID(models=[FakeLID([("eng", 0.9)]), FakeLID([("kor", 0.8)])])

    texts = ["Hello", "안녕", "World"]
    per_text = [lid_a.predict(t) for t in texts]
    batched = lid_b.predict_batch(texts)
    assert per_text == batched


@pytest.mark.unit
def test_robustlid_predict_batch_empty() -> None:
    lid = RobustLID(models=[FakeLID([])])
    assert lid.predict_batch([]) == []


@pytest.mark.unit
def test_robustlid_predict_batch_all_backends_empty_returns_sentinel() -> None:
    lid = RobustLID(models=[FakeLID([]), FakeLID([])])
    results = lid.predict_batch(["hello"])
    assert results == [("und_Zyyy", 0.0)]


@pytest.mark.unit
def test_robustlid_predict_batch_no_parallel_path() -> None:
    """With parallel=False we take the plain loop path."""
    backends = [_CountingBackend([("eng", 0.9)]) for _ in range(3)]
    lid = RobustLID(models=list(backends), parallel=False)
    lid.predict_batch(["a", "b"])
    for b in backends:
        assert b.batch_calls == 1


@pytest.mark.unit
def test_robustlid_predict_batch_parallel_uses_threadpool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With parallel=True and >1 models, predict_batch must dispatch through
    a thread pool (one submission per backend, not per text)."""
    from concurrent.futures import ThreadPoolExecutor

    original_map_count = {"calls": 0}
    real_init = ThreadPoolExecutor.__init__

    def counting_init(self: ThreadPoolExecutor, *args: object, **kwargs: object) -> None:
        original_map_count["calls"] += 1
        real_init(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", counting_init)

    backends = [_CountingBackend([("eng", 0.9)]) for _ in range(3)]
    lid = RobustLID(models=list(backends), parallel=True)
    lid.predict_batch(["a", "b", "c", "d"])
    # One pool for the entire batch (not 4, one per text).
    assert original_map_count["calls"] == 1


@pytest.mark.unit
def test_robustlid_predict_batch_low_memory_streams() -> None:
    """Low-memory batch mode loads each backend once for the whole batch."""
    from robust_lid import ensemble as ens

    construct_log: list[str] = []

    def make_factory(name: str) -> object:
        def _f() -> _CountingBackend:
            construct_log.append(name)
            return _CountingBackend([(name[:3], 1.0)])

        return _f

    factories = [make_factory(n) for n in ("eng", "kor", "jpn")]
    ens_module = ens
    orig_factories_fn = ens_module._default_factories
    ens_module._default_factories = lambda: factories  # type: ignore[assignment]

    try:
        lid = RobustLID(
            low_memory=True,
            weights=[1.0, 1.0, 1.0],
            script_weights=[{}, {}, {}],
            lang_weights=[{}, {}, {}],
        )
        lid.predict_batch(["t1", "t2", "t3", "t4"])
    finally:
        ens_module._default_factories = orig_factories_fn  # type: ignore[assignment]

    # Each factory invoked exactly once, despite 4 texts in the batch.
    assert construct_log == ["eng", "kor", "jpn"]
