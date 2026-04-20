from pathlib import Path
from unittest.mock import MagicMock

import fasttext
import numpy as np
import pytest

from robust_lid.models import (
    FastText176LID,
    FastTextLID,
    _patch_fasttext_for_numpy2,
)
from robust_lid.utils import ISOConverter, set_converter


@pytest.fixture(autouse=True)
def _inject_small_converter(small_converter: ISOConverter) -> None:
    set_converter(small_converter)


@pytest.mark.unit
def test_fasttext_constructor_calls_download_with_url_and_path(tmp_path: Path) -> None:
    download = MagicMock()
    loader = MagicMock(return_value=MagicMock())
    FastTextLID(
        model_url="https://example.test/m.bin",
        model_filename="m.bin",
        cache_dir=tmp_path,
        download_fn=download,
        model_loader=loader,
    )
    download.assert_called_once_with("https://example.test/m.bin", tmp_path / "m.bin")
    loader.assert_called_once_with(str(tmp_path / "m.bin"))


@pytest.mark.unit
def test_fasttext_predict_strips_label_prefix_and_normalizes(tmp_path: Path) -> None:
    mock_model = MagicMock()
    mock_model.predict.return_value = (("__label__en",), (0.95,))
    model = FastTextLID(
        model_url="unused",
        model_filename="m.bin",
        cache_dir=tmp_path,
        download_fn=lambda _url, _path: None,
        model_loader=lambda _path: mock_model,
    )

    result = model.predict("hello world")

    assert result == [("eng", pytest.approx(0.95))]
    mock_model.predict.assert_called_once()


@pytest.mark.unit
def test_fasttext_predict_replaces_newlines(tmp_path: Path) -> None:
    mock_model = MagicMock()
    mock_model.predict.return_value = ((), ())
    model = FastTextLID(
        model_url="unused",
        model_filename="m.bin",
        cache_dir=tmp_path,
        download_fn=lambda _url, _path: None,
        model_loader=lambda _path: mock_model,
    )
    model.predict("line1\nline2")
    args, _kwargs = mock_model.predict.call_args
    assert "\n" not in args[0]


@pytest.mark.unit
def test_fasttext_predict_swallows_errors_and_returns_empty(tmp_path: Path) -> None:
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("boom")
    model = FastTextLID(
        model_url="unused",
        model_filename="m.bin",
        cache_dir=tmp_path,
        download_fn=lambda _url, _path: None,
        model_loader=lambda _path: mock_model,
    )
    assert model.predict("text") == []


@pytest.mark.unit
def test_numpy2_patch_is_idempotent() -> None:
    cls = fasttext.FastText._FastText
    assert getattr(cls, "_robust_lid_patched", False) is True
    _patch_fasttext_for_numpy2()  # second call must not break
    assert cls._robust_lid_patched is True


@pytest.mark.unit
def test_numpy2_patch_returns_asarray_not_np_array_copy_false() -> None:
    """Regression: NumPy 2 raises on `np.array(copy=False)` in upstream
    fasttext-wheel. Our patch uses `np.asarray` so predict must succeed and
    return a numpy array."""

    class FakeBinding:
        def predict(self, *_args: object, **_kwargs: object) -> list[tuple[float, str]]:
            return [(0.9, "__label__en"), (0.1, "__label__de")]

    fake_model = fasttext.FastText._FastText.__new__(fasttext.FastText._FastText)
    fake_model.f = FakeBinding()
    labels, probs = fake_model.predict("hello", k=2)
    assert labels == ("__label__en", "__label__de")
    assert isinstance(probs, np.ndarray)
    assert probs.tolist() == [0.9, 0.1]


@pytest.mark.unit
def test_patched_predict_rejects_newline() -> None:
    class FakeBinding:
        def predict(self, *_a: object, **_kw: object) -> list[object]:
            return []

    fake = fasttext.FastText._FastText.__new__(fasttext.FastText._FastText)
    fake.f = FakeBinding()
    with pytest.raises(ValueError, match="one line at a time"):
        fake.predict("line1\nline2")


@pytest.mark.unit
def test_patched_predict_handles_empty_results() -> None:
    class FakeBinding:
        def predict(self, *_a: object, **_kw: object) -> list[object]:
            return []

    fake = fasttext.FastText._FastText.__new__(fasttext.FastText._FastText)
    fake.f = FakeBinding()
    labels, probs = fake.predict("hello")
    assert labels == ()
    assert probs.shape == (0,)


@pytest.mark.unit
def test_patched_predict_delegates_list_to_original() -> None:
    """When text is a list, patched_predict must delegate to the original
    multiline path (unpatched)."""

    class FakeBinding:
        def multilinePredict(  # noqa: N802 — matches fasttext C++ binding name
            self, _t: list[str], _k: int, _th: float, _oue: str
        ) -> tuple[list[list[str]], list[list[float]]]:
            return [["__label__en"]], [[0.9]]

        def predict(self, *_a: object, **_kw: object) -> list[object]:
            raise AssertionError("single-text path should not be hit for list input")

    fake = fasttext.FastText._FastText.__new__(fasttext.FastText._FastText)
    fake.f = FakeBinding()
    labels, probs = fake.predict(["hello"])
    assert labels == [["__label__en"]]
    assert probs == [[0.9]]


@pytest.mark.unit
def test_fasttext176_subclass_uses_correct_url(tmp_path: Path) -> None:
    download = MagicMock()
    FastText176LID(
        cache_dir=tmp_path,
        download_fn=download,
        model_loader=lambda _path: MagicMock(),
    )
    url_arg, path_arg = download.call_args[0]
    assert url_arg.endswith("lid.176.bin")
    assert path_arg.name == "lid.176.bin"
