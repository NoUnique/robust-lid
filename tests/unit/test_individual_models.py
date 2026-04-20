"""Cover the non-fastText backends that ship with their own lightweight models
(no network or 1GB+ binaries needed).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robust_lid import models
from robust_lid.models import (
    CLD2LID,
    CLD3_UNAVAILABLE_MSG,
    CLD3LID,
    FastText218eLID,
    GlotLID,
    LangdetectLID,
    LangidLID,
    _default_download,
    _emit_cld3_import_warning,
    is_cld3_available,
)
from robust_lid.utils import ISOConverter, set_converter


@pytest.fixture(autouse=True)
def _inject_small_converter(small_converter: ISOConverter) -> None:
    set_converter(small_converter)


@pytest.mark.unit
def test_langid_predict_english() -> None:
    results = LangidLID().predict("The quick brown fox jumps over the lazy dog.")
    assert results, "langid should return at least one prediction for English text"
    top_lang, top_prob = results[0]
    assert top_lang == "eng"
    assert isinstance(top_prob, float)


@pytest.mark.unit
def test_langid_returns_normalized_probability() -> None:
    """Regression: the raw langid.classify() returns negative log-probabilities.
    LangidLID must use the norm_probs=True identifier so the confidence is in
    [0, 1] — otherwise it corrupts the ensemble vote total."""
    results = LangidLID().predict("The quick brown fox jumps over the lazy dog.")
    _lang, prob = results[0]
    assert 0.0 <= prob <= 1.0


@pytest.mark.unit
def test_langid_predict_returns_empty_on_error() -> None:
    model = LangidLID()
    with patch.object(model._identifier, "classify", side_effect=RuntimeError("boom")):
        assert model.predict("text") == []


@pytest.mark.unit
def test_langdetect_predict_english() -> None:
    results = LangdetectLID().predict("The quick brown fox jumps over the lazy dog.")
    assert results
    codes = [lang for lang, _p in results]
    assert "eng" in codes


@pytest.mark.unit
def test_langdetect_predict_swallows_exceptions() -> None:
    with patch.object(models.langdetect, "detect_langs", side_effect=RuntimeError("boom")):
        assert LangdetectLID().predict("text") == []


@pytest.mark.unit
def test_langdetect_seed_is_configurable() -> None:
    model = LangdetectLID(seed=123)
    assert model.seed == 123


@pytest.mark.unit
def test_cld2_predict_english() -> None:
    results = CLD2LID().predict("The quick brown fox jumps over the lazy dog.")
    assert results
    codes = [lang for lang, _p in results]
    assert "eng" in codes


@pytest.mark.unit
def test_cld2_predict_swallows_exceptions() -> None:
    with patch.object(models.cld2, "detect", side_effect=RuntimeError("boom")):
        assert CLD2LID().predict("text") == []


@pytest.mark.unit
def test_cld3_disabled_when_gcld3_missing() -> None:
    with patch.object(models, "gcld3", None):
        model = CLD3LID()
        assert model.detector is None
        assert model.predict("hello") == []


@pytest.mark.unit
def test_is_cld3_available_reflects_gcld3_presence() -> None:
    with patch.object(models, "gcld3", None):
        assert is_cld3_available() is False
    with patch.object(models, "gcld3", MagicMock()):
        assert is_cld3_available() is True


@pytest.mark.unit
def test_import_warning_helper_emits_importwarning() -> None:
    with pytest.warns(ImportWarning, match="gcld3 is not installed"):
        _emit_cld3_import_warning()


@pytest.mark.unit
def test_cld3_warning_message_includes_install_hints() -> None:
    assert "protobuf-compiler" in CLD3_UNAVAILABLE_MSG
    assert "pip install" in CLD3_UNAVAILABLE_MSG


@pytest.mark.unit
def test_langid_supported_langs_is_non_empty_and_three_letter() -> None:
    set_converter(None)  # need the full ISO converter, not the 5-lang test one
    langs = LangidLID().supported_langs
    assert len(langs) > 50  # langid has 97 langs
    assert all(len(code) == 3 for code in langs)
    assert "eng" in langs
    assert "kor" in langs


@pytest.mark.unit
def test_langid_supported_scripts_covers_common_scripts() -> None:
    set_converter(None)
    scripts = LangidLID().supported_scripts
    assert {"Latn", "Cyrl", "Hang", "Hani", "Arab"} <= scripts


@pytest.mark.unit
def test_langdetect_supported_langs_is_55ish() -> None:
    set_converter(None)
    langs = LangdetectLID().supported_langs
    assert 40 < len(langs) < 70
    assert "eng" in langs


@pytest.mark.unit
def test_cld2_supported_langs_covers_majors() -> None:
    set_converter(None)
    langs = CLD2LID().supported_langs
    assert "eng" in langs
    assert "zho" in langs
    assert "heb" in langs


@pytest.mark.unit
def test_cld3_supported_langs_when_available() -> None:
    set_converter(None)
    fake_gcld3 = MagicMock()
    fake_gcld3.NNetLanguageIdentifier.return_value = MagicMock()
    with patch.object(models, "gcld3", fake_gcld3):
        langs = CLD3LID().supported_langs
        assert "eng" in langs
        assert "kor" in langs
        assert "heb" in langs  # "he" or "iw" normalizes to heb
        assert 90 < len(langs) < 120  # gcld3 lists 107 codes


@pytest.mark.unit
def test_cld3_supported_langs_empty_when_unavailable() -> None:
    with patch.object(models, "gcld3", None):
        assert CLD3LID().supported_langs == frozenset()


@pytest.mark.unit
def test_fasttext_supported_langs_derived_from_get_labels(tmp_path: Path) -> None:
    fake_model = MagicMock()
    fake_model.get_labels.return_value = [
        "__label__eng_Latn",
        "__label__kor_Hang",
        "__label__zho_Hans",
        "__label__xxx",  # unknown — should be dropped by normalize
    ]
    from robust_lid.models import FastTextLID

    model = FastTextLID(
        model_url="unused",
        model_filename="m.bin",
        cache_dir=tmp_path,
        download_fn=lambda _u, _p: None,
        model_loader=lambda _p: fake_model,
    )
    langs = model.supported_langs
    assert "eng" in langs
    assert "kor" in langs
    assert "zho" in langs
    assert "xxx" not in langs  # unknown dropped


@pytest.mark.unit
def test_lid_base_class_default_supported_langs_is_empty() -> None:
    """Base class default: empty set → script gating disabled for custom LIDs."""
    from tests.fixtures.fake_lid import FakeLID

    fake = FakeLID([("eng", 0.9)])
    assert fake.supported_langs == frozenset()
    assert fake.supported_scripts == frozenset()


@pytest.mark.unit
def test_cld3_init_failure_sets_none_detector() -> None:
    fake_gcld3 = MagicMock()
    fake_gcld3.NNetLanguageIdentifier.side_effect = RuntimeError("init boom")
    with patch.object(models, "gcld3", fake_gcld3):
        model = CLD3LID()
        assert model.detector is None


@pytest.mark.unit
def test_cld3_predict_reliable_result() -> None:
    fake_gcld3 = MagicMock()
    fake_result = MagicMock(is_reliable=True, language="en", probability=0.87)
    fake_detector = MagicMock()
    fake_detector.FindLanguage.return_value = fake_result
    fake_gcld3.NNetLanguageIdentifier.return_value = fake_detector

    with patch.object(models, "gcld3", fake_gcld3):
        model = CLD3LID()
        results = model.predict("hello world")
        assert results == [("eng", pytest.approx(0.87))]


@pytest.mark.unit
def test_cld3_predict_unreliable_returns_empty() -> None:
    fake_gcld3 = MagicMock()
    fake_result = MagicMock(is_reliable=False)
    fake_detector = MagicMock()
    fake_detector.FindLanguage.return_value = fake_result
    fake_gcld3.NNetLanguageIdentifier.return_value = fake_detector

    with patch.object(models, "gcld3", fake_gcld3):
        assert CLD3LID().predict("hello") == []


@pytest.mark.unit
def test_cld3_predict_swallows_exceptions() -> None:
    fake_gcld3 = MagicMock()
    fake_detector = MagicMock()
    fake_detector.FindLanguage.side_effect = RuntimeError("boom")
    fake_gcld3.NNetLanguageIdentifier.return_value = fake_detector

    with patch.object(models, "gcld3", fake_gcld3):
        assert CLD3LID().predict("hello") == []


@pytest.mark.unit
def test_fasttext218e_subclass_wires_correct_url(tmp_path: Path) -> None:
    download = MagicMock()
    FastText218eLID(cache_dir=tmp_path, download_fn=download, model_loader=lambda _p: MagicMock())
    url, path = download.call_args[0]
    assert "fasttext-language-identification" in url
    assert path.name == "lid.218e.bin"


@pytest.mark.unit
def test_glotlid_subclass_wires_correct_url(tmp_path: Path) -> None:
    download = MagicMock()
    GlotLID(cache_dir=tmp_path, download_fn=download, model_loader=lambda _p: MagicMock())
    url, path = download.call_args[0]
    assert "glotlid" in url
    assert path.name == "glotlid_v3.bin"


@pytest.mark.unit
def test_default_download_skips_when_file_exists(tmp_path: Path) -> None:
    existing = tmp_path / "model.bin"
    existing.write_bytes(b"already here")
    with patch.object(models, "requests") as mock_requests:
        _default_download("https://unused.test/m.bin", existing)
        mock_requests.get.assert_not_called()
    assert existing.read_bytes() == b"already here"


@pytest.mark.unit
def test_default_download_writes_streamed_chunks(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "model.bin"
    fake_response = MagicMock()
    fake_response.iter_content.return_value = [b"chunk1", b"", b"chunk2"]
    with patch.object(models.requests, "get", return_value=fake_response) as mock_get:
        _default_download("https://example.test/m.bin", target)

    mock_get.assert_called_once()
    _args, kwargs = mock_get.call_args
    assert kwargs["stream"] is True
    assert kwargs["timeout"] > 0
    fake_response.raise_for_status.assert_called_once()
    assert target.read_bytes() == b"chunk1chunk2"


@pytest.mark.unit
def test_default_download_raises_on_http_error(tmp_path: Path) -> None:
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = RuntimeError("404")
    with (
        patch.object(models.requests, "get", return_value=fake_response),
        pytest.raises(RuntimeError, match="404"),
    ):
        _default_download("https://example.test/m.bin", tmp_path / "m.bin")
