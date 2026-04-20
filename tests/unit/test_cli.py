"""Unit tests for the rlid CLI.

Mocks out the full RobustLID engine so we don't load 1.5 GB of fastText
models on every test — the CLI's routing and output formatting logic is
what we're verifying here.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from robust_lid import cli
from robust_lid.utils import ISOConverter, set_converter
from tests.fixtures.fake_lid import FakeLID


@pytest.fixture(autouse=True)
def _small_converter(small_converter: ISOConverter) -> None:
    set_converter(small_converter)


class _FakeEngine:
    def __init__(self, predictions: dict[str, tuple[str, float]] | None = None) -> None:
        self._preds = predictions or {}
        self._default = ("eng_Latn", 0.9)
        self.calls: list[str] = []

    def predict(self, text: str) -> tuple[str, float]:
        self.calls.append(text)
        return self._preds.get(text, self._default)


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> _FakeEngine:
    engine = _FakeEngine()
    monkeypatch.setattr(cli, "_build_engine", lambda _args: engine)
    return engine


@pytest.fixture(autouse=True)
def _stdin_non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    # By default treat stdin as a pipe so the "no args + TTY" help path
    # doesn't trigger unless a test explicitly wants it.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)


@pytest.mark.unit
def test_single_positional_arg_plain_output(
    fake_engine: _FakeEngine, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["Hello world"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out == "eng_Latn\t0.900\tHello world\n"
    assert fake_engine.calls == ["Hello world"]


@pytest.mark.unit
def test_multiple_positional_args(
    fake_engine: _FakeEngine, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["Hello", "안녕"])
    out = capsys.readouterr().out
    assert out.count("\n") == 2
    assert fake_engine.calls == ["Hello", "안녕"]


@pytest.mark.unit
def test_json_output(fake_engine: _FakeEngine, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--json", "Hello"])
    line = capsys.readouterr().out.strip()
    parsed = json.loads(line)
    assert parsed == {"text": "Hello", "lang": "eng_Latn", "confidence": 0.9}


@pytest.mark.unit
def test_json_preserves_non_ascii(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = _FakeEngine({"안녕": ("kor_Hang", 0.99)})
    monkeypatch.setattr(cli, "_build_engine", lambda _args: engine)
    cli.main(["--json", "안녕"])
    out = capsys.readouterr().out
    assert "안녕" in out  # ensure_ascii=False preserves Hangul


@pytest.mark.unit
def test_no_text_flag(fake_engine: _FakeEngine, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--no-text", "Hello"])
    out = capsys.readouterr().out
    assert out == "eng_Latn\t0.900\n"


@pytest.mark.unit
def test_file_input(
    fake_engine: _FakeEngine,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "texts.txt"
    input_path.write_text("line one\n\nline two\n", encoding="utf-8")
    cli.main(["-f", str(input_path)])
    assert fake_engine.calls == ["line one", "line two"]  # blank line skipped


@pytest.mark.unit
def test_stdin_input(
    fake_engine: _FakeEngine,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("first\nsecond\n"))
    cli.main([])
    assert fake_engine.calls == ["first", "second"]


@pytest.mark.unit
def test_help_shown_when_tty_and_no_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    rc = cli.main([])
    assert rc == 2
    out = capsys.readouterr().out
    assert "usage:" in out.lower()


@pytest.mark.unit
def test_list_backends(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # Replace factories with FakeLID-producing lambdas so we don't download models.
    fake_factories = {
        name: (lambda _n=name: FakeLID([(_n, 1.0)])) for name in cli._BACKEND_FACTORIES
    }
    monkeypatch.setattr(cli, "_BACKEND_FACTORIES", fake_factories)
    monkeypatch.setattr(cli, "default_backend_order", lambda: list(fake_factories))
    rc = cli.main(["--list-backends"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "BACKEND" in out
    assert "LANGS" in out
    assert "SCRIPTS" in out
    assert "langid" in out


@pytest.mark.unit
def test_models_subset_builds_filtered_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def predict(self, _text: str) -> tuple[str, float]:
            return ("eng_Latn", 1.0)

    # Swap factories to FakeLID + RobustLID to recorder.
    fake_factories = {
        "langid": lambda: FakeLID([]),
        "ft176": lambda: FakeLID([]),
        "glotlid": lambda: FakeLID([]),
    }
    monkeypatch.setattr(cli, "_BACKEND_FACTORIES", fake_factories)
    monkeypatch.setattr(cli, "RobustLID", _Recorder)

    cli.main(["--models", "langid,ft176,glotlid", "Hello"])

    models = captured["models"]
    assert isinstance(models, list) and len(models) == 3
    weights = captured["weights"]
    assert isinstance(weights, list) and len(weights) == 3
    # ft176 has a boost default (1.3); langid is 1.0
    assert weights[1] > weights[0]


@pytest.mark.unit
def test_unknown_backend_in_models_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "RobustLID", lambda **_: None)
    with pytest.raises(SystemExit, match="unknown backend"):
        cli.main(["--models", "bogus", "Hello"])


@pytest.mark.unit
def test_models_cld3_requires_gcld3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "is_cld3_available", lambda: False)
    with pytest.raises(SystemExit, match="cld3"):
        cli.main(["--models", "cld3", "Hello"])


@pytest.mark.unit
def test_models_uniform_skips_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def predict(self, _t: str) -> tuple[str, float]:
            return ("eng_Latn", 1.0)

    monkeypatch.setattr(cli, "_BACKEND_FACTORIES", {"langid": lambda: FakeLID([])})
    monkeypatch.setattr(cli, "RobustLID", _Recorder)
    cli.main(["--models", "langid", "--uniform", "Hello"])
    assert "weights" not in captured  # defaults-skip path passes only models


@pytest.mark.unit
def test_default_path_uses_robustlid_no_args(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {"called": False}

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            captured["called"] = True
            captured["kwargs"] = kwargs

        def predict(self, _t: str) -> tuple[str, float]:
            return ("eng_Latn", 1.0)

    monkeypatch.setattr(cli, "RobustLID", _Recorder)
    cli.main(["Hello"])  # no --models, no --uniform
    assert captured["called"] is True
    # Default path passes execution-mode kwargs but no weight overrides
    assert captured["kwargs"] == {"parallel": True, "low_memory": False}


@pytest.mark.unit
def test_uniform_builds_all_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def predict(self, _t: str) -> tuple[str, float]:
            return ("eng_Latn", 1.0)

    monkeypatch.setattr(cli, "RobustLID", _Recorder)
    monkeypatch.setattr(cli, "default_backend_order", lambda: ["a", "b", "c"])
    cli.main(["--uniform", "Hello"])
    assert captured["weights"] == [1.0, 1.0, 1.0]
    assert captured["script_weights"] == [{}, {}, {}]
    assert captured["lang_weights"] == [{}, {}, {}]


@pytest.mark.unit
def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "robust-lid" in out


@pytest.mark.unit
def test_verbose_sets_fasttext_env_var(
    fake_engine: _FakeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ROBUST_LID_FASTTEXT_VERBOSE", raising=False)
    cli.main(["--verbose", "Hello"])
    assert os.environ.get("ROBUST_LID_FASTTEXT_VERBOSE") == "1"


@pytest.mark.unit
def test_verbose_prints_stage_messages_to_stderr(
    fake_engine: _FakeEngine, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["--verbose", "Hello"])
    captured = capsys.readouterr()
    assert "eng_Latn" in captured.out  # main output unchanged
    assert "[rlid] building ensemble" in captured.err
    assert "[rlid] predicting" in captured.err
    assert "[rlid] done" in captured.err


@pytest.mark.unit
def test_low_memory_flag_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def predict(self, _t: str) -> tuple[str, float]:
            return ("eng_Latn", 1.0)

    monkeypatch.setattr(cli, "RobustLID", _Recorder)
    cli.main(["--low-memory", "Hello"])
    assert captured.get("low_memory") is True
    assert captured.get("parallel") is True  # default


@pytest.mark.unit
def test_no_parallel_flag_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Recorder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def predict(self, _t: str) -> tuple[str, float]:
            return ("eng_Latn", 1.0)

    monkeypatch.setattr(cli, "RobustLID", _Recorder)
    cli.main(["--no-parallel", "Hello"])
    assert captured.get("parallel") is False
    assert captured.get("low_memory") is False


@pytest.mark.unit
def test_low_memory_with_models_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_BACKEND_FACTORIES", {"langid": lambda: FakeLID([])})
    monkeypatch.setattr(cli, "RobustLID", lambda **_: None)
    with pytest.raises(SystemExit, match="incompatible"):
        cli.main(["--low-memory", "--models", "langid", "Hello"])


@pytest.mark.unit
def test_non_verbose_is_silent_on_stderr(
    fake_engine: _FakeEngine, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["Hello"])
    captured = capsys.readouterr()
    assert "eng_Latn" in captured.out
    assert captured.err == ""


@pytest.mark.unit
def test_verbose_with_list_backends_logs_and_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_factories = {
        name: (lambda _n=name: FakeLID([(_n, 1.0)])) for name in cli._BACKEND_FACTORIES
    }
    monkeypatch.setattr(cli, "_BACKEND_FACTORIES", fake_factories)
    monkeypatch.setattr(cli, "default_backend_order", lambda: list(fake_factories))
    cli.main(["--verbose", "--list-backends"])
    captured = capsys.readouterr()
    assert "BACKEND" in captured.out
    assert "[rlid] listing backends" in captured.err


@pytest.mark.unit
def test_print_backend_inventory_uses_supported_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _FakeWithSets(FakeLID):
        supported_langs = frozenset({"eng", "kor"})
        supported_scripts = frozenset({"Latn", "Hang"})

    monkeypatch.setattr(
        cli,
        "_BACKEND_FACTORIES",
        {"langid": lambda: _FakeWithSets([("eng", 1.0)])},
    )
    monkeypatch.setattr(cli, "default_backend_order", lambda: ["langid"])
    cli._print_backend_inventory()
    out = capsys.readouterr().out
    assert "langid" in out
    assert "2" in out  # two langs and two scripts
