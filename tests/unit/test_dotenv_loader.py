"""Unit tests for the tiny stdlib dotenv loader used by integration tests."""

from pathlib import Path

import pytest

from tests.integration.conftest import _parse_dotenv_line, load_dotenv


@pytest.mark.unit
def test_parse_simple_key_value() -> None:
    assert _parse_dotenv_line("HF_TOKEN=abc123") == ("HF_TOKEN", "abc123")


@pytest.mark.unit
def test_parse_strips_whitespace() -> None:
    assert _parse_dotenv_line("  KEY = value  ") == ("KEY", "value")


@pytest.mark.unit
def test_parse_strips_matching_quotes() -> None:
    assert _parse_dotenv_line('KEY="hello world"') == ("KEY", "hello world")
    assert _parse_dotenv_line("KEY='hello world'") == ("KEY", "hello world")


@pytest.mark.unit
def test_parse_leaves_mismatched_quotes_alone() -> None:
    assert _parse_dotenv_line('KEY="unterminated') == ("KEY", '"unterminated')


@pytest.mark.unit
def test_parse_comment_returns_none() -> None:
    assert _parse_dotenv_line("# this is a comment") is None


@pytest.mark.unit
def test_parse_blank_returns_none() -> None:
    assert _parse_dotenv_line("") is None
    assert _parse_dotenv_line("   ") is None


@pytest.mark.unit
def test_parse_missing_equals_returns_none() -> None:
    assert _parse_dotenv_line("NO_EQUALS_HERE") is None


@pytest.mark.unit
def test_parse_empty_key_returns_none() -> None:
    assert _parse_dotenv_line("=value") is None


@pytest.mark.unit
def test_load_dotenv_sets_env_and_returns_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n# comment\nBAZ='quoted value'\n\nBAD LINE\n")
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)

    parsed = load_dotenv(env_file)

    assert parsed == {"FOO": "bar", "BAZ": "quoted value"}
    import os

    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "quoted value"


@pytest.mark.unit
def test_load_dotenv_does_not_override_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=from_file\n")
    monkeypatch.setenv("FOO", "from_shell")

    load_dotenv(env_file)

    import os

    assert os.environ["FOO"] == "from_shell"


@pytest.mark.unit
def test_load_dotenv_missing_file_is_noop(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == {}
