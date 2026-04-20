"""Tests for the OS-level stderr suppressor and the quiet fastText loader.

These cover the fix for the fastText C++ load-time warning that was leaking
through Python-level redirects and cluttering CLI output.
"""

from __future__ import annotations

import os

import fasttext
import pytest

from robust_lid.models import (
    FASTTEXT_VERBOSE_ENV,
    _quiet_fasttext_load,
    _suppress_native_stderr,
)


@pytest.mark.unit
def test_suppress_native_stderr_hides_os_level_writes(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Writing directly to fd 2 inside the context must be swallowed."""
    os.write(2, b"visible_before\n")
    with _suppress_native_stderr():
        os.write(2, b"hidden_inside\n")
    os.write(2, b"visible_after\n")
    err = capfd.readouterr().err
    assert "visible_before" in err
    assert "hidden_inside" not in err
    assert "visible_after" in err


@pytest.mark.unit
def test_suppress_native_stderr_restores_on_exception() -> None:
    """Even if the body raises, fd 2 must be restored."""
    before = os.dup(2)
    try:
        with pytest.raises(RuntimeError, match="boom"), _suppress_native_stderr():
            raise RuntimeError("boom")
        after = os.dup(2)
        try:
            # dup'd fd refers to the same target as before
            assert os.fstat(after).st_ino == os.fstat(before).st_ino
        finally:
            os.close(after)
    finally:
        os.close(before)


@pytest.mark.unit
def test_quiet_fasttext_load_suppresses_by_default(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    def _fake_load(_path: str) -> str:
        os.write(2, b"fasttext-native-warning\n")
        return "model-handle"

    monkeypatch.setattr(fasttext, "load_model", _fake_load)
    monkeypatch.delenv(FASTTEXT_VERBOSE_ENV, raising=False)

    result = _quiet_fasttext_load("/unused/path")
    assert result == "model-handle"
    err = capfd.readouterr().err
    assert "fasttext-native-warning" not in err


@pytest.mark.unit
def test_quiet_fasttext_load_passthrough_when_env_set(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    def _fake_load(_path: str) -> str:
        os.write(2, b"fasttext-native-warning\n")
        return "model-handle"

    monkeypatch.setattr(fasttext, "load_model", _fake_load)
    monkeypatch.setenv(FASTTEXT_VERBOSE_ENV, "1")

    _quiet_fasttext_load("/unused/path")
    err = capfd.readouterr().err
    assert "fasttext-native-warning" in err
