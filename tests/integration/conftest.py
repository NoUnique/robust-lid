"""Shared setup for integration tests.

Loads the project-root `.env` (if present) so E2E tests can pick up
secrets like `HF_TOKEN` without shipping a dotenv dependency. Uses only
the standard library — the parser here is deliberately minimal (see
`_parse_dotenv_line`).

`.env` lives at the project root by convention: tools like
`python-dotenv`, `pytest-env`, and Docker all search from the project
root upward, so putting it anywhere else would break that ecosystem.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip()
    # Strip matching surrounding quotes (single or double).
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a dotenv file and merge into `os.environ` without overwriting
    values already set in the process. Returns the parsed mapping."""
    if not path.exists():
        return {}
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        item = _parse_dotenv_line(raw_line)
        if item is None:
            continue
        key, value = item
        parsed[key] = value
        # Existing env vars win so CI / shell exports aren't clobbered.
        os.environ.setdefault(key, value)
    return parsed


# Load on collection, before any fixture runs.
load_dotenv(PROJECT_ROOT / ".env")
