"""Command-line interface for robust-lid.

Entry points (see ``[project.scripts]`` in ``pyproject.toml``):

* ``rlid`` — short primary name
* ``robust-lid`` — long alias matching the package name

Examples::

    rlid "The quick brown fox"                # single text → "eng_Latn  0.987  The quick brown fox"
    echo "안녕" | rlid                        # stdin, one text per line
    rlid --file input.txt --json              # JSONL output
    rlid --models ft176,glotlid "Hello"       # backend subset
    rlid --uniform "Hello"                    # disable tuned defaults
    rlid --list-backends                      # inventory, then exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, TextIO

from .ensemble import (
    DEFAULT_LANG_WEIGHTS_BY_NAME,
    DEFAULT_SCRIPT_WEIGHTS_BY_NAME,
    DEFAULT_WEIGHTS_BY_NAME,
    RobustLID,
    default_backend_order,
)
from .models import (
    CLD2LID,
    CLD3LID,
    FASTTEXT_VERBOSE_ENV,
    FastText176LID,
    FastText218eLID,
    GlotLID,
    LangdetectLID,
    LangidLID,
    is_cld3_available,
)

if TYPE_CHECKING:
    from .models import LID

_BACKEND_FACTORIES: dict[str, type] = {
    "langid": LangidLID,
    "langdetect": LangdetectLID,
    "cld2": CLD2LID,
    "cld3": CLD3LID,
    "ft176": FastText176LID,
    "ft218e": FastText218eLID,
    "glotlid": GlotLID,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlid",
        description="Robust language identification — ensemble of 6-7 LID backends.",
        epilog=(
            "With no TEXT and no --file, reads one text per line from stdin. "
            "Output is '<lang>_<Script>\\t<confidence>\\t<text>' by default, "
            "or JSON Lines with --json."
        ),
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text(s) to classify. Omit to read from --file or stdin.",
    )
    parser.add_argument(
        "-f",
        "--file",
        metavar="PATH",
        help="Read texts from file (one text per line).",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output JSON Lines (one JSON object per line).",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Omit the input text from plain output (confidence + lang only).",
    )
    parser.add_argument(
        "--models",
        metavar="NAMES",
        help=(
            "Comma-separated backend subset from: "
            + ", ".join(_BACKEND_FACTORIES)
            + ". Tuned defaults are still applied by name."
        ),
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="Disable tuned defaults (scalar/script/lang weights all 1.0).",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help=(
            "Load backends one at a time per predict (peak RAM ≈ largest single "
            "model, ~1.2 GB) instead of keeping all ~2 GB resident. Much slower "
            "per call and disables script gating."
        ),
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable thread-pool parallel execution of backends (default: on).",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Print backend inventory (supported languages / scripts) and exit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Print per-stage progress to stderr and surface backend warnings "
            "(e.g. fastText's C++ load-time notice that is hidden by default)."
        ),
    )
    parser.add_argument("--version", action="version", version="robust-lid 0.1.0")
    return parser


def _vprint(msg: str, verbose: bool) -> None:
    """Emit a stage message to stderr when --verbose is set."""
    if verbose:
        print(f"[rlid] {msg}", file=sys.stderr, flush=True)


def _iter_inputs(args: argparse.Namespace) -> Iterator[str]:
    """Yield texts in order: positional args > --file > stdin."""
    if args.text:
        yield from args.text
        return
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if stripped:
                    yield stripped
        return
    for line in sys.stdin:
        stripped = line.rstrip("\n")
        if stripped:
            yield stripped


def _build_engine(args: argparse.Namespace) -> RobustLID:
    """Construct the engine honoring --models, --uniform, --low-memory, --no-parallel."""
    parallel = not args.no_parallel
    low_memory = args.low_memory

    if args.models:
        if low_memory:
            raise SystemExit(
                "rlid: --low-memory is incompatible with --models (it's wired to the "
                "default factory ensemble)."
            )
        names = [n.strip() for n in args.models.split(",") if n.strip()]
        unknown = [n for n in names if n not in _BACKEND_FACTORIES]
        if unknown:
            raise SystemExit(
                f"rlid: unknown backend(s): {unknown}. Valid: {sorted(_BACKEND_FACTORIES)}"
            )
        if "cld3" in names and not is_cld3_available():
            raise SystemExit(
                "rlid: cld3 requested but gcld3 is not installed. "
                "See README for install instructions."
            )
        models: list[LID] = [_BACKEND_FACTORIES[n]() for n in names]
        if args.uniform:
            return RobustLID(models=models, parallel=parallel)
        return RobustLID(
            models=models,
            weights=[DEFAULT_WEIGHTS_BY_NAME.get(n, 1.0) for n in names],
            script_weights=[dict(DEFAULT_SCRIPT_WEIGHTS_BY_NAME.get(n, {})) for n in names],
            lang_weights=[dict(DEFAULT_LANG_WEIGHTS_BY_NAME.get(n, {})) for n in names],
            parallel=parallel,
        )

    if args.uniform:
        n = len(default_backend_order())
        return RobustLID(
            weights=[1.0] * n,
            script_weights=[{}] * n,
            lang_weights=[{}] * n,
            parallel=parallel,
            low_memory=low_memory,
        )

    return RobustLID(parallel=parallel, low_memory=low_memory)


def _print_backend_inventory(out: TextIO | None = None) -> None:
    """Instantiate all available backends and print their supported sets."""
    stream: TextIO = out if out is not None else sys.stdout
    names = default_backend_order()
    rows: list[tuple[str, int, int]] = []
    for name in names:
        backend = _BACKEND_FACTORIES[name]()
        rows.append((name, len(backend.supported_langs), len(backend.supported_scripts)))
    width = max(len(n) for n in names)
    print(f"{'BACKEND':<{width}}  LANGS  SCRIPTS", file=stream)
    for name, nl, ns in rows:
        print(f"{name:<{width}}  {nl:>5}  {ns:>7}", file=stream)


def _format_plain(text: str, lang: str, confidence: float, show_text: bool) -> str:
    if show_text:
        return f"{lang}\t{confidence:.3f}\t{text}"
    return f"{lang}\t{confidence:.3f}"


def _format_json(text: str, lang: str, confidence: float) -> str:
    return json.dumps(
        {"text": text, "lang": lang, "confidence": round(confidence, 6)},
        ensure_ascii=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        # Re-enable fastText's native load-time warning so users can debug.
        os.environ[FASTTEXT_VERBOSE_ENV] = "1"

    if args.list_backends:
        _vprint("listing backends", args.verbose)
        _print_backend_inventory()
        return 0

    # No text, no file, and stdin is a TTY → user likely wants help.
    if not args.text and not args.file and sys.stdin.isatty():
        parser.print_help()
        return 2

    _vprint("building ensemble (downloads ~1.5 GB on first run)", args.verbose)
    t0 = time.monotonic()
    engine = _build_engine(args)
    _vprint(f"ensemble ready in {time.monotonic() - t0:.1f}s", args.verbose)

    count = 0
    _vprint("predicting", args.verbose)
    t1 = time.monotonic()
    for text in _iter_inputs(args):
        lang, confidence = engine.predict(text)
        count += 1
        if args.json:
            print(_format_json(text, lang, confidence))
        else:
            print(_format_plain(text, lang, confidence, show_text=not args.no_text))
    _vprint(f"done — {count} text(s) in {time.monotonic() - t1:.2f}s", args.verbose)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via entry points
    sys.exit(main())
