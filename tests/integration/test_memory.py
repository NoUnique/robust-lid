"""Memory-usage benchmark for the two execution modes.

Each mode is measured in its own subprocess so the RSS high-water mark is
clearly attributable and doesn't leak across measurements (once Python's
fastText model is loaded, CPython rarely returns that memory to the OS).

Linux-only (depends on ``/proc/self/status``). Runs under the ``slow`` +
``network`` marks because it instantiates real fastText models on first
use (~1.5 GB download).

Run with:  uv run pytest -m "slow and network" tests/integration/test_memory.py -s
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path("/proc/self/status").exists(),
    reason="Memory test reads /proc/self/status (Linux only).",
)


def _peak_rss_of_mode(mode: str) -> tuple[float, float]:
    """Spawn a child that loads RobustLID in ``mode`` and reports
    ``(current_rss_mb, peak_rss_mb)``. ``VmHWM`` is the process
    high-water mark — the actual peak we want to bound."""
    script = textwrap.dedent(
        f"""
        import gc
        from robust_lid import RobustLID

        def proc_status() -> tuple[float, float]:
            current = peak = 0.0
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        current = int(line.split()[1]) / 1024
                    elif line.startswith("VmHWM:"):
                        peak = int(line.split()[1]) / 1024
            return current, peak

        mode = "{mode}"
        if mode == "fast":
            lid = RobustLID()
        elif mode == "sequential":
            lid = RobustLID(parallel=False)
        elif mode == "low_memory":
            lid = RobustLID(low_memory=True)
        else:
            raise ValueError(mode)

        # One predict forces the full per-mode cost (load in low_memory).
        lid.predict("The quick brown fox jumps over the lazy dog.")
        gc.collect()
        current, peak = proc_status()
        print(f"{{current:.1f}} {{peak:.1f}}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = result.stdout.strip().splitlines()[-1].split()
    return float(parts[0]), float(parts[1])


@pytest.mark.slow
@pytest.mark.network
def test_low_memory_reduces_peak_rss() -> None:
    """Sanity: low_memory mode must have meaningfully lower peak RSS than the
    default eager mode. Measures VmHWM (high-water mark) — the actual peak
    the process ever reached, not the instantaneous post-predict RSS."""
    _fast_cur, fast_peak = _peak_rss_of_mode("fast")
    lm_cur, lm_peak = _peak_rss_of_mode("low_memory")
    print(
        f"\nPeak RSS — fast: {fast_peak:.0f} MB | "
        f"low_memory peak: {lm_peak:.0f} MB (steady-state {lm_cur:.0f} MB)"
    )
    # Eager ensemble keeps every fastText model resident; low_memory peaks
    # at the largest single model (ft218e ~ 1.2 GB) and drops between calls.
    assert lm_peak < fast_peak - 300, (
        f"low_memory peak ({lm_peak:.0f} MB) must be at least 300 MB below "
        f"fast peak ({fast_peak:.0f} MB) to justify the mode"
    )
    assert lm_cur < lm_peak, (
        f"low_memory steady-state RSS ({lm_cur:.0f} MB) should drop below its "
        f"peak ({lm_peak:.0f} MB) after the per-call model is released"
    )


@pytest.mark.slow
@pytest.mark.network
def test_fast_mode_rss_is_in_expected_range() -> None:
    """Absolute sanity-check — the default mode should fit within ~3.5 GB.
    If this fails we've probably regressed memory somewhere."""
    _cur, peak = _peak_rss_of_mode("fast")
    print(f"\nPeak RSS — fast: {peak:.0f} MB")
    assert 500 < peak < 3500, f"fast-mode peak RSS {peak:.0f} MB out of plausible range"
