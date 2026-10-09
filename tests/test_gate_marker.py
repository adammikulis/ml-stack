"""Whole-tree scans carry the `gate` marker: `scripts/test gate` runs them, full and fast do not."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCAN = "tests/test_no_data_files.py::test_the_tracked_tree_passes"


def collected(*options: str) -> str:
    done = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                           "-n", "0", *options, SCAN], cwd=REPO, capture_output=True, text=True, timeout=120)
    return done.stdout


def test_a_whole_tree_scan_is_left_out_of_the_default_tiers():
    assert SCAN not in collected()


@pytest.mark.parametrize("option", ["--gate", "--slow"])
def test_the_gate_and_the_everything_tier_collect_it(option):
    assert SCAN in collected(option)


def test_the_gate_step_asks_for_the_scans():
    runner = (REPO / "scripts" / "test").read_text(encoding="utf-8")
    steps = runner[runner.index("def gate_steps"):runner.index("STALE_HINTS")]
    assert '"--gate"' in steps and "tests/test_no_data_files.py" in steps and "tests/test_affected_scripts.py" in steps
