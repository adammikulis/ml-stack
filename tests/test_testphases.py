"""The phase line and its warning thresholds."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import testphases
from testphases import Phases


def test_a_run_with_small_overhead_prints_no_warning():
    text = testphases.line(Phases(wall=12.0, collection=1.0, setup=0.5, tests=10.0, teardown=0.5))
    assert "collection 1.0 s, session setup 0.5 s, tests 10.0 s, teardown 0.5 s" in text
    assert "WARNING" not in text


def test_more_than_five_seconds_of_fixed_overhead_warns():
    assert "WARNING" in testphases.line(Phases(wall=100.0, collection=3.0, setup=3.0, tests=94.0, teardown=0.0))


def test_a_quarter_of_wall_time_warns_once_it_is_over_the_floor():
    assert testphases.too_slow(Phases(wall=12.0, collection=2.0, setup=2.0, tests=8.0, teardown=0.0))
    assert not testphases.too_slow(Phases(wall=2.0, collection=1.5, setup=0.4, tests=0.1, teardown=0.0))


def test_the_recorder_writes_phases_the_runner_reads(tmp_path):
    target = tmp_path / "phases.json"
    (tmp_path / "test_one.py").write_text("def test_one():\n    assert True\n")
    env = {**os.environ, testphases.PHASES_ENV: str(target), "PYTHONPATH": str(Path(testphases.__file__).parent)}
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "testphases", "-p", "no:cacheprovider",
                           str(tmp_path / "test_one.py")], cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    phases = testphases.read(target)
    assert phases is not None
    assert phases.wall >= phases.tests >= 0.0
