"""`runtime status` names the processes that began before the selected runtime and still run older code.

The processes are real children of the test, found by the real process scan; only the verification
time is chosen, as the selected runtime's mark would record it.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from poolhouse import runtime_stale


@pytest.fixture
def daemon(tmp_path: Path):
    """A long-running child whose command line names a poolhouse program, and the runtime prefix it was not started from."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "poolhouse-fakedaemon"])
    try:
        deadline = time.monotonic() + 10
        while not any(p.pid == child.pid for p in runtime_stale.running()) and time.monotonic() < deadline:
            time.sleep(0.05)
        yield child, tmp_path / "runtimes" / "selected"
    finally:
        child.kill()
        child.wait()


def test_a_daemon_started_before_the_selected_runtime_is_named(daemon) -> None:
    child, prefix = daemon
    lines = runtime_stale.stale_lines(prefix, time.time() + 600)
    assert any(f"pid {child.pid} " in line and "poolhouse-fakedaemon" in line for line in lines)
    assert lines[-1].startswith("note")


def test_a_daemon_started_after_the_selected_runtime_is_not_named(daemon) -> None:
    child, prefix = daemon
    lines = runtime_stale.stale_lines(prefix, time.time() - 600)
    assert not any(f"pid {child.pid} " in line for line in lines)


def test_a_process_running_from_the_selected_runtime_is_not_named(daemon) -> None:
    child, _ = daemon
    prefix = Path(sys.executable).parent.parent
    old = runtime_stale.older_than(runtime_stale.running(), prefix, time.time() + 600)
    assert child.pid not in [p.pid for p in old]


def test_a_runtime_that_was_never_verified_names_nothing(daemon) -> None:
    _, prefix = daemon
    assert runtime_stale.stale_lines(prefix, 0.0) == []
