"""scripts/detach starts a command in a new session with its output in a file, and returns at once."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

DETACH = Path(__file__).resolve().parents[1] / "scripts" / "detach"


def test_a_detached_command_runs_in_its_own_session_and_the_caller_returns_at_once(tmp_path):
    log = tmp_path / "out.log"
    code = "import os,time; print(os.getpid(), os.getsid(0), os.getsid(os.getppid()), flush=True); time.sleep(1.5)"
    started = time.monotonic()
    done = subprocess.run([sys.executable, str(DETACH), str(log), sys.executable, "-c", code], timeout=10, check=False)
    assert done.returncode == 0 and time.monotonic() - started < 1.0          # did not wait for the 1.5 s command
    for _ in range(50):
        if log.exists() and log.read_text().strip():
            break
        time.sleep(0.1)
    pid, session, _ = (int(x) for x in log.read_text().split())
    assert session == pid or session != os.getsid(0)                           # a session of its own, not ours
    assert (log.stat().st_mode & 0o777) == 0o600


def test_it_refuses_a_call_without_a_command(tmp_path):
    done = subprocess.run([sys.executable, str(DETACH), str(tmp_path / "x.log")], capture_output=True, text=True, check=False)
    assert done.returncode == 2
