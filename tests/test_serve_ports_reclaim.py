"""A port is reclaimed from a server this manager recorded, never from one another
process started -- the rule that would have kept a twelve-hour ingest alive."""

import subprocess
import sys
import time

from ml_stack.serve import ports


def _sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_a_recorded_stale_server_is_reclaimed():
    child = _sleeper()
    try:
        assert ports.reclaim_port(1, recorded_pids=[child.pid]) is True
        child.wait(timeout=10)
        assert child.poll() is not None, "ended: it was ours"
    finally:
        if child.poll() is None:
            child.kill()


def test_a_server_another_process_started_is_left_alone(monkeypatch):
    child = _sleeper()
    monkeypatch.setattr(ports, "server_pids_on_port", lambda port: [child.pid])
    try:
        assert ports.reclaim_port(1) is False
        time.sleep(0.2)
        assert child.poll() is None, "still running: it was never ours to stop"
    finally:
        child.kill()
