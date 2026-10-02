"""A process killed in the middle of writing leaves the state and the log readable."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml_stack.sentinel.events import Bus, Event, EventLog, Severity
from ml_stack.sentinel.store import Store

SRC = str(Path(__file__).resolve().parent.parent / "src")

STATE_WRITER = (
    "import sys, itertools\n"
    "from ml_stack.sentinel.store import Store\n"
    "from ml_stack.sentinel.events import Bus\n"
    "s = Store(sys.argv[1], Bus())\n"
    "for i in itertools.count():\n"
    "    s.quarantine(('peer', f'{sys.argv[2]}-{i}'), 'x', None)\n")

LOG_WRITER = (
    "import sys, itertools\n"
    "from ml_stack.sentinel.events import Event, EventLog, Severity\n"
    "log = EventLog(__import__('pathlib').Path(sys.argv[1]) / 'events.log', max_bytes=4000)\n"
    "for i in itertools.count():\n"
    "    log.append(Event('t.x', Severity.INFO, 't', f'peer:{i}', {'n': i}))\n")


def _kill_after(code: str, args: list[str], seconds: float) -> None:
    proc = subprocess.Popen([sys.executable, "-c", code, *args],
                            env={**os.environ, "PYTHONPATH": SRC})
    time.sleep(seconds)
    proc.send_signal(signal.SIGKILL)
    proc.wait()


def test_a_record_written_before_the_head_landed_is_adopted_not_forked(tmp_path):
    log = EventLog(tmp_path / "events.log")
    for i in range(3):
        log.append(Event("t.x", Severity.INFO, "t", f"peer:{i}"))
    head = tmp_path / "events.log.head"
    before = head.read_bytes()
    log.append(Event("t.x", Severity.INFO, "t", "peer:3"))
    head.write_bytes(before)
    assert log.verify().ok
    log.append(Event("t.x", Severity.INFO, "t", "peer:4"))
    result = log.verify()
    assert result.ok and result.records == 5, result.problems


def test_a_line_appended_without_the_key_is_found(tmp_path):
    import hashlib
    import json

    log = EventLog(tmp_path / "events.log")
    for i in range(3):
        log.append(Event("t.x", Severity.INFO, "t", f"peer:{i}"))
    last = json.loads(log.path.read_text().splitlines()[-1])
    forged = {"version": 1, "ts": 1.0, "kind": "t.forged", "severity": "info", "source": "t",
              "subject": "", "evidence": {}, "seq": last["seq"] + 1, "prev": last["hash"]}
    body = json.dumps(forged, sort_keys=True, separators=(",", ":"))
    forged["hash"] = hashlib.sha256((last["hash"] + body).encode()).hexdigest()
    with log.path.open("a") as handle:
        handle.write(json.dumps(forged, sort_keys=True) + "\n")
    result = log.verify()
    assert not result.ok and any("edited" in p for p in result.problems)


@pytest.mark.slow
def test_a_killed_writer_never_leaves_a_bad_state(tmp_path):
    seen = 0
    for round_ in range(12):
        _kill_after(STATE_WRITER, [str(tmp_path / "s"), str(round_)], 0.4 + 0.07 * round_)
        after = Store(tmp_path / "s", Bus())
        assert not after.tampered, f"round {round_}"
        count = len(after.records())
        assert count >= seen
        seen = count
    assert seen > 0


@pytest.mark.slow
def test_a_killed_log_writer_leaves_a_chain_that_still_reads(tmp_path):
    for round_ in range(8):
        _kill_after(LOG_WRITER, [str(tmp_path)], 0.4 + 0.05 * round_)
        log = EventLog(tmp_path / "events.log", max_bytes=4000)
        result = log.verify()
        assert result.records > 0
        assert result.ok, f"round {round_}: {result.problems}"
