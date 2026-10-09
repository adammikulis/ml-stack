"""Many processes scanning one machine's pins never put more than one dialog on the screen,
and a pinned file that is simply gone raises none. The desktop tools are shims that record
when they start and end, so a stack of dialogs shows as overlapping records."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from poolhouse import home, sentinel

SRC = str(Path(__file__).resolve().parents[1] / "src")
CHILD = (
    "import threading\n"
    "from poolhouse import sentinel\n"
    "node = sentinel.default()\n"
    "for _ in range(SCANS):\n"
    "    node.scan(deep=True)\n"
    "for t in threading.enumerate():\n"
    "    if t.name == 'sentinel-heads-up':\n"
    "        t.join()\n")

@pytest.fixture
def desk(tmp_path):
    """Shims first on PATH that log ``start`` and ``end`` around a one second hold and answer
    with the first button; returns a function giving the child environment and the log."""
    if os.name == "nt":
        pytest.skip("the desktop shims are POSIX scripts")
    shims = tmp_path / "desk-shims"
    shims.mkdir()
    log = tmp_path / "desk.log"
    body = ('#!/bin/sh\necho "start $$ $*" >> "$DESK_LOG"\nsleep "${DESK_HOLD:-1}"\n'
            'echo "end $$" >> "$DESK_LOG"\n%s\n')
    for name, answer in (("osascript", "echo 'button returned:Later, gave up:false'"),
                         ("notify-send", "echo 0")):
        (shims / name).write_text(body % answer)
        (shims / name).chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE", "POOLHOUSE_AGENT", "POOLHOUSE_NONINTERACTIVE")}
    env.update(PATH=f"{shims}{os.pathsep}{env['PATH']}", DESK_LOG=str(log), POOLHOUSE_NOTIFY="system",
               PYTHONPATH=os.pathsep.join([SRC, str(Path(__file__).parent)]),
               POOLHOUSE_HOME=os.environ["POOLHOUSE_HOME"],
               PYTHON_KEYRING_BACKEND="onboard_support.FileKeyring",
               POOLHOUSE_TEST_KEYRING=str(tmp_path / "keyring.json"))
    return env, log


def lines(log: Path, word: str) -> list[str]:
    return [x for x in (log.read_text().splitlines() if log.exists() else [])
            if x.startswith(word)]


def peak(log: Path) -> int:
    live = top = 0
    for line in log.read_text().splitlines() if log.exists() else []:
        live += 1 if line.startswith("start") else -1
        top = max(top, live)
    return top


def scan_in_processes(env, processes: int, scans: int) -> None:
    code = CHILD.replace("SCANS", str(scans))
    running = [subprocess.Popen([sys.executable, "-c", code], env=env) for _ in range(processes)]
    assert [p.wait(timeout=120) for p in running] == [0] * processes


def pinned(count: int, kind: str = "model") -> list[Path]:
    node = sentinel.default()
    folder = home.home() / "runs" / "gemma-v1"
    folder.mkdir(parents=True)
    files = []
    for n in range(count):
        path = folder / f"part{n}.safetensors"
        path.write_bytes(b"weights %d" % n)
        node.manifest.pin(path, kind, source="trained:test")
        files.append(path)
    return files


def test_pinned_files_that_are_gone_raise_no_dialog_and_no_quarantine(desk):
    env, log = desk
    for path in pinned(4):
        path.unlink()
    scan_in_processes(env, processes=4, scans=3)
    node = sentinel.default()
    assert lines(log, "start") == []
    assert node.store.records() == []
    assert not [e for e in node.bus.log.read() if e.kind == "quarantine.move_refused"]
    assert node.manifest.pins() == {}


def test_four_changed_files_scanned_by_many_processes_are_one_dialog(desk):
    env, log = desk
    for path in pinned(4):
        path.write_bytes(b"tampered")
    scan_in_processes(env, processes=6, scans=3)
    assert len(lines(log, "start")) == 1
    assert peak(log) == 1
    assert len(sentinel.default().store.records(state=sentinel.State.QUARANTINED)) == 4


REVIEW = "import sys\nfrom poolhouse.sentinel.cli import command\nsys.exit(command(['review']))\n"


def review_without_a_terminal(env, **extra) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", REVIEW], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, env={**env, **extra}, timeout=120,
                          check=False)


def test_review_with_no_terminal_is_one_dialog_and_never_a_terminal_or_editor(desk):
    env, log = desk
    held = sentinel.default().store.quarantine(("peer", "10.0.0.7"), "peer.forged_traffic: x", {})
    done = review_without_a_terminal(env)
    assert done.returncode == 0
    started = lines(log, "start")
    text = log.read_text()
    assert len(started) == 1 and "10.0.0.7" in text and "Keep held" in text
    assert not any(word in text for word in ("Terminal", ".command", "TextEdit"))
    assert sentinel.default().store.get(held.id).state == sentinel.State.QUARANTINED


def test_review_with_no_terminal_started_by_an_agent_prints_the_list_and_shows_nothing(desk):
    env, log = desk
    sentinel.default().store.quarantine(("peer", "10.0.0.7"), "peer.forged_traffic: x", {})
    done = review_without_a_terminal(env, CLAUDECODE="1")
    assert done.returncode == 0 and "peer 10.0.0.7" in done.stdout
    assert lines(log, "start") == []


def test_notify_off_stops_every_dialog_of_every_process(desk):
    env, log = desk
    env = {**env, "POOLHOUSE_NOTIFY": "off"}
    for path in pinned(3):
        path.write_bytes(b"tampered")
    scan_in_processes(env, processes=3, scans=2)
    review_without_a_terminal(env)
    assert not log.exists()
    assert len(sentinel.default().store.records(state=sentinel.State.QUARANTINED)) == 3
