"""A record for a long command, shared by anything that detaches: record, alive, wait,
stop, status -- against a real sleeping child, in ``tmp_path``, never ``~/.ml-stack``."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from ml_stack import jobs
from ml_stack.lock import Busy


def _until(ready, what, seconds=30.0):
    deadline = time.time() + seconds
    while not ready():
        assert time.time() < deadline, f"timed out waiting until {what}"
        time.sleep(0.02)


def test_record_and_alive_track_a_real_child(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        job = jobs.record("bench", pid=child.pid, argv=["sweep", "--serve", "tiny"],
                          log="bench.log", home=tmp_path)
        assert job.kind == "bench" and job.pid == child.pid and job.home == tmp_path
        held = json.loads((tmp_path / "bench.json").read_text())
        assert held["pid"] == child.pid
        assert held["argv"] == ["sweep", "--serve", "tiny"]
        assert held["log"] == "bench.log" and held["started"]

        assert jobs.alive("bench", home=tmp_path) == child.pid
        assert jobs.alive("ingest", home=tmp_path) == 0, "a kind never recorded is never alive"
    finally:
        child.kill()
        child.wait(timeout=10)
    assert jobs.alive("bench", home=tmp_path) == 0, "the pid ended, whatever the file says"


def test_a_second_record_is_refused_while_the_first_is_alive(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        jobs.record("bench", pid=child.pid, argv=["sweep"], log="a.log", home=tmp_path)
        with pytest.raises(Busy, match=str(child.pid)):
            jobs.record("bench", pid=os.getpid(), argv=["sweep"], log="b.log", home=tmp_path)
        held = json.loads((tmp_path / "bench.json").read_text())
        assert held["pid"] == child.pid, "the refused record never overwrote the first"

        # a caller whose own concurrency is handled elsewhere opts out
        job = jobs.record("bench", pid=os.getpid(), argv=["sweep", "2"], log="b.log",
                          home=tmp_path, refuse_if_alive=False)
        assert job.pid == os.getpid()
    finally:
        child.kill()
        child.wait(timeout=10)


def test_wait_blocks_until_the_recorded_pid_ends_and_says_so(tmp_path, capsys):
    assert jobs.wait("train", home=tmp_path) == 0
    assert "no train job is running" in capsys.readouterr().out

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    jobs.record("train", pid=child.pid, argv=["run"], log="t.log", home=tmp_path)
    assert jobs.wait("train", home=tmp_path, every=0.05) == 0
    said = capsys.readouterr().out
    assert f"the train job (pid {child.pid}) has ended" in said
    assert child.wait(timeout=10) == 0


def test_stop_signals_the_recorded_pid_and_clears_the_record(tmp_path, capsys):
    assert jobs.stop("bench", home=tmp_path) == 1
    assert "no bench job is recorded" in capsys.readouterr().out

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    jobs.record("bench", pid=child.pid, argv=["sweep"], log="s.log", home=tmp_path)
    try:
        assert jobs.stop("bench", home=tmp_path) == 0
        child.wait(timeout=10)          # stop waits for it, and reaps it if it is a child
        said = capsys.readouterr().out
        assert f"stopped the bench job (pid {child.pid})" in said
        assert not (tmp_path / "bench.json").exists()
    finally:
        if child.poll() is None:
            child.kill()


def test_stop_says_so_when_the_pid_has_already_gone(tmp_path, capsys):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=10)
    jobs.record("bench", pid=child.pid, argv=[], log="", home=tmp_path)
    assert jobs.stop("bench", home=tmp_path) == 1
    assert "had already ended" in capsys.readouterr().out
    assert not (tmp_path / "bench.json").exists()


def test_stop_says_still_ending_when_the_child_holds_out(tmp_path, capsys):
    """A child that ignores SIGTERM for a while is reported as still ending, and its record
    stays -- so a caller checking `alive` first never starts a second job beside it."""
    deaf = ("import pathlib, signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "pathlib.Path(sys.argv[1]).write_text('ready')\n"
            "time.sleep(60)\n")
    ready = tmp_path / "ready"
    child = subprocess.Popen([sys.executable, "-c", deaf, str(ready)])
    jobs.record("ingest", pid=child.pid, argv=[], log="", home=tmp_path)
    try:
        _until(ready.is_file, "the child is ignoring SIGTERM")
        assert jobs.stop("ingest", home=tmp_path, wait=1.0) == 1
        said = capsys.readouterr().out
        assert "had not ended after 1s" in said
        assert (tmp_path / "ingest.json").exists(), "the record stays while it is still ending"
        assert jobs.alive("ingest", home=tmp_path) == child.pid
    finally:
        child.kill()
        child.wait(timeout=10)


def test_status_lists_every_kind_recorded_under_home(tmp_path, capsys):
    assert jobs.status(home=tmp_path) == 0
    assert "no job is recorded" in capsys.readouterr().out

    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        ended.wait(timeout=10)
        jobs.record("bench", pid=live.pid, argv=["sweep", "--serve", "tiny"],
                    log="bench.log", home=tmp_path)
        jobs.record("ingest", pid=ended.pid, argv=["notes.pdf"], log="ingest.log",
                    home=tmp_path, refuse_if_alive=False)
        assert jobs.status(home=tmp_path) == 0
        said = capsys.readouterr().out
        assert f"bench: running (pid {live.pid})" in said
        assert f"ingest: ended (pid {ended.pid})" in said
        assert "sweep --serve tiny" in said and "bench.log" in said
    finally:
        live.kill()
        live.wait(timeout=10)


@pytest.mark.slow
def test_detach_starts_a_real_child_in_its_own_session_and_stop_ends_it(
        tmp_path, capsys, monkeypatch):
    """The one launcher, driven: a module run detached writes its header, keeps running
    when this process would not, is recorded for `status`, and `stop` ends it."""
    module = tmp_path / "sleeper" / "__init__.py"
    module.parent.mkdir()
    (module.parent / "__main__.py").write_text(
        "import sys, time\nprint('awake', flush=True)\ntime.sleep(120)\n")
    module.write_text("")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    log = tmp_path / "logs" / "sleeper.log"
    ran = jobs.detach("sleeper", ["--quietly"], log=log, lines=["commit: 0f1e2d3"],
                      kind="sleeper", home=tmp_path)
    try:
        assert ran.pid > 0 and ran.log == log
        assert ran.command[1:] == ("-m", "sleeper", "--quietly")
        header = log.read_text().splitlines()
        assert header[:3] == ["argv: --quietly", f"started: {ran.started}", "commit: 0f1e2d3"]
        assert os.getsid(ran.pid) == ran.pid, "a session of its own, owned by no terminal"
        assert jobs.alive("sleeper", home=tmp_path) == ran.pid

        _until(lambda: "awake" in log.read_text(), "the child never wrote to the log")
        assert jobs.status(home=tmp_path) == 0
        assert f"sleeper: running (pid {ran.pid})" in capsys.readouterr().out

        assert jobs.stop("sleeper", home=tmp_path, wait=10.0) == 0
        assert jobs.alive("sleeper", home=tmp_path) == 0
        assert not (tmp_path / "sleeper.json").exists()
    finally:
        with contextlib.suppress(OSError):
            os.kill(ran.pid, signal.SIGKILL)


def test_detach_without_a_kind_writes_no_record(tmp_path, monkeypatch):
    """`mcp.detached` and the fleet daemon keep records of their own; `status` is not theirs."""
    seen: dict = {}

    class Child:
        pid = 4242

        def __init__(self, command, **kw):
            seen["command"], seen["kw"] = list(command), kw

    monkeypatch.setattr(subprocess, "Popen", Child)
    ran = jobs.detach("ml_stack.hub", ["fetch", "hf:pellard/larch/larch.gguf"],
                      log=tmp_path / "logs" / "fetch.log")

    assert ran.pid == 4242 and seen["command"][1:3] == ["-m", "ml_stack.hub"]
    assert seen["kw"]["stdin"] is subprocess.DEVNULL
    assert seen["kw"]["stderr"] is subprocess.STDOUT
    assert seen["kw"]["env"]["PYTHONUNBUFFERED"] == "1"
    assert seen["kw"].get("start_new_session") or "creationflags" in seen["kw"]
    assert list(tmp_path.glob("*.json")) == [], "no record without a kind"
    assert jobs.status(home=tmp_path) == 0
