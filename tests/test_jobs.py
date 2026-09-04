"""A record for a long command, shared by anything that detaches: record, alive, wait,
stop, status -- against a real sleeping child, in ``tmp_path``, never ``~/.ml-stack``."""

from __future__ import annotations

import json
import os
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
