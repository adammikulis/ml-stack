"""A training run is hours, so it detaches and is recorded like every other long command.

Against a real sleeping child in ``tmp_path``, never a real fine-tune and never a real
``~/.ml-stack``: what is under test is the record and the refusal, not the weights.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from ml_stack import jobs
from ml_stack.train import run as train_run


def _sleeper(seconds: float = 60.0) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def _argv(tmp_path: Path) -> list[str]:
    return ["--recipe", "tool-calls", "--data", str(tmp_path / "data"),
            "--out", str(tmp_path / "out"), "--detach"]


def test_detach_records_the_run_as_this_machines_train_job(tmp_path, monkeypatch, capsys):
    """`detach` writes the pid, the argv and the log through `ml_stack.jobs` under the kind
    `train`, and that record is what `wait`, `stop` and `ml-stack-jobs status` read."""
    monkeypatch.setattr(train_run, "HOME", tmp_path)
    started = {}
    popen = subprocess.Popen

    def sleeping(command, **kw):
        started["command"] = list(command)
        started["child"] = held = popen([sys.executable, "-c", "import time; time.sleep(60)"])
        return held

    monkeypatch.setattr(subprocess, "Popen", sleeping)
    argv = _argv(tmp_path)
    try:
        assert train_run.main(argv) == 0
        assert started["command"][1:3] == ["-m", "ml_stack.train.run"], "the module, not a shell"
        assert "--detach" not in started["command"], "the child does not detach again"
        held = jobs.held("train", home=tmp_path / "jobs")
        assert held["pid"] == started["child"].pid
        assert held["argv"] == [a for a in argv if a != "--detach"]
        assert held["log"].endswith(".log") and Path(held["log"]).is_file()
        assert "argv:" in Path(held["log"]).read_text(), "the log says what it is"
        assert jobs.alive("train", home=tmp_path / "jobs") == started["child"].pid
        assert "detached; the log is" in capsys.readouterr().out
    finally:
        started["child"].kill()
        started["child"].wait(timeout=10)


def test_a_second_detach_is_refused_while_one_is_still_training(tmp_path, monkeypatch,
                                                                capsys):
    """Two fine-tunes on one machine share one GPU: both are slower and neither
    measurement is believable, so the second is refused rather than queued."""
    monkeypatch.setattr(train_run, "HOME", tmp_path)
    child = _sleeper()
    try:
        jobs.record("train", pid=child.pid, argv=[], home=tmp_path / "jobs")
        assert train_run.main(_argv(tmp_path)) == 2
        said = capsys.readouterr().err
        assert f"a training run (pid {child.pid}) is still going" in said
        assert "wait" in said and "stop" in said
        held = jobs.held("train", home=tmp_path / "jobs")
        assert held["pid"] == child.pid, "the refused run never overwrote the record"
    finally:
        child.kill()
        child.wait(timeout=10)


def test_status_names_the_recorded_training_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_run, "HOME", tmp_path)
    assert train_run.main(["status"]) == 0
    assert "no job is recorded" in capsys.readouterr().out

    child = _sleeper()
    try:
        jobs.record("train", pid=child.pid, argv=["--recipe", "tool-calls"],
                    log="t.log", home=tmp_path / "jobs")
        assert train_run.main(["status"]) == 0
        said = capsys.readouterr().out
        assert f"train: running (pid {child.pid})" in said
        assert "--recipe tool-calls" in said and "t.log" in said
    finally:
        child.kill()
        child.wait(timeout=10)


def test_wait_blocks_until_the_recorded_run_has_ended(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_run, "HOME", tmp_path)
    assert train_run.main(["wait"]) == 0
    assert "no train job is running" in capsys.readouterr().out

    child = _sleeper(2.0)
    jobs.record("train", pid=child.pid, argv=[], home=tmp_path / "jobs")
    done: list[int] = []
    thread = threading.Thread(
        target=lambda: done.append(train_run.wait(home=tmp_path, every=1.0)))
    thread.start()
    thread.join(timeout=30)
    child.wait(timeout=10)
    assert done == [0] and not thread.is_alive()
    assert f"the train job (pid {child.pid}) has ended" in capsys.readouterr().out


def test_stop_ends_the_recorded_run_and_clears_the_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_run, "HOME", tmp_path)
    assert train_run.main(["stop"]) == 1
    assert "no train job is recorded" in capsys.readouterr().out

    child = _sleeper()
    try:
        jobs.record("train", pid=child.pid, argv=[], home=tmp_path / "jobs")
        assert train_run.main(["stop"]) == 0
        child.wait(timeout=10)
        assert f"stopped the train job (pid {child.pid})" in capsys.readouterr().out
        assert not (tmp_path / "jobs" / "train.json").exists()
    finally:
        if child.poll() is None:
            child.kill()
