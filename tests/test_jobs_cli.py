"""``ml-stack-jobs``: what this machine has recorded, waiting on one, stopping one --
against a real sleeping child, in ``tmp_path``, never ``~/.ml-stack``."""

from __future__ import annotations

import subprocess
import sys

import pytest

from ml_stack import jobs


def test_status_lists_what_is_recorded_under_the_home_it_is_given(tmp_path, capsys):
    assert jobs.main(["status", "--home", str(tmp_path)]) == 0
    assert "no job is recorded" in capsys.readouterr().out

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        jobs.record("ingest", pid=child.pid, argv=["book.pdf", "--out", "shelf"],
                    log="ingest.log", home=tmp_path)
        assert jobs.main(["status", "--home", str(tmp_path)]) == 0
        said = capsys.readouterr().out
        assert f"ingest: running (pid {child.pid})" in said
        assert "book.pdf --out shelf" in said and "ingest.log" in said
    finally:
        child.kill()
        child.wait(timeout=10)


def test_wait_blocks_on_the_named_kind_and_says_when_it_has_ended(tmp_path, capsys):
    assert jobs.main(["wait", "bench", "--home", str(tmp_path)]) == 0
    assert "no bench job is running" in capsys.readouterr().out

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"])
    jobs.record("bench", pid=child.pid, argv=["sweep"], log="b.log", home=tmp_path)
    assert jobs.main(["wait", "bench", "--home", str(tmp_path)]) == 0
    assert f"the bench job (pid {child.pid}) has ended" in capsys.readouterr().out
    assert child.wait(timeout=10) == 0


def test_stop_ends_the_named_kind_and_clears_its_record(tmp_path, capsys):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    jobs.record("train", pid=child.pid, argv=["run"], log="t.log", home=tmp_path)
    try:
        assert jobs.main(["stop", "train", "--home", str(tmp_path)]) == 0
        child.wait(timeout=10)
        assert f"stopped the train job (pid {child.pid})" in capsys.readouterr().out
        assert not (tmp_path / "train.json").exists()
    finally:
        if child.poll() is None:                # pragma: no cover - only on a hung child
            child.kill()


def test_wait_and_stop_need_a_kind_and_an_unknown_word_is_refused(tmp_path, capsys):
    assert jobs.main(["wait", "--home", str(tmp_path)]) == 2
    assert "needs a KIND" in capsys.readouterr().err
    assert jobs.main(["stop", "--home", str(tmp_path)]) == 2
    assert "needs a KIND" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        jobs.main(["tail", "bench", "--home", str(tmp_path)])
