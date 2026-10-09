"""Writes under the real state root: failed when this session owns them, warned when others might."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import pytest
from conftest import changed_files, file_mtimes
from state_attribution import Watch, _is_poolhouse, live_poolhouse_processes

OTHER = [(4242, "poolhouse-bench prepare")]
WRITE = "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text('escaped')"


def watch(root, rows):
    return Watch(root, file_mtimes, changed_files, lambda: rows)


def escape(root, name="bench/graph.ladybug"):
    """A child that ignores the moved HOME and writes into the 'real' root, as a buggy test would."""
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-c", WRITE, str(root / name)], check=True,
                   env={"HOME": str(root / "elsewhere")})


def test_a_write_by_the_session_on_an_idle_machine_fails(tmp_path):
    seen = watch(tmp_path, [])
    escape(tmp_path)
    failed, possible, _ = seen.verdict()
    assert failed == ["bench/graph.ladybug"] and possible == []
    assert "was written during the run" in seen.settle()


def test_the_same_write_with_another_live_writer_is_a_warning(tmp_path):
    seen = watch(tmp_path, OTHER)
    escape(tmp_path)
    with pytest.warns(UserWarning, match=r"4242 poolhouse-bench prepare.*bench/graph\.ladybug"):
        assert seen.settle() == ""


def test_a_writer_seen_only_at_the_end_still_makes_it_a_warning(tmp_path):
    rows: list = []
    seen = Watch(tmp_path, file_mtimes, changed_files, lambda: rows)
    escape(tmp_path)
    rows.extend(OTHER)
    failed, possible, _ = seen.verdict()
    assert failed == [] and possible == ["bench/graph.ladybug"]


def test_an_empty_lock_file_is_never_a_state_write(tmp_path):
    (tmp_path / "keystore").mkdir()
    (tmp_path / "keystore/flight.lock").write_bytes(b"")
    seen = watch(tmp_path, [])
    (tmp_path / "keystore/flight.lock").write_bytes(b"")
    (tmp_path / "keystore/state.lock").write_bytes(b"")
    (tmp_path / "broker.lock").write_bytes(b"")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert seen.settle() == ""


def test_a_lock_file_with_content_is_still_a_write(tmp_path):
    seen = watch(tmp_path, [])
    (tmp_path / "x.lock").write_text("data")
    assert seen.verdict()[0] == ["x.lock"]


def test_nothing_changed_is_quiet(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert watch(tmp_path, OTHER).settle() == ""


@pytest.mark.parametrize("argv,expected", [
    (["/venv/bin/poolhouse-bench", "prepare"], True),
    (["/venv/bin/python", "-m", "poolhouse.broker"], True),
    (["python3", "/repos/ml-stack/scripts/test", "all"], False),
    (["vim", "poolhouse/README.md"], False)])
def test_poolhouse_processes_are_recognised_by_how_they_were_started(argv, expected):
    assert _is_poolhouse(argv) is expected


def test_this_session_is_not_its_own_other_writer():
    assert os.getpid() not in {pid for pid, _ in live_poolhouse_processes()}
