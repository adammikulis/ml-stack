"""Who may open a store, and when.

The cross-process claims are driven with real processes and real advisory locks, because the
whole module exists for what happens between processes and a same-process test would prove
nothing about it.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ml_stack.graph import access
from ml_stack.graph.access import (LockError, holder, lock_path, pid_alive, reading,
                                   recover_stale, release_all, write_lock, writing)


class Fake:
    """A store that is a text file, so the tests are about access and nothing else."""

    opened = 0
    closed = 0

    def __init__(self, path):
        self.path = Path(path)
        Fake.opened += 1
        self.shut = False

    def close(self):
        if not self.shut:
            Fake.closed += 1
            self.shut = True


@pytest.fixture(autouse=True)
def _clean():
    Fake.opened = Fake.closed = 0
    release_all()
    yield
    release_all()


def a_store(tmp_path):
    path = tmp_path / "g.store"
    path.write_text("rows", encoding="utf-8")
    return path


def test_a_live_process_is_alive_and_a_freed_pid_is_not():
    assert pid_alive(os.getpid())
    assert not pid_alive(2 ** 22 - 1)
    assert not pid_alive(0) and not pid_alive(-1)


def test_the_lease_records_who_holds_it(tmp_path):
    path = a_store(tmp_path)
    with write_lock(path):
        who = holder(path)
        assert who is not None and who.pid == os.getpid() and who.alive
        assert "alive" in who.describe()
    # and gives it back
    assert holder(path) is None


def test_taking_the_same_lock_twice_in_one_thread_is_not_a_deadlock(tmp_path):
    path = a_store(tmp_path)
    with write_lock(path, timeout_s=1):
        with write_lock(path, timeout_s=1):
            assert holder(path).pid == os.getpid()
        assert holder(path) is not None      # still held by the outer one


def test_another_process_holding_it_is_named_rather_than_guessed(tmp_path):
    path = a_store(tmp_path)
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(Path(__file__).parent.parent / 'src')!r})
        from ml_stack.graph.access import write_lock
        with write_lock({str(path)!r}, timeout_s=5):
            print("held", flush=True)
            time.sleep(8)
    """)
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        who = holder(path)
        assert who is not None and who.pid == child.pid and who.alive
        with pytest.raises(LockError, match=f"pid={child.pid}"):
            with write_lock(path, timeout_s=0.5):
                pass
    finally:
        child.kill()
        child.wait()


def test_a_dead_owners_record_is_cleared_and_a_live_ones_is_not(tmp_path):
    path = a_store(tmp_path)
    lock_path(path).write_text(json.dumps(
        {"pid": 2 ** 22 - 1, "host": os.uname().nodename, "since": time.time()}), encoding="utf-8")
    assert holder(path).alive is False
    assert recover_stale(path) is True
    assert holder(path) is None

    lock_path(path).write_text(json.dumps(
        {"pid": os.getpid(), "host": os.uname().nodename, "since": time.time()}), encoding="utf-8")
    assert recover_stale(path) is False
    assert holder(path).pid == os.getpid()


def test_a_dead_owner_does_not_keep_the_next_writer_out(tmp_path):
    """The kernel drops the lock with the process; only our record survives."""
    path = a_store(tmp_path)
    script = textwrap.dedent(f"""
        import sys, os, signal, time
        sys.path.insert(0, {str(Path(__file__).parent.parent / 'src')!r})
        from ml_stack.graph.access import write_lock
        with write_lock({str(path)!r}, timeout_s=5):
            print("held", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
    """)
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "held"
    child.wait()
    with write_lock(path, timeout_s=5):
        assert holder(path).pid == os.getpid()


def test_readers_in_one_process_share_a_handle(tmp_path):
    path = a_store(tmp_path)
    with reading(path, Fake) as first:
        with reading(path, Fake) as second:
            assert first is second
    assert Fake.opened == 1


def test_reading_a_store_that_is_not_there_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        with reading(tmp_path / "nothing.store", Fake):
            pass


def test_a_writer_takes_the_file_back_from_a_cached_reader(tmp_path):
    path = a_store(tmp_path)
    with reading(path, Fake):
        pass                                  # handle stays cached
    assert Fake.closed == 0
    with writing(path, Fake):
        assert Fake.closed == 1               # the reader was let go for the writer
    assert Fake.closed == 2                   # and the writer closed its own


def test_what_must_happen_before_a_write_happens_inside_the_lock(tmp_path):
    path = a_store(tmp_path)
    seen = []

    def note(p):
        seen.append((Path(p), holder(p) is not None))

    with writing(path, Fake, before=note):
        pass
    assert seen == [(path, True)], "the snapshot hook ran outside the lock"


def test_releasing_closes_what_this_process_is_holding(tmp_path):
    path = a_store(tmp_path)
    with reading(path, Fake):
        pass
    assert release_all() == [str(path)]
    assert Fake.closed == 1
