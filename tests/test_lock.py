"""One runner at a time, so nobody hand-writes a waiter again.

The shell loop this replaces could not work: `until ! pgrep -f "...bench"` matches the
waiting shell's own command line, so the condition never goes false. Two sat spinning for an
afternoon and the draft-head comparison queued behind them never ran -- silently, with its
log file never created.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from ml_stack.lock import Busy, only_one


def test_a_second_holder_is_refused_rather_than_allowed_to_overlap(tmp_path):
    with only_one(tmp_path / "l", wait=False):
        with pytest.raises(Busy) as why:
            with only_one(tmp_path / "l", wait=False):
                raise AssertionError("two runs held the same lock at once")
    assert str(os.getpid()) in str(why.value), "says who has it, for a stalled machine"


def test_the_lock_is_released_when_the_block_ends(tmp_path):
    with only_one(tmp_path / "l"):
        pass
    with only_one(tmp_path / "l", wait=False):
        pass


def test_it_is_released_even_when_the_run_raises(tmp_path):
    with pytest.raises(ValueError):
        with only_one(tmp_path / "l"):
            raise ValueError("a run that failed still has to let the next one in")
    with only_one(tmp_path / "l", wait=False):
        pass


def test_waiting_is_announced_rather_than_silent(tmp_path):
    """A wait nobody can see is indistinguishable from a hang -- which is exactly how the
    stuck waiters read."""
    said = []
    other = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import time
            from ml_stack.lock import only_one
            with only_one({str(tmp_path / 'l')!r}):
                print("held", flush=True)
                time.sleep(1.5)
        """)], stdout=subprocess.PIPE, text=True)
    try:
        assert other.stdout.readline().strip() == "held"
        with only_one(tmp_path / "l", timeout=10, announce=said.append):
            pass
    finally:
        other.wait(timeout=10)
    assert said and "waiting for" in said[0]


def test_a_bounded_wait_gives_up_and_says_so(tmp_path):
    with only_one(tmp_path / "l"):
        with pytest.raises(Busy, match="still held"):
            with only_one(tmp_path / "l", timeout=0.2, announce=lambda _: None):
                pass


def test_only_the_measuring_subcommands_take_it():
    """`show` reads the store and touches no GPU; making it queue behind a run would be a new
    way to hang."""
    from ml_stack.graph.bench import MEASURING

    assert set(MEASURING) == {"run", "sweep", "drafts"}
    assert "show" not in MEASURING and "prepare" not in MEASURING
