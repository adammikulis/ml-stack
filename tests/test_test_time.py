"""The full tier's time is a ratchet (scripts/testtime.py): it may fall, and a rise fails."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import testtime  # noqa: E402
from testtime import Timing  # noqa: E402


def seeded(root: Path, recorded: Timing, now: Timing) -> None:
    (root / "tests").mkdir()
    testtime.save(root / testtime.RECORD, recorded)
    testtime.save(root / testtime.LAST, now)


def test_a_run_within_ten_percent_passes_and_one_over_it_fails(tmp_path):
    seeded(tmp_path, Timing(300, 2000, 16, 9000), Timing(329, 2190, 16, 9000))
    assert testtime.check(tmp_path) == 0
    testtime.save(tmp_path / testtime.LAST, Timing(331, 2000, 16, 9000))
    assert testtime.check(tmp_path) == 1
    testtime.save(tmp_path / testtime.LAST, Timing(300, 2201, 16, 9000))
    assert testtime.check(tmp_path) == 1


def test_wall_time_on_fewer_workers_is_not_compared_but_cpu_time_is(tmp_path):
    seeded(tmp_path, Timing(300, 2000, 16, 9000), Timing(900, 2000, 4, 9000))
    assert testtime.check(tmp_path) == 0
    testtime.save(tmp_path / testtime.LAST, Timing(900, 2500, 4, 9000))
    assert testtime.check(tmp_path) == 1


def test_update_lowers_the_record_and_refuses_to_raise_it(tmp_path):
    seeded(tmp_path, Timing(300, 2000, 16, 9000), Timing(250, 1800, 16, 9000))
    assert testtime.check(tmp_path, update=True) == 0
    assert testtime.load(tmp_path / testtime.RECORD) == Timing(250, 1800, 16, 9000)
    testtime.save(tmp_path / testtime.LAST, Timing(400, 1800, 16, 9000))
    assert testtime.check(tmp_path, update=True) == 1
    assert testtime.load(tmp_path / testtime.RECORD).wall_s == 250
    assert testtime.check(tmp_path, update=True, allow_increase=True) == 0
    assert testtime.load(tmp_path / testtime.RECORD).wall_s == 400


def test_the_cpu_seconds_are_the_sum_of_the_testcases():
    xml = '<testsuite><testcase classname="a" name="x" time="1.5"/><testcase name="y" time="2"/></testsuite>'
    assert testtime.cpu_seconds(xml) == (3.5, 2)


def test_nothing_measured_is_not_a_pass(tmp_path):
    (tmp_path / "tests").mkdir()
    assert testtime.check(tmp_path) == 2


@pytest.mark.parametrize("bad", ["", "not json", "{}"])
def test_an_unreadable_record_is_none(tmp_path, bad):
    path = tmp_path / "r.json"
    path.write_text(bad)
    assert testtime.load(path) is None


def test_wall_time_is_not_compared_when_the_machine_was_busy(tmp_path):
    seeded(tmp_path, Timing(300, 2000, 16, 9000), Timing(900, 2000, 16, 9000, load=400.0))
    assert testtime.check(tmp_path) == 0
    testtime.save(tmp_path / testtime.LAST, Timing(900, 2000, 16, 9000, load=0.0))
    assert testtime.check(tmp_path) == 1
