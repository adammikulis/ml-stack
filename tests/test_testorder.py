"""The longest files start first, a long test leads its file, and nothing else moves."""

from __future__ import annotations

import json
from types import SimpleNamespace

import testorder

HISTORY = {
    "tests/test_a.py::one": {"cpu": 1.0, "wall": 1.0, "n": 3},
    "tests/test_b.py::slow": {"cpu": 90.0, "wall": 200.0, "n": 3},
    "tests/test_b.py::quick": {"cpu": 1.0, "wall": 1.0, "n": 3},
    "tests/test_c.py::mid": {"cpu": 30.0, "wall": 30.0, "n": 3},
}


def test_the_longest_file_comes_first_and_its_long_test_leads_it():
    order = testorder.snapshot(HISTORY)
    ids = ["tests/test_a.py::one", "tests/test_b.py::quick", "tests/test_b.py::slow", "tests/test_c.py::mid"]
    assert testorder.ordered(ids, order) == [
        "tests/test_b.py::slow", "tests/test_b.py::quick", "tests/test_c.py::mid", "tests/test_a.py::one"]


def test_tests_inside_a_file_keep_their_order_and_stay_together():
    order = testorder.snapshot(HISTORY)
    ids = ["tests/test_a.py::x[1]", "tests/test_c.py::p", "tests/test_a.py::x[2]", "tests/test_c.py::q"]
    got = testorder.ordered(ids, order)
    assert got == ["tests/test_c.py::p", "tests/test_c.py::q", "tests/test_a.py::x[1]", "tests/test_a.py::x[2]"]


def test_a_file_the_history_never_saw_counts_as_the_default():
    order = testorder.snapshot(HISTORY)
    got = testorder.ordered(["tests/test_a.py::one", "tests/test_new.py::t", "tests/test_b.py::quick"], order)
    assert got[0] == "tests/test_b.py::quick"          # test_b is 91 s of history
    assert got.index("tests/test_new.py::t") < got.index("tests/test_a.py::one")   # 30 s default beats 1 s


def test_the_plugin_reorders_collected_items_from_the_snapshot_and_does_nothing_without_one(tmp_path, monkeypatch):
    snapshot = tmp_path / "order.json"
    snapshot.write_text(json.dumps(testorder.snapshot(HISTORY)))
    items = [SimpleNamespace(nodeid=n) for n in ("tests/test_a.py::one", "tests/test_b.py::slow")]
    monkeypatch.delenv("DEV_TEST_ORDER", raising=False)
    testorder.pytest_collection_modifyitems(None, items)
    assert [i.nodeid for i in items] == ["tests/test_a.py::one", "tests/test_b.py::slow"]
    monkeypatch.setenv("DEV_TEST_ORDER", str(snapshot))
    testorder.pytest_collection_modifyitems(None, items)
    assert [i.nodeid for i in items] == ["tests/test_b.py::slow", "tests/test_a.py::one"]
    monkeypatch.setenv("DEV_TEST_ORDER", str(tmp_path / "missing.json"))
    testorder.pytest_collection_modifyitems(None, items)
    assert [i.nodeid for i in items] == ["tests/test_b.py::slow", "tests/test_a.py::one"]
