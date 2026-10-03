"""Every surface where outside input reaches a model, a tool, a process, a path or the network
has a row in the coverage map, every test the map names exists, and the uncovered count only
falls. The checks are the ones `scripts/redteam_coverage.py --check` runs."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@cache
def coverage():
    spec = importlib.util.spec_from_file_location("redteam_coverage", REPO / "scripts/redteam_coverage.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["redteam_coverage"] = module
    spec.loader.exec_module(module)
    return module


@cache
def state():
    cov = coverage()
    found = cov.discover()
    owner, problems = cov.resolve(found, cov.load_map())
    return found, owner, problems, cov.rows(found, owner)


def test_the_tree_passes_the_gate():
    found, owner, problems, table = state()
    failures = coverage().check(found, table, owner, problems)
    assert failures == [], "\n".join(failures)


def test_every_kind_of_surface_is_found():
    kinds = {row["kind"] for row in state()[3]}
    assert kinds >= {"route", "handler", "mcp-tool", "chat-tool", "slash-command", "cli", "spawn",
                     "listener", "egress", "human", "desktop", "keystore", "context", "parser"}


def test_a_new_surface_without_a_row_fails_and_the_message_says_what_to_add():
    cov = coverage()
    found, _, _, _ = state()
    fresh = dict(found)
    cov.add(fresh, "route", "fleet/api.py:/brand-new", "fleet/api.py:Handler.do_GET")
    owner, problems = cov.resolve(fresh, {"group": []})
    failures = cov.check(fresh, cov.rows(fresh, owner), owner, problems)
    mine = [f for f in failures if "route:fleet/api.py:/brand-new" in f]
    assert mine and "coverage-map.toml" in mine[0]
    assert '[[group]]\nmatch = ["route:fleet/api.py:/brand-new"]' in mine[0]
    assert "tests = [" in mine[0]


def test_a_test_that_no_longer_exists_fails():
    cov = coverage()
    found, _, _, _ = state()
    table = {"group": [{"match": ["*"], "status": "covered",
                        "tests": ["tests/test_redteam_coverage.py::test_gone_for_good",
                                  "tests/test_not_a_file.py"]}]}
    _, problems = cov.resolve(found, table)
    assert any("test_gone_for_good" in p for p in problems)
    assert any("test_not_a_file.py does not exist" in p for p in problems)


def test_a_group_that_matches_nothing_or_miscounts_fails():
    cov = coverage()
    found, _, _, _ = state()
    table = {"group": [{"match": ["route:nothing/*"], "status": "n/a", "note": "x"},
                       {"match": ["mcp-tool:*"], "status": "n/a", "note": "x", "count": 1}]}
    _, problems = cov.resolve(found, table)
    assert any("matches no surface" in p for p in problems)
    assert any("expected 1 surfaces" in p for p in problems)


@pytest.mark.parametrize("group,needle", [
    ({"match": ["*"], "status": "covered"}, "needs at least one test"),
    ({"match": ["*"], "status": "uncovered"}, "needs a note"),
    ({"match": ["*"], "status": "n/a"}, "needs a note"),
    ({"match": ["*"], "status": "fine", "note": "x"}, "status must be one of"),
])
def test_a_row_must_say_why(group, needle):
    assert any(needle in p for p in coverage().group_problems(group, 1))


def test_the_uncovered_count_may_only_fall(monkeypatch):
    cov = coverage()
    found, owner, problems, table = state()
    held = cov.totals(table)
    monkeypatch.setattr(cov, "ratchet", lambda: {"uncovered": held["uncovered"] - 1,
                                                 "partial": held["partial"]})
    failures = cov.check(found, table, owner, problems)
    assert any("uncovered surfaces rose" in f for f in failures)
    monkeypatch.setattr(cov, "ratchet", lambda: {"uncovered": held["uncovered"] + 1,
                                                 "partial": held["partial"]})
    assert any("uncovered fell" in f for f in cov.check(found, table, owner, problems))


def test_the_written_map_is_the_one_the_tree_produces():
    cov = coverage()
    assert cov.OUT.read_text(encoding="utf-8") == cov.render(state()[3])
