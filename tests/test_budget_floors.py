"""A floor is a count that may not fall: the tests the suite collects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from gates import _floors  # noqa: E402
from test_budget_rises import scoreboard  # noqa: E402

FLOORS = "floors-only-rise"
RECORDED = json.loads((REPO / "budgets.json").read_text(encoding="utf-8"))


def test_the_collected_count_is_recorded_as_a_floor() -> None:
    least = RECORDED.get(FLOORS, {})
    assert _floors.NAME in least, (
        f"budgets.json records no floor for {_floors.NAME} -- run scripts/budgets --update"
    )
    assert isinstance(least[_floors.NAME], int) and least[_floors.NAME] > 0


@pytest.mark.slow
def test_the_suite_collects_at_least_what_is_recorded() -> None:
    reason = _floors.skip(REPO)
    if reason:
        pytest.skip(reason)
    counted, broke = _floors.collect(REPO)
    least = RECORDED[FLOORS][_floors.NAME]
    assert broke == 0, f"{broke} modules failed to collect; their tests ran nowhere"
    assert counted >= least, (
        f"{_floors.NAME}: {counted} collected, {least} is the floor -- "
        f"{least - counted} tests stopped running. {_floors.describe()}"
    )


@pytest.mark.slow
def test_a_module_that_stops_importing_takes_its_tests_out_of_the_count(tmp_path) -> None:
    """The fortnight of tests nobody noticed had gone, in miniature."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_kept.py").write_text("def test_one():\n    pass\n\n\ndef test_two():\n"
                                        "    pass\n", encoding="utf-8")
    whole = _floors.collect(tmp_path)
    (tests / "test_lost.py").write_text("import nothing_is_named_this\n\n\n"
                                        "def test_three():\n    pass\n", encoding="utf-8")
    assert whole == (2, 0)
    assert _floors.collect(tmp_path) == (2, 1)


def test_a_missing_extra_leaves_the_count_uncounted_rather_than_low(monkeypatch) -> None:
    monkeypatch.setattr(_floors, "installed", lambda name: False)
    reason = _floors.skip(REPO)
    assert "absent" in reason and "ladybug" in reason


def test_a_floor_that_has_fallen_is_named_and_fails_the_scoreboard(monkeypatch) -> None:
    module = scoreboard()
    monkeypatch.setattr(module._floors, "skip", lambda root: "")
    monkeypatch.setattr(module._floors, "collect", lambda root: (40, 0))
    lines, code = module.floor_rows({_floors.NAME: 50})
    assert code == 1
    assert any("collected 40" in line and "-10" in line for line in lines)


def test_a_count_above_the_floor_asks_to_be_recorded(monkeypatch) -> None:
    module = scoreboard()
    monkeypatch.setattr(module._floors, "skip", lambda root: "")
    monkeypatch.setattr(module._floors, "collect", lambda root: (60, 0))
    lines, code = module.floor_rows({_floors.NAME: 50})
    assert code == 0
    assert any("--update" in line for line in lines)


def test_a_module_that_did_not_collect_stops_the_count_being_recorded(monkeypatch, capsys,
                                                                     tmp_path):
    module = scoreboard()
    monkeypatch.setattr(module, "BUDGETS", tmp_path / "budgets.json")
    monkeypatch.setattr(module._floors, "skip", lambda root: "")
    monkeypatch.setattr(module._floors, "collect", lambda root: (40, 2))
    monkeypatch.setattr(module.gates, "run", lambda root: {})
    monkeypatch.setattr(module.gates, "hard", set)
    monkeypatch.setattr(module.gates, "unrunnable", dict)
    assert module.main(["--update"]) == 1
    assert "2 modules failed to collect" in capsys.readouterr().out
