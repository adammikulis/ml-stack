"""The debt falls on a schedule the scoreboard reports and nothing refuses a branch over."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from test_budget_rises import scoreboard  # noqa: E402

ANCHOR = "ratchet"
START = {"date": "2026-01-01", "total": 2000}


def test_the_anchor_is_recorded_with_a_date_and_a_total() -> None:
    held = json.loads((REPO / "budgets.json").read_text(encoding="utf-8"))
    start = held.get(ANCHOR, {})
    assert set(start) == {"date", "total"}
    assert date.fromisoformat(start["date"]) and isinstance(start["total"], int)


def test_the_debt_due_falls_by_the_rate_every_month() -> None:
    module = scoreboard()
    begins = date.fromisoformat(START["date"])
    assert module.due(START, begins) == 2000
    assert module.due(START, begins + timedelta(days=module.MONTH)) == 1900
    assert module.due(START, begins + timedelta(days=module.MONTH * 10)) == 1000


def test_the_schedule_reaches_zero_and_stays_there() -> None:
    module = scoreboard()
    begins = date.fromisoformat(START["date"])
    months = int(module.MONTH / module.RATCHET)
    assert module.due(START, begins + timedelta(days=months)) == 0
    assert module.due(START, begins + timedelta(days=months * 3)) == 0


def test_a_debt_at_or_under_the_line_is_on_track() -> None:
    module = scoreboard()
    when = date.fromisoformat(START["date"]) + timedelta(days=module.MONTH)
    said = "\n".join(module.ratchet_lines(1900, START, when))
    assert "on track" in said
    assert "1900 now" in said


def test_a_debt_over_the_line_says_how_far_over_and_still_returns_a_report() -> None:
    module = scoreboard()
    when = date.fromisoformat(START["date"]) + timedelta(days=module.MONTH)
    said = "\n".join(module.ratchet_lines(1950, START, when))
    assert "50 over today's 1900" in said
    assert "on track" not in said


def test_the_next_months_target_and_the_date_it_reaches_zero_are_both_named() -> None:
    module = scoreboard()
    begins = date.fromisoformat(START["date"])
    said = "\n".join(module.ratchet_lines(2000, START, begins))
    assert "1900 due by 2026-01-31" in said
    assert "zero by 2027-08-24" in said


def test_a_ratchet_already_recorded_is_not_moved_by_a_later_debt(monkeypatch, tmp_path):
    module = scoreboard()
    held = tmp_path / "budgets.json"
    held.write_text(json.dumps({"broad-excepts": 9, ANCHOR: START}), encoding="utf-8")
    monkeypatch.setattr(module, "BUDGETS", held)
    assert module.anchor(9, date(2026, 6, 1)) == START


def test_a_tree_with_no_ratchet_yet_starts_one_at_todays_debt(monkeypatch, tmp_path):
    module = scoreboard()
    held = tmp_path / "budgets.json"
    held.write_text(json.dumps({"broad-excepts": 9}), encoding="utf-8")
    monkeypatch.setattr(module, "BUDGETS", held)
    assert module.anchor(9, date(2026, 6, 1)) == {"date": "2026-06-01", "total": 9}
