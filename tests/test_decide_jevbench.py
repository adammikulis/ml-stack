"""JevBench's record format as cases, and the scoring of a run, on a tiny synthetic file in that
format (no download, no model): the real logprob decider against the fake chat server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from decide_fakes import logprob_handler

from ml_stack.decide import jevbench
from ml_stack.decide.logprob import LogprobDecider

ROWS = [
    {"id": "n1", "family": "policy", "state": "Receipt present.", "labels": ["no", "yes"],
     "expected": "yes", "question": {"type": "noul", "instructions": "Refund allowed?",
                                     "criteria": {"false": "missing", "true": "established"}}},
    {"id": "n2", "family": "policy", "state": "No receipt.", "labels": ["no", "yes"],
     "expected": "no", "question": {"type": "noul", "instructions": "Refund allowed?",
                                    "criteria": {"false": "missing", "true": "established"}}},
    {"id": "c1", "family": "intent", "state": "Where is my parcel?", "group": "g",
     "labels": ["track", "cancel", "other"], "expected": "track",
     "question": {"type": "choice", "instructions": "Intent?",
                  "criteria": {"track": "where", "cancel": "stop", "other": "none"}}},
    {"id": "s1", "family": "ordinal", "state": "Icon is misaligned.", "labels": ["0", "1", "2"],
     "expected": 0, "question": {"type": "score", "instructions": "Impact?",
                                 "criteria": ["cosmetic", "workaround", "blocked"]}},
]


@pytest.fixture
def cases(tmp_path: Path):
    path = tmp_path / "tiny.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    return jevbench.read_part(path, "easy")


def test_records_become_cases_with_the_rubric_as_option_text(cases):
    by = {c.id: c for c in cases}
    assert [(o.name, o.description) for o in by["n1"].options] == [("no", "missing"),
                                                                    ("yes", "established")]
    assert [o.description for o in by["s1"].options] == ["cosmetic", "workaround", "blocked"]
    assert by["s1"].label == "0" and by["c1"].group == "g" and by["n1"].group == "n1"
    assert "type:score" in by["s1"].tags and "tier:easy" in by["s1"].tags


def test_an_item_without_an_answer_is_refused():
    with pytest.raises(ValueError, match="no expected"):
        jevbench.case_of({**ROWS[0], "expected": None}, "easy")


def test_a_run_scores_per_type_and_counts_a_failed_answer_as_wrong(cases, server):
    # the fake model always picks the first option with 0.8, so n1 (label yes) is wrong,
    # n2, c1 and s1 are right; one server error is a miss, not a dropped item
    inner = logprob_handler(lambda user: {"A": 0.8, "B": 0.1, "C": 0.1} if "Intent" in user
                            or "Impact" in user else {"A": 0.8, "B": 0.2})

    def flaky(method, path, body):
        if method == "POST" and b"Impact?" in body:
            return 500, b"{}"
        return inner(method, path, body)

    decider = LogprobDecider(server(flaky).base_url)
    out = jevbench.run(lambda c: decider.decide(c.question, c.state, c.options), cases)
    assert [o.decision is None for o in out] == [False, False, False, True]
    summary = jevbench.summarize(out)
    assert summary["all"]["n"] == 4 and summary["all"]["errors"] == 1
    assert summary["all"]["accuracy"] == pytest.approx(2 / 4)
    assert summary["type:noul"]["accuracy"] == pytest.approx(1 / 2)
    assert summary["type:choice"]["accuracy"] == 1.0
    assert summary["type:score"]["accuracy"] == 0.0 and summary["type:score"]["errors"] == 1
    assert summary["type:noul"]["brier"] == pytest.approx((1.28 + 0.08) / 2)


def test_the_rules_baseline_is_the_first_option(cases):
    out = jevbench.run(lambda c: jevbench.first_option_rules(c).decide(
        c.question, c.state, c.options), cases)
    assert [o.decision.choice for o in out] == ["no", "no", "track", "0"]
    assert jevbench.summarize(out)["all"]["accuracy"] == pytest.approx(3 / 4)


def test_the_pins_cover_the_three_public_files():
    assert [(p.tier, p.size) for p in jevbench.PARTS] == [("easy", 37220), ("original", 57237),
                                                          ("hard", 651848)]
    assert len(jevbench.COMMIT) == 40 and all(len(p.sha256) == 64 for p in jevbench.PARTS)
