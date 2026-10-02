"""Reports: what is counted, how they read back, and which attacks a new run lost ground on."""

from __future__ import annotations

import json

import pytest

from ml_stack.redteam.report import Attempt, Report, compare, markdown, summary


def attempt(attack_id: str, succeeded: bool, **kwargs) -> Attempt:
    return Attempt("chat", "system-prompt-extraction", attack_id, succeeded, **kwargs)


def test_the_summary_counts_attempts_successes_and_blocks_per_class_and_arm():
    report = Report(attempts=[
        attempt("a", True, arm="server", seconds=1.0),
        attempt("b", False, arm="server", seconds=3.0, blocked=True),
        attempt("c", False, arm="daemon", seconds=2.0, attempted=True),
    ])
    rows = {row["arm"]: row for row in summary(report)}
    assert (rows["server"]["attempts"], rows["server"]["succeeded"],
            rows["server"]["blocked"], rows["server"]["median_s"]) == (2, 1, 1, 2.0)
    assert (rows["daemon"]["attempts"], rows["daemon"]["attempted"]) == (1, 1)


def test_a_report_read_back_from_json_has_the_same_attempts_and_meta(tmp_path):
    report = Report({"model": "m.gguf", "ref": "abc1234"}, [attempt("a", True, detail="x")])
    path = tmp_path / "run.json"
    path.write_text(report.to_json(), encoding="utf-8")
    again = Report.load(path)
    assert again.attempts == report.attempts and again.meta == report.meta


def test_a_report_of_another_schema_is_refused():
    with pytest.raises(ValueError, match="version"):
        Report.from_json(json.dumps({"version": 99, "meta": {}, "attempts": []}))


def test_an_attack_that_failed_before_and_succeeds_now_is_a_regression():
    old = Report(attempts=[attempt("a", False), attempt("b", True), attempt("c", False)])
    new = Report(attempts=[attempt("a", True), attempt("b", False), attempt("d", True)])
    got = compare(old, new)
    assert [key[2] for key in got["regressed"]] == ["a"]
    assert [key[2] for key in got["fixed"]] == ["b"]
    assert [key[2] for key in got["new"]] == ["d"] and [key[2] for key in got["gone"]] == ["c"]


def test_one_success_among_repeats_of_an_attack_counts_as_the_attack_succeeding():
    old = Report(attempts=[attempt("a", False), attempt("a", False)])
    new = Report(attempts=[attempt("a", False), attempt("a", True)])
    assert [key[2] for key in compare(old, new)["regressed"]] == ["a"]


def test_the_same_attack_on_another_arm_is_a_different_attack():
    old = Report(attempts=[attempt("a", False, arm="server")])
    new = Report(attempts=[attempt("a", True, arm="daemon")])
    got = compare(old, new)
    assert not got["regressed"] and len(got["new"]) == 1


def test_the_markdown_lists_each_success_by_name_and_where_the_run_came_from():
    text = markdown(Report({"model": "m.gguf"}, [attempt("garak-03", True, arm="server",
                                                         detail="the key is in the reply")]))
    assert "model: m.gguf" in text and "`garak-03`" in text and "[server]" in text
