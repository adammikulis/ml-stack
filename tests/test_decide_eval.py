"""The metrics: Brier, ECE with reliability bins, abstention curves, percentiles."""

from __future__ import annotations

import math

import pytest

from ml_stack.decide import Rule, RulesDecider
from ml_stack.decide.cases import Case
from ml_stack.decide.eval import (
    abstention_curve,
    brier,
    ece,
    evaluate,
    percentile,
    reliability,
    score,
    top_of,
)
from ml_stack.decide.types import DecideError, Decision, options_of


def test_brier_is_zero_when_perfect_and_two_when_confidently_wrong():
    assert brier([[1.0, 0.0]], [0]) == 0.0
    assert brier([[1.0, 0.0]], [1]) == pytest.approx(2.0)
    assert brier([[0.5, 0.5]], [0]) == pytest.approx(0.5)
    assert brier([[0.7, 0.2, 0.1]], [0]) == pytest.approx(0.09 + 0.04 + 0.01)
    with pytest.raises(ValueError):
        brier([], [])


def test_ece_is_the_gap_between_confidence_and_accuracy_weighted_by_bin_size():
    rows = [[0.9, 0.1]] * 10
    assert ece(rows, [0] * 10) == pytest.approx(0.1)
    assert ece(rows, [0] * 9 + [1]) == pytest.approx(0.0)
    assert ece([[0.5, 0.5]] * 4, [0, 1, 0, 1]) == pytest.approx(0.0)
    assert ece([], []) == 0.0


def test_reliability_bins_report_range_count_confidence_and_accuracy():
    bins = reliability([[0.95, 0.05], [0.92, 0.08], [0.55, 0.45]], [0, 1, 0])
    assert [(b["low"], b["n"]) for b in bins] == [(0.5, 1), (0.9, 2)]
    assert bins[1]["accuracy"] == 0.5
    assert bins[1]["confidence"] == pytest.approx(0.935)


def test_a_probability_of_exactly_one_falls_in_the_last_bin():
    assert reliability([[1.0, 0.0]], [0])[0]["high"] == 1.0


def test_the_abstention_curve_trades_coverage_for_accuracy():
    rows = [[0.95, 0.05], [0.9, 0.1], [0.6, 0.4], [0.55, 0.45]]
    labels = [0, 0, 1, 1]
    curve = {c["threshold"]: c for c in abstention_curve(rows, labels, [0.0, 0.8, 0.99])}
    assert curve[0.0]["coverage"] == 1.0 and curve[0.0]["accuracy"] == 0.5
    assert curve[0.8]["coverage"] == 0.5 and curve[0.8]["accuracy"] == 1.0
    assert curve[0.99]["coverage"] == 0.0 and math.isnan(curve[0.99]["accuracy"])


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.95) == 5
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)
    assert math.isnan(percentile([], 0.5))


def test_top_of_takes_the_first_of_a_tie():
    assert top_of([0.5, 0.5]) == 0 and top_of([0.2, 0.8]) == 1


def cases():
    opts = options_of(["deny", "allow"])
    return [Case("q", "rm -rf /", opts, "deny", tags=("a",)),
            Case("q", "ls", opts, "allow", tags=("a", "b")),
            Case("q", "ls -la", opts, "deny", tags=("b",))]


def test_evaluate_scores_a_decider_and_breaks_accuracy_down_by_tag():
    d = RulesDecider([Rule("deny", pattern="rm"), Rule("allow", pattern="ls")])
    r = evaluate(d, cases())
    assert (r.n, r.errors) == (3, 0)
    assert r.accuracy == pytest.approx(2 / 3)
    assert r.by_tag == {"a": {"n": 2, "accuracy": 1.0}, "b": {"n": 2, "accuracy": 0.5}}
    assert r.brier == pytest.approx(2.0 / 3)
    assert "acc=0.667" in r.line()
    assert r.public()["backend"] == "rules"


def test_cases_a_decider_fails_on_are_counted_not_hidden():
    class Flaky:
        def decide(self, question, state, options, **kw):
            if "ls -la" in str(state):
                raise DecideError("timeout")
            return Decision("deny", {"deny": 0.9, "allow": 0.1}, 0.8, False, 3.0, "flaky", "m")

    r = evaluate(Flaky(), cases())
    assert (r.n, r.errors) == (2, 1)
    assert r.latency_ms["p50"] == 3.0


def test_scoring_nothing_is_an_error():
    with pytest.raises(DecideError):
        score([], [])
