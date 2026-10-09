"""Options, decisions, calibration and rules: the parts of poolhouse.decide that need no model."""

from __future__ import annotations

import asyncio
import math

import pytest

from poolhouse.decide import Calibration, Layered, Rule, RulesDecider
from poolhouse.decide.calibrate import fit, fit_isotonic, fit_temperature, nll, rescale
from poolhouse.decide.types import DecideError, Option, decision_from, options_of


def test_options_keep_their_order_and_take_descriptions_from_a_mapping():
    got = options_of({"b": "second letter", "a": ""})
    assert [o.name for o in got] == ["b", "a"]
    assert got[0].description == "second letter"


def test_options_refuse_fewer_than_two_duplicates_and_stray_descriptions():
    with pytest.raises(ValueError, match="at least two"):
        options_of(["only"])
    with pytest.raises(ValueError, match="unique"):
        options_of(["a", "a"])
    with pytest.raises(ValueError, match="not offered"):
        options_of(["a", "b"], {"c": "x"})
    with pytest.raises(ValueError, match="empty"):
        options_of(["a", " "])


def test_descriptions_argument_overrides_those_in_the_options():
    got = options_of([Option("a", "old"), Option("b")], {"a": "new"})
    assert got[0].description == "new"


def test_decision_normalises_and_reports_confidence_margin_entropy_and_top_k():
    d = decision_from([3.0, 1.0, 0.0], options_of(["x", "y", "z"]))
    assert d.choice == "x"
    assert d.scores == pytest.approx({"x": 0.75, "y": 0.25, "z": 0.0})
    assert d.confidence == pytest.approx(0.75)
    assert d.margin == pytest.approx(0.5)
    assert d.entropy == pytest.approx(-(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))
    assert d.certainty == pytest.approx((3 * 0.75 - 1) / 2)
    assert d.top_k(2) == [("x", pytest.approx(0.75)), ("y", pytest.approx(0.25))]


def test_a_tie_goes_to_the_option_offered_first():
    d = decision_from([1.0, 1.0], options_of(["first", "second"]))
    assert d.choice == "first"
    assert d.top_k(1)[0][0] == "first"


def test_abstains_only_below_the_threshold():
    opts = options_of(["a", "b"])
    assert decision_from([0.6, 0.4], opts, abstain_below=0.7).abstained
    assert not decision_from([0.6, 0.4], opts, abstain_below=0.6).abstained
    assert not decision_from([0.6, 0.4], opts).abstained


def test_a_distribution_that_is_not_one_is_refused():
    opts = options_of(["a", "b"])
    for bad in ([0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0], [1.0]):
        with pytest.raises(DecideError):
            decision_from(bad, opts)


def test_temperature_above_one_flattens_and_below_one_sharpens():
    p = [0.8, 0.2]
    assert rescale(p, 2.0)[0] < 0.8 < rescale(p, 0.5)[0]
    assert rescale(p, 1.0) == pytest.approx(p)
    with pytest.raises(ValueError):
        rescale(p, 0.0)


def test_fit_temperature_recovers_overconfidence():
    # always 99% sure, right 3 times in 4: the fitted temperature must flatten it
    rows = [[0.99, 0.01]] * 4
    labels = [0, 0, 0, 1]
    t = fit_temperature(rows, labels)
    assert t > 1.5
    assert nll(rows, labels, t) < nll(rows, labels, 1.0)
    assert rescale(rows[0], t)[0] == pytest.approx(0.75, abs=0.02)


def test_fit_temperature_leaves_a_calibrated_model_alone():
    rows = [[0.75, 0.25]] * 4
    labels = [0, 0, 0, 1]
    assert fit_temperature(rows, labels) == pytest.approx(1.0, abs=0.05)


def test_fit_temperature_checks_its_inputs():
    with pytest.raises(ValueError):
        fit_temperature([[0.5, 0.5]], [0, 1])
    with pytest.raises(ValueError):
        nll([], [])


def test_isotonic_is_monotone_and_pools_violations():
    knots = fit_isotonic([0.5, 0.6, 0.7, 0.8, 0.9], [False, True, False, True, True])
    ys = [y for _, y in knots]
    assert ys == sorted(ys)
    assert knots == [(0.5, 0.0), (0.7, 0.5), (0.9, 1.0)]


def test_calibration_round_trips_and_is_applied_by_a_decider():
    cal = fit([[0.99, 0.01]] * 4, [0, 0, 0, 1], isotonic=True)
    again = Calibration.from_public(cal.public())
    assert again == cal
    out = cal.apply([0.99, 0.01])
    assert sum(out) == pytest.approx(1.0)
    assert out[0] < 0.99
    with pytest.raises(ValueError):
        Calibration.from_public({"temperature": -1})


def test_isotonic_keeps_the_chosen_option_and_shares_the_rest_in_proportion():
    cal = Calibration(1.0, ((0.5, 0.4), (1.0, 0.9)))
    out = cal.apply([0.6, 0.3, 0.1])
    assert out[0] == pytest.approx(0.4 + (0.6 - 0.5) / 0.5 * 0.5)
    assert out[1] / out[2] == pytest.approx(3.0)
    assert sum(out) == pytest.approx(1.0)


RULES = [Rule("deny", pattern=r"\brm\s+-rf\b"), Rule("allow", one_of=("ls",), field="command"),
         Rule("deny", contains=("drop table",), weight=2.0)]


def test_rules_pick_the_option_a_pattern_a_word_or_an_allowed_value_votes_for():
    rules = RulesDecider(RULES, default="allow")
    assert rules.decide("q", "please RM -RF /tmp", ["allow", "deny"]).choice == "deny"
    assert rules.decide("q", "DROP TABLE users", ["allow", "deny"]).choice == "deny"
    got = rules.decide("q", {"command": "ls"}, ["allow", "deny"])
    assert got.choice == "allow" and got.details["matched"]
    nothing = rules.decide("q", "hello", ["allow", "deny"])
    assert nothing.choice == "allow" and not nothing.details["matched"]


def test_rules_with_no_match_and_no_default_are_uniform_and_can_abstain():
    got = RulesDecider(RULES).decide("q", "hello", ["allow", "deny"], abstain_below=0.9)
    assert got.scores == {"allow": 0.5, "deny": 0.5}
    assert got.abstained


def test_rules_weigh_competing_votes_and_ignore_rules_for_absent_options():
    rules = RulesDecider([Rule("a", contains=("x",), weight=1.0), Rule("b", contains=("x",),
                                                                       weight=3.0),
                          Rule("zzz", contains=("x",), weight=9.0)])
    assert rules.decide("q", "x", ["a", "b"]).choice == "b"


def test_a_rule_can_be_case_sensitive():
    rules = RulesDecider([Rule("a", contains=("Key",))], default="b", case_sensitive=True)
    assert rules.decide("q", "key", ["a", "b"]).choice == "b"
    assert rules.decide("q", "Key", ["a", "b"]).choice == "a"


def test_layered_lets_a_matching_rule_overrule_the_learned_decider():
    learned = RulesDecider([Rule("allow", contains=("rm",))])
    layered = Layered(RulesDecider([Rule("deny", pattern="rm -rf")]), learned)
    assert layered.decide("q", "rm -rf /", ["allow", "deny"]).backend == "rules"
    assert layered.decide("q", "rm file", ["allow", "deny"]).choice == "allow"
    got = asyncio.run(layered.adecide("q", "rm -rf /", ["allow", "deny"]))
    assert got.choice == "deny"
    assert [d.choice for d in layered.decide_many([])] == []


def test_state_and_question_are_size_limited():
    with pytest.raises(ValueError, match="limit"):
        RulesDecider([]).decide("q", "x" * 200_001, ["a", "b"])
