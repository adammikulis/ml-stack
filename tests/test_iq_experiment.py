"""The IQ against K-quant experiment: its plan, its checks and its decision rule."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("iq_vs_kquant", ROOT / "scripts/experiments/iq_vs_kquant.py")
exp = importlib.util.module_from_spec(spec)
sys.modules["iq_vs_kquant"] = exp
spec.loader.exec_module(exp)


@pytest.fixture
def config():
    return exp.load()


def test_the_plan_alternates_the_builds_and_never_thinks_for_decisions_or_speed(config):
    plan = exp.cells(config)
    assert len(plan) == 24
    firsts = [c.quant for c in plan if c.task == "jevbench"]
    assert firsts == ["iq", "kquant", "kquant", "iq", "iq", "kquant"]
    assert {c.thinking for c in plan if c.task != "generation"} == {"off"}
    assert {c.thinking for c in plan if c.task == "generation"} == {"off", "on"}


def test_every_command_parses_with_the_real_parsers_and_holds_the_settings_constant(config):
    out = Path("/x")
    for cell in exp.cells(config):
        argv = exp.command(config, cell, out)
        text = " ".join(argv)
        if cell.task != "jevbench":
            assert "--no-draft" in argv and "--no-profile" in argv and "--no-queue" in argv
        if cell.task == "generation":
            assert ("--reasoning-budget 0" in text) == (cell.thinking == "off")
            assert "--temperature 0" in text and "--serve-arg=1234" in text
    assert exp._flags(config) == []


def test_check_refuses_a_config_that_breaks_the_protocol(config):
    assert exp.problems({**config, "repeats": 2}) and exp.problems({**config, "temperature": 0.7})
    assert exp.problems({**config, "draft": "mtp"})
    assert exp.problems({**config, "decision_rule": {}})


def metrics(acc, n=100, seconds=30.0, half=None):
    return {"accuracy": acc, "n": n, "seconds_per_question": seconds, "half_width": half}


def results(iq, kq, task="generation", thinking="off"):
    out = {"cells": {}, "build": "b1"}
    for quant, series in (("iq", iq), ("kquant", kq)):
        for r, m in enumerate(series, 1):
            out["cells"][f"{quant}{r}"] = {"cell": {"pair": "p", "quant": quant, "task": task,
                                                    "thinking": thinking, "repeat": r},
                                           "metrics": m}
    return out


@pytest.fixture
def one_pair(config):
    return {**config, "pairs": [{"id": "p", "thinking": ["off"], "tasks": ["generation"]}]}


def verdict(config, iq, kq):
    return exp.judge(config, results(iq, kq), "p", "generation", "off")["verdict"]


def test_iq_is_worse_when_the_intervals_are_three_points_apart(one_pair):
    iq = [metrics(60, half=2), metrics(61, half=2), metrics(59, half=2)]
    kq = [metrics(70, half=2), metrics(71, half=2), metrics(69, half=2)]
    assert verdict(one_pair, iq, kq) == "worse"


def test_iq_is_worse_when_it_is_twenty_percent_slower_in_two_of_three_repeats(one_pair):
    kq = [metrics(70, half=5, seconds=30)] * 3
    slow = [metrics(70, half=5, seconds=40), metrics(70, half=5, seconds=37),
            metrics(70, half=5, seconds=30)]
    assert verdict(one_pair, slow, kq) == "worse"
    one = [metrics(70, half=5, seconds=40), metrics(70, half=5, seconds=31),
           metrics(70, half=5, seconds=30)]
    assert verdict(one_pair, one, kq) == "inconclusive"


def test_equal_builds_are_no_worse_and_a_short_series_is_incomplete(one_pair):
    same = [metrics(70, half=5), metrics(71, half=5), metrics(69, half=5)]
    assert verdict(one_pair, same, same) == "no worse"
    assert verdict(one_pair, same[:2], same) == "incomplete"


def test_the_interval_is_the_wider_of_item_and_repeat_spread():
    mean, _low, high = exp.interval([60.0, 70.0, 80.0], 1.0)
    assert mean == 70 and high - mean == pytest.approx(4.303 * 10 / 3 ** 0.5, rel=1e-3)
    assert exp.interval([60.0, 60.0, 60.0], 3.0) == (60.0, 57.0, 63.0)
    assert exp.wilson_half_width(50, 100) == pytest.approx(9.6, abs=0.3)


def test_speed_is_judged_on_decode_rate(config):
    def speed(tps):
        return {"speed": {"512": {"decode_tps": tps}, "4096": {"decode_tps": tps}}}

    fast = results([speed(20)] * 3, [speed(30)] * 3, task="speed")
    got = exp.judge(config, fast, "p", "speed", "off")
    assert got["verdict"] == "worse"
    even = results([speed(29)] * 3, [speed(30)] * 3, task="speed")
    assert exp.judge(config, even, "p", "speed", "off")["verdict"] == "no worse"


def test_the_table_names_each_group_and_the_build(one_pair):
    same = [metrics(70, half=5)] * 3
    text = exp.table(exp.analyse(one_pair, results(same, same)))
    assert "Build: b1" in text and "| p | generation | off |" in text and "no worse" in text
    json.dumps(exp.analyse(one_pair, results(same, same)))
