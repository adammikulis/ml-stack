"""`ml_stack.suite`: a measurement run over several seeds, and what its record carries.

Every suite here is invented and counts rather than computes, so the tests measure the
harness and not a model. Nothing reads or writes outside ``tmp_path``.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from ml_stack import suite as suites
from ml_stack.lock import Busy, only_one


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch):
    """Each test registers into a registry of its own."""
    monkeypatch.setattr(suites, "_SUITES", {})


@pytest.fixture()
def lock(tmp_path):
    return tmp_path / "suite.lock"


def counting(**over):
    """A suite that returns the seed and a constant, so every number is predictable."""
    def measure(*, backend, seed, width=10):
        return {"width": float(width), "seed": float(seed)}
    return suites.register("counting", "the seed and the width", **over)(measure)


# ------------------------------------------------------------------------ the registry


def test_a_suite_is_registered_once_and_listed_with_what_it_measures():
    counting()
    assert suites.known() == {"counting": "the seed and the width"}
    assert suites.registered("counting").about == "the seed and the width"
    with pytest.raises(ValueError, match="already registered"):
        counting()


def test_an_unknown_suite_names_what_there_is_instead():
    counting()
    with pytest.raises(KeyError, match="counting"):
        suites.registered("nothing-like-it")


def test_a_suite_refuses_a_backend_it_does_not_declare(lock):
    counting(backends=("counting",))
    assert suites.run("counting", backend="counting", seeds=(0,), lock=lock).backend == "counting"
    with pytest.raises(ValueError, match="runs on counting"):
        suites.run("counting", backend="other", seeds=(0,), lock=lock)


def test_no_seeds_is_refused_because_one_run_measures_no_spread(lock):
    counting()
    with pytest.raises(ValueError, match="at least one seed"):
        suites.run("counting", backend="counting", seeds=(), lock=lock)


# --------------------------------------------------------------------- what a run says


def test_every_metric_carries_its_mean_spread_and_how_many_seeds_made_it(lock):
    counting()
    out = suites.run("counting", backend="counting", seeds=(0, 2, 4), width=7, lock=lock)
    assert out.metrics["width"] == {"mean": 7.0, "std": 0.0, "n": 3, "values": [7.0, 7.0, 7.0]}
    assert out.metrics["seed"]["mean"] == 2.0 and out.metrics["seed"]["n"] == 3
    assert out.metrics["seed"]["std"] == pytest.approx(2.0)
    assert out.seeds == [0, 2, 4] and out.config == {"width": 7}
    assert out.failures == [] and out.seconds > 0


def test_a_seed_that_raised_is_reported_and_left_out_of_the_mean(lock):
    """Averaging two of three numbers and calling it three is the failure this prevents."""
    def measure(*, backend, seed):
        if seed == 1:
            raise RuntimeError("this seed is unlucky")
        return {"value": 10.0}
    suites.register("flaky", "one seed always fails")(measure)

    out = suites.run("flaky", backend="counting", seeds=(0, 1, 2), lock=lock)
    assert out.metrics["value"]["n"] == 2, "what arrived"
    assert out.seeds == [0, 1, 2], "what was asked for"
    assert out.failures == ["seed 1: RuntimeError: this seed is unlucky"]


def test_the_reading_says_the_spread_the_failures_and_a_dirty_tree(lock):
    counting()
    out = suites.run("counting", backend="counting", seeds=(0, 1), lock=lock)
    out.dirty = True
    out.failures = ["seed 9: RuntimeError: nope"]
    out.busy_pct_at_start = 91.0
    said = out.said()
    assert "counting [counting]" in said and "a dirty tree" in said
    assert "seed" in said and "(n=2)" in said
    assert "91% busy" in said and "FAILED seed 9" in said


# ------------------------------------------------------------------- what is written down


def test_the_file_is_written_after_every_seed_not_only_at_the_end(tmp_path, lock):
    """A seed can run for hours; a run killed in its third must leave the two it paid for."""
    seen = []

    def measure(*, backend, seed):
        seen.append(sorted(p.name for p in (tmp_path / "out").glob("*.json")))
        return {"value": float(seed)}
    suites.register("slow", "records what was on disk when it started")(measure)

    suites.run("slow", backend="counting", seeds=(0, 1, 2), out_dir=tmp_path / "out", lock=lock)
    assert seen[0] == [], "nothing before the first seed"
    assert len(seen[1]) == 1 and len(seen[2]) == 1, "and one file from the seed before"

    [written] = (tmp_path / "out").glob("*.json")
    held = json.loads(written.read_text())
    assert held["metrics"]["value"]["n"] == 3 and held["suite"] == "slow"
    assert held["seeds"] == [0, 1, 2]


def test_two_configurations_of_one_suite_do_not_overwrite_each_other(tmp_path, lock):
    counting()
    for width, seeds in ((3, (0, 1)), (4, (0, 1)), (4, (0, 1, 2))):
        suites.run("counting", backend="counting", seeds=seeds, width=width,
                   out_dir=tmp_path / "out", lock=lock)
    names = sorted(p.name for p in (tmp_path / "out").glob("*.json"))
    assert len(names) == 3, names
    assert any("width_3" in n for n in names) and any("seeds_3" in n for n in names)


def test_the_name_says_the_suite_the_backend_and_the_commit():
    made = suites.file_name("counting", "torch", "abc1234", {"width": 7}, (0, 1))
    assert made.startswith("counting.torch.abc1234.width_7-seeds_2.")
    assert made.endswith(".json")


# ----------------------------------------------------------------------------- the rest


def test_two_runs_take_turns_rather_than_timing_each_other(lock):
    counting()
    with only_one(lock):
        with pytest.raises(Busy):
            suites.run("counting", backend="counting", seeds=(0,), wait=False, lock=lock)


def test_provenance_reads_the_commit_and_whether_the_tree_was_edited(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*words):
        subprocess.run(["git", *words], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "nobody@example.invalid")
    git("config", "user.name", "Nobody")
    (repo / "a.txt").write_text("one")
    git("add", "a.txt")
    git("commit", "-qm", "one")

    sha, dirty = suites.provenance(repo)
    assert len(sha) >= 7 and not dirty
    (repo / "a.txt").write_text("two")
    assert suites.provenance(repo) == (sha, True)


def test_a_backend_that_reports_no_peak_is_zero_rather_than_an_error():
    assert suites.peak_memory_bytes("counting") == 0


def test_the_busy_reading_is_zero_when_no_vendor_tool_answers(monkeypatch):
    monkeypatch.setattr("ml_stack.fleet.telemetry.gpu_telemetry", lambda: {})
    assert suites.busy_pct() == 0.0
    monkeypatch.setattr("ml_stack.fleet.telemetry.gpu_telemetry",
                        lambda: {"gpu_util_pct": 73.5})
    assert suites.busy_pct() == 73.5


# ----------------------------------------------------------------------- the command

def a_module(tmp_path, monkeypatch, text):
    """A module on the path that registers a suite when it is imported."""
    (tmp_path / "invented_suites.py").write_text(text, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import sys
    sys.modules.pop("invented_suites", None)
    return "invented_suites"


MODULE = '''
from ml_stack.suite import register

@register("weighing", "the width it was given")
def weighing(*, backend, seed, width=10, tight=False):
    return {"width": float(width), "tight": float(tight)}
'''


def test_list_names_every_suite_a_module_registered(tmp_path, monkeypatch, capsys):
    name = a_module(tmp_path, monkeypatch, MODULE)
    assert suites.main(["--import", name, "list"]) == 0
    assert "weighing  the width it was given" in capsys.readouterr().out


def test_list_with_nothing_registered_says_how_to_register_one(capsys):
    assert suites.main(["list"]) == 1
    assert "--import" in capsys.readouterr().err


def test_run_takes_its_seeds_and_its_settings_and_writes_the_result(tmp_path, monkeypatch,
                                                                    capsys):
    name = a_module(tmp_path, monkeypatch, MODULE)
    monkeypatch.setattr(suites, "LOCK", tmp_path / "suite.lock")
    code = suites.main(["--import", name, "run", "weighing", "--backend", "counting",
                        "--seed", "0", "--seed", "1", "--out", str(tmp_path / "out"),
                        "--set", "width=512", "--set", "tight=true"])
    said = capsys.readouterr().out
    assert code == 0 and "weighing [counting]" in said
    assert "512.00000" in said and "(n=2)" in said
    [written] = (tmp_path / "out").glob("*.json")
    held = json.loads(written.read_text())
    assert held["config"] == {"width": 512, "tight": True}
    assert held["metrics"]["tight"]["mean"] == 1.0


def test_run_says_what_went_wrong_rather_than_raising(tmp_path, monkeypatch, capsys):
    name = a_module(tmp_path, monkeypatch, MODULE)
    monkeypatch.setattr(suites, "LOCK", tmp_path / "suite.lock")
    assert suites.main(["--import", name, "run", "nothing-like-it"]) == 2
    assert "no suite called" in capsys.readouterr().err
    assert suites.main(["--import", name, "run", "weighing", "--set", "width"]) == 2
    assert "takes k=v" in capsys.readouterr().err


def test_a_failed_seed_makes_the_command_exit_nonzero(tmp_path, monkeypatch, capsys):
    name = a_module(tmp_path, monkeypatch, '''
from ml_stack.suite import register

@register("flaky", "fails on the second seed")
def flaky(*, backend, seed):
    if seed == 1:
        raise RuntimeError("this seed is unlucky")
    return {"value": 1.0}
''')
    monkeypatch.setattr(suites, "LOCK", tmp_path / "suite.lock")
    code = suites.main(["--import", name, "run", "flaky", "--backend", "counting",
                        "--seed", "0", "--seed", "1"])
    assert code == 1 and "FAILED seed 1" in capsys.readouterr().out


def test_a_setting_reads_as_the_number_or_boolean_it_looks_like():
    assert suites._value("512") == 512 and isinstance(suites._value("512"), int)
    assert suites._value("0.5") == 0.5
    assert suites._value("TRUE") is True and suites._value("false") is False
    assert suites._value("q8_0") == "q8_0"
