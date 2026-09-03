"""Standard benchmark sets through lm-evaluation-harness, with the harness faked.

Every fixture is invented and lives in ``tmp_path``: the harness is a recorded results dict in
its own shape, no dataset is downloaded, no model is served and nothing reaches a port.
"""

from __future__ import annotations

import json
import os

import pytest

import ml_stack.graph.bench as bench
from ml_stack.graph.bench import standard

URL = "http://127.0.0.1:1/v1/chat/completions"


def recorded(task: str, n: int, original: int, **metrics) -> dict:
    """One task's results in the harness's own shape (``results[task]["metric,filter"]``)."""
    return {
        "results": {task: {"alias": task, "sample_len": n, **metrics}},
        "group_subtasks": {task: []},
        "configs": {task: {"task": task}},
        "versions": {task: 4.0},
        "n-shot": {task: 0},
        "n-samples": {task: {"original": original, "effective": n}},
    }


@pytest.fixture
def faked(monkeypatch):
    """`standard._evaluate` answering from a script keyed on the harness task, recording
    every call's kwargs; `_version` pinned."""
    calls, script = [], {}

    def evaluate(**kwargs):
        calls.append(kwargs)
        return script[kwargs["tasks"][0]]

    monkeypatch.setattr(standard, "_evaluate", evaluate)
    monkeypatch.setattr(standard, "_version", lambda: "0.4.99")
    return calls, script


# -- the sets ----------------------------------------------------------------------------

def test_every_short_name_maps_to_a_harness_task_and_the_metric_it_reports():
    assert standard.SETS["gsm8k"].task == "gsm8k_cot_llama"
    assert standard.SETS["gsm8k"].key == "exact_match,strict-match"
    assert standard.SETS["mmlu_pro"].task == "mmlu_pro"
    assert standard.SETS["mmlu_pro"].key == "exact_match,custom-extract"
    assert standard.SETS["ifeval"].task == "ifeval"
    assert standard.SETS["ifeval"].key == "prompt_level_strict_acc,none"
    assert standard.SETS["humaneval"].task == "humaneval_instruct"
    assert standard.SETS["humaneval"].key == "pass@1,create_test"
    assert standard.SETS["humaneval"].unsafe is True
    for one in standard.SETS.values():
        assert one.max_gen_toks >= 4096


# -- --dry-run ---------------------------------------------------------------------------

def test_dry_run_prints_the_harness_kwargs_and_writes_nothing(tmp_path, monkeypatch, capsys):
    def never(**kwargs):
        raise AssertionError("the harness was called on a dry run")

    monkeypatch.setattr(standard, "_evaluate", never)
    out = tmp_path / "quill.json"
    assert standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "gsm8k,ifeval",
                          "--limit", "5", "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    said = capsys.readouterr().out
    plan = json.loads(said)
    assert [p["set"] for p in plan] == ["gsm8k", "ifeval"]
    first = plan[0]["simple_evaluate"]
    assert first["model"] == "local-chat-completions"
    assert first["model_args"]["base_url"] == URL
    assert first["model_args"]["model"] == "quill-2b"
    assert first["model_args"]["num_concurrent"] == 1
    assert first["model_args"]["tokenized_requests"] is False
    assert first["model_args"]["max_retries"] >= 1
    assert first["tasks"] == ["gsm8k_cot_llama"]
    assert first["limit"] == 5
    assert first["apply_chat_template"] is True
    assert first["gen_kwargs"]["max_gen_toks"] >= 4096
    assert first["confirm_run_unsafe_code"] is False
    assert "chat_template_kwargs" not in first["gen_kwargs"]
    assert not (bench.HOME / "measuring.lock").exists(), "a dry run takes no lock"


# -- a run -------------------------------------------------------------------------------

def test_a_run_writes_one_json_in_the_comparison_shape(tmp_path, faked, monkeypatch):
    calls, script = faked
    script["gsm8k_cot_llama"] = recorded(
        "gsm8k_cot_llama", 5, 1319, **{"exact_match,strict-match": 0.8,
                                        "exact_match_stderr,strict-match": 0.2,
                                        "exact_match,flexible-extract": 1.0,
                                        "exact_match_stderr,flexible-extract": 0.0})
    script["ifeval"] = recorded(
        "ifeval", 5, 541, **{"prompt_level_strict_acc,none": 0.6,
                             "prompt_level_strict_acc_stderr,none": 0.24,
                             "inst_level_strict_acc,none": 0.7,
                             "prompt_level_loose_acc,none": 0.8,
                             "inst_level_loose_acc,none": 0.9})
    ticks = iter([100.0, 112.5, 200.0, 203.25])
    monkeypatch.setattr(standard, "_clock", lambda: next(ticks))
    out = tmp_path / "quill.json"

    assert standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "gsm8k,ifeval",
                          "--limit", "5", "--out", str(out), "--label", "quill tight",
                          "--no-think"]) == 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["label"] == "quill tight"
    assert doc["url"] == URL and doc["model"] == "quill-2b"
    assert doc["limit"] == 5 and doc["think"] is False
    assert doc["made_at"].endswith("Z") and "T" in doc["made_at"]
    assert doc["harness"] == {"name": "lm-eval", "version": "0.4.99"}
    assert doc["sets"]["gsm8k"] == {"score": 0.8, "metric": "exact_match",
                                    "filter": "strict-match", "stderr": 0.2, "n": 5,
                                    "seconds": 12.5, "task": "gsm8k_cot_llama"}
    assert doc["sets"]["ifeval"] == {"score": 0.6, "metric": "prompt_level_strict_acc",
                                     "filter": "none", "stderr": 0.24, "n": 5,
                                     "seconds": 3.25, "task": "ifeval"}
    assert list(doc["sets"]) == ["gsm8k", "ifeval"]
    assert [c["tasks"] for c in calls] == [["gsm8k_cot_llama"], ["ifeval"]]
    assert all(c["limit"] == 5 for c in calls)


def test_no_limit_scores_the_whole_set_and_records_what_was_scored(tmp_path, faked):
    calls, script = faked
    script["ifeval"] = recorded("ifeval", 541, 541,
                                **{"prompt_level_strict_acc,none": 0.84})
    out = tmp_path / "all.json"
    assert standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "ifeval",
                          "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["limit"] is None
    assert doc["sets"]["ifeval"]["n"] == 541
    assert doc["sets"]["ifeval"]["stderr"] is None
    assert calls[0]["limit"] is None
    assert doc["label"] == "quill-2b", "the label falls back to the model"


def test_mmlu_pro_is_a_group_whose_n_is_the_sum_of_its_subjects(tmp_path, faked):
    calls, script = faked
    script["mmlu_pro"] = {
        "results": {
            "mmlu_pro": {"alias": "mmlu_pro", "sample_len": None,
                         "exact_match,custom-extract": 0.5,
                         "exact_match_stderr,custom-extract": 0.1},
            "mmlu_pro_biology": {"alias": " - biology", "sample_len": 3,
                                 "exact_match,custom-extract": 1.0},
            "mmlu_pro_math": {"alias": " - math", "sample_len": 3,
                              "exact_match,custom-extract": 0.0},
        },
        "groups": {"mmlu_pro": {"exact_match,custom-extract": 0.5}},
        "group_subtasks": {"mmlu_pro": ["mmlu_pro_biology", "mmlu_pro_math"]},
        "n-samples": {"mmlu_pro_biology": {"original": 717, "effective": 3},
                      "mmlu_pro_math": {"original": 1351, "effective": 3}},
    }
    out = tmp_path / "m.json"
    assert standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "mmlu_pro",
                          "--limit", "3", "--out", str(out)]) == 0
    got = json.loads(out.read_text(encoding="utf-8"))["sets"]["mmlu_pro"]
    assert got["score"] == 0.5 and got["n"] == 6 and got["filter"] == "custom-extract"


# -- thinking ----------------------------------------------------------------------------

def test_the_think_switch_rides_in_gen_kwargs_and_its_absence_is_the_server_default(
        tmp_path, faked):
    calls, script = faked
    script["ifeval"] = recorded("ifeval", 1, 541, **{"prompt_level_strict_acc,none": 1.0})
    base = ["--url", URL, "--model", "quill-2b", "--tasks", "ifeval", "--limit", "1"]

    out = tmp_path / "a.json"
    assert standard.main([*base, "--out", str(out), "--no-think"]) == 0
    assert calls[-1]["gen_kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "think_end_token" not in calls[-1]["model_args"]
    assert json.loads(out.read_text())["think"] is False

    out = tmp_path / "b.json"
    assert standard.main([*base, "--out", str(out), "--think"]) == 0
    assert calls[-1]["gen_kwargs"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert calls[-1]["model_args"]["think_end_token"] == "</think>"
    assert json.loads(out.read_text())["think"] is True

    out = tmp_path / "c.json"
    assert standard.main([*base, "--out", str(out)]) == 0
    assert "chat_template_kwargs" not in calls[-1]["gen_kwargs"]
    assert json.loads(out.read_text())["think"] == "server default"


# -- humaneval ---------------------------------------------------------------------------

def test_humaneval_confirms_unsafe_code_and_lets_code_eval_run(tmp_path, faked, monkeypatch):
    calls, script = faked
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    seen = []

    def evaluate(**kwargs):
        seen.append(os.environ.get("HF_ALLOW_CODE_EVAL"))
        calls.append(kwargs)
        return recorded("humaneval_instruct", 164, 164, **{"pass@1,create_test": 0.75})

    monkeypatch.setattr(standard, "_evaluate", evaluate)
    out = tmp_path / "h.json"
    assert standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "humaneval",
                          "--out", str(out)]) == 0
    assert calls[0]["confirm_run_unsafe_code"] is True
    assert seen == ["1"]
    got = json.loads(out.read_text(encoding="utf-8"))["sets"]["humaneval"]
    assert got == {"score": 0.75, "metric": "pass@1", "filter": "create_test", "stderr": None,
                   "n": 164, "seconds": got["seconds"], "task": "humaneval_instruct"}


# -- the lock ----------------------------------------------------------------------------

def test_no_queue_is_refused_with_3_while_another_measurement_holds_the_lock(
        tmp_path, faked, capsys):
    from ml_stack.lock import only_one

    calls, script = faked
    script["ifeval"] = recorded("ifeval", 1, 541, **{"prompt_level_strict_acc,none": 1.0})
    out = tmp_path / "held.json"
    with only_one(bench.HOME / "measuring.lock", wait=False):
        assert standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "ifeval",
                              "--limit", "1", "--out", str(out), "--no-queue"]) == 3
    assert calls == [] and not out.exists()
    assert "measuring.lock" in capsys.readouterr().err

    assert standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "ifeval",
                          "--limit", "1", "--out", str(out), "--no-queue"]) == 0
    assert out.exists() and len(calls) == 1


def test_the_lock_is_let_go_after_a_run_and_after_a_harness_failure(tmp_path, faked,
                                                                    monkeypatch):
    from ml_stack.lock import only_one

    calls, script = faked

    def failing(**kwargs):
        raise RuntimeError("the harness could not reach the model")

    monkeypatch.setattr(standard, "_evaluate", failing)
    out = tmp_path / "f.json"
    with pytest.raises(RuntimeError):
        standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "ifeval",
                       "--out", str(out)])
    assert not out.exists()
    with only_one(bench.HOME / "measuring.lock", wait=False):
        pass


# -- refusals ----------------------------------------------------------------------------

def test_a_set_the_module_does_not_know_is_refused_by_name(tmp_path, faked, capsys):
    calls, _ = faked
    code = standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "gsm8k,arc_easy",
                          "--out", str(tmp_path / "x.json")])
    assert code == 2 and calls == []
    err = capsys.readouterr().err
    assert "arc_easy" in err and "gsm8k, mmlu_pro, ifeval, humaneval" in err


def test_a_metric_the_harness_did_not_report_is_a_failure_not_a_zero(tmp_path, faked):
    calls, script = faked
    script["ifeval"] = recorded("ifeval", 1, 541, **{"inst_level_loose_acc,none": 1.0})
    with pytest.raises(standard.HarnessShape, match="prompt_level_strict_acc,none"):
        standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "ifeval",
                       "--limit", "1", "--out", str(tmp_path / "x.json")])


def test_the_default_out_lives_under_the_bench_home(tmp_path, faked, capsys):
    calls, script = faked
    script["ifeval"] = recorded("ifeval", 1, 541, **{"prompt_level_strict_acc,none": 1.0})
    assert standard.main(["--url", URL, "--model", "quill-2b", "--tasks", "ifeval",
                          "--limit", "1", "--label", "Quill/tight"]) == 0
    written = list((bench.HOME / "standard").glob("*.json"))
    assert len(written) == 1 and written[0].name.startswith("quill-tight-")
    assert str(written[0]) in capsys.readouterr().out
