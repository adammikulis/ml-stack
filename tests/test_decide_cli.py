"""``ml-stack-decide`` against a fake chat server."""

from __future__ import annotations

import json

import pytest
from decide_fakes import logprob_handler

from ml_stack.decide.cases import read_cases
from ml_stack.decide_cli import main


def leaning(user: str) -> dict[str, float]:
    destructive = any(w in user for w in ("delete", "rm -rf", "DROP", "push", "transfer"))
    return {"A": 0.1, "B": 0.1, "C": 0.8} if destructive else {"A": 0.8, "B": 0.1, "C": 0.1}


@pytest.fixture
def url(server):
    return server(logprob_handler(leaning)).base_url


def test_ask_prints_the_choice_and_every_probability(url, capsys):
    code = main(["ask", "Is it risky?", "--state", "delete everything", "--option", "safe",
                 "--option", "mild=small change", "--option", "risky", "--url", url,
                 "--backend", "logprob"])
    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines()[0].startswith("risky  (confidence 0.800, logprob")
    assert "safe" in out and "mild" in out


def test_ask_json_and_the_abstain_flag(url, capsys):
    main(["ask", "q", "--state", "ls", "--option", "a", "--option", "b", "--url", url,
          "--backend", "logprob", "--abstain-below", "0.95", "--json"])
    got = json.loads(capsys.readouterr().out)
    assert got["abstained"] is True and got["choice"] == "a"


def test_ask_reads_the_state_from_a_file(url, tmp_path, capsys):
    f = tmp_path / "s.txt"
    f.write_text("please delete it")
    main(["ask", "q", "--state", f"@{f}", "--option", "a", "--option", "b", "--option", "c",
          "--url", url, "--backend", "logprob", "--json"])
    assert json.loads(capsys.readouterr().out)["choice"] == "c"


def test_a_backend_that_is_down_is_an_error_line_and_exit_two(capsys):
    code = main(["ask", "q", "--state", "s", "--option", "a", "--option", "b", "--url",
                 "http://127.0.0.1:9", "--backend", "logprob"])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_export_then_check_round_trips_the_guard_cases(tmp_path, capsys):
    path = tmp_path / "guards.jsonl"
    assert main(["export-cases", "--out", str(path)]) == 0
    assert len(read_cases(path)) >= 300
    assert main(["check-cases", str(path)]) == 0
    out = capsys.readouterr().out
    assert "injected" in out and "reversible" in out


def test_eval_writes_a_report_per_backend(url, tmp_path, capsys):
    out = tmp_path / "r.json"
    code = main(["eval", "guards", "--limit", "20", "--url", url, "--backend", "logprob",
                 "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text())[0]
    assert report["n"] == 20 and report["backend"] == "logprob"
    assert "acc=" in capsys.readouterr().out


def test_calibrate_fits_on_some_groups_and_scores_on_others(url, tmp_path, capsys):
    out = tmp_path / "cal.json"
    code = main(["calibrate", "guards", "--limit", "120", "--url", url, "--backend",
                 "logprob", "--isotonic", "--out", str(out), "--seed", "3"])
    assert code == 0
    saved = json.loads(out.read_text())
    assert saved["calibration"]["temperature"] > 0 and saved["calibration"]["knots"]
    assert saved["holdout"]["after"]["n"] == saved["holdout"]["before"]["n"] > 0
    assert len(saved["data_hash"]) == 64
    assert "held-out" in capsys.readouterr().out


def test_a_calibration_file_is_applied_by_the_other_commands(url, tmp_path, capsys):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"calibration": {"temperature": 10.0, "knots": [], "n": 1}}))
    main(["ask", "q", "--state", "ls", "--option", "a", "--option", "b", "--option", "c",
          "--url", url, "--backend", "logprob", "--calibration", str(cal), "--json"])
    assert json.loads(capsys.readouterr().out)["confidence"] < 0.5


def test_fetch_without_yes_downloads_nothing(capsys):
    assert main(["fetch"]) == 0
    out = capsys.readouterr().out
    assert "Apache-2.0" in out and "nothing downloaded" in out


def test_fetch_names_a_pinned_base_and_downloads_nothing_without_yes(capsys):
    assert main(["fetch", "--base", "gemma-4-e2b"]) == 0
    out = capsys.readouterr().out
    assert "gemma-4-e2b: 4 files, 10.28 GB" in out and "nothing downloaded" in out
