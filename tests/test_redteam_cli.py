"""The command: checking the corpus, comparing two runs, and refusing what it does not know."""

from __future__ import annotations

from ml_stack.redteam.cli import main
from ml_stack.redteam.report import Attempt, Report


def saved(tmp_path, name: str, *wins: bool) -> str:
    report = Report({"model": "m"}, [Attempt("chat", "jailbreak-template", f"t{n}", won)
                                     for n, won in enumerate(wins)])
    path = tmp_path / name
    path.write_text(report.to_json(), encoding="utf-8")
    return str(path)


def test_the_corpus_command_passes_on_the_files_that_ship(capsys):
    assert main(["corpus"]) == 0
    assert "matches" in capsys.readouterr().out


def test_comparing_a_run_that_lost_ground_exits_one_and_names_the_attack(tmp_path, capsys):
    assert main(["compare", saved(tmp_path, "a.json", False, True),
                 saved(tmp_path, "b.json", True, True)]) == 1
    assert "regressed  chat / jailbreak-template / t0" in capsys.readouterr().out


def test_comparing_a_run_that_only_improved_exits_zero(tmp_path, capsys):
    assert main(["compare", saved(tmp_path, "a.json", True), saved(tmp_path, "b.json", False)]) == 0
    assert "fixed" in capsys.readouterr().out


def test_an_unknown_scenario_is_refused_before_anything_starts(capsys):
    assert main(["run", "--scenarios", "chat,nonsense"]) == 2
    assert "nonsense" in capsys.readouterr().err
