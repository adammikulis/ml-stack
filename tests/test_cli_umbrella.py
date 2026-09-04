"""``ml-stack <word>`` runs ``ml-stack-<word>``: the ``git foo`` -> ``git-foo`` pattern."""

from __future__ import annotations

import sys
import tomllib
import types
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from ml_stack import cli

REPO = Path(__file__).resolve().parent.parent


def _fake_target(monkeypatch, name: str, calls: list, *, returns: int = 0,
                 about: str = "Counts quinces.") -> str:
    """A module on ``sys.modules`` whose ``main`` records its args; returns its ``module:attr``."""
    module = types.ModuleType(name)

    def main(argv=None):
        argv = list(argv or [])
        if argv == ["--help"]:
            print(f"usage: ml-stack-{name} [-h]\n\n{about}\n\noptions:\n  -h, --help")
            raise SystemExit(0)
        calls.append((name, argv))
        return returns

    module.main = main
    monkeypatch.setitem(sys.modules, name, module)
    return f"{name}:main"


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Three invented commands registered as entry points, and no pyproject to fall back on."""
    calls: list = []
    points = [
        EntryPoint("ml-stack-quince", _fake_target(monkeypatch, "quince", calls, returns=3),
                   "console_scripts"),
        EntryPoint("ml-stack-train", _fake_target(monkeypatch, "train", calls, about="Trains."),
                   "console_scripts"),
        EntryPoint("ml-stack-train-run",
                   _fake_target(monkeypatch, "train_run", calls, about="Runs one."),
                   "console_scripts"),
        EntryPoint("ml-stack", "ml_stack.cli:main", "console_scripts"),
        EntryPoint("unrelated-tool", "quince:main", "console_scripts"),
    ]
    monkeypatch.setattr(cli, "entry_points", lambda group: points)
    monkeypatch.setattr(cli, "PYPROJECT", tmp_path / "absent.toml")
    return calls


def test_a_word_runs_the_hyphenated_command_in_process(fake):
    assert cli.main(["quince", "--ripe", "2"]) == 3
    assert fake == [("quince", ["--ripe", "2"])]


def test_words_join_with_hyphens_longest_match_first(fake):
    assert cli.main(["train", "run", "--steps", "1"]) == 0
    assert cli.main(["train", "--dry-run"]) == 0
    assert fake == [("train_run", ["--steps", "1"]), ("train", ["--dry-run"])]


def test_the_targets_help_is_returned_unchanged(fake, capsys):
    with pytest.raises(SystemExit) as left:
        cli.main(["quince", "--help"])
    assert left.value.code == 0
    assert capsys.readouterr().out.startswith("usage: ml-stack-quince")


def test_an_unknown_word_exits_2_naming_the_nearest(fake, capsys):
    assert cli.main(["quinc"]) == 2
    err = capsys.readouterr().err
    assert "quinc" in err and "quince" in err
    assert fake == []


def test_list_prints_every_command_with_the_first_line_of_its_help(fake, capsys):
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "quince" in out and "Counts quinces." in out
    assert "train run" in out and "Runs one." in out
    assert "unrelated" not in out
    assert fake == []


def test_help_lists_the_subcommands(fake, capsys):
    with pytest.raises(SystemExit) as left:
        cli.main(["--help"])
    assert left.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    for word in ("quince", "train", "train run"):
        assert word in out


def test_bare_and_its_flags_reach_the_app(fake, monkeypatch):
    seen: list = []
    monkeypatch.setattr("ml_stack.fleet.launch.main", lambda argv: seen.append(argv) or 0)
    assert cli.main([]) == 0
    assert cli.main(["--port", "8771", "--no-browser"]) == 0
    assert seen == [[], ["--port", "8771", "--no-browser"]]
    assert fake == []


def test_an_uninstalled_checkout_dispatches_from_its_pyproject(monkeypatch, tmp_path):
    calls: list = []
    target = _fake_target(monkeypatch, "medlar", calls)
    (tmp_path / "pyproject.toml").write_text(
        f'[project.scripts]\nml-stack = "ml_stack.cli:main"\nml-stack-medlar = "{target}"\n')
    monkeypatch.setattr(cli, "entry_points", lambda group: [])
    monkeypatch.setattr(cli, "PYPROJECT", tmp_path / "pyproject.toml")
    assert cli.main(["medlar", "x"]) == 0
    assert calls == [("medlar", ["x"])]


def test_every_command_in_pyproject_resolves_to_a_main():
    table = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    words = cli.commands()
    for name, target in table.items():
        if not name.startswith(cli.PREFIX):
            continue
        word = name[len(cli.PREFIX):]
        assert word in words, f"{name} is not a subcommand of ml-stack"
        assert callable(cli.load(words[word])), f"{name} -> {target} has no main"
    assert "app" in words and "bench" in words and "train-run" in words


def test_help_lists_every_command_and_hands_a_named_one_its_own_help(capsys):
    from ml_stack import cli

    assert cli.help_main([]) == 0
    out = capsys.readouterr().out
    assert "bench" in out and "do" in out and "usage: ml-stack help" in out and "asks itself" not in out
    code = cli.help_main(["bench"])
    assert code == 0 and "usage: ml-stack-bench" in capsys.readouterr().out
    assert cli.main(["help", "do"]) == 0
    assert "usage: ml-stack-do" in capsys.readouterr().out
    assert cli.help_main(["benc"]) == 2
    assert "not a command" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.help_main(["--help"])
