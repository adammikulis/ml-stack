"""Nobody raises a budget by accident, and no agent raises one at all."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import gates  # noqa: E402

HOOK = REPO / "scripts" / "hooks" / "budgets-only-fall"


def scoreboard():
    """scripts/budgets loaded as a module."""
    where = REPO / "scripts" / "budgets"
    loader = importlib.machinery.SourceFileLoader("_ml_stack_budgets", str(where))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def one(path: str = "src/ml_stack/a.py", detail: str = "") -> gates.Finding:
    return gates.Finding(path, 1, detail)


@pytest.fixture
def tool(tmp_path, monkeypatch):
    """The scoreboard pointed at a budgets.json of its own, over a tree it is told about."""
    module = scoreboard()
    held = tmp_path / "budgets.json"
    held.write_text(json.dumps({"broad-excepts": 2, "print-calls": 0}, indent=2) + "\n",
                    encoding="utf-8")
    monkeypatch.setattr(module, "BUDGETS", held)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.gates, "run", lambda root: {
        "broad-excepts": [one("src/ml_stack/a.py"), one("src/ml_stack/b.py"),
                          one("src/ml_stack/c.py")],
        "print-calls": [],
    })
    monkeypatch.setattr(module.gates, "hard", set)
    monkeypatch.setattr(module.gates, "unrunnable", dict)
    monkeypatch.setattr(module.gates, "checkers", list)
    monkeypatch.setattr(module, "owners", lambda: {"broad-excepts": "", "print-calls": ""})
    monkeypatch.setattr(module, "descriptions",
                        lambda: {"broad-excepts": "a bare except", "print-calls": "a print"})
    return module, held


def test_an_agent_cannot_raise_a_budget_even_with_allow_increase(tool, monkeypatch, capsys):
    module, held = tool
    monkeypatch.setenv("CLAUDECODE", "1")
    before = held.read_text(encoding="utf-8")
    assert module.main(["--update", "--allow-increase"]) == 1
    assert held.read_text(encoding="utf-8") == before
    said = capsys.readouterr().out
    assert "broad-excepts: 2 allowed, 3 found -- up 1." in said
    assert "src/ml_stack/a.py:1" in said
    assert "No agent raises a budget" in said
    assert "repository owner, at his own terminal" in said


def test_a_person_at_a_terminal_can_raise_a_budget(tool, monkeypatch):
    module, held = tool
    monkeypatch.delenv("CLAUDECODE", raising=False)
    assert module.main(["--update", "--allow-increase"]) == 0
    assert json.loads(held.read_text(encoding="utf-8"))["broad-excepts"] == 3


def test_a_rise_without_the_flag_is_refused_at_a_terminal_too(tool, monkeypatch, capsys):
    module, held = tool
    monkeypatch.delenv("CLAUDECODE", raising=False)
    before = held.read_text(encoding="utf-8")
    assert module.main(["--update"]) == 1
    assert held.read_text(encoding="utf-8") == before
    assert "--allow-increase" in capsys.readouterr().out


def test_a_fall_is_recorded_whoever_runs_it(tool, monkeypatch):
    module, held = tool
    held.write_text(json.dumps({"broad-excepts": 9, "print-calls": 0}) + "\n",
                    encoding="utf-8")
    monkeypatch.setenv("CLAUDECODE", "1")
    assert module.main(["--update"]) == 0
    assert json.loads(held.read_text(encoding="utf-8"))["broad-excepts"] == 3


def test_the_table_prints_the_total_and_how_it_has_moved(tool, capsys):
    module, _ = tool
    assert module.main([]) == 1
    lines = capsys.readouterr().out.splitlines()
    total = [line for line in lines if line.startswith("total")]
    assert total, "the scoreboard prints no total"
    fields = total[0].split()
    assert fields[1:] == ["2", "3", "+1"]


def git(root: Path, *args: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                          check=True, env={**os.environ, **env})


WALLS = '''"""A checker that allows nothing, so the hook has a hard gate to read."""

NAME = "walls"
OWNER = ""
HARD = True


def describe() -> str:
    return "a wall"


def find(root):
    return []
'''


@pytest.fixture
def checkout(tmp_path):
    """A repository holding budgets.json, the hook, and the checkers the hook reads."""
    root = tmp_path / "repo"
    (root / "scripts" / "hooks").mkdir(parents=True)
    shutil.copytree(REPO / "scripts" / "gates", root / "scripts" / "gates",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "scripts" / "gates" / "walls.py").write_text(WALLS, encoding="utf-8")
    shutil.copy2(HOOK, root / "scripts" / "hooks" / HOOK.name)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    git(root, "config", "user.email", "nobody@example.invalid")
    git(root, "config", "user.name", "A Tester")
    (root / "budgets.json").write_text(
        json.dumps({"broad-excepts": 2, "print-calls": 0, "walls": 4}, indent=2) + "\n",
        encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "before")
    return root


def commit_with(root: Path, numbers: dict[str, int], **env) -> subprocess.CompletedProcess:
    """Stage that budgets.json and run the hook over it."""
    (root / "budgets.json").write_text(json.dumps(numbers, indent=2) + "\n",
                                       encoding="utf-8")
    git(root, "add", "budgets.json")
    kept = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    return subprocess.run([sys.executable, str(root / "scripts" / "hooks" / HOOK.name)],
                          cwd=root, capture_output=True, text=True, env={**kept, **env})


def test_a_hand_edited_rise_is_refused_at_the_commit(checkout):
    done = commit_with(checkout, {"broad-excepts": 40, "print-calls": 0, "walls": 4},
                       CLAUDECODE="1")
    assert done.returncode == 1
    assert "broad-excepts: 2 -> 40, up 38" in done.stderr
    assert "No agent raises a budget" in done.stderr


def test_a_metric_deleted_so_it_can_be_re_added_is_refused(checkout):
    done = commit_with(checkout, {"print-calls": 0, "walls": 4}, CLAUDECODE="1")
    assert done.returncode == 1
    assert "broad-excepts: 2 allowed, now absent" in done.stderr


def test_dropping_the_line_a_hard_gate_should_never_have_had_is_the_fix(checkout):
    done = commit_with(checkout, {"broad-excepts": 2, "print-calls": 0}, CLAUDECODE="1")
    assert done.returncode == 0, done.stderr


def test_a_fall_is_committed_without_a_word(checkout):
    done = commit_with(checkout, {"broad-excepts": 1, "print-calls": 0, "walls": 4},
                       CLAUDECODE="1")
    assert (done.returncode, done.stderr) == (0, "")


def test_a_person_raises_it_on_purpose_and_an_agent_still_cannot(checkout):
    raised = {"broad-excepts": 40, "print-calls": 0, "walls": 4}
    assert commit_with(checkout, raised, ML_STACK_BUDGET_RISE="yes").returncode == 0
    refused = commit_with(checkout, raised, ML_STACK_BUDGET_RISE="yes", CLAUDECODE="1")
    assert refused.returncode == 1
    assert "No agent raises a budget" in refused.stderr


def test_a_budgets_json_nobody_staged_is_not_this_hook_s_business(checkout):
    (checkout / "notes.txt").write_text("nothing to do with budgets\n", encoding="utf-8")
    git(checkout, "add", "notes.txt")
    kept = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    done = subprocess.run([sys.executable, str(checkout / "scripts" / "hooks" / HOOK.name)],
                          cwd=checkout, capture_output=True, text=True,
                          env={**kept, "CLAUDECODE": "1"})
    assert done.returncode == 0


def test_the_hook_reads_the_checkout_it_is_committing_in(checkout, tmp_path):
    """Installed as a symlink, the hook sits in one checkout and the commit is in another."""
    elsewhere = tmp_path / "elsewhere" / "hooks"
    elsewhere.mkdir(parents=True)
    shutil.copy2(HOOK, elsewhere / HOOK.name)
    kept = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    (checkout / "budgets.json").write_text(
        json.dumps({"broad-excepts": 2, "print-calls": 0}, indent=2) + "\n", encoding="utf-8")
    git(checkout, "add", "budgets.json")
    done = subprocess.run([sys.executable, str(elsewhere / HOOK.name)], cwd=checkout,
                          capture_output=True, text=True, env={**kept, "CLAUDECODE": "1"})
    assert done.returncode == 0, done.stderr


def test_the_branch_is_compared_against_a_revision_without_a_commit_hook(checkout):
    """The same refusal, reached the way CI reaches it."""
    (checkout / "budgets.json").write_text(
        json.dumps({"broad-excepts": 40, "print-calls": 0, "walls": 4}, indent=2) + "\n",
        encoding="utf-8")
    kept = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    done = subprocess.run(
        [sys.executable, str(checkout / "scripts" / "hooks" / HOOK.name),
         "--against", "HEAD"],
        cwd=checkout, capture_output=True, text=True, env=kept)
    assert done.returncode == 1
    assert "broad-excepts: 2 -> 40, up 38" in done.stderr
