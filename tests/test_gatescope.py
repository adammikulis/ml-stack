"""`scripts/gatescope.py` lets the gate stop when a diff reaches none of its checks, and only then.

Each case builds a small repository with a `0.2dev` branch, so the decision comes from real files
and a real `git diff`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import gatescope  # noqa: E402

TREE = {
    "src/poolhouse/__init__.py": "",
    "src/poolhouse/mod.py": "VALUE = 1\n",
    "tests/test_mod.py": "from poolhouse import mod\n\n\ndef test_it():\n    assert mod.VALUE\n",
    "HANDOFF.md": "# Handoff\n",
    "docs/guide.md": "A guide nothing tests.\n",
    "docs/commands.md": "generated\n",
}


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", *args],
                   cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    for rel, text in TREE.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    git(tmp_path, "init", "-q", "-b", "0.2dev")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "checkout", "-q", "-b", "work")
    return tmp_path


def edit(root: Path, rel: str, text: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)
    git(root, "add", rel)
    git(root, "commit", "-q", "-m", f"edit {rel}")


def test_a_change_to_prose_no_test_names_does_not_reach_the_gate(repo):
    edit(repo, "HANDOFF.md", "# Handoff\nmore\n")
    edit(repo, "docs/guide.md", "A longer guide.\n")
    assert "reaches no test" in gatescope.unreached(repo, "0.2dev")


def test_a_source_change_reaches_the_gate(repo):
    edit(repo, "src/poolhouse/mod.py", "VALUE = 2\n")
    assert gatescope.unreached(repo, "0.2dev") == ""


def test_prose_next_to_a_source_change_still_runs_the_gate(repo):
    edit(repo, "HANDOFF.md", "# Handoff\nmore\n")
    edit(repo, "src/poolhouse/mod.py", "VALUE = 2\n")
    assert gatescope.unreached(repo, "0.2dev") == ""


def test_prose_a_test_names_reaches_the_gate(repo):
    edit(repo, "tests/test_mod.py", TREE["tests/test_mod.py"] + "\n\ndef test_doc():\n    assert 'docs/guide.md'\n")
    edit(repo, "docs/guide.md", "Changed.\n")
    assert gatescope.unreached(repo, "0.2dev") == ""


def test_a_clean_tree_and_an_undiffable_base_run_the_gate(repo):
    assert gatescope.unreached(repo, "0.2dev") == ""
    edit(repo, "HANDOFF.md", "# Handoff\nmore\n")
    assert gatescope.unreached(repo, "no-such-branch") == ""


def load_runner():
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader

    loader = SourceFileLoader("scripts_test_runner", str(REPO / "scripts" / "test"))
    module = module_from_spec(spec_from_loader("scripts_test_runner", loader))
    sys.modules["scripts_test_runner"] = module
    loader.exec_module(module)
    return module


def test_the_gate_stops_for_a_diff_that_reaches_nothing_and_full_runs_it_anyway(monkeypatch, capsys):
    from argparse import Namespace

    runner = load_runner()
    ran = []
    monkeypatch.setattr(runner.gatescope, "unreached", lambda root, base: "the change reaches no test (x)")
    monkeypatch.setattr(runner, "gate_steps", lambda workers, rest: ran.append("steps") or [])
    assert runner.gate(Namespace(full=False, base="0.2dev", workers=0, artifact_output=False), []) == 0
    assert ran == [] and "gate: skipped" in capsys.readouterr().out
    assert runner.gate(Namespace(full=True, base="0.2dev", workers=0, artifact_output=False), []) == 0
    assert ran == ["steps"]
