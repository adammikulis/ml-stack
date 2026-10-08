"""scripts/worktrees flags a branch over the hygiene limits; weakened-assertions flags a weaker guard test."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def load(name: str, where: Path):
    loader = importlib.machinery.SourceFileLoader(name, str(where))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


worktrees = load("_ml_stack_worktrees", REPO / "scripts" / "worktrees")
hook = load("_ml_stack_weakened", REPO / "scripts" / "hooks" / "weakened-assertions")
HOOK = REPO / "scripts" / "hooks" / "weakened-assertions"


@pytest.mark.parametrize("ahead,behind,age,expect", [
    (10, 0, 0.0, 0), (11, 0, 0.0, 1), (31, 0, 0.0, 1), (0, 20, 0.0, 0), (0, 21, 0.0, 1),
    (0, 1, 1.5, 1), (0, 0, 9.0, 0), (5, 1, 0.5, 0)])
def test_flags_follow_the_limits(ahead, behind, age, expect) -> None:
    assert len(worktrees.flags_for(ahead, behind, age)) == expect


def test_over_thirty_ahead_names_the_split() -> None:
    assert "split" in worktrees.flags_for(31, 0, 0.0)[0]


def sh(cwd: Path, *args: str, **kw) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", **kw.pop("env", {})}
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, check=False, **kw)


def commit(cwd: Path, name: str, msg: str = "c") -> None:
    (cwd / name).write_text(name)
    sh(cwd, "git", "add", name)
    assert sh(cwd, "git", "commit", "-qm", msg, "--no-verify").returncode == 0


def test_worktrees_reports_a_branch_over_ten_commits_and_exits_one(tmp_path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    sh(main, "git", "init", "-q", "-b", "0.2dev")
    commit(main, "base")
    side = tmp_path / "side"
    sh(main, "git", "worktree", "add", "-q", "-b", "topic", str(side))
    out = sh(side, sys.executable, str(REPO / "scripts" / "worktrees"))
    assert out.returncode == 0, out.stdout
    for i in range(11):
        commit(side, f"f{i}")
    out = sh(side, sys.executable, str(REPO / "scripts" / "worktrees"))
    assert out.returncode == 1
    assert "FLAG topic  +11 -0" in out.stdout
    assert "commits>10" in out.stdout


BEFORE = "def test_denied():\n    assert deny()\n    assert other()\n"


def diff_for(path: str, removed: list[str], added: list[str]) -> str:
    body = "".join(f"-{x}\n" for x in removed) + "".join(f"+{x}\n" for x in added)
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n{body}"


def found(path: str, removed: list[str], added: list[str], before: str = "") -> list[str]:
    return hook.weakened(diff_for(path, removed, added), lambda p: before)


def test_a_deleted_assert_in_a_guard_test_is_reported() -> None:
    out = found("tests/test_bash_guard.py", ["    assert other()"], [])
    assert out == ["tests/test_bash_guard.py: 1 fewer assertions"]


def test_an_assert_replaced_by_pass_is_reported() -> None:
    out = found("tests/test_redteam_x.py", ["    assert deny()"], ["    pass"])
    assert "replaced by pass" in out[0]


def test_a_removed_test_function_is_reported() -> None:
    out = found("tests/test_authority.py", ["def test_denied():"], [])
    assert "test function test_denied removed" in out[0]


@pytest.mark.parametrize("line", ["@pytest.mark.skip", "@pytest.mark.xfail", "    pytest.skip('x')",
                                  "@pytest.mark.skipif(True, reason='x')"])
def test_an_added_skip_is_reported(line) -> None:
    assert found("tests/test_auth_login.py", [], [line])


def test_an_assert_reworded_in_place_is_not_reported() -> None:
    assert found("tests/test_guard.py", ["    assert a"], ["    assert a == 1"]) == []


def test_an_ordinary_test_file_is_not_covered() -> None:
    assert found("tests/test_other.py", ["    assert a"], []) == []


def test_a_redteam_marked_file_is_covered_whatever_it_is_named() -> None:
    marked = "import pytest\npytestmark = pytest.mark.redteam\ndef test_a():\n    assert 1\n"
    assert found("tests/test_other.py", ["    assert 1"], [], before=marked)


def test_the_commit_message_line_accepts_it(tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    sh(repo, "git", "init", "-q")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_guard.py").write_text(BEFORE)
    sh(repo, "git", "add", "tests")
    sh(repo, "git", "commit", "-qm", "base", "--no-verify")
    (repo / "tests" / "test_guard.py").write_text("def test_denied():\n    assert deny()\n")
    sh(repo, "git", "add", "tests")
    plain = tmp_path / "plain"
    plain.write_text("chore: weaken\n")
    signed = tmp_path / "signed"
    signed.write_text("chore: weaken\n\nReviewed-by-second: someone\n")
    refused = sh(repo, sys.executable, str(HOOK), "--staged", "--message", str(plain))
    assert refused.returncode == 1
    assert "1 fewer assertions" in refused.stderr
    assert sh(repo, sys.executable, str(HOOK), "--staged", "--message", str(signed)).returncode == 0
