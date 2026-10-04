"""The hooks that keep agents out of the primary checkout, run as scripts with JSON on stdin."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from ml_stack import harnesshook, harnesspolicy

HOOKS = Path(__file__).resolve().parent.parent / "scripts" / "hooks"
BASH, EDIT = HOOKS / "claude-bash-guard", HOOKS / "claude-edit-guard"
START, COMMIT = HOOKS / "claude-subagent-start", HOOKS / "primary-only"
BLOCKED, ALLOWED = 2, 0
CLEAN = {k: v for k, v in os.environ.items()
         if k not in ("MLSTACK_GUARD", "CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE")}


def git(where: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(where), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A primary checkout on `dev` holding the opt-in marker, and a linked worktree on `work`."""
    primary = (tmp_path / "primary").resolve()
    (primary / "scripts" / "hooks").mkdir(parents=True)
    (primary / "scripts" / "hooks" / "primary-only").write_text("", encoding="utf-8")
    (primary / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "dev", str(primary)], check=True)
    git(primary, "add", "-A")
    git(primary, "commit", "-qm", "chore: start")
    inside = primary / ".claude" / "worktrees" / "inside"
    git(primary, "worktree", "add", "-q", "-b", "work", str(tmp_path / "work"))
    git(primary, "worktree", "add", "-q", "-b", "nested", str(inside))
    return primary, (tmp_path / "work").resolve(), inside.resolve()


def hook(script: Path, event: dict, **env: str) -> tuple[int, str]:
    done = subprocess.run([str(script)], input=json.dumps(event), text=True, capture_output=True,
                          env={**CLEAN, **env})
    assert done.returncode in (BLOCKED, ALLOWED), done.stderr
    return done.returncode, done.stderr if done.returncode == BLOCKED else done.stdout


def bash(command: str, cwd: Path, **env: str) -> int:
    code, said = hook(BASH, {"tool_name": "Bash", "cwd": str(cwd), "tool_input": {"command": command}}, **env)
    if code == BLOCKED:
        assert "git worktree add" in said or "pip install -e" in said or "--ff-only" in said, said
    return code


def edit(path: Path, cwd: Path, tool: str = "Write", key: str = "file_path", **env: str) -> int:
    code, said = hook(EDIT, {"tool_name": tool, "cwd": str(cwd),
                             "tool_input": {key: str(path), "content": "y = 2\n"}}, **env)
    if code == BLOCKED:
        assert "git worktree add -b" in said, said
    return code


@pytest.mark.parametrize("tool, key", [("Write", "file_path"), ("Edit", "file_path"),
                                       ("MultiEdit", "file_path"), ("NotebookEdit", "notebook_path")])
def test_a_write_into_the_primary_checkout_is_refused(repo, tool, key):
    primary, work, _ = repo
    assert edit(primary / "a.py", work, tool, key) == BLOCKED
    assert edit(primary / "new" / "deep" / "b.py", primary, tool, key) == BLOCKED


def test_a_write_in_any_other_tree_or_outside_the_repository_is_allowed(repo, tmp_path):
    primary, work, inside = repo
    assert edit(work / "a.py", work) == ALLOWED
    assert edit(inside / "a.py", primary) == ALLOWED
    assert edit(tmp_path / "scratch.txt", primary) == ALLOWED
    assert edit(Path.home() / ".claude" / "x.md", primary) == ALLOWED


def test_the_edit_guard_can_be_switched_off(repo):
    primary, work, _ = repo
    assert edit(primary / "a.py", work, MLSTACK_GUARD="off") == ALLOWED


@pytest.mark.parametrize("command", [
    "git add a.py", "git commit -m x", "git checkout work", "git switch work", "git reset --hard",
    "git restore a.py", "git stash", "git rebase work", "git cherry-pick abc", "git am p.patch",
    "git apply p.patch", "git merge work", "git merge --no-ff work", "git merge --ff-only a b",
    "git merge --ff-only origin/work:x", "FOO=1 git commit -m x", "echo hi && git add a.py",
    "git -c core.editor=true commit", "(git commit -m x)", "cd . && git stash",
])
def test_a_tree_changing_git_command_in_the_primary_checkout_is_refused(repo, command):
    primary, _, _ = repo
    assert bash(command, primary) == BLOCKED


@pytest.mark.parametrize("command", [
    "git merge --ff-only work", "git merge --ff-only work && git push origin dev",
    "git worktree remove ../work", "git worktree add -b x ../x dev", "git worktree prune",
    "git worktree list", "git branch -d work", "git fetch origin", "git status", "git log --oneline",
    "git diff HEAD", "echo 'git commit'", "grep -rn 'git add' docs/", "git branch --show-current",
])
def test_landing_and_reading_are_allowed_in_the_primary_checkout(repo, command):
    primary, _, _ = repo
    assert bash(command, primary) == ALLOWED


@pytest.mark.parametrize("command", [
    "git add a.py", "git commit -m x", "git reset --hard dev", "git rebase dev", "git stash",
    "git checkout -b more", "git merge dev",
])
def test_an_agent_in_its_own_worktree_may_do_all_of_it(repo, command):
    _, work, inside = repo
    assert bash(command, work) == ALLOWED
    assert bash(command, inside) == ALLOWED


def test_the_directory_a_compound_command_runs_in_is_followed(repo):
    primary, work, _ = repo
    assert bash(f"cd {work} && git commit -m x", primary) == ALLOWED
    assert bash(f"git -C {work} commit -m x", primary) == ALLOWED
    assert bash(f"cd {primary} && git commit -m x", work) == BLOCKED
    assert bash(f"git -C {primary} add a.py", work) == BLOCKED


def test_the_bash_guard_can_be_switched_off(repo):
    primary, _, _ = repo
    assert bash("git commit -m x", primary, MLSTACK_GUARD="off") == ALLOWED


@pytest.mark.parametrize("command", [
    "pip install -e .", "pip3 install -e '.[test]'", "uv pip install -e .", "pip install --editable .",
    "python3 -m pip install --editable=.", "FOO=1 pip install -e .",
])
def test_an_editable_install_of_a_worktree_is_refused(repo, command):
    _, work, inside = repo
    assert bash(command, work) == BLOCKED
    assert bash(command, inside) == BLOCKED


@pytest.mark.parametrize("command", [
    "pip install -e .", "pip install -e '.[test]'", "pip install requests", "pip install -r r.txt",
])
def test_an_install_that_points_at_the_primary_checkout_is_allowed(repo, command):
    primary, work, _ = repo
    assert bash(command, primary) == ALLOWED
    if "-e" not in command:
        assert bash(command, work) == ALLOWED
    assert bash(f"pip install -e {primary}", work) == ALLOWED


def test_a_repository_without_the_marker_is_left_alone(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    (other / "f.py").write_text("", encoding="utf-8")
    assert edit(other / "f.py", other) == ALLOWED
    assert bash("git commit -m x", other) == ALLOWED


def test_a_subagent_is_told_the_rule_and_the_development_branch(repo):
    primary, work, _ = repo
    code, said = hook(START, {"hook_event_name": "SubagentStart", "cwd": str(work)})
    context = json.loads(said)["hookSpecificOutput"]
    assert code == ALLOWED and context["hookEventName"] == "SubagentStart"
    assert "development branch dev" in context["additionalContext"]
    assert str(primary) in context["additionalContext"]


def commit(cwd: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(COMMIT)], cwd=cwd, text=True, capture_output=True, env={**CLEAN, **env})


def test_an_agent_cannot_commit_in_the_primary_checkout(repo):
    primary, work, _ = repo
    refused = commit(primary, CLAUDECODE="1")
    assert refused.returncode == 1 and "git worktree add -b" in refused.stderr
    assert commit(primary).returncode == 1
    assert commit(work, CLAUDECODE="1").returncode == 0
    assert commit(primary, CLAUDECODE="1", MLSTACK_GUARD="off").returncode == 0


def test_an_agent_cannot_commit_on_the_development_branch_in_another_tree(repo):
    _, work, _ = repo
    assert commit(work, CLAUDECODE="1").returncode == 0
    git(work, "checkout", "-q", "--ignore-other-worktrees", "dev")
    refused = commit(work, CLAUDECODE="1")
    assert refused.returncode == 1 and "the development branch" in refused.stderr


def test_a_person_at_a_terminal_commits_anywhere(repo):
    primary, _, _ = repo
    master, slave = os.openpty()
    try:
        done = subprocess.run([str(COMMIT)], cwd=primary, stdin=slave, stdout=slave,
                              stderr=subprocess.PIPE, env=CLEAN)
    finally:
        os.close(master)
        os.close(slave)
    assert done.returncode == 0


def test_the_harness_policy_applies_the_same_rule_to_codex_and_local_sessions(repo):
    primary, work, _ = repo
    patch = f"*** Begin Patch\n*** Update File: {primary / 'a.py'}\n*** End Patch"
    for name, args in [("Write", {"file_path": str(primary / "a.py")}),
                       ("apply_patch", {"input": patch}),
                       ("shell", {"command": ["git", "commit", "-m", "x"]}),
                       ("Bash", {"command": "git add a.py"})]:
        verdict = harnesspolicy.primary_decision(name, args, str(primary))
        assert verdict and verdict.action == "deny" and "git worktree add" in verdict.reason, name
    for name, args in [("Write", {"file_path": str(work / "a.py")}),
                       ("Bash", {"command": "git add a.py"})]:
        assert harnesspolicy.primary_decision(name, args, str(work)) is None


def test_the_harness_hook_denies_a_write_to_the_primary_checkout(repo):
    primary, work, _ = repo
    event = {"tool_name": "Write", "cwd": str(work), "tool_input": {"file_path": str(primary / "a.py")}}
    out = harnesshook.pre(event, harnesshook.Rail("plan-and-go", "t"))["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "git worktree add" in out["permissionDecisionReason"]
