"""The hooks that protect checkout and branch changes, run as scripts with JSON on stdin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ml_stack import harnesshook, harnesspolicy, worktreerules

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
    done = subprocess.run([sys.executable, str(script)], input=json.dumps(event), text=True, capture_output=True,
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
def test_a_write_into_the_primary_development_checkout_is_allowed(repo, tool, key):
    primary, work, _ = repo
    assert edit(primary / "a.py", work, tool, key) == ALLOWED
    assert edit(primary / "new" / "deep" / "b.py", primary, tool, key) == ALLOWED


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
    "git checkout work", "git switch work", "git reset --hard",
    "git restore a.py", "git stash", "git rebase work", "git cherry-pick abc", "git am p.patch",
    "git apply p.patch", "git merge work", "git merge --no-ff work", "git merge --ff-only a b",
    "git merge --ff-only origin/work:x", "cd . && git stash",
])
def test_a_tree_changing_git_command_in_the_primary_checkout_is_refused(repo, command):
    primary, _, _ = repo
    assert bash(command, primary) == BLOCKED


@pytest.mark.parametrize("command", [
    "git add a.py", "git commit -m x", "FOO=1 git commit -m x", "echo hi && git add a.py",
    "git -c core.editor=true commit", "(git commit -m x)",
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


LANDING_WITH_REDIRECTS = [
    "git merge --ff-only work 2>&1 | tail -1", "git merge --ff-only work >/dev/null 2>&1",
    "git merge --ff-only work &> out", "git merge --ff-only work &>out", "git merge --ff-only work 2>out",
    "git merge --ff-only work 2> out", "git merge --ff-only work > out 2>&1", "git merge --ff-only work >> out",
    "git merge --ff-only work 2>/dev/null; git status", "git merge --ff-only work < /dev/null",
    "git merge --ff-only work <<EOF\nbody\nEOF", "git worktree remove ../work 2>&1",
    "git worktree remove ../work >/dev/null 2>&1", "git branch -d work 2>&1", "git branch -d work 2>/dev/null",
]


@pytest.mark.parametrize("command", LANDING_WITH_REDIRECTS)
def test_a_redirection_is_not_an_argument_of_the_command_it_follows(repo, command):
    primary, _, _ = repo
    assert bash(command, primary) == ALLOWED


def test_a_landing_merge_after_cd_to_the_primary_checkout_may_carry_redirections(repo):
    primary, work, _ = repo
    assert bash(f"cd {primary} && git merge --ff-only work 2>&1 | tail -1; git status", work) == ALLOWED
    assert bash(f"cd {primary} && git merge --ff-only work 2>&1 | tail -1; git commit -m x", work) == ALLOWED


@pytest.mark.parametrize("command", [
    "git merge --no-ff work 2>&1",
    "git merge --ff-only a b 2>&1", "git merge work &> out",
])
def test_a_redirection_does_not_hide_a_refused_command(repo, command):
    primary, _, _ = repo
    assert bash(command, primary) == BLOCKED


def test_a_redirection_after_a_cd_or_an_install_is_still_read(repo):
    primary, work, _ = repo
    assert bash(f"cd {work} 2>&1 && git commit -m x", primary) == ALLOWED
    assert bash(f"git -C {primary} add a.py 2>&1", work) == ALLOWED
    assert bash("pip install -e . 2>&1 | tail -1", work) == BLOCKED
    assert bash("pip install -e . >log", work) == BLOCKED
    assert bash("pip install requests 2>&1", work) == ALLOWED


def test_the_directory_a_compound_command_runs_in_is_followed(repo):
    primary, work, _ = repo
    assert bash(f"cd {work} && git commit -m x", primary) == ALLOWED
    assert bash(f"git -C {work} commit -m x", primary) == ALLOWED
    assert bash(f"cd {primary} && git commit -m x", work) == ALLOWED
    assert bash(f"git -C {primary} add a.py", work) == ALLOWED


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


@pytest.mark.parametrize("command", ["pip install -e .", "pip install -e '.[test]'"])
def test_an_editable_install_of_the_primary_checkout_is_refused(repo, command):
    primary, work, _ = repo
    assert bash(command, primary) == BLOCKED
    assert bash(f"pip install -e {primary}", work) == BLOCKED


@pytest.mark.parametrize("command", ["pip install requests", "pip install -r r.txt"])
def test_noneditable_installs_are_allowed(repo, command):
    primary, work, _ = repo
    assert bash(command, primary) == ALLOWED
    assert bash(command, work) == ALLOWED


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
    return subprocess.run([sys.executable, str(COMMIT)], cwd=cwd, text=True, capture_output=True, env={**CLEAN, **env})


def test_an_agent_can_commit_in_the_primary_development_checkout(repo):
    primary, work, _ = repo
    refused = commit(primary, CLAUDECODE="1")
    assert refused.returncode == 0
    assert commit(primary).returncode == 0
    assert commit(work, CLAUDECODE="1").returncode == 0
    assert commit(primary, CLAUDECODE="1", MLSTACK_GUARD="off").returncode == 0


def test_an_agent_cannot_commit_on_the_development_branch_in_another_tree(repo):
    _, work, _ = repo
    assert commit(work, CLAUDECODE="1").returncode == 0
    git(work, "checkout", "-q", "--ignore-other-worktrees", "dev")
    refused = commit(work, CLAUDECODE="1")
    assert refused.returncode == 1 and "the development branch" in refused.stderr


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="requires a POSIX pseudoterminal")
def test_a_person_at_a_terminal_commits_anywhere(repo):
    primary, _, _ = repo
    master, slave = os.openpty()
    try:
        done = subprocess.run([sys.executable, str(COMMIT)], cwd=primary, stdin=slave, stdout=slave,
                              stderr=subprocess.PIPE, env=CLEAN)
    finally:
        os.close(master)
        os.close(slave)
    assert done.returncode == 0


def test_person_terminal_policy_permits_primary_and_worktree(repo, monkeypatch):
    primary, work, _ = repo
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert worktreerules.commit_refusal(primary, {}) == ""
    assert worktreerules.commit_refusal(work, {}) == ""


def test_the_harness_policy_applies_the_same_rule_to_codex_and_local_sessions(repo):
    primary, work, _ = repo
    patch = f"*** Begin Patch\n*** Update File: {primary / 'a.py'}\n*** End Patch"
    for name, args in [("Write", {"file_path": str(primary / "a.py")}),
                       ("apply_patch", {"input": patch}),
                       ("shell", {"command": ["git", "commit", "-m", "x"]}),
                       ("Bash", {"command": "git add a.py"})]:
        verdict = harnesspolicy.primary_decision(name, args, str(primary))
        assert verdict is None, name
    for name, args in [("Write", {"file_path": str(work / "a.py")}),
                       ("Bash", {"command": "git add a.py"})]:
        assert harnesspolicy.primary_decision(name, args, str(work)) is None


def test_the_harness_hook_preserves_primary_checkout_ownership_checks(repo, monkeypatch):
    primary, work, _ = repo
    event = {"tool_name": "Write", "cwd": str(work), "tool_input": {"file_path": str(primary / "a.py")}}
    monkeypatch.setattr(harnesshook.harness_claims, "conflict",
                        lambda *args: "file area is claimed by another worker")
    out = harnesshook.pre(event, harnesshook.Rail("plan-and-go", "t"))["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "claimed by another worker" in out["permissionDecisionReason"]


@pytest.mark.parametrize("name", ["a; touch pwned", "$(touch pwned)", "`touch pwned`", "-C", "a b\nc"])
def test_a_hostile_path_is_one_argv_item_and_never_reaches_a_shell(repo, tmp_path, name):
    primary, _, _ = repo
    here = tmp_path / "cwd"
    here.mkdir()
    worktreerules.checkouts(here / name)
    worktreerules.commit_refusal(str(here / name), {"CLAUDECODE": "1"})
    assert worktreerules.bash_refusal(f"cd '{here / name}' && git add a.py", str(here)) == ""
    assert worktreerules.edit_refusal([str(primary / name / "f.py")], str(here)) == ""
    assert not (here / "pwned").exists() and not Path("pwned").exists()


@pytest.mark.parametrize("variable", ["GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"])
def test_hook_repository_environment_does_not_change_checkout_identity(repo, monkeypatch, variable):
    primary, work, _ = repo
    selected = subprocess.run(["git", "-C", str(work), "rev-parse", "--absolute-git-dir"],
                              check=True, capture_output=True, text=True).stdout.strip()
    value = {"GIT_DIR": selected, "GIT_WORK_TREE": str(work),
             "GIT_COMMON_DIR": str(primary / ".git"), "GIT_INDEX_FILE": str(work / "other-index")}[variable]
    monkeypatch.setenv(variable, value)
    assert worktreerules.checkouts(primary) == (primary, primary)
    assert worktreerules.checkouts(work) == (work, primary)
    assert worktreerules.commit_refusal(work, {"ML_STACK_AGENT": "test"}) == ""
    assert worktreerules.commit_refusal(primary, {"ML_STACK_AGENT": "test"}) == ""


def test_read_only_stash_commands_are_allowed_in_the_primary_checkout(repo):
    primary, _, _ = repo
    assert worktreerules.bash_refusal("git stash list", str(primary)) == ""
    assert worktreerules.bash_refusal("git stash show -p", str(primary)) == ""
    assert worktreerules.bash_refusal("git stash", str(primary))
    assert worktreerules.bash_refusal("git stash pop", str(primary))


def test_main_branch_edits_and_commits_are_refused(repo):
    primary, work, _ = repo
    git(primary, "checkout", "-q", "-b", "main")
    assert worktreerules.edit_refusal([str(primary / "a.py")], str(primary))
    assert worktreerules.bash_refusal("git add a.py", str(primary))
    assert worktreerules.bash_refusal("git commit -m x", str(primary))
    assert commit(primary, CLAUDECODE="1").returncode == 1
    assert edit(work / "a.py", work) == ALLOWED


def test_claimed_primary_write_reaches_ownership_reservation(repo, monkeypatch):
    primary, _, _ = repo
    reserved = []
    monkeypatch.setattr(harnesshook, "decide",
                        lambda *args, **kwargs: harnesspolicy.Decision("allow", "reversible", "owned write"))
    monkeypatch.setattr(harnesshook.harness_claims, "conflict", lambda *args: "")
    monkeypatch.setattr(harnesshook.harness_claims, "reserve", lambda *args: reserved.append(args))
    event = {"tool_name": "Write", "cwd": str(primary),
             "tool_input": {"file_path": str(primary / "a.py")}}
    out = harnesshook.pre(event, harnesshook.Rail("plan-and-go", "worker", roots=[str(primary)]))
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert len(reserved) == 1 and reserved[0][3] == "worker"


def test_linked_main_checkout_edits_and_staging_are_refused(repo):
    primary, work, _ = repo
    git(work, "checkout", "-q", "-b", "main")
    assert worktreerules.edit_refusal([str(work / "a.py")], str(primary))
    assert worktreerules.bash_refusal(f"git -C {work} add a.py", str(primary))
    assert worktreerules.bash_refusal("git commit -m x", str(work))
    assert commit(work, CLAUDECODE="1").returncode == 1
