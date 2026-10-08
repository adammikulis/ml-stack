"""Whether git will run a repository's hooks, against real repositories, real git config and the real SessionStart hook."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ml_stack import doctor, hookcheck
from ml_stack.checks import ask

ROOT = Path(__file__).resolve().parent.parent
NAMES = ("pre-commit", "commit-msg", "pre-push", "post-merge", "claude-bash-guard")


@pytest.fixture(autouse=True)
def plain_git(tmp_path, monkeypatch):
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "Ada Lovelace")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "ada@invented.example")


def git(repo, *words, check=True):
    done = subprocess.run(["git", "-C", str(repo), *words], capture_output=True, text=True, check=check)
    return done.stdout.strip()


def make_repo(where: Path) -> Path:
    where.mkdir(parents=True)
    git(where, "init", "-q", "-b", "dev")
    for name in NAMES:
        path = where / "scripts" / "hooks" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!/bin/sh\nexit 0\n# {name}\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    shutil.copy(ROOT / "scripts" / "install-hooks.sh", where / "scripts" / "install-hooks.sh")
    git(where, "add", "scripts")
    git(where, "commit", "-q", "-m", "chore: scripts")
    return where


@pytest.fixture
def repo(tmp_path):
    made = make_repo(tmp_path / "repo")
    subprocess.run(["sh", "scripts/install-hooks.sh"], cwd=made, check=True, capture_output=True)
    return made


def whats(repo) -> list[str]:
    return [p.what for p in hookcheck.inspect(repo)]


def test_a_repository_with_its_hooks_installed_has_no_problems(repo):
    assert hookcheck.inspect(repo) == []


def test_a_hooks_path_naming_a_missing_directory_is_reported_with_the_exact_repair(repo):
    git(repo, "config", "core.hooksPath", str(repo / "gone"))
    problems = hookcheck.inspect(repo)
    assert [p.kind for p in problems].count("config") == 1
    config = next(p for p in problems if p.kind == "config")
    assert config.repair == f"git -C {shlex.quote(str(repo))} config --local --unset-all core.hookspath"
    assert any("which does not exist" in p.what for p in problems)
    subprocess.run(shlex.split(config.repair), check=True)
    assert hookcheck.inspect(repo) == []


def test_a_hooks_path_naming_the_repositorys_own_hooks_directory_is_fine(repo):
    git(repo, "config", "core.hooksPath", str(repo / ".git" / "hooks"))
    assert hookcheck.inspect(repo) == []


@pytest.mark.parametrize("key,value", [
    ("core.fsmonitor", "/bin/sh"),
    ("core.sshCommand", "ssh -o ProxyCommand=x"),
    ("alias.st", "!sh -c 'id'"),
    ("url.https://evil.example/.insteadOf", "https://github.example/"),
    ("remote.origin.pushurl", "https://evil.example/r.git"),
])
def test_config_that_runs_a_command_or_redirects_a_remote_is_reported_and_never_edited(repo, key, value):
    git(repo, "config", key, value)
    before = (repo / ".git" / "config").read_text()
    found = [p for p in hookcheck.inspect(repo) if p.kind == "config"]
    assert len(found) == 1 and key.split(".")[0] in found[0].what
    assert (repo / ".git" / "config").read_text() == before
    subprocess.run(shlex.split(found[0].repair), check=True)
    assert hookcheck.inspect(repo) == []


def test_an_alias_that_is_not_a_shell_command_and_a_disabled_fsmonitor_are_fine(repo):
    git(repo, "config", "alias.st", "status -s")
    git(repo, "config", "core.fsmonitor", "false")
    assert hookcheck.inspect(repo) == []


def test_the_worktree_config_is_read_too(repo, tmp_path):
    git(repo, "config", "extensions.worktreeConfig", "true")
    other = tmp_path / "second"
    git(repo, "worktree", "add", "-q", "-b", "side", str(other))
    git(other, "config", "--worktree", "core.sshCommand", "ssh -v")
    found = [p for p in hookcheck.inspect(other) if p.kind == "config"]
    assert len(found) == 1 and "the worktree" in found[0].what and "--worktree" in found[0].repair
    assert [p for p in hookcheck.inspect(repo) if p.kind == "config"] == []


def test_a_hook_that_is_missing_or_foreign_is_not_installed_and_one_that_cannot_run_is_reported(repo):
    hooks = repo / ".git" / "hooks"
    (hooks / "pre-commit").unlink()
    (hooks / "commit-msg").unlink()
    (hooks / "commit-msg").write_text("#!/bin/sh\nexit 0\n")
    (hooks / "commit-msg").chmod(0o755)
    assert whats(repo) == ["not installed: pre-commit, commit-msg"]
    (hooks / "commit-msg").write_text("#!/bin/sh\nexec scripts/hooks/commit-msg\n")
    (hooks / "commit-msg").chmod(0o644)
    (hooks / "pre-commit").symlink_to("../../scripts/hooks/pre-commit")
    [problem] = hookcheck.inspect(repo)
    assert "commit-msg" in problem.what and "not executable" in problem.what and problem.repair.startswith("chmod +x ")


def test_a_wrapper_that_execs_the_shipped_script_counts(repo):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.unlink()
    hook.write_text('#!/bin/sh\nexport NAMES_GRAPH=/nowhere\nexec "$(git rev-parse --show-toplevel)/scripts/hooks/pre-commit" "$@"\n')
    hook.chmod(0o755)
    assert hookcheck.inspect(repo) == []


def test_a_guard_script_that_differs_from_head_is_reported_with_a_restore(repo):
    guard = repo / "scripts" / "hooks" / "claude-bash-guard"
    guard.write_text("#!/bin/sh\nexit 0\n# weakened\n")
    found = [p for p in hookcheck.inspect(repo) if p.kind == "tampered"]
    assert [p.what for p in found] == ["scripts/hooks/claude-bash-guard differs from HEAD"]
    subprocess.run(shlex.split(found[0].repair), check=True)
    assert hookcheck.inspect(repo) == []
    guard.unlink()
    assert whats(repo) == ["scripts/hooks/claude-bash-guard is missing"]


def test_the_doctor_finding_offers_only_the_installer_and_leaves_config_for_a_person(repo, tmp_path):
    git(repo, "config", "core.hooksPath", str(tmp_path / "gone"))
    before = (repo / ".git" / "config").read_text()
    found = doctor.hooks_of(repo)
    assert not found.good and found.fix == [] and "--unset-all core.hookspath" in found.note
    ask([found], yes=True)
    assert (repo / ".git" / "config").read_text() == before
    git(repo, "config", "--unset-all", "core.hooksPath")
    shutil.rmtree(repo / ".git" / "hooks")
    missing = doctor.hooks_of(repo)
    assert not missing.good and missing.fix == ["sh", "scripts/install-hooks.sh"] and missing.cwd == str(repo)
    ask([missing], yes=True)
    assert doctor.hooks_of(repo).good


# -- the SessionStart hook ------------------------------------------------------------

@pytest.fixture
def session(tmp_path, repo):
    """The SessionStart hook of a repository that holds a copy of the real hook scripts."""
    for name in ("claude-session-start", "workspace_hook.py"):
        shutil.copy(ROOT / "scripts" / "hooks" / name, repo / "scripts" / "hooks" / name)
    (repo / "src").symlink_to(ROOT / "src")
    git(repo, "add", "scripts")
    git(repo, "commit", "-q", "-m", "chore: session hook")
    calls = tmp_path / "calls.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ml-stack-workspace"
    fake.write_text(f"#!{sys.executable}\nimport json, sys\n"
                    f"open({str(calls)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\nprint('inbox line')\n")
    fake.chmod(0o700)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "ML_STACK_HOME": str(tmp_path / "state"),
           "ML_STACK_RUNTIME_ENSURE": "off", "PYTHONPATH": str(ROOT / "src")}

    def start():
        return subprocess.run([sys.executable, str(repo / "scripts" / "hooks" / "claude-session-start")],
                              input=json.dumps({"session_id": "native-1", "model": "claude-sonnet-5-5"}),
                              text=True, capture_output=True, timeout=60, env=env)

    def announced():
        rows = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return [r for r in rows if r[:2] == ["announce", "blocked"]]
    return start, announced


def test_a_healthy_repository_adds_no_board_line_and_no_context_line(session):
    start, announced = session
    done = start()
    assert done.returncode == 0 and "git hooks" not in done.stdout and announced() == []


def test_broken_hooks_are_announced_once_and_shown_to_the_agent_every_time(session, repo, tmp_path):
    start, announced = session
    git(repo, "config", "core.hooksPath", str(tmp_path / "gone"))
    first = start()
    context = json.loads(first.stdout)["hookSpecificOutput"]["additionalContext"]
    assert first.returncode == 0 and "core.hooksPath" in context and "--unset-all core.hookspath" in context
    [row] = announced()
    assert "--agent" in row and row[row.index("--agent") + 1] == "claude" and "git hooks in repo" in row[2] and len(row[2]) <= 190
    second = start()
    assert "core.hooksPath" in second.stdout and len(announced()) == 1
    git(repo, "config", "--unset-all", "core.hooksPath")
    assert start().returncode == 0 and len(announced()) == 1
    git(repo, "config", "core.hooksPath", str(tmp_path / "gone"))
    start()
    assert len(announced()) == 2
    recorded = json.loads(next((tmp_path / "state" / "hookcheck").glob("*.json")).read_text())
    assert recorded["version"] == 1
