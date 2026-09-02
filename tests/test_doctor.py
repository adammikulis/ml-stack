"""``ml-stack-doctor`` reads the repositories and the working state, and says what is wrong.

Everything it reads is built here: a repository is ``git init`` in ``tmp_path`` with an
invented author, a worktree is added to it, a venv is a shell script that prints a path,
a bench home is a store written by ``GraphStore`` beside a lock and some logs, and the
managed llama.cpp is a directory holding a script that answers ``--help``. Nothing under
``~`` is opened, and nothing is committed to anything real.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from ml_stack import doctor
from ml_stack.doctor import (ahead_of, bench_of, builds_of, hooks_of, install_of, look, main,
                             repositories, status_of, worktrees_of)

REPO = Path(__file__).resolve().parent.parent
AUTHOR = ("Ada Lovelace", "ada@invented.example")


@pytest.fixture(autouse=True)
def plain_git(tmp_path, monkeypatch):
    """git with no global or system config: no hooksPath, no template, no real identity."""
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_AUTHOR_NAME", AUTHOR[0])
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", AUTHOR[1])
    monkeypatch.setenv("GIT_COMMITTER_NAME", AUTHOR[0])
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", AUTHOR[1])


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True)
    return done.stdout.strip()


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_repo(where: Path, *, hooks_dir: str = "scripts/hooks", commits: int = 1) -> Path:
    """An invented repository shipping its hooks the way the real ones do."""
    where.mkdir(parents=True)
    git(where, "init", "-q", "-b", "main")
    for name in ("no-real-names", "commit-msg", "pre-commit"):
        _script(where / hooks_dir / name, "exit 0\n")
    if hooks_dir == "scripts/hooks":
        shutil.copy(REPO / "scripts" / "install-hooks.sh", where / "scripts" / "install-hooks.sh")
    (where / ".gitignore").write_text(".venv/\n")
    for n in range(commits):
        (where / f"note-{n}.txt").write_text(f"commit {n}\n")
        git(where, "add", "-A")
        git(where, "commit", "-q", "-m", f"commit {n}")
    return where


def by_name(findings, name: str):
    found = [f for f in findings if f.name == name]
    assert found, f"no finding named {name!r} among {[f.name for f in findings]}"
    return found[0]


# -- hooks ---------------------------------------------------------------------------

def test_hooks_not_installed_is_named_with_the_installer_as_the_fix(tmp_path):
    repo = make_repo(tmp_path / "quenlow")
    found = hooks_of(repo)
    assert not found.good
    assert found.said == "not installed: pre-commit, commit-msg"
    assert found.fix == f"cd {repo} && sh scripts/install-hooks.sh"


def test_the_offered_fix_installs_them_and_the_next_look_is_good(tmp_path):
    repo = make_repo(tmp_path / "quenlow")
    subprocess.run(hooks_of(repo).fix, shell=True, check=True, capture_output=True)
    found = hooks_of(repo)
    assert found.good
    assert found.said == "installed, from scripts/hooks"
    assert os.readlink(repo / ".git" / "hooks" / "pre-commit") == "../../scripts/hooks/no-real-names"


def test_an_untracked_wrapper_that_execs_the_shipped_script_counts_as_installed(tmp_path):
    """A machine points the tracked hook at its own database with a wrapper; that is the
    documented shape, and it must not read as 'not installed'."""
    repo = make_repo(tmp_path / "quenlow")
    for hook, script in (("pre-commit", "no-real-names"), ("commit-msg", "commit-msg")):
        _script(repo / ".git" / "hooks" / hook,
                'export NAMES_GRAPH=/nowhere\nexec "$(git rev-parse --show-toplevel)/'
                f'scripts/hooks/{script}" "$@"\n')
    assert hooks_of(repo).good


def test_a_repository_shipping_hooks_under_services_gets_a_link_fix(tmp_path):
    repo = make_repo(tmp_path / "pellard", hooks_dir="services/hooks")
    found = hooks_of(repo)
    assert not found.good
    assert found.fix == (f"cd {repo} && ln -sf ../../services/hooks/pre-commit .git/hooks/pre-commit"
                         " && ln -sf ../../services/hooks/commit-msg .git/hooks/commit-msg")
    subprocess.run(found.fix, shell=True, check=True)
    assert hooks_of(repo).said == "installed, from services/hooks"


def test_a_hook_pointing_somewhere_else_is_not_installed(tmp_path):
    repo = make_repo(tmp_path / "quenlow")
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    os.symlink("/somewhere/else/pre-commit", hooks / "pre-commit")
    _script(hooks / "commit-msg", "exit 0\n")
    assert hooks_of(repo).said == "not installed: pre-commit, commit-msg"


def test_a_repository_shipping_no_hooks_has_nothing_to_say(tmp_path):
    where = tmp_path / "bare"
    where.mkdir()
    git(where, "init", "-q")
    assert hooks_of(where) is None


# -- the working tree and the branch -----------------------------------------------------

def test_a_clean_tree_is_clean_and_a_dirty_one_lists_what_is_not(tmp_path):
    repo = make_repo(tmp_path / "quenlow")
    assert status_of(repo).good
    assert status_of(repo).said == "clean"
    (repo / "note-0.txt").write_text("changed\n")
    (repo / "new.txt").write_text("new\n")
    found = status_of(repo)
    assert not found.good
    assert found.said == "2 not committed: note-0.txt, new.txt"
    assert found.fix == ""


def test_ahead_of_origin_is_a_note_and_never_a_push(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    repo = make_repo(tmp_path / "quenlow")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-q", "-u", "origin", "main")
    assert ahead_of(repo).said == "main is in step with origin/main"
    (repo / "later.txt").write_text("later\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "later")
    found = ahead_of(repo)
    assert found.good
    assert found.said == "main is 1 commit(s) ahead of origin/main"
    assert "not pushed" in found.note and found.fix == ""
    assert git(repo, "rev-list", "--count", "origin/main..main") == "1"


def test_no_upstream_is_said_rather_than_counted(tmp_path):
    repo = make_repo(tmp_path / "quenlow")
    found = ahead_of(repo)
    assert found.good
    assert found.said == "main, no upstream to compare with"


# -- worktrees -----------------------------------------------------------------------

def test_a_worktree_pinned_behind_head_is_noted_with_its_commit(tmp_path):
    repo = make_repo(tmp_path / "quenlow", commits=3)
    pinned = git(repo, "rev-parse", "HEAD~2")
    git(repo, "worktree", "add", "-q", "--detach", str(tmp_path / "pin"), pinned)
    found = worktrees_of(repo)
    assert len(found) == 1
    assert found[0].name == "quenlow: worktree pin"
    assert found[0].good
    assert found[0].said == f"stale: holds {pinned[:7]}, 2 commit(s) behind HEAD"
    assert str(tmp_path / "pin") in found[0].note


def test_a_worktree_at_head_is_not_stale_and_none_says_nothing(tmp_path):
    repo = make_repo(tmp_path / "quenlow")
    assert worktrees_of(repo) == []
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "worktree", "add", "-q", "--detach", str(tmp_path / "pin"), head)
    assert worktrees_of(repo)[0].said == f"holds {head[:7]}, at HEAD"


# -- the editable install --------------------------------------------------------------

def fake_python(repo: Path, prints: str, *, fails: bool = False) -> Path:
    body = "echo 'ModuleNotFoundError: No module named ml_stack' >&2\nexit 1\n" if fails \
        else f"echo {prints}\n"
    return _script(repo / ".venv" / "bin" / "python", body)


def test_an_install_under_the_checkout_is_good_and_reports_the_path(tmp_path):
    checkout = tmp_path / "checkout"
    ask = checkout / "src" / "ml_stack" / "graph" / "ask.py"
    ask.parent.mkdir(parents=True)
    ask.write_text("")
    repo = make_repo(tmp_path / "quenlow")
    fake_python(repo, str(ask))
    found = install_of(repo, checkout=checkout)
    assert found.good
    assert found.said == str(ask)
    assert found.fix == ""


def test_an_install_in_site_packages_is_a_copy_with_pip_e_as_the_fix(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    copy = tmp_path / "venv" / "lib" / "site-packages" / "ml_stack" / "graph" / "ask.py"
    copy.parent.mkdir(parents=True)
    copy.write_text("")
    repo = make_repo(tmp_path / "quenlow")
    python = fake_python(repo, str(copy))
    found = install_of(repo, checkout=checkout)
    assert not found.good
    assert found.said == str(copy)
    assert found.fix == f"{python} -m pip install -e {checkout}"
    assert "imports a copy" in found.note


def test_an_import_that_fails_says_why(tmp_path):
    repo = make_repo(tmp_path / "quenlow")
    python = fake_python(repo, "", fails=True)
    found = install_of(repo, checkout=tmp_path / "checkout")
    assert not found.good
    assert found.said == f"{python}: import ml_stack fails -- ModuleNotFoundError: No module named ml_stack"


# -- the bench -----------------------------------------------------------------------

def dead_pid() -> int:
    """A pid that was a process and is one no longer."""
    child = subprocess.Popen(["true"])
    child.wait()
    return child.pid


def make_bench(home: Path, *, runs=(), empties=(), lock_pid=None, logs=()) -> Path:
    """A bench home: a runs store, a lock naming ``lock_pid``, and logs by stamp."""
    from ml_stack.graph.store import GraphStore

    home.mkdir(parents=True)
    if runs or empties:
        with GraphStore(home / "runs.ladybug") as store:
            for n, at in enumerate(runs):
                store.put_doc(f"bench:invented:{n}", {"at": at, "label": "invented", "rows": []})
            for n in empties:
                store.put_doc(f"bench:hollow:{n}", {})
    if lock_pid is not None:
        (home / "measuring.json").write_text(json.dumps({
            "pid": lock_pid, "argv": ["sweep"], "started": "2026-08-30T10:00:00",
            "log": str(home / "logs" / (logs[-1] if logs else "none.log"))}))
    for name in logs:
        (home / "logs").mkdir(exist_ok=True)
        (home / "logs" / name).write_text("measuring...\n")
    return home


def test_a_missing_bench_home_says_nothing(tmp_path):
    assert bench_of(tmp_path / "none") == []


def test_empty_runs_are_counted_and_named_with_forget_as_the_fix(tmp_path):
    home = make_bench(tmp_path / "bench", runs=("2026-08-30T09:00:00",), empties=(1, 2))
    found = by_name(bench_of(home), "bench: runs")
    assert not found.good
    assert found.said == "2 run(s) read back as nothing: bench:hollow:1, bench:hollow:2"
    assert found.fix == f"ml-stack-bench forget --empty --kept {home / 'runs.ladybug'}"


def test_a_lock_whose_pid_is_dead_is_stale_and_its_removal_is_the_fix(tmp_path):
    pid = dead_pid()
    home = make_bench(tmp_path / "bench", lock_pid=pid)
    found = by_name(bench_of(home), "bench: measuring")
    assert not found.good
    assert found.said == f"stale lock: {home / 'measuring.json'} names pid {pid}, which is not running"
    assert found.fix == f"rm -f {home / 'measuring.json'}"


def test_a_lock_whose_pid_is_alive_is_a_measurement_in_progress(tmp_path):
    home = make_bench(tmp_path / "bench", lock_pid=os.getpid(), logs=("sweep-x-20260830T100000.log",))
    found = by_name(bench_of(home), "bench: measuring")
    assert found.good
    assert found.said == f"pid {os.getpid()} since 2026-08-30T10:00:00: ml-stack-bench sweep"
    # the log it is writing is not a run that died
    assert by_name(bench_of(home), "bench: logs").good


def test_a_log_newer_than_the_newest_run_with_no_run_kept_is_a_run_that_died(tmp_path):
    home = make_bench(tmp_path / "bench", runs=("2026-08-30T09:00:00", "2026-08-30T12:00:00"),
                      logs=("sweep-kept-20260830T080000.log", "sweep-died-20260830T130000.log"))
    found = by_name(bench_of(home), "bench: logs")
    assert not found.good
    assert found.said == ("1 log(s) newer than the newest kept run, with no run kept: "
                          "sweep-died-20260830T130000.log")
    assert found.fix == ""


def test_every_log_with_a_run_after_it_is_good(tmp_path):
    home = make_bench(tmp_path / "bench", runs=("2026-08-30T12:00:00",),
                      logs=("sweep-kept-20260830T080000.log",))
    found = by_name(bench_of(home), "bench: logs")
    assert found.good
    assert found.said == "1 run(s) kept; every log has one"


def test_a_log_without_a_stamp_is_dated_by_its_mtime(tmp_path):
    home = make_bench(tmp_path / "bench", runs=("2026-08-30T12:00:00",), logs=("odd.log",))
    old = time.mktime(time.strptime("2026-08-30T11:00:00", "%Y-%m-%dT%H:%M:%S"))
    os.utime(home / "logs" / "odd.log", (old, old))
    assert by_name(bench_of(home), "bench: logs").good
    later = time.mktime(time.strptime("2026-08-30T13:00:00", "%Y-%m-%dT%H:%M:%S"))
    os.utime(home / "logs" / "odd.log", (later, later))
    assert not by_name(bench_of(home), "bench: logs").good


# -- the managed llama.cpp -------------------------------------------------------------

def make_build(where: Path, *, commit: str, days_old: int, answers: bool = True) -> Path:
    built = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - days_old * 86400))
    _script(where / "llama-server", "echo 'usage: llama-server [options]'\n" if answers else "exit 1\n")
    (where / "BUILD.json").write_text(json.dumps({"commit": commit, "built_at": built}))
    return where


def test_current_answers_help_and_a_fresh_build_is_good(tmp_path):
    current = make_build(tmp_path / "builds" / "abc1234", commit="abc1234", days_old=3)
    found = builds_of(current, tmp_path / "named")
    assert [f.name for f in found] == ["llama.cpp: current"]
    assert found[0].good
    assert found[0].said == "abc1234, 3d old, answers --help"


def test_a_build_older_than_fourteen_days_is_noted_with_build_as_the_fix(tmp_path):
    current = make_build(tmp_path / "builds" / "abc1234", commit="abc1234", days_old=20)
    found = builds_of(current, tmp_path / "named")[0]
    assert not found.good
    assert found.said == "abc1234, 20d old, answers --help"
    assert found.fix == "ml-stack-serve build"
    assert builds_of(current, tmp_path / "named", stale_days=30)[0].good


def test_no_current_and_a_current_that_does_not_answer_are_both_told_to_build(tmp_path):
    found = builds_of(tmp_path / "current", tmp_path / "named")[0]
    assert (found.good, found.said, found.fix) == (False, "not built yet", "ml-stack-serve build")
    broken = make_build(tmp_path / "broken", commit="bad0000", days_old=1, answers=False)
    found = builds_of(broken, tmp_path / "named")[0]
    assert not found.good
    assert found.said == f"{broken / 'llama-server'} does not answer --help"
    assert found.fix == "ml-stack-serve build"


def test_named_builds_are_listed_beside_current(tmp_path):
    current = make_build(tmp_path / "builds" / "abc1234", commit="abc1234", days_old=1)
    named = tmp_path / "named"
    named.mkdir()
    fork = make_build(tmp_path / "builds" / "fork111", commit="fork111", days_old=1)
    os.symlink(fork, named / "fork")
    (named / "gone").mkdir()                     # no server in it: not a build
    found = builds_of(current, named)
    assert found[1].name == "llama.cpp: named builds"
    assert found[1].said == "fork (fork111)"


# -- the command ---------------------------------------------------------------------

def test_repositories_are_those_given_or_the_known_ones_that_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(doctor, "CHECKOUT", tmp_path / "absent")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert repositories() == [tmp_path]
    assert repositories([str(tmp_path / "a"), str(tmp_path / "a")]) == [tmp_path / "a"]


@pytest.fixture
def everything(tmp_path, monkeypatch):
    """A whole working state, all of it wrong: hooks missing, a dead lock, a stale build."""
    repo = make_repo(tmp_path / "quenlow")
    checkout = tmp_path / "checkout"
    ask = checkout / "src" / "ml_stack" / "graph" / "ask.py"
    ask.parent.mkdir(parents=True)
    ask.write_text("")
    fake_python(repo, str(ask))
    home = make_bench(tmp_path / "bench", lock_pid=dead_pid())
    current = make_build(tmp_path / "builds" / "old0000", commit="old0000", days_old=40)
    from ml_stack.serve import binary
    monkeypatch.setattr(binary, "MANAGED_CURRENT", current)
    monkeypatch.setattr(binary, "MANAGED_NAMED", tmp_path / "named")
    monkeypatch.setattr(doctor, "CHECKOUT", checkout)
    return repo, home


def test_look_names_everything_and_main_exits_one_until_it_is_fixed(everything, capsys):
    repo, home = everything
    found = look([repo], bench_home=home)
    assert [f.name for f in found] == [
        "quenlow: hooks", "quenlow: working tree", "quenlow: branch",
        "quenlow: editable install", "bench: measuring", "llama.cpp: current"]
    assert [f.name for f in found if not f.good] == [
        "quenlow: hooks", "bench: measuring", "llama.cpp: current"]
    assert main(["--repo", str(repo), "--bench-home", str(home)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("ml-stack: the repositories and the working state\n")
    assert "  ! quenlow: hooks: not installed: pre-commit, commit-msg" in out
    assert f"      fix: rm -f {home / 'measuring.json'}" in out
    assert "ok  quenlow: working tree: clean" in out


def test_yes_runs_the_fixes_it_can_and_does_not_touch_the_build(everything, capsys, monkeypatch):
    """Installing the hooks and removing the dead lock are safe; ``ml-stack-serve build``
    is a compile, and the test replaces it with a record of having been asked."""
    repo, home = everything
    asked: list[str] = []
    real = subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd == "ml-stack-serve build":
            asked.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    assert main(["--repo", str(repo), "--bench-home", str(home), "--yes"]) == 1
    assert asked == ["ml-stack-serve build"]
    assert not (home / "measuring.json").exists()
    assert hooks_of(repo).good
    found = look([repo], bench_home=home)
    assert [f.name for f in found if not f.good] == ["llama.cpp: current"]


def test_a_path_that_is_not_a_repository_is_said(tmp_path, capsys, monkeypatch):
    from ml_stack.serve import binary
    monkeypatch.setattr(binary, "MANAGED_CURRENT", tmp_path / "no-current")
    monkeypatch.setattr(binary, "MANAGED_NAMED", tmp_path / "no-named")
    where = tmp_path / "plain"
    where.mkdir()
    found = look([where], bench_home=tmp_path / "no-bench")
    assert found[0].said == f"{where} is not a git repository"
    assert not found[0].good
