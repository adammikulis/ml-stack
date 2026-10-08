"""Hostile paths, refs and arguments reach the runtime deploy's processes as literal argv, never through a shell."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from test_runtime_deploy import builder, commit, plan_for, world  # noqa: F401

from ml_stack import runtime, runtime_deploy, runtime_launchers
from ml_stack.fleet import runtime_wheel, updates

pytestmark = pytest.mark.slow

HOSTILE = "a b;touch PWNED;$(touch PWNED)`touch PWNED`"


def untouched(*roots: Path) -> bool:
    return not any(path.name == "PWNED" for root in roots for path in root.rglob("PWNED")) \
        and not (Path.cwd() / "PWNED").exists()


def test_a_checkout_path_with_shell_punctuation_reaches_git_as_one_argument(world, tmp_path):  # noqa: F811
    repo, _, _ = world
    odd = tmp_path / HOSTILE
    repo.rename(odd)
    head = commit(odd, "a")
    assert runtime_deploy.resolve_commit(odd) == head
    assert runtime_wheel._run(["git", "-C", str(odd), "rev-parse", "HEAD"], 30) == head
    assert untouched(tmp_path)


@pytest.mark.parametrize("ref", ["--output=PWNED", "-h", "--help", "HEAD; touch PWNED", "$(touch PWNED)"])
def test_a_ref_that_looks_like_an_option_or_a_command_names_no_commit(world, tmp_path, ref):  # noqa: F811
    repo, _, _ = world
    commit(repo, "a")
    with pytest.raises(runtime_deploy.DeployError):
        runtime_deploy.resolve_commit(repo, ref)
    assert untouched(tmp_path)


def test_forward_hands_hostile_arguments_to_the_selected_interpreter_unchanged(tmp_path, monkeypatch):
    chosen = runtime.Runtime(tmp_path / "prefix", "a" * 40, "0", runtime.identity())
    seen = []
    monkeypatch.setattr(runtime, "available", lambda: chosen)
    monkeypatch.setattr(os, "execve", lambda path, argv, env: seen.append((path, argv, env)))
    runtime.forward("ml_stack.fleet.launch", ["--x; touch PWNED", "$(id)", "`id`"])
    _, argv, env = seen[0]
    assert argv == [str(chosen.python), "-I", "-m", "ml_stack.fleet.launch", "--x; touch PWNED", "$(id)", "`id`"]
    assert "PYTHONPATH" not in env


def test_a_state_root_with_shell_punctuation_still_builds_verifies_and_installs_launchers(world, tmp_path, monkeypatch):  # noqa: F811
    repo, _, _ = world
    odd = tmp_path / HOSTILE
    odd.mkdir()
    monkeypatch.setenv("ML_STACK_HOME", str(odd / "state"))
    launchers = odd / "bin"
    launchers.mkdir(mode=0o700)
    head = commit(repo, "a")
    outcome = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert outcome.action == "switched" and outcome.commit == head
    chosen = runtime.selected()
    assert runtime.verify(chosen) == chosen
    assert runtime_launchers.install(launchers, chosen)
    done = subprocess.run([str(launchers / "ml-stack-workspace")], capture_output=True, text=True, timeout=60)
    assert done.stdout.strip() == f"help {head[:7]}"
    assert untouched(tmp_path)


def test_the_smoke_runs_hooks_from_a_source_path_with_punctuation_as_one_argument(world, tmp_path):  # noqa: F811
    repo, launchers, _ = world
    commit(repo, "a")
    seen = []

    def build(plan, stage, prefix):
        built = builder()(plan, stage, prefix)
        odd = stage.parent / HOSTILE / "source"
        (odd / "scripts" / "hooks").mkdir(parents=True)
        for name in runtime_deploy.HOOKS:
            (odd / "scripts" / "hooks" / name).write_text(f"import sys\nopen(r'{tmp_path}/hook-ran', 'a').write(sys.argv[0] + '\\n')\n")
        seen.append(odd)
        return built

    plan = plan_for(repo, launchers)
    chosen = build(plan, tmp_path, runtime.directory() / plan.commit / ("c" * 32))
    runtime_deploy.smoke(chosen, seen[0], tmp_path)
    assert len((tmp_path / "hook-ran").read_text().splitlines()) == len(runtime_deploy.HOOKS)
    assert untouched(tmp_path)


def test_the_launcher_directory_with_shell_punctuation_is_one_path(world, tmp_path):  # noqa: F811
    repo, launchers, _ = world
    commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    odd = tmp_path / HOSTILE
    odd.mkdir(mode=0o700)
    written = runtime_launchers.install(odd, runtime.selected())
    assert written and all(path.parent == odd for path in written)
    assert untouched(tmp_path)


def test_the_tracker_hands_a_hostile_checkout_path_to_the_runtime_command_as_one_argument(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(updates.subprocess, "run",
                        lambda argv, **kwargs: seen.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""))
    updates.pip_install(tmp_path / HOSTILE)
    assert seen[0][-2:] == ["--checkout", str(tmp_path / HOSTILE)] and "sh" not in seen[0][:2]
