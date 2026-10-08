"""Continuous runtime deploy: build, smoke, atomic switch, recovery, rollback, collection, claims and the board."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack import runtime, runtime_board, runtime_cli, runtime_deploy, runtime_store
from ml_stack.fleet import runtime_wheel
from ml_stack.lock import only_one
from ml_stack.workspace.claims import Claims
from ml_stack.workspace.identity import AGENT, Identity

pytestmark = pytest.mark.slow


def git(repo, *words):
    return subprocess.run(["git", "-C", str(repo), *words], capture_output=True, text=True, check=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}).stdout.strip()


def commit(repo, name, text="x"):
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", f"chore: {name}")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def world(tmp_path, monkeypatch):
    clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.delenv("ML_STACK_RUNTIME_ENSURE", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "dev")
    launchers = tmp_path / "bin"
    launchers.mkdir(mode=0o700)
    return repo, launchers, tmp_path


def wheel(stage, commit_id):
    path = stage / commit_id / "ml_stack-0.1.0-py3-none-any.whl"
    path.parent.mkdir(parents=True)
    contents = {
        "ml_stack/__init__.py": b"",
        "ml_stack/workspace/__init__.py": b"",
        "ml_stack/workspace/cli.py": f"def main():\n    print('help {commit_id[:7]}')\n    return 0\n".encode(),
        "ml_stack-0.1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: ml-stack\nVersion: 0.1.0\n",
        "ml_stack-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        "ml_stack-0.1.0.dist-info/entry_points.txt": b"[console_scripts]\nml-stack-workspace = ml_stack.workspace.cli:main\n",
        "ml_stack-0.1.0.dist-info/RECORD": b"",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return path


def builder(hook_code=0, log=None):
    def build(plan, stage):
        source = stage / "source" / "scripts" / "hooks"
        source.mkdir(parents=True)
        for name in runtime_deploy.HOOKS:
            (source / name).write_text(f"import sys\nsys.exit({hook_code})\n")
        built = wheel(stage, plan.commit)
        runtime_wheel.stamp(built, plan.commit, plan.checkout)
        if log is not None:
            log.append(plan.commit)
        return runtime_wheel.prepare(built, plan.commit, timeout=120)
    return build


def plan_for(repo, launchers, **kw):
    return runtime_deploy.Plan(repo, runtime_deploy.resolve_commit(repo), launchers, wait_s=kw.pop("wait_s", 2), **kw)


def launcher_output(launchers):
    done = subprocess.run([sys.executable, str(launchers / "ml-stack-workspace")], capture_output=True, text=True,
                          timeout=60, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    return done.stdout.strip()


def test_ensure_builds_smokes_and_switches_launchers_and_selection(world):
    repo, launchers, _ = world
    first = commit(repo, "a")
    built = []
    outcome = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(log=built))
    assert (outcome.action, outcome.commit, built) == ("switched", first, [first]), outcome.detail
    assert runtime_store.selection()["commit"] == first
    assert launcher_output(launchers) == f"help {first[:7]}"
    again = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(log=built))
    assert again.action == "current" and built == [first]
    second = commit(repo, "b")
    assert runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(log=built)).action == "switched"
    assert launcher_output(launchers) == f"help {second[:7]}"
    assert {c.commit for c in runtime_store.candidates()} == {first, second}


def test_a_failed_smoke_leaves_the_selection_and_launchers_and_records_the_failure(world):
    repo, launchers, _ = world
    first = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    before = (launchers / "ml-stack-workspace").read_bytes()
    second = commit(repo, "b")
    outcome = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(hook_code=3))
    assert outcome.action == "failed" and "claude-session-start" in outcome.detail
    assert runtime_store.selection()["commit"] == first
    assert (launchers / "ml-stack-workspace").read_bytes() == before
    assert not (runtime.directory() / second).exists() or not list((runtime.directory() / second).iterdir())
    assert runtime_store.read_state()["last_failure"]["commit"] == second


def test_a_launcher_whose_target_is_gone_runs_the_newest_verified_older_runtime(world):
    repo, launchers, _ = world
    first = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    commit(repo, "b")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    newest = runtime_store.selection()
    import shutil
    shutil.rmtree(newest["prefix"])
    assert launcher_output(launchers) == f"help {first[:7]}"


def test_ensure_replaces_a_corrupt_selection_with_a_fallback_then_builds_head(world):
    repo, launchers, _ = world
    first = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    second = commit(repo, "b")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    import shutil
    shutil.rmtree(runtime_store.selection()["prefix"])
    third = commit(repo, "c")
    outcome = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert outcome.action == "switched" and "in place of a broken selection" in outcome.detail
    assert runtime_store.selection()["commit"] == third
    assert first != second != third


def test_ensure_without_a_new_commit_recovers_a_corrupt_selection_from_the_fallback(world):
    repo, launchers, _ = world
    first = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    commit(repo, "b")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    import shutil
    shutil.rmtree(runtime_store.selection()["prefix"])
    outcome = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(hook_code=3))
    assert outcome.action == "recovered" and runtime_store.selection()["commit"] == first
    assert launcher_output(launchers) == f"help {first[:7]}"


def test_rollback_selects_the_earlier_runtime_and_holds_the_commit_until_head_moves(world):
    repo, launchers, _ = world
    first = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    second = commit(repo, "b")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert runtime_deploy.rollback(plan_for(repo, launchers)).commit == first
    assert launcher_output(launchers) == f"help {first[:7]}"
    held = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert held.action == "held" and held.commit == second
    third = commit(repo, "c")
    assert runtime_deploy.ensure(plan_for(repo, launchers), builder=builder()).commit == third


def test_runtimes_below_the_floor_are_replaced_and_rejected(world):
    repo, launchers, _ = world
    old = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    (repo / "packaging").mkdir()
    floor = commit(repo, "packaging/runtime-floor", old + "\n")
    assert runtime_deploy.floor_of(repo, floor) == old
    unrelated = subprocess.run(["git", "-C", str(repo), "commit-tree", "-m", "x", "HEAD^{tree}"],
                               capture_output=True, text=True, check=True).stdout.strip()
    assert runtime_deploy.meets(repo, old, floor) and not runtime_deploy.meets(repo, floor, old)
    assert not runtime_deploy.meets(repo, floor, unrelated)
    plan = runtime_deploy.Plan(repo, floor, launchers, floor=floor, wait_s=2)
    assert runtime_deploy.healthy(plan) is None
    outcome = runtime_deploy.ensure(plan, builder=builder())
    assert outcome.action == "switched" and runtime_store.selection()["commit"] == floor
    assert [c.commit for c in runtime_store.candidates()] == [floor]


def test_collection_keeps_the_newest_verified_and_never_a_tree_a_process_runs_in(world):
    _, _, _ = world
    root = runtime.directory()
    prefixes = []
    for index in range(6):
        prefix = root / f"{index:040x}" / f"{index:032x}"
        prefix.mkdir(parents=True)
        runtime_store.mark_verified(runtime.Runtime(prefix, f"{index:040x}", "0.1.0", runtime.identity()))
        os.utime(prefix / runtime_store.MARK, None)
        data = json.loads((prefix / runtime_store.MARK).read_text())
        data["verified_at"] = 1000.0 + index
        (prefix / runtime_store.MARK).write_text(json.dumps(data))
        prefixes.append(prefix)
    live = subprocess.Popen(["sleep", "30"], cwd=prefixes[0])
    try:
        time.sleep(0.2)
        gone = runtime_store.collect(set(), keep=2)
    finally:
        live.kill()
        live.wait()
    assert set(gone) == {prefixes[1], prefixes[2], prefixes[3]}
    assert prefixes[0].exists() and prefixes[4].exists() and prefixes[5].exists()


def test_a_live_owner_of_the_install_claim_blocks_and_a_dead_owner_is_recovered(world, tmp_path):
    base = tmp_path / "claims"
    base.mkdir()
    store = Claims(base, 600)
    target = tmp_path / "launchers"
    target.mkdir()
    store.claim(Identity("someone-else", AGENT), "install", str(target), {"pid": os.getpid(), "ttl_s": 600})
    with pytest.raises(runtime_deploy.DeployError, match="claimed by someone-else"), \
            runtime_deploy.owning([target], wait_s=0.2, store=store):
        pass
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    store.release(Identity("someone-else", AGENT), "install", str(target))
    store.claim(Identity("gone", AGENT), "install", str(target), {"pid": dead.pid, "ttl_s": 600})
    with runtime_deploy.owning([target], wait_s=0.2, store=store):
        assert store.who("install", str(target))["owner"].startswith("runtime-deploy:")
    assert store.who("install", str(target)) is None


def test_two_ensures_do_not_build_twice(world):
    repo, launchers, _ = world
    commit(repo, "a")
    built = []
    runtime_deploy._prepare_root()
    with only_one(runtime.directory() / "deploy.lock", note="other build"):
        outcome = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(log=built))
    assert outcome.action == "busy" and built == []


def test_status_reports_selection_and_build_state(world, capsys):
    repo, launchers, _ = world
    head = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert runtime_cli.main(["status", "--checkout", str(repo), "--launchers", str(launchers), "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["selected"] == head and record["current"] and record["healthy"]
    assert runtime_cli.main(["status", "--checkout", str(repo), "--launchers", str(launchers)]) == 0
    assert "healthy" in capsys.readouterr().out


def test_outcomes_reach_the_board_as_announcements_and_failures_as_incidents(world, monkeypatch, tmp_path):
    repo, launchers, _ = world
    commit(repo, "a")
    kit = Kit(Path(os.environ["ML_STACK_WORKSPACE_HOME"]))
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    first = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert runtime_board.announce(first, "", agent="runtime-agent", label="runtime-cd")
    second_commit = commit(repo, "b")
    failed = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(hook_code=3))
    assert failed.action == "failed"
    assert runtime_board.announce(failed, first.commit, agent="runtime-agent", label="runtime-cd")
    assert not runtime_board.announce(runtime_deploy.Outcome("current", second_commit), "", agent="runtime-agent")
    rows = [row for row in kit.ws.bus.outbox("runtime-agent", 20) if row["to"] == "#announcements"]
    assert len(rows) == 2
    texts = json.dumps(rows, default=str)
    assert first.commit[:7] in texts and "diagnostic=" in texts and second_commit[:7] in texts
    assert "stays selected" in texts


def test_the_device_profile_carries_the_runtime_commit(world):
    from ml_stack.workspace import device_metadata
    assert device_metadata.normalize({"runtime_commit": "a" * 40})["runtime_commit"] == "a" * 40
    assert device_metadata.normalize({})["runtime_commit"] == ""
    with pytest.raises(ValueError):
        device_metadata.normalize({"runtime_commit": "rm -rf /"})


def test_the_daemon_follows_the_selected_runtime_only_when_idle(world, monkeypatch):
    from ml_stack.fleet import updates
    assert updates.follow_runtime(idle=lambda: True) is None
    repo, launchers, _ = world
    head = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    monkeypatch.setattr(updates, "_installed_commit", lambda: "b" * 40)
    restarts, idle = [], []
    monkeypatch.setattr(updates, "restart_after_update", lambda: restarts.append(updates.selected_commit()) or "restarted")
    thread = updates.follow_runtime(idle=lambda: bool(idle), schedule=updates.UpdateSchedule(interval=0.05, first_after_s=0))
    time.sleep(0.4)
    assert restarts == []
    idle.append(True)
    thread.join(timeout=10)
    assert restarts == [head]


def test_background_ensure_is_cheap_when_current_and_starts_one_build_otherwise(world, capsys):
    repo, launchers, _ = world
    commit(repo, "a")
    argv = ["ensure", "--background", "--checkout", str(repo), "--launchers", str(launchers)]
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert runtime_cli.main(argv) == 0 and "is current" in capsys.readouterr().out
    commit(repo, "b")
    with only_one(runtime.directory() / "deploy.lock", note="busy"):
        assert runtime_cli.main(argv) == 0
    assert "already running" in capsys.readouterr().out


def test_a_launcher_with_no_usable_runtime_says_so_and_exits(world):
    repo, launchers, _ = world
    commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    import shutil
    shutil.rmtree(runtime.directory())
    runtime.directory().mkdir(parents=True)
    done = subprocess.run([sys.executable, str(launchers / "ml-stack-workspace")], capture_output=True, text=True, timeout=60)
    assert done.returncode != 0 and "no usable ml-stack runtime" in done.stderr


def test_a_command_without_a_recorded_or_named_launcher_directory_or_checkout_writes_nothing(world, capsys, tmp_path):
    repo, launchers, _ = world
    commit(repo, "a")
    assert runtime_cli.main(["ensure", "--checkout", str(repo)]) == 1
    assert "pass --launchers" in capsys.readouterr().err
    assert runtime_cli.main(["ensure", "--checkout", str(tmp_path), "--launchers", str(launchers)]) == 1
    assert "not a git checkout" in capsys.readouterr().err
    assert not list(launchers.iterdir()) and not runtime.directory().exists()
