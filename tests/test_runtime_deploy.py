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

from poolhouse import runtime, runtime_board, runtime_cli, runtime_deploy, runtime_store
from poolhouse.fleet import runtime_wheel
from poolhouse.lock import only_one
from poolhouse.workspace.claims import Claims
from poolhouse.workspace.identity import AGENT, Identity

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
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.delenv("POOLHOUSE_RUNTIME_ENSURE", raising=False)
    monkeypatch.setenv("POOLHOUSE_WORKSPACE_AGENT", "runtime-agent")
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "dev")
    (repo / "src" / "poolhouse").mkdir(parents=True)
    (repo / "src" / "poolhouse" / "__init__.py").write_text("")
    launchers = tmp_path / "bin"
    launchers.mkdir(mode=0o700)
    return repo, launchers, tmp_path


def wheel(stage, commit_id):
    path = stage / commit_id / "poolhouse-0.1.0-py3-none-any.whl"
    path.parent.mkdir(parents=True)
    contents = {
        "poolhouse/__init__.py": b"",
        "poolhouse/workspace/__init__.py": b"",
        "poolhouse/workspace/cli.py": f"def main():\n    print('help {commit_id[:7]}')\n    return 0\n".encode(),
        "poolhouse-0.1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: poolhouse\nVersion: 0.1.0\n",
        "poolhouse-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        "poolhouse-0.1.0.dist-info/entry_points.txt": b"[console_scripts]\npoolhouse-workspace = poolhouse.workspace.cli:main\n",
        "poolhouse-0.1.0.dist-info/RECORD": b"",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return path


def builder(hook_code=0, log=None, epoch=0):
    def build(plan, stage, prefix):
        source = stage / "source" / "scripts" / "hooks"
        source.mkdir(parents=True)
        if epoch:
            (stage / "source" / "packaging").mkdir()
            (stage / "source" / "packaging" / "runtime-floor").write_text(f"{epoch}\n")
        for name in runtime_deploy.HOOKS:
            (source / name).write_text(f"import sys\nsys.exit({hook_code})\n")
        built = wheel(stage, plan.commit)
        runtime_wheel.stamp(built, plan.commit, plan.checkout)
        if log is not None:
            log.append(plan.commit)
        return runtime_wheel.prepare(built, plan.commit, timeout=120, target=runtime_wheel.Target(prefix))
    return build


def plan_for(repo, launchers, **kw):
    return runtime_deploy.Plan(repo, runtime_deploy.resolve_commit(repo), launchers, wait_s=kw.pop("wait_s", 2), **kw)


def launcher_output(launchers):
    done = subprocess.run([sys.executable, str(launchers / "poolhouse-workspace")], capture_output=True, text=True,
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
    before = (launchers / "poolhouse-workspace").read_bytes()
    second = commit(repo, "b")
    outcome = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(hook_code=3))
    assert outcome.action == "failed" and "claude-session-start" in outcome.detail
    assert runtime_store.selection()["commit"] == first
    assert (launchers / "poolhouse-workspace").read_bytes() == before
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
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(epoch=1))
    assert runtime_store.epoch_of(runtime_store.candidates()[0].prefix) == 1
    (repo / "packaging").mkdir()
    head = commit(repo, "packaging/runtime-floor", "2\n")
    assert runtime_deploy.floor_of(repo, head) == 2 and runtime_deploy.floor_of(repo, old) == 0
    plan = runtime_deploy.Plan(repo, head, launchers, floor=2, wait_s=2)
    assert runtime_deploy.healthy(plan) is None
    outcome = runtime_deploy.ensure(plan, builder=builder(epoch=2))
    assert outcome.action == "switched" and runtime_store.selection()["commit"] == head
    assert [c.commit for c in runtime_store.candidates()] == [head]


def test_a_floor_that_is_not_an_integer_is_an_error(world):
    repo, _, _ = world
    (repo / "packaging").mkdir()
    head = commit(repo, "packaging/runtime-floor", "5967b960\n")
    with pytest.raises(runtime_deploy.DeployError, match="runtime-floor"):
        runtime_deploy.floor_of(repo, head)


def test_collection_keeps_the_newest_verified_and_never_a_tree_a_process_runs_in(world):
    _, _, _ = world
    root = runtime.directory()
    prefixes = []
    for index in range(6):
        prefix = root / f"{index:040x}" / f"{index:032x}"
        prefix.mkdir(parents=True)
        runtime_store.mark_created(prefix, "runtime-agent", "ensure")
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
            runtime_deploy.owning([target], wait_s=0.2, who=Identity("deployer", AGENT), store=store):
        pass
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    store.release(Identity("someone-else", AGENT), "install", str(target))
    store.claim(Identity("gone", AGENT), "install", str(target), {"pid": dead.pid, "ttl_s": 600})
    with runtime_deploy.owning([target], wait_s=0.2, who=Identity("deployer", AGENT), store=store):
        assert store.who("install", str(target))["owner"] == "deployer"
    assert store.who("install", str(target)) is None


def test_two_ensures_do_not_build_twice(world):
    repo, launchers, _ = world
    commit(repo, "a")
    built = []
    runtime_deploy.prepare_root()
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
    kit = Kit(Path(os.environ["POOLHOUSE_WORKSPACE_HOME"]))
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "state"))
    first = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert runtime_board.announce(first, "", agent="runtime-agent")
    second_commit = commit(repo, "b")
    failed = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(hook_code=3))
    assert failed.action == "failed"
    assert runtime_board.announce(failed, first.commit, agent="runtime-agent")
    assert not runtime_board.announce(runtime_deploy.Outcome("current", second_commit), "", agent="runtime-agent")
    rows = [row for row in kit.ws.bus.outbox("runtime-agent", 20) if row["to"] == "#announcements"]
    assert len(rows) == 2
    texts = json.dumps(rows, default=str)
    assert first.commit[:7] in texts and "diagnostic=" in texts and second_commit[:7] in texts
    assert "stays selected" in texts


def test_the_device_profile_carries_the_runtime_commit(world):
    from poolhouse.workspace import device_metadata
    assert device_metadata.normalize({"runtime_commit": "a" * 40})["runtime_commit"] == "a" * 40
    assert device_metadata.normalize({})["runtime_commit"] == ""
    with pytest.raises(ValueError):
        device_metadata.normalize({"runtime_commit": "rm -rf /"})


def test_the_daemon_follows_the_selected_runtime_only_when_idle(world, monkeypatch):
    from poolhouse.fleet import updates
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
    done = subprocess.run([sys.executable, str(launchers / "poolhouse-workspace")], capture_output=True, text=True, timeout=60)
    assert done.returncode != 0 and "no usable poolhouse runtime" in done.stderr


def test_a_command_with_an_invalid_checkout_writes_nothing(world, capsys, tmp_path):
    repo, launchers, _ = world
    commit(repo, "a")
    assert runtime_cli.main(["ensure", "--checkout", str(tmp_path), "--launchers", str(launchers)]) == 1
    assert "not a poolhouse source checkout" in capsys.readouterr().err
    assert not list(launchers.iterdir()) and not runtime.directory().exists()


def test_a_current_runtime_answers_without_taking_the_build_lock(world):
    repo, launchers, _ = world
    commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    with only_one(runtime.directory() / "deploy.lock", note="other build"):
        assert runtime_deploy.ensure(plan_for(repo, launchers), builder=builder()).action == "current"


def tree_at(root, commit_id, verified_at, name, **options):
    importable, created = options.get("importable", True), options.get("created", True)
    prefix = root / commit_id / f"{len(name):032x}"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.write_text(f"#!/bin/sh\necho {name}\n")
    python.chmod(0o700)
    if importable:
        (prefix / "lib" / "python3.13" / "site-packages" / "poolhouse").mkdir(parents=True)
        (prefix / "lib" / "python3.13" / "site-packages" / "poolhouse" / "__init__.py").write_text("")
    (prefix / "verified.json").write_text(json.dumps({"commit": commit_id, "version": "0", "identity": "x",
                                                      "verified_at": verified_at}))
    if created:
        (prefix / "created.json").write_text(json.dumps({"tool": "poolhouse-runtime", "agent": "a"}))
    return prefix


def render_launcher(tmp_path, root, python="/nonexistent/bin/python"):
    path = tmp_path / "poolhouse-probe"
    path.write_text(runtime.LAUNCHER.format(root=str(root), python=python, arguments="['-c', 'pass']", name=path.name))
    path.chmod(0o700)
    return path


def run_launcher(path, cwd=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60, cwd=cwd, env=env)


def test_launcher_fallback_prefers_the_selection_honours_the_hold_and_survives_a_bad_timestamp(tmp_path):
    root = tmp_path / "root"
    a, b, c = "a" * 40, "b" * 40, "c" * 40
    tree_at(root, a, "garbage", "tree-a")
    chosen = tree_at(root, b, 100.0, "tree-b")
    tree_at(root, c, 200.0, "tree-c")
    launcher = render_launcher(tmp_path, root)
    (root / "selected.json").write_text(json.dumps({"prefix": str(chosen), "commit": b}))
    assert run_launcher(launcher).stdout.strip() == "tree-b"
    (root / "deploy.json").write_text(json.dumps({"held": {"commit": b, "reason": "rolled back"}}))
    assert run_launcher(launcher).stdout.strip() == "tree-c"


def test_launcher_recovery_spawn_is_validated_logged_rate_limited_and_ignores_the_working_directory(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    source = tmp_path / "source"
    (source / "src" / "poolhouse").mkdir(parents=True)
    (source / "src" / "poolhouse" / "__init__.py").write_text("")
    marker, shadow_marker = tmp_path / "marker", tmp_path / "shadow"
    (source / "src" / "poolhouse" / "runtime_cli.py").write_text(
        f"import pathlib\npathlib.Path({str(marker)!r}).open('a').write('ran\\n')\nprint('recovery output')\n")
    shadow = tmp_path / "shadow-cwd" / "poolhouse"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")
    (shadow / "runtime_cli.py").write_text(f"import pathlib\npathlib.Path({str(shadow_marker)!r}).write_text('x')\n")
    launcher = render_launcher(tmp_path, root)
    (root / "deploy.json").write_text(json.dumps({"checkout": 5}))
    done = run_launcher(launcher, cwd=shadow.parent)
    assert done.returncode != 0 and "Traceback" not in done.stderr and not marker.exists()
    (root / "deploy.json").write_text(json.dumps({"checkout": str(tmp_path / "elsewhere")}))
    run_launcher(launcher, cwd=shadow.parent)
    assert not marker.exists()
    (root / "deploy.json").write_text(json.dumps({"checkout": str(source)}))
    run_launcher(launcher, cwd=shadow.parent)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and not (root / "ensure.log").exists():
        time.sleep(0.1)
    time.sleep(1)
    run_launcher(launcher, cwd=shadow.parent)
    time.sleep(1)
    assert marker.read_text() == "ran\n" and not shadow_marker.exists()
    assert "recovery output" in (root / "ensure.log").read_text()


def test_launcher_directories_are_absolute_and_a_relative_recorded_one_is_refused(world, monkeypatch, tmp_path):
    runtime_deploy.prepare_root()
    runtime_store.write_state({"launchers": "relative/bin"})
    with pytest.raises(runtime_deploy.DeployError, match="absolute"):
        runtime_cli._launchers("")
    monkeypatch.chdir(tmp_path)
    assert runtime_cli._launchers("bin") == tmp_path.resolve() / "bin"


def test_a_checkout_comes_only_from_the_flag_or_the_record_and_must_be_an_poolhouse_source(world, monkeypatch, tmp_path):
    repo, _, _ = world
    commit(repo, "a")
    monkeypatch.chdir(repo)
    with pytest.raises(runtime_deploy.DeployError):
        runtime_cli._checkout("")
    with pytest.raises(runtime_deploy.DeployError, match="poolhouse"):
        runtime_cli._checkout(str(tmp_path))
    assert runtime_cli._checkout(str(repo)) == repo.resolve()
    runtime_deploy.prepare_root()
    runtime_store.write_state({"checkout": str(tmp_path)})
    with pytest.raises(runtime_deploy.DeployError):
        runtime_cli._checkout("")


def test_a_failure_recording_state_after_the_selection_is_published_keeps_the_tree(world, monkeypatch):
    repo, launchers, _ = world
    head = commit(repo, "a")

    def refuse(update, root=None):
        raise OSError("disk full")

    monkeypatch.setattr(runtime_store, "write_state", refuse)
    built = runtime_deploy.build_and_switch(plan_for(repo, launchers), builder())
    assert built.prefix.is_dir() and runtime_store.selection()["commit"] == head
    assert launcher_output(launchers) == f"help {head[:7]}"


def test_a_failure_before_the_selection_restores_the_previous_launchers_and_discards_the_tree(world, monkeypatch):
    repo, launchers, _ = world
    first = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    commit(repo, "b")

    def refuse(chosen):
        raise OSError("cannot publish")

    monkeypatch.setattr(runtime, "publish", refuse)
    outcome = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert outcome.action == "failed"
    assert launcher_output(launchers) == f"help {first[:7]}"


def test_an_unusable_selection_never_stops_the_callers_that_would_forward_to_it(world, tmp_path):
    from poolhouse import jobs
    runtime_deploy.prepare_root()
    (runtime.directory() / "selected.json").write_text("{not json")
    assert runtime.available() is None
    assert runtime.forward("json.tool", []) is False
    started = jobs.detach("calendar", ["2001"], log=tmp_path / "out.log")
    assert started.command[0] == sys.executable


def test_forward_does_not_verify_when_this_process_runs_from_the_selected_prefix(world, monkeypatch):
    runtime_deploy.prepare_root()
    (runtime.directory() / "selected.json").write_text(json.dumps({"prefix": sys.prefix, "commit": "a" * 40}))
    monkeypatch.setattr(runtime, "verify", lambda row: pytest.fail("verified"))
    assert runtime.forward("json.tool", []) is False


def test_a_rollback_hold_does_not_block_recovery_and_background_does_not_spawn_when_held(world, capsys, monkeypatch):
    repo, launchers, _ = world
    first = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    commit(repo, "b")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    third = commit(repo, "c")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    runtime_deploy.rollback(plan_for(repo, launchers))
    monkeypatch.setattr(runtime_cli, "_start_background", lambda argv: pytest.fail("spawned while held"))
    argv = ["ensure", "--background", "--checkout", str(repo), "--launchers", str(launchers)]
    assert runtime_cli.main(argv) == 0 and "held" in capsys.readouterr().out
    assert runtime_cli.main(["status", "--checkout", str(repo), "--launchers", str(launchers)]) == 0
    assert "held" in capsys.readouterr().out
    import shutil
    shutil.rmtree(runtime_store.selection()["prefix"])
    outcome = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(hook_code=3))
    assert outcome.action == "recovered"
    assert runtime_store.selection()["commit"] == first
    assert runtime_deploy.healthy(plan_for(repo, launchers)) is not None
    assert third not in {runtime_store.selection()["commit"]}


def test_discarding_a_tree_unlaunches_it_even_when_deletion_fails(world, monkeypatch):
    import shutil
    runtime_deploy.prepare_root()
    prefix = tree_at(runtime.directory(), "d" * 40, 5.0, "tree-d")
    monkeypatch.setattr(shutil, "rmtree", lambda *a, **k: None)
    runtime_store.discard(prefix)
    assert prefix.exists() and not (prefix / "verified.json").exists() and runtime_store.rejected(prefix)


def test_board_text_carries_only_full_hex_commits(world, monkeypatch):
    posted = []
    monkeypatch.setattr(runtime_board.cli, "main", lambda argv: posted.append(argv) or 0)
    good = runtime_deploy.Outcome("switched", "a" * 40)
    assert runtime_board.announce(good, "x; rm -rf /", agent="someone")
    assert "none" in posted[-1][2] and "rm" not in posted[-1][2]
    assert not runtime_board.announce(runtime_deploy.Outcome("switched", "z" * 40), "", agent="someone")


def test_the_session_start_refresh_is_bounded_and_the_smoke_environment_disables_it(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("workspace_hook_under_test", Path(__file__).resolve().parents[1] / "scripts/hooks/workspace_hook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = []
    monkeypatch.delenv("POOLHOUSE_RUNTIME_ENSURE", raising=False)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: seen.append(k.get("timeout")) or subprocess.CompletedProcess(a, 0, "", ""))
    module.refresh_runtime("SessionStart")
    assert seen and all(t is not None and t <= 6 for t in seen) and len(seen) == 1
    env = runtime_deploy._clean_environment(Path("/tmp/x"), Path("/tmp/y"))
    assert env["POOLHOUSE_RUNTIME_ENSURE"] == "off"


def test_a_claim_is_released_only_when_its_pid_started_after_the_claim_was_taken(tmp_path, monkeypatch):
    from poolhouse.workspace import claims as claims_module
    from poolhouse.workspace.claims import Claims
    store = Claims(tmp_path, 600)
    target = tmp_path / "t"
    target.mkdir()
    store.claim(Identity("holder", AGENT), "install", str(target), {"pid": os.getpid(), "ttl_s": 600})
    since = store.who("install", str(target))["since"]
    monkeypatch.setattr(claims_module, "started_at", lambda pid: since - 3600)
    assert store.who("install", str(target))["owner"] == "holder"
    monkeypatch.setattr(claims_module, "started_at", lambda pid: since + 3600)
    assert store.who("install", str(target)) is None


def test_status_and_a_tracker_ensure_need_no_recorded_launcher_directory(world, capsys):
    repo, launchers, _ = world
    head = commit(repo, "a")
    assert runtime_cli.main(["status", "--checkout", str(repo)]) == 0
    assert not runtime.directory().exists()
    outcome = runtime_deploy.ensure(runtime_deploy.Plan(repo, head, None, wait_s=2), builder=builder())
    assert outcome.action == "switched" and runtime_store.selection()["commit"] == head
    assert not list(launchers.iterdir())


def test_tests_cannot_write_launchers_outside_the_state_root_or_the_temporary_directory(world):
    from poolhouse import runtime_launchers
    fake = runtime.Runtime(Path("/nonexistent"), "a" * 40, "0", runtime.identity())
    import sysconfig
    real = Path(sysconfig.get_path("scripts", vars={"base": sys.base_prefix, "platbase": sys.base_prefix})) / "poolhouse-never"
    with pytest.raises(OSError, match="test"):
        runtime.write_launcher(real, "poolhouse.cli", "main", fake)
    with pytest.raises(OSError, match="test"):
        runtime_launchers.install(real.parent, fake)
    assert not real.exists()


def test_the_install_claim_is_held_by_the_acting_agent_and_rollback_needs_one(world, monkeypatch):
    repo, launchers, _ = world
    commit(repo, "a")
    seen = []
    real = runtime_deploy.owning

    def spy(paths, **kwargs):
        seen.append(kwargs["who"].id if kwargs["who"] else None)
        return real(paths, **kwargs)

    monkeypatch.setattr(runtime_deploy, "owning", spy)
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    assert seen and all(owner.startswith("runtime-agent") for owner in seen)
    commit(repo, "b")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    monkeypatch.delenv("POOLHOUSE_WORKSPACE_AGENT")
    with pytest.raises(runtime_deploy.DeployError, match="authenticated agent"):
        runtime_deploy.rollback(plan_for(repo, launchers))


def test_without_an_agent_ensure_and_recovery_run_unclaimed_and_post_nothing(world, monkeypatch):
    repo, launchers, _ = world
    commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    second = commit(repo, "b")
    monkeypatch.delenv("POOLHOUSE_WORKSPACE_AGENT")
    seen = []
    real = runtime_deploy.owning

    def spy(paths, **kwargs):
        seen.append(kwargs["who"])
        return real(paths, **kwargs)

    monkeypatch.setattr(runtime_deploy, "owning", spy)
    assert runtime_deploy.ensure(plan_for(repo, launchers), builder=builder()).action == "switched"
    import shutil
    shutil.rmtree(runtime_store.selection()["prefix"])
    assert runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(hook_code=3)).action == "recovered"
    assert seen and all(who is None for who in seen)
    assert not runtime_board.announce(runtime_deploy.Outcome("switched", second), "")


def test_the_acting_identity_is_the_workspace_agent_and_collides_with_its_harness_claim(world):
    from poolhouse.workspace import limits
    repo, launchers, _ = world
    commit(repo, "a")
    who = runtime_deploy.acting(plan_for(repo, launchers))
    base = limits.root()
    store = Claims(base, 600)
    target = world[2] / "area"
    target.mkdir()
    store.claim(Identity(who.id, AGENT), "install", str(target), {"ttl_s": 600})
    other = Identity("somebody-else", AGENT)
    with runtime_deploy.owning([target], wait_s=0.1, who=who, store=store):
        assert store.who("install", str(target))["owner"] == who.id
        with pytest.raises(runtime_deploy.DeployError), runtime_deploy.owning([target], wait_s=0.1, who=other, store=store):
            pass


def test_the_acting_identity_is_the_physical_owner_and_needs_the_claim_capability(world, monkeypatch):
    from types import SimpleNamespace

    from poolhouse.workspace import harness_remote, project_connection
    repo, launchers, _ = world
    commit(repo, "a")
    info = {"id": "runtime-agent", "role": "agent", "can": ["claim", "send"], "project": {"key": "pk"}}
    remote = SimpleNamespace(host="h", project_id="pk", call=lambda name, token: info)
    monkeypatch.setattr(runtime_deploy.cli, "_project_connection", lambda cwd=None: {"host": "h"})
    monkeypatch.setattr(runtime_deploy.cli, "_context",
                        lambda args, connection: (project_connection.BoardWorkspace(remote, "t"), "t"))
    got = runtime_deploy.acting(plan_for(repo, launchers))
    assert got.id == harness_remote.physical_owner(remote, Identity("runtime-agent", AGENT)).id
    info["can"] = ["send"]
    assert runtime_deploy.acting(plan_for(repo, launchers)) is None
    info["can"], info["project"] = ["claim"], {"key": "other"}
    assert runtime_deploy.acting(plan_for(repo, launchers)) is None


def test_a_symlinked_family_directory_is_never_listed_collected_or_deleted(world, tmp_path):
    runtime_deploy.prepare_root()
    outside = tmp_path / "outside"
    prefix = tree_at(outside, "a" * 40, 5.0, "tree-x")
    (runtime.directory() / ("a" * 40)).symlink_to(outside / ("a" * 40))
    assert runtime_store.trees() == [] and runtime_store.candidates() == []
    runtime_store.collect(set(), keep=0)
    assert prefix.exists() and (prefix / "verified.json").exists()


def test_the_launcher_fallback_needs_this_tools_creation_record(tmp_path):
    root = tmp_path / "root"
    tree_at(root, "a" * 40, 100.0, "tree-a", created=False)
    launcher = render_launcher(tmp_path, root)
    done = run_launcher(launcher)
    assert done.returncode != 0 and "no usable" in done.stderr
    tree_at(root, "b" * 40, 50.0, "tree-b")
    assert run_launcher(launcher).stdout.strip() == "tree-b"


def test_status_lists_trees_it_does_not_manage_and_never_collects_them(world, capsys):
    runtime_deploy.prepare_root()
    foreign = foreign_trees(runtime.directory())
    live = subprocess.Popen(["sleep", "30"], cwd=foreign[2])
    try:
        time.sleep(0.2)
        repo, launchers, _ = world
        commit(repo, "a")
        assert runtime_cli.main(["status", "--checkout", str(repo), "--launchers", str(launchers), "--json"]) == 0
        record = json.loads(capsys.readouterr().out)
        listed = {row["path"]: row for row in record["unmanaged"]}
        assert set(listed) == {str(p) for p in foreign} and listed[str(foreign[2])]["in_use"] and not listed[str(foreign[0])]["in_use"]
        assert listed[str(foreign[0])]["bytes"] > 0
        assert runtime_cli.main(["status", "--checkout", str(repo), "--launchers", str(launchers)]) == 0
        assert "unmanaged" in capsys.readouterr().out
        runtime_store.collect(set(), keep=0)
    finally:
        live.kill()
        live.wait()
    assert all(p.exists() for p in foreign)


def test_a_failed_first_switch_restores_the_original_launcher_bytes_and_removes_new_ones(world, monkeypatch):
    repo, launchers, _ = world
    (launchers / "poolhouse-workspace").write_text("original")
    (launchers / "poolhouse-workspace").chmod(0o700)
    commit(repo, "a")
    monkeypatch.setattr(runtime, "publish", lambda chosen: (_ for _ in ()).throw(OSError("cannot publish")))
    assert runtime_deploy.ensure(plan_for(repo, launchers), builder=builder()).action == "failed"
    assert (launchers / "poolhouse-workspace").read_text() == "original"
    assert [p.name for p in launchers.iterdir()] == ["poolhouse-workspace"]


def test_a_commit_that_failed_is_not_rebuilt_for_ten_minutes_unless_forced_by_an_agent(world, monkeypatch):
    repo, launchers, _ = world
    commit(repo, "a")
    built = []
    assert runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(hook_code=3, log=built)).action == "failed"
    again = runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(log=built))
    assert again.action == "backoff" and len(built) == 1 and again.ok
    assert runtime_deploy.ensure(plan_for(repo, launchers), force_build=True, builder=builder(log=built)).action == "switched"
    runtime_store.write_state({"last_failure": {"commit": "f" * 40, "at": time.time(), "detail": "x"}})
    commit(repo, "b")
    monkeypatch.delenv("POOLHOUSE_WORKSPACE_AGENT")
    with pytest.raises(runtime_deploy.DeployError, match="authenticated agent"):
        runtime_deploy.ensure(plan_for(repo, launchers), force_build=True, builder=builder())


def test_an_unset_state_root_is_refused_while_a_test_runs(world, monkeypatch, tmp_path):
    monkeypatch.delenv("POOLHOUSE_HOME")
    with pytest.raises(OSError, match="POOLHOUSE_HOME"):
        runtime.confine_tests(tmp_path / "bin")


def test_a_malformed_git_file_does_not_stop_the_session_hook(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("workspace_hook_d8", Path(__file__).resolve().parents[1] / "scripts/hooks/workspace_hook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / ".git").write_bytes(b"\xff\xfe not text")
    assert module.primary_checkout(tmp_path) == tmp_path
    (tmp_path / ".git").unlink()
    (tmp_path / ".git").mkdir()
    assert module.primary_checkout(tmp_path) == tmp_path


def test_the_background_log_is_private_and_the_runtimes_root_is_owner_only(world):
    repo, _, _ = world
    commit(repo, "a")
    runtime_cli._start_background(["status", "--checkout", str(repo)])
    time.sleep(0.5)
    assert (runtime.directory() / "ensure.log").stat().st_mode & 0o777 == 0o600
    assert runtime.directory().stat().st_mode & 0o777 == 0o700


def foreign_trees(root):
    """Trees another tool made: one unmarked, one marked by another tool, one with a live process inside."""
    made = []
    for index, marker in enumerate((None, {"tool": "other-tool", "agent": "someone"}, None)):
        prefix = root / f"{index + 1:040x}" / f"{index + 1:032x}"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin" / "python").write_text("#!/bin/sh\n")
        (prefix / "lib" / "python3.13" / "site-packages" / "poolhouse").mkdir(parents=True)
        (prefix / "lib" / "python3.13" / "site-packages" / "poolhouse" / "__init__.py").write_text("")
        if marker:
            (prefix / "created.json").write_text(json.dumps(marker))
        os.utime(prefix, (1000.0, 1000.0))
        made.append(prefix)
    return made


def snapshot(prefixes):
    return {str(path): path.stat().st_size for prefix in prefixes for path in prefix.rglob("*") if path.is_file()}


def test_foreign_trees_survive_ensure_rollback_recovery_and_collection(world):
    repo, launchers, _ = world
    runtime_deploy.prepare_root()
    foreign = foreign_trees(runtime.directory())
    before = snapshot(foreign)
    live = subprocess.Popen(["sleep", "60"], cwd=foreign[2])
    try:
        commit(repo, "a")
        runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
        commit(repo, "b")
        runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
        commit(repo, "c")
        runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
        runtime_deploy.rollback(plan_for(repo, launchers))
        import shutil
        shutil.rmtree(runtime_store.selection()["prefix"])
        runtime_deploy.ensure(plan_for(repo, launchers), builder=builder(hook_code=3))
        runtime_store.collect(set(), keep=0)
        for prefix in foreign:
            runtime_store.reject(prefix, "x")
            runtime_store.discard(prefix)
    finally:
        live.kill()
        live.wait()
    assert snapshot(foreign) == before
    assert all(prefix.exists() and not (prefix / "rejected").exists() for prefix in foreign)
    assert not {str(p) for p in foreign} & {str(c.prefix) for c in runtime_store.candidates()}
    assert all(not runtime_store.ours(prefix) for prefix in foreign)


def test_a_tree_this_tool_created_is_marked_before_anything_else_and_stale_unverified_ones_are_collected(world):
    repo, launchers, _ = world
    head = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    prefix = Path(runtime_store.selection()["prefix"])
    marker = json.loads((prefix / "created.json").read_text())
    assert marker["tool"] == "poolhouse-runtime" and marker["command"] == "ensure" and marker["agent"] == "runtime-agent"
    assert marker["pid"] == os.getpid()
    ours = runtime.directory() / head / ("e" * 32)
    ours.mkdir(parents=True)
    runtime_store.mark_created(ours, "runtime-agent", "ensure")
    os.utime(ours, (1000.0, 1000.0))
    assert ours in runtime_store.collect({prefix})
