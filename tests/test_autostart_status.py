"""`status` and `verify` compare the installed units with the prepared ones; agents can read the result and cannot change it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from autostart_support import Recorder, environment, seams, staged, write_launcher
from workspace_kit import Kit, clean_env, cli

from ml_stack import home
from ml_stack.fleet import autostart_check, autostart_guard, autostart_ledger
from ml_stack.fleet.autostart_apply import install
from ml_stack.fleet.autostart_manifest import launcher_problem
from ml_stack.fleet.autostart_prepare import platform_of
from ml_stack.lock import only_one
from ml_stack.workspace import autostart_status


@pytest.fixture
def user(tmp_path, monkeypatch):
    return environment(tmp_path, monkeypatch)


def installed(tmp_path, roles=("pool-daemon",)):
    done = staged(tmp_path, roles=roles)
    recorder = Recorder()
    got = install(done.path, seams=seams(recorder, typed=lambda _p: done.manifest.id))
    assert got.ok, got.lines
    return done, recorder


def state(recorder) -> autostart_check.Report:
    return autostart_check.check(runner=recorder, platform="linux")


def test_nothing_prepared_nothing_installed(tmp_path, user):
    assert autostart_check.check().state == "not-prepared"


def test_a_prepared_manifest_that_is_not_installed_is_missing(tmp_path, user):
    staged(tmp_path)
    assert autostart_check.check(platform="linux").state == "missing"


def test_an_expired_preparation_is_not_a_preparation(tmp_path, user):
    done = staged(tmp_path)
    assert autostart_check.prepared(now=done.manifest.expires + 1, platform="linux") is None


def test_an_install_that_matches_its_manifest_is_current(tmp_path, user):
    _, recorder = installed(tmp_path)
    report = state(recorder)
    assert report.state == "current" and report.reasons == () and dict(report.roles) == {"pool-daemon": "current"}


def test_status_reads_and_changes_nothing(tmp_path, user):
    _, recorder = installed(tmp_path)
    before = sorted((p.relative_to(tmp_path).as_posix(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file())
    state(recorder)
    assert sorted((p.relative_to(tmp_path).as_posix(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()) == before


def test_an_edited_unit_is_drifted(tmp_path, user):
    _, recorder = installed(tmp_path)
    unit = user / ".config" / "systemd" / "user" / "ml-stack-traind.service"
    unit.write_text(unit.read_text() + "# edited\n")
    report = state(recorder)
    assert report.state == "drifted" and "edited" in report.reasons[0]


def test_a_removed_unit_is_drifted(tmp_path, user):
    _, recorder = installed(tmp_path)
    (user / ".config" / "systemd" / "user" / "ml-stack-traind.service").unlink()
    assert state(recorder).state == "drifted"


def test_a_missing_exec_target_is_drifted(tmp_path, user):
    done, recorder = installed(tmp_path)
    Path(done.manifest.roles[0].argv[0]).unlink()
    report = state(recorder)
    assert report.state == "drifted" and "missing" in " ".join(report.reasons)


def test_an_exec_target_that_is_no_longer_a_runtime_launcher_is_drifted(tmp_path, user):
    done, recorder = installed(tmp_path)
    target = Path(done.manifest.roles[0].argv[0])
    target.write_text("#!/bin/sh\necho hi\n")
    assert "not a runtime launcher" in " ".join(state(recorder).reasons)


def test_a_unit_the_service_manager_does_not_hold_is_drifted(tmp_path, user):
    _, recorder = installed(tmp_path)
    recorder.alive = False
    report = state(recorder)
    assert report.state == "drifted" and "not loaded" in report.reasons[0]


def test_a_newer_different_preparation_makes_the_install_stale(tmp_path, user):
    _, recorder = installed(tmp_path)
    staged(tmp_path, roles=("pool-daemon",), slots=2)
    assert state(recorder).state == "stale"


def test_an_identical_new_preparation_is_not_stale(tmp_path, user):
    _, recorder = installed(tmp_path)
    staged(tmp_path, roles=("pool-daemon",))
    assert state(recorder).state == "current"


def test_the_launcher_path_stays_launchable_while_the_selected_runtime_is_switched(tmp_path, user):
    done, recorder = installed(tmp_path, roles=("pool-daemon", "runtime-ensure"))
    launcher = Path(done.manifest.roles[0].argv[0])
    unit_hashes = {p.name: p.read_bytes() for p in (user / ".config" / "systemd" / "user").iterdir()}
    stop, failures = threading.Event(), []

    def watch():
        while not stop.is_set():
            problem = launcher_problem(launcher)
            if problem:
                failures.append(problem)

    thread = threading.Thread(target=watch)
    thread.start()
    try:
        for number in range(60):
            write_launcher(launcher.parent, launcher.name, tag=f"runtime-{number}")
            ran = subprocess.run([str(launcher), "--help"], capture_output=True, text=True, timeout=30)
            assert ran.returncode == 0 and ran.stdout.startswith(f"runtime-{number}"), ran.stderr
    finally:
        stop.set()
        thread.join()
    assert failures == []
    assert {p.name: p.read_bytes() for p in (user / ".config" / "systemd" / "user").iterdir()} == unit_hashes
    assert state(recorder).state == "current"


def test_the_cli_reports_and_verify_fails_unless_current(tmp_path, user, monkeypatch):
    for verb, expect in (("status", 0), ("verify", 1)):
        done = subprocess.run([sys.executable, "-m", "ml_stack.fleet.autostart", verb, "--json"],
                              capture_output=True, text=True, timeout=60,
                              env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
        assert done.returncode == expect, done.stderr
        assert json.loads(done.stdout)["state"] == "not-prepared"


def test_workspace_status_shows_the_state_and_posts_one_line_per_drift(tmp_path, user, monkeypatch):
    installed(tmp_path)
    kit = Kit(clean_env(monkeypatch, tmp_path))
    token = kit.agent("worker")
    first = autostart_status.fields(kit.ws, token)
    spent = autostart_ledger.read()["reported"]
    again = autostart_status.fields(kit.ws, token)
    assert autostart_ledger.read()["reported"] == spent != {}
    assert first["autostart"]["state"] == "drifted" == again["autostart"]["state"]
    lines = [m for m in kit.ws.bus.outbox("worker", 50) if "autostart drifted" in json.dumps(m)]
    assert len(lines) == 1
    posted = autostart_ledger.read()["reported"]["at"]
    later = autostart_status.fields(kit.ws, token, now=posted + autostart_status.REPEAT_S + 1)
    assert autostart_ledger.read()["reported"]["at"] == posted + autostart_status.REPEAT_S + 1
    assert later["autostart"]["reasons"]


def test_the_workspace_status_command_carries_the_autostart_field(tmp_path, user, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    done = cli(kit.base, kit.owner, "status", "--json")
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1])["autostart"]["state"] in autostart_check.STATES


def test_the_device_report_names_the_state(tmp_path, user):
    assert autostart_check.state_word(now=1.0) == "not-prepared"
    staged(tmp_path, platform=platform_of())
    assert autostart_check.state_word(now=2.0) == "not-prepared"
    assert autostart_check.state_word(now=1000.0) == "missing"


def test_the_ledger_is_versioned_and_defaults_empty(tmp_path, user):
    assert autostart_ledger.read()["installed"] == {}
    autostart_ledger.write({"installed": {}, "used": ["x"], "last": "", "reported": {}})
    assert json.loads(home.state("autostart", "ledger.json").read_text())["version"] == 1


def test_a_second_daemon_on_the_same_root_is_told_it_is_not_alone(tmp_path):
    with autostart_guard.running(tmp_path) as first:
        assert first is True
        with autostart_guard.running(tmp_path) as second:
            assert second is False
    with autostart_guard.running(tmp_path) as after:
        assert after is True
    with only_one(tmp_path / "traind.lock", wait=False):
        pass


def test_a_log_over_its_limit_is_copied_aside_and_emptied(tmp_path):
    log = tmp_path / "traind.log"
    for number in range(5):
        log.write_text(f"{number}" * 100)
        assert autostart_guard.rotate(log, max_bytes=50, keep=3)
        assert log.read_text() == ""
    assert sorted(p.name for p in tmp_path.iterdir()) == ["traind.log", "traind.log.1", "traind.log.2", "traind.log.3"]
    assert (tmp_path / "traind.log.1").read_text() == "4" * 100
    log.write_text("short")
    assert not autostart_guard.rotate(log, max_bytes=50, keep=3) and log.read_text() == "short"
