"""`install` and `rollback` against real files in a temporary account home, with the service manager recorded."""

from __future__ import annotations

import getpass
import json
import shlex
import sys
from pathlib import Path

import pytest
from autostart_support import Recorder, environment, seams, staged

from poolhouse import authority, home
from poolhouse.fleet import autostart_ledger
from poolhouse.fleet.autostart_apply import install, rollback
from poolhouse.person import HumanRequired


@pytest.fixture
def user(tmp_path, monkeypatch):
    return environment(tmp_path, monkeypatch)


def unit_dir(user: Path) -> Path:
    return user / ".config" / "systemd" / "user"


def run_install(done, recorder=None, **more):
    recorder = recorder or Recorder()
    return install(done.path, seams=seams(recorder, typed=lambda _prompt: done.manifest.id, **more)), recorder


def test_install_places_the_staged_bytes_loads_them_and_records_it(tmp_path, user):
    done = staged(tmp_path)
    outcome, recorder = run_install(done)
    assert outcome.ok, outcome.lines
    for role in done.manifest.roles:
        for unit in role.units:
            placed = unit_dir(user) / unit.name
            assert placed.read_bytes() == (done.path.parent / unit.name).read_bytes()
            assert oct(placed.stat().st_mode & 0o777) == "0o644"
    assert ["systemctl", "--user", "enable", "--now", "poolhouse-traind.service"] in recorder.ran
    assert ["systemctl", "--user", "enable", "--now", "poolhouse-runtime-ensure.timer"] in recorder.ran
    ledger = autostart_ledger.read()
    assert ledger["version"] == 1 and done.manifest.id in ledger["used"]
    assert set(ledger["installed"]) == {"pool-daemon", "runtime-ensure"}
    assert "rollback" in outcome.rollback and sys.executable in outcome.rollback
    row = [r for r in authority.audit_rows() if r["event"] == "autostart.install"][-1]
    assert row["by"] == "person" and row["manifest"] == done.manifest.id
    assert {u["sha256"] for u in row["units"]} == {u.sha256 for r in done.manifest.roles for u in r.units}


def test_install_backs_up_what_it_replaces_and_rollback_restores_it(tmp_path, user):
    done = staged(tmp_path, roles=("pool-daemon",))
    old = unit_dir(user) / "poolhouse-traind.service"
    old.parent.mkdir(parents=True)
    old.write_text("[Service]\nExecStart=/old\n")
    outcome, recorder = run_install(done)
    assert outcome.ok
    backup = home.state("autostart", "backups", done.manifest.id)
    assert (backup / "poolhouse-traind.service").read_text() == "[Service]\nExecStart=/old\n"
    assert json.loads((backup / "index.json").read_text())["version"] == 1
    assert old.read_bytes() != b"[Service]\nExecStart=/old\n"
    undone = rollback(seams=seams(recorder, typed=lambda _p: done.manifest.id))
    assert undone.ok, undone.lines
    assert old.read_text() == "[Service]\nExecStart=/old\n"
    assert autostart_ledger.read()["installed"] == {}
    assert [r for r in authority.audit_rows() if r["event"] == "autostart.rollback"]


def test_rollback_of_a_first_install_removes_the_files_and_unloads(tmp_path, user):
    done = staged(tmp_path)
    _, recorder = run_install(done)
    undone = rollback(seams=seams(recorder, typed=lambda _p: done.manifest.id))
    assert undone.ok
    assert not any(unit_dir(user).glob("poolhouse-*"))
    assert ["systemctl", "--user", "disable", "--now", "poolhouse-traind.service"] in recorder.ran
    assert recorder.alive is False


def test_a_failed_load_puts_the_previous_unit_back_and_spends_nothing(tmp_path, user):
    done = staged(tmp_path, roles=("pool-daemon",))
    old = unit_dir(user) / "poolhouse-traind.service"
    old.parent.mkdir(parents=True)
    old.write_text("old\n")
    outcome, _ = run_install(done, Recorder(fail_load=True))
    assert not outcome.ok and "load refused" in outcome.lines[0]
    assert old.read_text() == "old\n"
    assert autostart_ledger.read()["used"] == [] and autostart_ledger.read()["installed"] == {}


def test_a_unit_that_does_not_answer_its_health_probe_is_undone(tmp_path, user):
    done = staged(tmp_path, roles=("pool-daemon",))
    outcome, _ = run_install(done, probe=lambda health: False)
    assert not outcome.ok and "health probe" in outcome.lines[0]
    assert not (unit_dir(user) / "poolhouse-traind.service").exists()
    assert autostart_ledger.read()["installed"] == {}


def test_an_expired_manifest_is_refused_and_nothing_is_written(tmp_path, user):
    done = staged(tmp_path)
    outcome, recorder = run_install(done, now=done.manifest.expires + 1)
    assert not outcome.ok and "expired" in outcome.lines[0]
    assert not unit_dir(user).exists() and recorder.ran == []


def test_a_manifest_prepared_on_another_device_is_refused(tmp_path, user):
    done = staged(tmp_path, device="d" * 64)
    outcome, _ = run_install(done)
    assert not outcome.ok and "another device" in " ".join(outcome.lines)


def test_replaying_an_installed_manifest_is_refused(tmp_path, user):
    done = staged(tmp_path)
    assert run_install(done)[0].ok
    again, _ = run_install(done)
    assert not again.ok and "already installed" in again.lines[0]


def test_changing_a_staged_unit_between_prepare_and_install_is_refused(tmp_path, user):
    done = staged(tmp_path)
    target = done.path.parent / "poolhouse-traind.service"
    target.write_bytes(target.read_bytes() + b"ExecStartPost=/bin/sh -c evil\n")
    outcome, recorder = run_install(done)
    assert not outcome.ok and "changed after it was prepared" in " ".join(outcome.lines)
    assert not unit_dir(user).exists() and recorder.ran == []


def test_a_unit_and_manifest_edited_together_still_differ_from_what_prepare_renders(tmp_path, user):
    done = staged(tmp_path, roles=("pool-daemon",))
    target = done.path.parent / "poolhouse-traind.service"
    forged = target.read_bytes() + b"ExecStartPost=/bin/sh -c evil\n"
    target.write_bytes(forged)
    import hashlib

    raw = json.loads(done.path.read_text())
    raw["roles"][0]["units"][0]["sha256"] = hashlib.sha256(forged).hexdigest()
    done.path.write_text(json.dumps(raw))
    outcome, _ = run_install(done)
    assert not outcome.ok and "does not match the manifest" in " ".join(outcome.lines)


def test_a_consistently_forged_environment_is_refused(tmp_path, user):
    import hashlib
    from dataclasses import replace

    from poolhouse.fleet.autostart_manifest import parse
    from poolhouse.fleet.autostart_units import render

    done = staged(tmp_path, roles=("pool-daemon",))
    raw = json.loads(done.path.read_text())
    raw["roles"][0]["environment"]["PATH"] = "/tmp/evil"
    forged = render(replace(parse(raw).roles[0], units=()), "linux", "user", getpass.getuser())["service"]
    (done.path.parent / "poolhouse-traind.service").write_bytes(forged)
    raw["roles"][0]["units"][0]["sha256"] = hashlib.sha256(forged).hexdigest()
    done.path.write_text(json.dumps(raw))
    outcome, recorder = run_install(done)
    assert not outcome.ok and "environment is not the allowed one" in " ".join(outcome.lines)
    assert recorder.ran == []


def test_an_agent_is_refused_before_a_manifest_is_even_opened(tmp_path, user):
    with pytest.raises(HumanRequired):
        install(tmp_path / "missing.json", seams=seams(Recorder(), env={"CLAUDECODE": "1"}))


def edit(done, change):
    raw = json.loads(done.path.read_text())
    change(raw)
    done.path.write_text(json.dumps(raw))


@pytest.mark.parametrize("destination", [
    "{user}/.config/systemd/user/../../../.bashrc",
    "{user}/.config/systemd/user/../poolhouse-traind.service",
    "/etc/systemd/system/poolhouse-traind.service",
    "{user}/.config/systemd/user/other.service",
])
def test_a_destination_outside_the_allowed_set_is_refused(tmp_path, user, destination):
    done = staged(tmp_path, roles=("pool-daemon",))
    edit(done, lambda raw: raw["roles"][0]["units"][0].update(destination=destination.format(user=user)))
    outcome, recorder = run_install(done)
    assert not outcome.ok and recorder.ran == []
    assert not (user / ".bashrc").exists()


def test_a_symlinked_unit_directory_is_refused(tmp_path, user):
    done = staged(tmp_path, roles=("pool-daemon",))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (user / ".config" / "systemd").mkdir(parents=True)
    (user / ".config" / "systemd" / "user").symlink_to(elsewhere)
    outcome, _ = run_install(done)
    assert not outcome.ok and "symbolic link" in " ".join(outcome.lines)
    assert list(elsewhere.iterdir()) == []


def test_a_symlinked_destination_file_is_refused(tmp_path, user):
    done = staged(tmp_path, roles=("pool-daemon",))
    victim = tmp_path / "victim"
    victim.write_text("keep")
    unit_dir(user).mkdir(parents=True)
    (unit_dir(user) / "poolhouse-traind.service").symlink_to(victim)
    outcome, _ = run_install(done)
    assert not outcome.ok and victim.read_text() == "keep"


@pytest.mark.parametrize("change", [
    lambda raw: raw["roles"][0]["argv"].__setitem__(0, "/bin/sh"),
    lambda raw: raw["roles"][0]["argv"].extend(["-c", "evil"]),
    lambda raw: raw["roles"][0]["environment"].update(PATH="/tmp/evil"),
    lambda raw: raw["roles"][0].update(workdir="/tmp"),
    lambda raw: raw.update(platform="darwin"),
])
def test_a_manifest_edited_to_run_something_else_is_refused(tmp_path, user, change):
    done = staged(tmp_path, roles=("pool-daemon",))
    edit(done, change)
    outcome, recorder = run_install(done)
    assert not outcome.ok and recorder.ran == []
    assert not unit_dir(user).exists()


def test_a_missing_or_replaced_exec_target_is_refused_at_install(tmp_path, user):
    done = staged(tmp_path, roles=("pool-daemon",))
    Path(done.manifest.roles[0].argv[0]).write_text("#!/bin/sh\nexit 0\n")
    outcome, _ = run_install(done)
    assert not outcome.ok and "not a runtime launcher" in " ".join(outcome.lines)


def test_a_staging_directory_others_can_write_is_refused(tmp_path, user):
    done = staged(tmp_path, roles=("pool-daemon",))
    done.path.parent.chmod(0o777)
    outcome, _ = run_install(done)
    assert not outcome.ok and "not private" in " ".join(outcome.lines)


@pytest.mark.parametrize("marker", ["CLAUDECODE", "POOLHOUSE_NONINTERACTIVE", "POOLHOUSE_AGENT"])
def test_an_agent_is_refused_install_and_rollback_before_anything_is_read(tmp_path, user, marker):
    done = staged(tmp_path)
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    recorder = Recorder()
    agent = seams(recorder, typed=lambda _p: done.manifest.id, env={marker: "1"})
    with pytest.raises(HumanRequired):
        install(done.path, seams=agent)
    with pytest.raises(HumanRequired):
        rollback(seams=agent)
    assert recorder.ran == [] and not unit_dir(user).exists()
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before


def test_no_terminal_means_no_install(tmp_path, user):
    done = staged(tmp_path)
    with pytest.raises(HumanRequired):
        install(done.path, seams=seams(Recorder(), terminal=(False, False)))


def test_the_person_must_type_the_manifest_id(tmp_path, user):
    done = staged(tmp_path)
    with pytest.raises(HumanRequired):
        install(done.path, seams=seams(Recorder(), typed=lambda _p: "yes"))
    assert not unit_dir(user).exists()


def test_a_system_manifest_needs_the_flag_and_only_then(tmp_path, user):
    if sys.platform != "darwin":
        pytest.skip("the system directories are checked as they are on macOS")
    done = staged(tmp_path, platform="darwin", scope="system")
    plain = install(done.path, seams=seams(Recorder(), "darwin", "system", typed=lambda _p: done.manifest.id))
    assert not plain.ok and "--system" in " ".join(plain.lines)
    user_scope = staged(tmp_path, platform="darwin")
    flagged = install(user_scope.path, system=True, seams=seams(Recorder(), "darwin", typed=lambda _p: user_scope.manifest.id))
    assert not flagged.ok and "--system" in " ".join(flagged.lines)


def test_a_system_install_asks_the_operating_system_and_builds_a_quoted_command(tmp_path, user):
    if sys.platform != "darwin":
        pytest.skip("the system directories are checked as they are on macOS")
    done = staged(tmp_path, roles=("pool-daemon",), platform="darwin", scope="system")
    asked = []
    recorder = Recorder()
    outcome = install(done.path, system=True, seams=seams(
        recorder, "darwin", "system", typed=lambda _p: done.manifest.id, ask=lambda cmd, why: (asked.append(cmd) or (True, ""))))
    assert not outcome.ok
    command = asked[0]
    words = shlex.split(command.replace("&&", " ").replace("{", " ").replace("}", " ").replace(";", " "))
    assert "/Library/LaunchDaemons/com.poolhouse.traind.plist" in words
    assert str(done.path.parent / "com.poolhouse.traind.plist") in words
    assert "sudo" not in command and recorder.ran == []
    assert not Path("/Library/LaunchDaemons/com.poolhouse.traind.plist").exists()


def test_the_unit_runs_as_the_user_even_when_installed_system_wide(tmp_path, user):
    import plistlib

    done = staged(tmp_path, roles=("pool-daemon",), platform="darwin", scope="system")
    body = plistlib.loads((done.path.parent / "com.poolhouse.traind.plist").read_bytes())
    import getpass

    assert body["UserName"] == getpass.getuser() != "root"


def test_a_concurrent_install_is_refused_not_interleaved(tmp_path, user):
    from poolhouse.lock import only_one

    done = staged(tmp_path)
    with only_one(home.state("autostart", "apply.lock"), wait=False):
        outcome, _ = run_install(done)
    assert not outcome.ok and "another" in outcome.lines[0]

