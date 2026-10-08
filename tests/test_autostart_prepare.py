"""`prepare` stages unit files and a manifest and installs nothing; the manifest is strict about what it holds."""

from __future__ import annotations

import json
import plistlib
from pathlib import Path

import pytest
from autostart_support import environment, launchers, staged, write_launcher

from ml_stack import home
from ml_stack.files import sha256_file
from ml_stack.fleet import autostart_manifest as manifest_mod
from ml_stack.fleet.autostart_manifest import ManifestError, launcher_problem, parse
from ml_stack.fleet.autostart_prepare import PrepareError, Spec, prepare


@pytest.fixture
def user(tmp_path, monkeypatch):
    return environment(tmp_path, monkeypatch)


def test_prepare_stages_files_and_installs_nothing(tmp_path, user):
    done = staged(tmp_path, platform="darwin", agent={"agent": "claude-code", "label": "lane"})
    folder = done.path.parent
    assert folder.is_relative_to(home.state("autostart", "staging"))
    assert not (user / "Library").exists()
    raw = json.loads(done.path.read_text())
    assert raw["version"] == 1
    assert raw["preparer"] == {"agent": "claude-code", "label": "lane"}
    assert raw["expires"] - raw["created"] == 24 * 3600
    assert raw["platform"] == "darwin" and raw["device"] == home.device_id()
    for role in raw["roles"]:
        for unit in role["units"]:
            assert sha256_file(folder / unit["name"]) == unit["sha256"]
            assert unit["destination"] == str(user / "Library" / "LaunchAgents" / unit["name"])
        assert Path(role["argv"][0]).is_absolute()
        assert Path(role["argv"][0]).is_relative_to(home.state("bin"))
        assert Path(role["workdir"]).is_relative_to(home.state())
        assert set(role["environment"]) <= set(manifest_mod.ENV_ALLOWED)
    assert oct(folder.stat().st_mode & 0o777) == "0o700"
    assert done.command.endswith(f"install --manifest {done.path}")
    assert done.command.count("\n") == 0


def test_the_manifest_carries_no_secret_from_the_environment(tmp_path, user, monkeypatch):
    for name in ("HF_TOKEN", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY", "ML_STACK_TOKEN"):
        monkeypatch.setenv(name, f"secret-{name}")
    done = staged(tmp_path, platform="linux")
    for path in done.path.parent.iterdir():
        text = path.read_text()
        assert "secret-" not in text, path.name


def test_each_platform_renders_a_restart_policy_and_the_argv_list(tmp_path, user):
    mac = staged(tmp_path, platform="darwin").manifest
    pool = next(r for r in mac.roles if r.role == "pool-daemon")
    plist = plistlib.loads((staged_dir(mac) / pool.units[0].name).read_bytes())
    assert plist["ProgramArguments"] == list(pool.argv)
    assert plist["KeepAlive"] == {"SuccessfulExit": False} and plist["ThrottleInterval"] == 30
    assert plist["RunAtLoad"] is True and "UserName" not in plist
    ensure = next(r for r in mac.roles if r.role == "runtime-ensure")
    timed = plistlib.loads((staged_dir(mac) / ensure.units[0].name).read_bytes())
    assert timed["StartInterval"] == 3600 and timed["RunAtLoad"] is True and "KeepAlive" not in timed

    nix = staged(tmp_path, platform="linux").manifest
    units = {u.name: (staged_dir(nix) / u.name).read_text() for r in nix.roles for u in r.units}
    service = units["ml-stack-traind.service"]
    assert "Restart=on-failure" in service and "RestartSec=30" in service
    assert "StartLimitBurst=5" in service and "NoNewPrivileges=yes" in service and "User=" not in service
    assert "OnUnitActiveSec=3600" in units["ml-stack-runtime-ensure.timer"]
    assert "Type=oneshot" in units["ml-stack-runtime-ensure.service"]

    win = staged(tmp_path, platform="win32").manifest
    xml = (staged_dir(win) / "com.ml-stack.traind.xml").read_bytes().decode("utf-16")
    for tag in ("<LogonTrigger>", "<RestartOnFailure>", "<Exec>", "<RunLevel>LeastPrivilege</RunLevel>"):
        assert tag in xml


def staged_dir(manifest) -> Path:
    return home.state("autostart", "staging", manifest.id)


def test_unattended_ensure_takes_only_its_own_arguments(tmp_path, user):
    done = staged(tmp_path, roles=("runtime-ensure",), platform="linux")
    role = done.manifest.roles[0]
    assert role.argv[1:] == ("runtime", "ensure") and Path(role.argv[0]).name == "ml-stack"
    assert not {"ML_STACK_WORKSPACE_AGENT", "ML_STACK_WORKSPACE_TOKEN"} & set(role.environment)


def test_arguments_with_spaces_quotes_and_percent_stay_single_argv_elements(tmp_path, user):
    odd = home.state("bin dir", "it's \"q\" $x %h")
    launchers(odd)
    done = prepare(Spec(("pool-daemon",), launchers=odd, platform="linux", report="a b;c"))
    argv = done.manifest.roles[0].argv
    assert argv[0] == str(odd / "ml-stack-traind") and argv[-1] == "--report=a b;c"
    text = (staged_dir(done.manifest) / "ml-stack-traind.service").read_text()
    line = next(row for row in text.splitlines() if row.startswith("ExecStart="))
    assert "%%h" in line and "$$x" in line and '\\"q\\"' in line and "%h" not in line.replace("%%h", "")
    plist = plistlib.loads((staged_dir(prepare(Spec(("pool-daemon",), launchers=odd, platform="darwin")).manifest)
                            / "com.ml-stack.traind.plist").read_bytes())
    assert plist["ProgramArguments"][0] == str(odd / "ml-stack-traind")


@pytest.mark.parametrize("bad", [{"labels": ("a b",)}, {"labels": ("x;rm -rf",)},
                                  {"report": "two\nlines"}, {"slots": 0}, {"slots": 99}])
def test_prepare_refuses_option_values_that_are_not_plain(tmp_path, user, bad):
    with pytest.raises(PrepareError):
        prepare(Spec(("pool-daemon",), launchers=launchers(home.state("bin")), platform="linux", **bad))


def test_a_path_with_a_newline_cannot_be_prepared(tmp_path, user):
    odd = home.state("bin", "line\nbreak")
    launchers(odd)
    with pytest.raises(ValueError):
        prepare(Spec(("pool-daemon",), launchers=odd, platform="linux"))
    assert not home.state("autostart", "staging").exists()


def test_the_exec_target_must_be_a_launcher_the_runtime_tooling_wrote(tmp_path, user):
    bin_dir = home.state("bin")
    launchers(bin_dir)
    target = bin_dir / "ml-stack-traind"
    assert launcher_problem(target) == ""
    target.write_text("#!/bin/sh\nexec python -m ml_stack.cli.daemon\n")
    assert "not a runtime launcher" in launcher_problem(target)
    write_launcher(bin_dir, "ml-stack-traind", mode=0o777)
    assert "writable by others" in launcher_problem(target)
    target.unlink()
    assert "missing" in launcher_problem(target)
    target.symlink_to(bin_dir / "ml-stack-runtime")
    assert "not a regular file" in launcher_problem(target)
    with pytest.raises(PrepareError):
        prepare(Spec(("pool-daemon",), launchers=bin_dir, platform="linux"))


def test_a_launcher_inside_a_checkout_or_outside_the_runtime_directory_is_refused(tmp_path, user):
    checkout = home.state("work")
    (checkout / ".git").mkdir(parents=True)
    path = write_launcher(checkout / "bin", "ml-stack-traind")
    assert "source checkout" in launcher_problem(path)
    odd = write_launcher(home.state("bin"), "ml-stack-traind")
    odd.write_text(odd.read_text().replace(str(home.state("runtimes")), "/opt/elsewhere"))
    assert "outside" in launcher_problem(odd)


def test_the_launcher_the_runtime_tooling_writes_is_accepted(tmp_path, user):
    from ml_stack import runtime

    path = home.state("bin") / "ml-stack"
    path.parent.mkdir(parents=True)
    path.write_text(runtime.LAUNCHER.format(root=str(runtime.directory()), python="/x/bin/python",
                                            arguments=repr(["-m", "ml_stack.cli"]), name="ml-stack"))
    path.chmod(0o700)
    assert launcher_problem(path) == ""


def test_prepare_needs_the_launcher_directory_named(tmp_path, user):
    with pytest.raises(PrepareError, match="--launchers"):
        prepare(Spec(("pool-daemon",), platform="linux"))


def test_a_relative_launcher_directory_is_refused(tmp_path, user):
    assert "absolute" in launcher_problem(Path("ml-stack-traind"))


@pytest.mark.parametrize("roles", [(), ("landing",), ("pool-daemon", "bogus")])
def test_there_is_no_role_for_landing_or_anything_unnamed(tmp_path, user, roles):
    with pytest.raises(PrepareError):
        prepare(Spec(roles, launchers=launchers(home.state("bin")), platform="linux"))


def test_system_scope_is_not_offered_on_windows(tmp_path, user):
    with pytest.raises(PrepareError):
        prepare(Spec(("pool-daemon",), scope="system", launchers=launchers(home.state("bin")), platform="win32"))


def test_the_parser_refuses_what_the_preparer_never_writes(tmp_path, user):
    raw = staged(tmp_path, platform="linux").manifest.to_dict()
    for change in ({"version": 2}, {"id": "x"}, {"expires": raw["created"] + 90000}, {"scope": "root"},
                   {"roles": []}, {"platform": "plan9"}):
        with pytest.raises(ManifestError):
            parse({**raw, **change})
    role = raw["roles"][0]
    for change in ({"role": "landing"}, {"environment": {"HF_TOKEN": "x"}}, {"argv": "ml-stack-traind"},
                   {"units": []}):
        with pytest.raises(ManifestError):
            parse({**raw, "roles": [{**role, **change}]})
    assert parse(raw).id == raw["id"]


def test_the_daemon_port_in_the_health_probe_is_the_daemon_default():
    from ml_stack.fleet import daemon

    assert manifest_mod.DAEMON_PORT == daemon.DEFAULT_PORT
