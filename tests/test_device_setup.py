"""The one device setup command: its decisions per platform, and that what it runs on the host is fixed argv."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import pytest

from poolhouse import device_host, device_setup, node_binary, node_join

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("old, expected", [
    ("", "[wsl2]\nnetworkingMode=mirrored\n"),
    ("[wsl2]\nmemory=8GB\n", "[wsl2]\nnetworkingMode=mirrored\nmemory=8GB\n"),
    ("[wsl2]\nnetworkingMode=NAT\nmemory=8GB\n", "[wsl2]\nnetworkingMode=mirrored\nmemory=8GB\n"),
    ("[experimental]\nx=1\n", "[experimental]\nx=1\n\n[wsl2]\nnetworkingMode=mirrored\n"),
    ("[wsl2]\nmemory=8GB\n[experimental]\nnetworkingMode=nat\n", "[wsl2]\nnetworkingMode=mirrored\nmemory=8GB\n[experimental]\nnetworkingMode=nat\n"),
])
def test_wslconfig_gets_mirrored_networking_and_nothing_else_moves(old, expected):
    new, changed = device_setup.mirrored_config(old)
    assert (new, changed) == (expected, True)
    assert device_setup.mirrored_config(new) == (new, False)


def test_writing_the_config_keeps_the_old_file_and_a_second_run_changes_nothing(tmp_path):
    config = tmp_path / ".wslconfig"
    config.write_text("[wsl2]\nmemory=8GB\n", encoding="utf-8")
    kept = device_setup.write_config(config, now=0)
    assert (tmp_path / kept).read_text(encoding="utf-8") == "[wsl2]\nmemory=8GB\n"
    assert "networkingMode=mirrored" in config.read_text(encoding="utf-8")
    assert device_setup.write_config(config) == "unchanged"
    assert len(list(tmp_path.iterdir())) == 2
    absent = tmp_path / "new" / ".wslconfig"
    absent.parent.mkdir()
    assert device_setup.write_config(absent) == "created"


def test_the_node_comes_from_the_runtime_then_a_binary_in_the_repository_then_cargo_then_rust(tmp_path):
    runtime_prefix = tmp_path / "rt"
    repo = tmp_path / "repo"
    assert device_setup.node_source(repo, selected=None, cargo=None) == device_setup.Source("rustup")
    assert device_setup.node_source(repo, selected=None, cargo="/bin/cargo") == device_setup.Source("build")
    built = repo / "app" / "target" / "release" / node_binary.name()
    built.parent.mkdir(parents=True)
    built.write_text("x", encoding="utf-8")
    assert device_setup.node_source(repo, selected=None, cargo=None) == device_setup.Source("found", str(built))
    chosen = node_binary.location(runtime_prefix)
    chosen.parent.mkdir(parents=True)
    chosen.write_text("x", encoding="utf-8")
    assert device_setup.node_source(repo, selected=runtime_prefix, cargo=None).kind == "runtime"


def needs(**over):
    base = {"kind": "mac", "mode": "mirrored", "config_change": False, "firewall_missing": False, "source": device_setup.Source("found", "/n")}
    return device_setup.Needs(**{**base, **over})


def test_the_plan_lists_only_what_will_change_and_always_the_node_and_the_check():
    nothing = device_setup.plan(needs())
    assert len(nothing) == 2 and "/n" in nothing[0] and "join check" in nothing[1]
    windows = "\n".join(device_setup.plan(needs(kind="windows", firewall_missing=True, source=device_setup.Source("rustup"))))
    assert "firewall" not in windows and "inbound allow rules" in windows and "Rust" in windows
    assert "wslconfig" not in windows
    assert str(node_join.DEFAULT_PORT) in windows and str(node_join.DEFAULT_BEACON_PORT) in windows


def test_wsl_in_nat_mode_or_with_an_unmirrored_config_ends_the_plan_at_a_restart():
    for wsl in (needs(kind="wsl", mode="nat"), needs(kind="wsl", config_change=True)):
        steps = device_setup.plan(wsl)
        assert wsl.restart_wsl and "restart WSL" in steps[-1] and "join check" not in "\n".join(steps)
    assert not needs(kind="wsl").restart_wsl and not needs(kind="windows", mode="nat").restart_wsl
    assert any(".wslconfig" in s for s in device_setup.plan(needs(kind="wsl", config_change=True)))
    assert not any(".wslconfig" in s for s in device_setup.plan(needs(kind="mac", config_change=True)))


@pytest.mark.parametrize("platform, version, kind", [
    ("win32", "", "windows"), ("darwin", "", "mac"), ("linux", "Linux 5.15 microsoft-standard-WSL2", "wsl"), ("linux", "Linux 6.1 generic", "linux")])
def test_the_kind_of_machine_comes_from_the_platform_and_the_kernel(platform, version, kind):
    assert device_host.host_kind(platform, version) == kind


def test_the_firewall_script_covers_both_ports_and_the_vm_and_travels_encoded():
    script = device_host.rules_script()
    assert f"-LocalPort {node_join.DEFAULT_PORT}" in script and f"-LocalPort {node_join.DEFAULT_BEACON_PORT}" in script
    assert device_host.HYPERV_ID in script and "-Profile Private" in script
    assert base64.b64decode(base64.b64encode(script.encode("utf-16-le"))).decode("utf-16-le") == script


def test_a_dry_run_lists_the_changes_and_makes_none(monkeypatch, capsys):
    monkeypatch.setattr(device_setup, "survey", lambda repo: needs(kind="windows", firewall_missing=True))
    monkeypatch.setattr(device_host, "add_firewall", lambda: pytest.fail("a dry run changed the machine"))
    assert device_setup.main(["--dry-run"]) == 0
    assert "inbound allow rules" in capsys.readouterr().out


def test_without_a_yes_and_without_a_person_nothing_changes(monkeypatch, capsys):
    monkeypatch.setattr(device_setup, "survey", lambda repo: needs(kind="windows", firewall_missing=True))
    monkeypatch.setattr(device_host, "add_firewall", lambda: pytest.fail("changed without a yes"))
    monkeypatch.setattr(device_setup.sys.stdin, "isatty", lambda: False)
    assert device_setup.main([]) == 1
    assert "Nothing changed." in capsys.readouterr().out


def test_wsl_that_needs_mirroring_writes_the_config_and_restarts_before_anything_else(monkeypatch, tmp_path):
    config = tmp_path / ".wslconfig"
    config.write_text("[wsl2]\nmemory=8GB\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(device_setup, "survey", lambda repo: needs(kind="wsl", mode="nat", config_change=True))
    monkeypatch.setattr(device_host, "wslconfig_path", lambda kind: config)
    monkeypatch.setattr(device_host, "restart_wsl", lambda: calls.append("restart"))
    monkeypatch.setattr(device_setup, "binary_of", lambda *a: pytest.fail("started the node before the restart"))
    assert device_setup.main(["--yes"]) == 0
    assert calls == ["restart"] and "networkingMode=mirrored" in config.read_text(encoding="utf-8")
    assert any(p.name.startswith(".wslconfig.poolhouse-backup") for p in tmp_path.iterdir())


def test_a_missing_firewall_rule_after_the_prompt_is_not_ready_with_the_fix(monkeypatch, capsys):
    monkeypatch.setattr(device_setup, "survey", lambda repo: needs(kind="windows", firewall_missing=True))
    monkeypatch.setattr(device_host, "add_firewall", lambda: None)
    monkeypatch.setattr(device_host, "firewall_missing", lambda: True)
    assert device_setup.main(["--yes"]) == 1
    assert "NOT READY" in capsys.readouterr().err


def test_pairing_without_a_code_is_refused_before_the_node_is_asked(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(node_join.PAIR_CODE_ENV, raising=False)
    assert node_join.main(["pair", "--host", "192.0.2.1", "--state", str(tmp_path)]) == 1
    assert node_join.PAIR_CODE_ENV in capsys.readouterr().err


def _recorded(monkeypatch, stdout=""):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(device_host.subprocess, "run", run)
    return calls


def test_a_hostile_windows_profile_path_reaches_wslpath_as_one_argument_and_no_command_goes_through_a_shell(monkeypatch):
    hostile = "C:\\Users\\x'; rm -rf ~; echo '$(id)"
    calls = _recorded(monkeypatch, stdout=hostile)
    device_host.wslconfig_path("wsl")
    assert calls[1][0] == ["wslpath", "-u", hostile]
    for argv, kwargs in calls:
        assert isinstance(argv, list) and not kwargs.get("shell")


def test_every_command_the_host_runs_is_a_fixed_argv_with_no_input_from_outside(monkeypatch):
    calls = _recorded(monkeypatch)
    device_host.restart_wsl()
    device_host.firewall_missing()
    device_host.add_firewall()
    device_host.install_rust("windows")
    device_host.install_rust("linux")
    argvs = [argv for argv, _ in calls]
    assert argvs[0] == ["wsl.exe", "--shutdown"]
    assert argvs[-1][:2] == ["sh", "-c"] and device_host.RUSTUP_URL in argvs[-1][2] and "--proto '=https'" in argvs[-1][2]
    assert argvs[-2][0] == "winget"
    assert all(not kwargs.get("shell") for _, kwargs in calls)
