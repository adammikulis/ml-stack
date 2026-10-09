"""scripts/agent-sandbox: the profiles it derives, the staged files and manifest, and the probes."""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import agent_sandbox_probe as probe  # noqa: E402
import agent_sandbox_profile as profile  # noqa: E402
import agent_sandbox_stage as stage  # noqa: E402


@pytest.fixture
def layout(tmp_path):
    home = tmp_path / "home"
    return profile.Layout(
        home=home, state=home / ".ml-stack", primary=tmp_path / "ml-stack",
        worktrees=(tmp_path / "ml-stack-feature",), scratch=(tmp_path / "scratch",), platform="darwin")


def test_claude_settings_close_the_escape_hatches(layout):
    sandbox = profile.claude_settings(layout)["sandbox"]
    assert sandbox["enabled"] and sandbox["failIfUnavailable"]
    assert sandbox["allowUnsandboxedCommands"] is False
    assert sandbox["autoAllowBashIfSandboxed"] is False


def test_only_the_gated_runtime_deploy_commands_run_outside_the_sandbox(layout):
    sandbox = profile.claude_settings(layout)["sandbox"]
    assert sandbox["excludedCommands"] == ["ml-stack runtime ensure", "ml-stack runtime rollback", "ml-stack runtime restart-host"]
    assert str(layout.state / "runtimes") in sandbox["filesystem"]["denyWrite"]


def test_every_worktree_is_writable_and_its_git_config_and_hooks_are_not(layout, tmp_path):
    files = profile.claude_settings(layout)["sandbox"]["filesystem"]
    for checkout in (layout.primary, layout.worktrees[0]):
        assert str(checkout) in files["allowWrite"]
        for name in ("scripts/hooks", ".claude", ".git/config", ".git/hooks"):
            assert str(checkout / name) in files["denyWrite"]
    shared = layout.primary / ".git"
    assert str(shared) in files["allowWrite"]
    assert str(shared / "config") in files["denyWrite"]
    assert str(shared / "hooks") in files["denyWrite"]


def test_tokens_are_denied_except_the_harness_own(layout):
    files = profile.claude_settings(layout)["sandbox"]["filesystem"]
    tokens = str(layout.state / "workspace" / "tokens")
    assert tokens in files["denyRead"] and tokens in files["denyWrite"]
    assert files["allowRead"] == [tokens + "/claude-code"]
    for name in (".ssh", ".aws", ".config/gh"):
        assert str(layout.home / name) in files["denyRead"]


def test_runtimes_transcripts_and_persistence_paths_are_not_writable(layout):
    denied = profile.claude_settings(layout)["sandbox"]["filesystem"]["denyWrite"]
    for path in (layout.state / "runtimes", layout.home / ".claude", layout.home / "Library/LaunchAgents",
                 layout.home / ".zshrc", layout.home / ".pyenv/versions"):
        assert str(path) in denied


def test_linux_denies_systemd_user_directories_instead_of_launch_agents(layout):
    linux = profile.Layout(layout.home, layout.state, layout.primary, platform="linux")
    denied = profile.claude_settings(linux)["sandbox"]["filesystem"]["denyWrite"]
    assert str(linux.home / ".config/systemd") in denied
    assert str(linux.home / "Library/LaunchAgents") not in denied
    assert "allowLocalBinding" not in profile.claude_settings(linux)["sandbox"]["network"]


def test_network_allows_only_the_listed_hosts(layout):
    network = profile.claude_settings(layout)["sandbox"]["network"]
    assert network["allowedDomains"] == list(profile.HOSTS)
    assert "127.0.0.1:8770" in set(network["allowedDomains"])
    assert {"pypi.org", "*.hf.co"} <= set(network["allowedDomains"])
    assert "example.com" not in set(network["allowedDomains"])


def test_scrubbed_environment_names_messaging_and_search_path_variables(layout):
    deny = profile.claude_settings(layout)["sandbox"]["credentials"]["envVars"]["deny"]
    for name in ("CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN", "ML_STACK_HOME",
                 "PYTHONPATH", "GIT_CONFIG_GLOBAL", "SSH_AUTH_SOCK"):
        assert name in deny


def test_file_tools_are_refused_the_hook_directory(layout):
    deny = profile.claude_settings(layout)["permissions"]["deny"]
    assert f"Edit(/{layout.primary}/scripts/hooks/**)" in deny
    assert f"Write(/{layout.home}/.claude/**)" in deny


def test_codex_fragment_parses_and_lists_the_checkouts(layout):
    parsed = tomllib.loads(profile.codex_toml(layout))
    assert parsed["sandbox_mode"] == "workspace-write"
    roots = parsed["sandbox_workspace_write"]["writable_roots"]
    assert str(layout.primary) in roots and str(layout.worktrees[0]) in roots
    assert parsed["sandbox_workspace_write"]["network_access"] is False


def test_srt_settings_carry_the_same_filesystem_lists(layout):
    srt = profile.srt_settings(layout)
    claude = profile.claude_settings(layout)["sandbox"]["filesystem"]
    assert srt["filesystem"]["denyWrite"] == claude["denyWrite"]
    assert "api.anthropic.com" in set(srt["network"]["allowedDomains"])


def test_prepare_stages_files_with_matching_digests(layout):
    target = stage.prepare(layout, "device-1", now=1000.0)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["device_id"] == "device-1"
    assert manifest["expires"] == 1000.0 + stage.LIFETIME
    for item in manifest["files"]:
        assert stage.digest((target / item["name"]).read_bytes()) == item["sha256"]
    assert {item["destination"] for item in manifest["files"]} >= {
        "/Library/Application Support/ClaudeCode/managed-settings.json"}


def test_install_command_is_one_sudo_line_carrying_the_digests(layout):
    stage.prepare(layout, "device-1")
    command = stage.install_command(layout.state)
    assert command.startswith("sudo sh -c ") and "\n" not in command
    for item in stage.load(layout.state)["files"]:
        assert item["sha256"] in command
    assert "install -m 644" in command


def test_installed_reports_absent_match_and_differs(layout, tmp_path, monkeypatch):
    destination = tmp_path / "managed-settings.json"
    monkeypatch.setitem(stage.DESTINATIONS["darwin"], "claude-managed-settings.json", str(destination))
    target = stage.prepare(layout, "device-1")
    assert (str(destination), "absent") in stage.installed(layout.state)
    destination.write_bytes((target / "claude-managed-settings.json").read_bytes())
    assert (str(destination), "match") in stage.installed(layout.state)
    destination.write_text("{}")
    assert (str(destination), "differs") in stage.installed(layout.state)


def test_staged_files_expire(layout):
    stage.prepare(layout, "device-1", now=1000.0)
    assert not stage.expired(layout.state, now=1000.0 + stage.LIFETIME - 1)
    assert stage.expired(layout.state, now=1000.0 + stage.LIFETIME + 1)
    assert not stage.expired(layout.state / "elsewhere")


def test_command_line_prepares_and_reports_without_touching_destinations(tmp_path):
    base = [sys.executable, str(ROOT / "scripts" / "agent-sandbox"),
            "--home", str(tmp_path / "home"), "--state", str(tmp_path / "state")]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    (tmp_path / "home").mkdir()
    prepared = subprocess.run([*base, "prepare"], capture_output=True, text=True, env=env, check=False,
                              cwd=ROOT)
    assert prepared.returncode == 0, prepared.stderr
    assert "sudo sh -c" in prepared.stdout
    reported = subprocess.run([*base, "status"], capture_output=True, text=True, env=env, check=False,
                              cwd=ROOT)
    assert reported.returncode == 0
    assert "managed-settings.json" in reported.stdout
    refused = subprocess.run([*base, "install"], capture_output=True, text=True, env=env, check=False,
                             cwd=ROOT)
    assert refused.returncode != 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
def test_a_write_probe_passes_only_where_the_write_is_refused(tmp_path):
    shut = tmp_path / "shut"
    shut.mkdir()
    shut.chmod(0o500)
    try:
        assert probe._denied_write("shut", shut / "x").status == probe.PASS
        assert probe._denied_write("open", tmp_path / "x").status == probe.FAIL
    finally:
        shut.chmod(0o700)
    assert probe._denied_write("missing", tmp_path / "none" / "x").status == probe.SKIP
    assert not (tmp_path / "x").exists()


def test_a_read_probe_fails_when_the_file_reads(tmp_path):
    secret = tmp_path / "token"
    secret.write_text("x")
    assert probe._denied_read("token", secret).status == probe.FAIL
    assert probe._denied_read("token", tmp_path / "none").status == probe.SKIP


def test_connect_state_tells_refused_from_open():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert probe.connect_state("127.0.0.1", port) == "open"
    assert probe.connect_state("127.0.0.1", port) == "refused"


def test_environment_probe_fails_for_a_scrubbed_variable_that_is_set():
    results = {item.name: item.status for item in probe.environment_probes({"PYTHONPATH": "/x"})}
    assert results["env PYTHONPATH absent"] == probe.FAIL
    assert results["env CLAUDE_CODE_MESSAGING_SOCKET absent"] == probe.PASS


def test_git_probe_pushes_to_a_local_bare_repository():
    assert probe.git_probe().status == probe.PASS


def test_inside_needs_a_refusal_and_no_open_line():
    passed, failed = probe.Result("a", probe.PASS), probe.Result("b", probe.FAIL)
    assert probe.inside([passed])
    assert not probe.inside([passed, failed])
    assert not probe.inside([probe.Result("c", probe.SKIP)])
    assert probe.render([passed]).startswith("PASS  a")


def test_the_install_command_installs_the_staged_files_when_run_without_sudo(layout, tmp_path, monkeypatch):
    places = {name: str(tmp_path / "root" / name) for name in stage.DESTINATIONS["darwin"]}
    monkeypatch.setitem(stage.DESTINATIONS, "darwin", places)
    stage.prepare(layout, "device-1")
    argv = shlex.split(stage.install_command(layout.state))[1:]
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert all(state == "match" for _, state in stage.installed(layout.state))
