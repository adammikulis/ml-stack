"""The nudge hooks written into Claude Code's and Codex's settings, and the check that they are there."""

from __future__ import annotations

import argparse
import json
import tomllib

import pytest

from ml_stack import agent_hooks, authority
from ml_stack.person import HumanRequired
from ml_stack.workspace import hooks_cli, onboard

CODEX_BEFORE = '''model = "gpt-6.1-sol"

[features]
hooks = true
multi_agent = true

[mcp_servers.docs]
url = "https://example.invalid/mcp"

[[hooks.PostToolUse]]
matcher = ".*"

[[hooks.PostToolUse.hooks]]
type = "command"
command = "python -m ml_stack.harnesshook post --label codex"
timeout = 30

[hooks.state]

[agents]
max_threads = 32
'''


@pytest.fixture(autouse=True)
def machine(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv(authority.FLOOR_ENV, raising=False)
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    for name in (*authority.DELEGATING, "ML_STACK_NONINTERACTIVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(agent_hooks.shutil, "which", lambda name: f"/bin/{name}")
    return tmp_path


def commands(data, event):
    return [h["command"] for g in data["hooks"][event] for h in g["hooks"]]


def test_claude_hooks_are_idempotent_keep_other_hooks_and_replace_earlier_nudges(machine):
    path = machine / "settings.json"
    path.write_text(json.dumps({"model": "x", "hooks": {
        "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "other-tool"}]},
                        {"matcher": "*", "hooks": [{"type": "command", "command": "sh ~/.ml-stack/hooks/claude-nudge.sh"}]}],
        "Stop": [{"hooks": [{"type": "command", "command": "keep-me"},
                            {"type": "command", "command": "ml-stack-workspace nudge --agent old --hook stop"}]}]}}))
    assert agent_hooks.install_claude(path) == ["PostToolUse", "Stop", "UserPromptSubmit"]
    first = path.read_text()
    agent_hooks.install_claude(path)
    assert path.read_text() == first
    data = json.loads(first)
    assert data["model"] == "x"
    assert commands(data, "PostToolUse") == ["other-tool", "ml-stack-workspace nudge --agent claude --hook post"]
    assert commands(data, "Stop") == ["keep-me", "ml-stack-workspace nudge --agent claude --hook stop"]
    assert commands(data, "UserPromptSubmit") == ["ml-stack-workspace nudge --agent claude --hook prompt"]
    assert data["hooks"]["PostToolUse"][-1]["matcher"] == "*"


def test_codex_hooks_keep_every_other_setting_and_do_not_double_the_launcher_nudge(machine):
    path = machine / "config.toml"
    path.write_text(CODEX_BEFORE)
    assert agent_hooks.install_codex(path) == ["Stop", "UserPromptSubmit"]
    text = path.read_text()
    assert text.startswith(CODEX_BEFORE.rstrip("\n"))
    data = tomllib.loads(text)
    assert data["model"] == "gpt-6.1-sol" and data["agents"] == {"max_threads": 32}
    assert commands(data, "Stop") == ["ml-stack-workspace nudge --agent codex --hook stop"]
    assert commands(data, "UserPromptSubmit") == ["ml-stack-workspace nudge --agent codex --hook prompt"]
    assert commands(data, "PostToolUse") == ["python -m ml_stack.harnesshook post --label codex"]
    agent_hooks.install_codex(path)
    assert path.read_text() == text


def test_codex_hooks_write_the_post_hook_when_no_launcher_hook_nudges(machine):
    path = machine / "config.toml"
    assert agent_hooks.install_codex(path) == ["PostToolUse", "Stop", "UserPromptSubmit"]
    data = tomllib.loads(path.read_text())
    assert data["features"] == {"hooks": True}
    assert data["hooks"]["PostToolUse"][0]["matcher"] == ".*"
    assert commands(data, "PostToolUse") == ["ml-stack-workspace nudge --agent codex --hook post"]


def test_codex_turns_the_hooks_feature_on_and_replaces_an_earlier_notify_nudge(machine):
    path = machine / "config.toml"
    path.write_text('notify = ["ml-stack-workspace", "nudge", "--agent", "codex"]\nmodel = "m"\n\n'
                    '[features]\nhooks = false\nshell_snapshot = true\n')
    agent_hooks.install_codex(path)
    data = tomllib.loads(path.read_text())
    assert "notify" not in data and data["model"] == "m"
    assert data["features"] == {"hooks": True, "shell_snapshot": True}
    path.write_text('[features]\nshell_snapshot = true\n')
    agent_hooks.install_codex(path)
    assert tomllib.loads(path.read_text())["features"] == {"hooks": True, "shell_snapshot": True}


def test_a_codex_config_that_is_not_toml_is_left_untouched(machine):
    path = machine / "config.toml"
    path.write_text("model = [unclosed\n")
    with pytest.raises(ValueError):
        agent_hooks.install_codex(path)
    assert path.read_text() == "model = [unclosed\n"


def test_install_writes_only_the_agents_on_this_machine(machine, monkeypatch):
    (machine / "home" / ".claude").mkdir(parents=True)
    monkeypatch.setattr(agent_hooks.shutil, "which", lambda name: None)
    assert list(agent_hooks.install()) == ["claude-code"]
    assert not (machine / "home" / ".codex").exists()
    assert agent_hooks.paths()["claude-code"].is_file()


def test_install_names_every_agent_it_is_told_to(machine):
    where = {"claude-code": machine / "a.json", "codex": machine / "b.toml"}
    assert set(agent_hooks.install(where=where)) == {"claude-code", "codex"}
    assert agent_hooks.install_report()[0].startswith("claude-code: message-board hooks (PostToolUse, Stop")


def test_the_install_command_follows_the_registry(machine, monkeypatch):
    args = argparse.Namespace(settings=str(machine / "s.json"), codex_config="", only=[], agent="")
    authority.set_state(["workspace.setup"], authority.PERSON, by="lead")
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(HumanRequired):
        hooks_cli.install(args, None)
    assert not (machine / "s.json").exists()
    authority.set_state(["workspace.setup"], authority.DELEGATED, by="lead")
    assert hooks_cli.install(args, None) == 0
    assert (machine / "s.json").is_file()


def test_findings_say_missing_stale_and_current_per_agent(machine):
    where = {"claude-code": machine / "a.json", "codex": machine / "b.toml"}
    missing = agent_hooks.findings(where)
    assert [f.name for f in missing] == ["claude-code: message-board hooks", "codex: message-board hooks"]
    assert not any(f.good for f in missing) and "PostToolUse missing" in missing[0].said
    assert all(f.fix == ["ml-stack-workspace", "install-hooks"] for f in missing)
    agent_hooks.install(where=where)
    assert all(f.good for f in agent_hooks.findings(where))
    where["claude-code"].write_text(where["claude-code"].read_text().replace("--agent claude", "--agent old"))
    stale = agent_hooks.findings(where)
    assert not stale[0].good and "PostToolUse stale" in stale[0].said and stale[1].good
    where["codex"].write_text(where["codex"].read_text().replace("hooks = true", "hooks = false"))
    assert "the hooks feature is off" in agent_hooks.findings(where)[1].said


def test_findings_report_an_unreadable_file_and_a_missing_binary(machine, monkeypatch):
    where = {"claude-code": machine / "a.json"}
    where["claude-code"].write_text("{broken")
    assert "cannot be read" in agent_hooks.findings(where)[0].said
    agent_hooks.install_claude(machine / "ok.json")
    monkeypatch.setattr(agent_hooks.shutil, "which", lambda name: None)
    found = agent_hooks.findings({"claude-code": machine / "ok.json"})
    assert "ml-stack-workspace is not on PATH" in found[0].said


def test_the_workspace_doctor_carries_the_hook_findings(machine, monkeypatch):
    from workspace_kit import Kit

    monkeypatch.setenv("ML_STACK_WORKSPACE_HOME", str(machine / "ws"))
    monkeypatch.setattr(authority, "require_person", lambda *a, **k: None)
    kit = Kit(machine / "ws")
    hooks = [f for f in onboard.doctor(kit.ws) if "message-board hooks" in f.what]
    assert [f.ok for f in hooks] == [False, False]
    assert all("missing" in f.what and f.fix == "ml-stack-workspace install-hooks" for f in hooks)


def test_the_install_command_writes_both_agents_into_named_files(machine, capsys, monkeypatch):
    monkeypatch.setattr(authority, "require_person", lambda *a, **k: None)
    args = argparse.Namespace(settings=str(machine / "s.json"), codex_config=str(machine / "c.toml"), only=[], agent="")
    assert hooks_cli.install(args, None) == 0
    out = capsys.readouterr().out
    assert "claude-code: PostToolUse, Stop, UserPromptSubmit" in out and "codex: PostToolUse, Stop" in out
    assert (machine / "s.json").is_file() and (machine / "c.toml").is_file()
    args.only = ["codex"]
    assert hooks_cli.install(args, None) == 0
    assert "claude-code" not in capsys.readouterr().out
