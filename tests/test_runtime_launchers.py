"""Owned runtime launcher and existing hook cutover boundaries."""

import json
import os
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from poolhouse import runtime, runtime_launchers


@pytest.fixture
def chosen(tmp_path, monkeypatch):
    value = runtime.Runtime(tmp_path / "prefix", "a" * 40, "0.2.2", runtime.identity())
    monkeypatch.setattr(runtime, "selected", lambda: value)
    monkeypatch.setattr(runtime, "verify", lambda row: row)
    return value


def test_regular_owned_console_gateway_executes_selected_python_with_literal_arguments(tmp_path, chosen):
    target = tmp_path / "poolhouse-workspace"
    target.write_text("old")
    target.chmod(0o755)
    chosen.python.parent.mkdir(parents=True)
    (chosen.prefix / "lib" / "python3.13" / "site-packages" / "poolhouse").mkdir(parents=True)
    (chosen.prefix / "lib" / "python3.13" / "site-packages" / "poolhouse" / "__init__.py").write_text("")
    chosen.python.write_text("#!/usr/bin/env python3\nimport json,sys,os\nprint(json.dumps([sys.argv[1:],os.environ.get('PYTHONPATH')]))\n")
    chosen.python.chmod(0o700)
    runtime.gateway(target, "poolhouse.workspace.cli", "main")
    arguments = ["inbox", "--label", "literal $(echo unsafe)"]
    done = subprocess.run([sys.executable, str(target), *arguments], capture_output=True,
                          text=True, check=True, env={**os.environ, "PYTHONPATH": "hostile-source"})
    argv, source = json.loads(done.stdout)
    assert argv[0] == "-I" and argv[-3:] == arguments and source is None
    assert "poolhouse.workspace.cli" in argv[2]
    assert "sys.argv[0]='poolhouse-workspace'" in argv[2]


def test_console_preflight_rejects_symlinks_and_foreign_owners_before_writing(tmp_path, chosen, monkeypatch):
    first = tmp_path / "poolhouse"
    first.write_text("original")
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    (tmp_path / "poolhouse-workspace").symlink_to(victim)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=json.dumps({"poolhouse": "poolhouse.cli:main", "poolhouse-workspace": "poolhouse.workspace.cli:main"})))
    with pytest.raises(OSError, match="symbolic"):
        runtime_launchers.install(tmp_path)
    assert first.read_text() == "original" and victim.read_text() == "untouched"
    (tmp_path / "poolhouse-workspace").unlink()
    monkeypatch.setattr(os, "getuid", lambda: first.stat().st_uid + 1)
    with pytest.raises(OSError, match="account"):
        runtime_launchers.install(tmp_path)
    assert first.read_text() == "original"


@pytest.mark.parametrize("value", ["os:system", "poolhouse.cli:main;echo", "poolhouse.cli:main.extra"])
def test_console_metadata_refuses_foreign_or_executable_entrypoints(tmp_path, chosen, monkeypatch, value):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=json.dumps({"poolhouse": value})))
    with pytest.raises(ValueError, match="belong"):
        runtime_launchers.install(tmp_path)
    assert not (tmp_path / "poolhouse").exists()


def test_existing_claude_hooks_preserve_authorization_arguments_and_settings(tmp_path, chosen):
    settings = tmp_path / "settings.json"
    command = "/old/bin/python -m poolhouse.harnesshook pre --role actor --label helper --protect /repo"
    document = {"permissions": {"deny": ["unchanged"]}, "hooks": {
        "PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": command,
         "timeout": 10, "additionalContextLimit": 250}, {"type": "command", "command": "unrelated guard"}]}]}}
    settings.write_text(json.dumps(document))
    assert runtime_launchers.claude_hooks(settings) == 1
    updated = json.loads(settings.read_text())
    hook = updated["hooks"]["PreToolUse"][0]["hooks"][0]
    assert hook["command"] == command.replace("/old/bin/python", str(chosen.python))
    hook["command"] = command
    assert updated == document


def test_codex_command_changes_only_interpreter_and_matching_path(tmp_path, chosen):
    settings = tmp_path / "config.toml"
    command = "PATH=/old/bin:$PATH /old/bin/python -m poolhouse.harnesshook post --label codex"
    text = '[hooks.post]\ncommand = ' + json.dumps(command) + '\ntimeout = 10\nadditionalContextLimit = 250\ntrusted_hash = "existing-review"\n'
    settings.write_text(text)
    receipt = runtime_launchers.codex_hooks(settings)
    assert receipt["changed"] == 1 and receipt["pending_trust"] is True
    assert "revalidation" in receipt["detail"]
    result = tomllib.loads(settings.read_text())["hooks"]["post"]
    assert result.pop("command") == command.replace("/old/bin", str(chosen.python.parent))
    assert result == {"timeout": 10, "additionalContextLimit": 250, "trusted_hash": "existing-review"}


@pytest.mark.parametrize("command", ["/old/bin/python -m foreign.hook post", "echo /old/bin/python -m poolhouse.harnesshook post", "/old/bin/python -m poolhouse.harnesshook post && echo other", "PATH=/unrelated:$PATH /old/bin/python -m poolhouse.harnesshook post"])
def test_unknown_shell_shapes_are_unchanged(command, chosen):
    assert runtime_launchers.hook_command(command, chosen.python) == command


@pytest.mark.parametrize("format", ["claude", "codex"])
def test_concurrent_settings_edit_is_preserved_and_cutover_refused(tmp_path, chosen, monkeypatch, format):
    settings = tmp_path / "settings"
    command = "/old/bin/python -m poolhouse.harnesshook post --label codex"
    if format == "claude":
        settings.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": command}]}]}}))
        cutover = runtime_launchers.claude_hooks
    else:
        settings.write_text('[hooks.post]\ncommand = ' + json.dumps(command) + '\n')
        cutover = runtime_launchers.codex_hooks
    original_protect = runtime.protect

    def intervening_protect(path):
        original_protect(path)
        settings.write_text("concurrent user edit")

    monkeypatch.setattr(runtime, "protect", intervening_protect)
    with pytest.raises(OSError, match="changed during"):
        cutover(settings)
    assert settings.read_text() == "concurrent user edit"
