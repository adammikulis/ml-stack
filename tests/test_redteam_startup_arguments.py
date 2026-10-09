"""Startup and binary information arguments remain literal argv values."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from poolhouse.fleet import wsl, wsl_startup
from poolhouse.serve import backend


@pytest.mark.parametrize("action", ["status", "configure"])
def test_guest_owner_uses_literal_executable_and_json(monkeypatch, action):
    executable = "C:/tools/python name; $(payload).exe"
    monkeypatch.setenv("POOLHOUSE_WINDOWS_PYTHON", executable)
    run = Mock(return_value=SimpleNamespace(stdout='{"installed": true}'))
    monkeypatch.setattr(wsl_startup.subprocess, "run", run)
    hostile = "label; $(payload) & command"
    assert wsl_startup.call(action, labels=[hostile])["installed"]
    assert run.call_args.args[0] == [executable, "-m", "poolhouse.fleet.autostart", "windows-owner"]
    assert json.loads(run.call_args.kwargs["input"])["labels"] == [hostile]
    assert not run.call_args.kwargs.get("shell")


def test_wsl_start_preserves_hostile_arguments_and_service_setup(monkeypatch):
    calls = []
    child = Mock()
    child.wait.return_value = 0
    monkeypatch.setattr(wsl, "_read", lambda *args: "/mnt/c/python; $(payload).exe")
    monkeypatch.setattr(wsl, "_bridge", lambda *args: None)
    local_ui = Mock()
    bridge_factory = Mock(return_value=local_ui)
    monkeypatch.setattr(wsl.wsl_ui, "LocalUIBridge", bridge_factory)
    monkeypatch.setattr(wsl, "command", lambda *args: ["wsl.exe", "--exec", *args])
    run = Mock()
    monkeypatch.setattr(wsl.subprocess, "run", run)
    monkeypatch.setattr(wsl, "launch", lambda argv, **kwargs: calls.append((argv, kwargs)) or child)
    monkeypatch.delenv("POOLHOUSE_HOME", raising=False)
    monkeypatch.delenv("POOLHOUSE_CACHE", raising=False)
    payload = "label; $(payload) & command"
    assert wsl.start(["--label", payload], executable="/usr/bin/python name") == 0
    run.assert_not_called()
    bridge_factory.assert_called_once_with(["wsl.exe", "--exec", "/usr/bin/python name", "-m",
                                            "poolhouse.fleet.wsl_ui", "8770"], 8770)
    local_ui.start.assert_called_once()
    local_ui.close.assert_called_once()
    assert calls[0][0][-2:] == ["--label", payload]
    assert "POOLHOUSE_WINDOWS_PYTHON=/mnt/c/python; $(payload).exe" in calls[0][0]
    assert not calls[0][1].get("shell")


@pytest.mark.parametrize("devices, option", [(False, "--help"), (True, "--list-devices")])
def test_binary_information_uses_one_literal_path_and_fixed_flag(monkeypatch, devices, option):
    path = Path("C:/tools/binary name; $(payload).exe")
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="information", stderr=""))
    monkeypatch.setattr(backend.subprocess, "run", run)
    monkeypatch.setattr(backend, "child_env", lambda path: {})
    assert backend._binary_info(path, devices=devices, timeout=2).strip() == "information"
    assert run.call_args.args[0] == [str(path), option]
    assert run.call_args.kwargs["timeout"] == 2
    assert not run.call_args.kwargs.get("shell")
