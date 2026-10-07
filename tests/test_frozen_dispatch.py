"""Frozen module commands and helper scripts dispatch without starting another daemon."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_stack.fleet import frozen_dispatch

ROOT = Path(__file__).resolve().parents[1]


def test_registered_cli_uses_the_source_registry_and_literal_arguments(monkeypatch):
    from ml_stack import cli

    calls = []
    monkeypatch.setattr(cli, "load", lambda target: lambda argv: calls.append((target, argv)) or 7)
    monkeypatch.setattr(sys, "argv", ["frozen-app"])
    args = ["broker", "--name", "literal;$(touch unwanted)"]
    assert frozen_dispatch.main(["-m", "ml_stack.serve.cli", *args]) == 7
    assert calls == [(cli.commands()["serve"], args)]
    assert sys.argv == ["ml_stack.serve.cli", *args]


@pytest.mark.parametrize("module", ["ml_stack.workspace.localcoding", "ml_stack.workspace.remote_workers",
                                     "ml_stack.workspace.issuepump", "ml_stack.serve.mlx_tree_server"])
def test_worker_modules_use_their_maintained_main_entrypoint(monkeypatch, module):
    calls = []
    monkeypatch.setattr(sys, "argv", ["frozen-app"])
    monkeypatch.setattr(frozen_dispatch, "_target", lambda module: None)
    monkeypatch.setattr(frozen_dispatch.runpy, "run_module", lambda module, **kwargs: calls.append((module, kwargs)))
    assert frozen_dispatch.main(["-m", module, "literal argument"]) == 0
    assert calls == [(module, {"run_name": "__main__", "alter_sys": True})]
    assert sys.argv == [module, "literal argument"]


@pytest.mark.parametrize("script", ["_watchdog", "socket_relay"])
def test_owned_helper_script_paths_dispatch_packaged_modules(monkeypatch, script):
    calls = []
    root = Path(frozen_dispatch.__file__).resolve().parents[1]
    path = root / "serve" / (script + ".py")
    monkeypatch.setattr(sys, "argv", ["frozen-app"])
    monkeypatch.setattr(frozen_dispatch, "_target", lambda module: None)
    monkeypatch.setattr(frozen_dispatch.runpy, "run_module", lambda module, **kwargs: calls.append(module))
    assert frozen_dispatch.main([str(path), "1", "2", "3", "4"]) == 0
    assert calls == ["ml_stack.serve." + script]


def test_harness_hook_keeps_its_module_failure_handler_and_event_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["frozen-app"])
    monkeypatch.setattr(frozen_dispatch, "_target", lambda module: pytest.fail("hook loaded CLI registry"))
    monkeypatch.setattr(frozen_dispatch.runpy, "run_module",
                        lambda module, **kwargs: calls.append((module, list(sys.argv))))
    assert frozen_dispatch.main(["-m", "ml_stack.harnesshook", "post", "--role", "worker"]) == 0
    assert calls == [("ml_stack.harnesshook", ["ml_stack.harnesshook", "post", "--role", "worker"])]


@pytest.mark.parametrize("args", [["-m"], ["-m", "os"], ["-m", "ml_stack.unknown"],
                                  ["-m", "ml_stack.serve.cli;touch unwanted"],
                                  ["-m", "ml_stack../serve.cli"], ["/tmp/foreign.py"],
                                  ["-c", "print('untrusted')"]])
def test_unsupported_invocations_refuse_before_any_module_or_daemon_runs(monkeypatch, args):
    monkeypatch.setattr(frozen_dispatch, "_target", lambda module: pytest.fail("loaded rejected target"))
    monkeypatch.setattr(frozen_dispatch.runpy, "run_module", lambda *a, **kw: pytest.fail("ran rejected module"))
    monkeypatch.setattr(frozen_dispatch.importlib, "import_module", lambda module: pytest.fail("started daemon"))
    assert frozen_dispatch.main(args) == 2


def test_library_without_a_command_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["frozen-app"])
    monkeypatch.setattr(frozen_dispatch, "_target", lambda module: None)
    monkeypatch.setattr(frozen_dispatch.runpy, "run_module", lambda *a, **kw: pytest.fail("ran library module"))
    assert frozen_dispatch.main(["-m", "ml_stack.fleet.device"]) == 2


def test_normal_interface_arguments_still_launch_the_daemon(monkeypatch):
    calls = []
    monkeypatch.setattr(frozen_dispatch.importlib, "import_module",
                        lambda module: SimpleNamespace(main=lambda args: calls.append((module, args)) or 0))
    assert frozen_dispatch.main(["--no-browser", "--port", "9876"]) == 0
    assert calls == [("ml_stack.fleet.launch", ["--no-browser", "--port", "9876"])]


@pytest.mark.parametrize("module,args,expected", [
    ("ml_stack.serve.cli", ["broker", "--help"], "broker"),
    ("ml_stack.mcp", ["--help"], "ml-stack-mcp"),
])
def test_packaged_launcher_dispatches_actual_command_parsers(module, args, expected):
    done = subprocess.run([sys.executable, str(ROOT / "packaging/launcher-headless.py"), "-m", module, *args],
                          capture_output=True, text=True, timeout=20,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stdout + done.stderr
    assert expected in done.stdout
