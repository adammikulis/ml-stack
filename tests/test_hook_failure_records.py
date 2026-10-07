"""Hook failure tracing without daemon, graph or private input."""

import io
import json
import os
import re
import subprocess
import sys

import pytest

from ml_stack import harnesshook, hook_diagnostics


def _record() -> dict:
    paths = list(hook_diagnostics.directory().glob("*.json"))
    assert len(paths) == 1
    assert paths[0].stat().st_mode & 0o077 == 0
    return json.loads(paths[0].read_text())


def test_failed_event_emits_same_reference_on_both_streams_without_input(monkeypatch, capsys):
    monkeypatch.setenv("PRIVATE_TOKEN", "fixture-private-token")
    out = io.StringIO()
    secret = "fixture-private-token"
    assert harnesshook.run(["post", "--unsupported", secret], io.StringIO("PRIVATE TOOL INPUT"), out) == 0
    stderr = capsys.readouterr().err
    held = _record()
    assert held["stage"] == "options" and held["event"] == "post"
    assert held["id"] in stderr and held["id"] in out.getvalue()
    assert held["runtime"]["version"] and held["runtime"]["package"]
    assert "PRIVATE TOOL INPUT" not in json.dumps(held)
    assert secret not in stderr + out.getvalue() + json.dumps(held)
    assert "frames" in held and all("line" in frame and "code" not in frame for frame in held["frames"])
    assert hook_diagnostics.main([held["id"]]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["id"] == held["id"] and inspected["runtime"] == held["runtime"]


@pytest.mark.parametrize("event, status", [("pre", 2), ("post", 0), ("stop", 2)])
def test_import_failure_is_traced_before_heavy_hook_imports(tmp_path, event, status):
    code = """
import importlib.abc, runpy, sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ml_stack.harness_claims':
            raise ImportError('missing fixture import sk-' + 'a' * 30)
sys.meta_path.insert(0, Refuse())
sys.argv = ['harnesshook', sys.argv[1]]
runpy.run_module('ml_stack.harnesshook', run_name='__main__')
"""
    environment = dict(os.environ, ML_STACK_HOME=str(tmp_path / "state"))
    done = subprocess.run([sys.executable, "-c", code, event], env=environment,
                          input="{}", capture_output=True, text=True, timeout=20, check=False)
    assert done.returncode == status, done.stderr
    answer = json.loads(done.stdout)
    assert "diagnostic=" in done.stdout and "stage=bootstrap" in done.stderr
    if event == "post":
        assert answer["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    elif event == "pre":
        assert answer["hookSpecificOutput"]["permissionDecision"] == "deny"
    else:
        assert answer["decision"] == "block"
    assert "sk-" + "a" * 30 not in done.stdout + done.stderr
    held = json.loads(next((tmp_path / "state" / "hook-diagnostics").glob("*.json")).read_text())
    assert held["stage"] == "bootstrap" and held["runtime"]["python"]


def test_unwritable_diagnostics_do_not_change_event_admission(monkeypatch, tmp_path, capsys):
    root = tmp_path / "state"
    root.write_text("fixture")
    monkeypatch.setenv("ML_STACK_HOME", str(root))
    out = io.StringIO()
    assert harnesshook.run(["post", "--unsupported", "fixture"], io.StringIO("{}"), out) == 0
    assert "diagnostic storage unavailable" in capsys.readouterr().err
    assert "notification unavailable" in json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]
    assert harnesshook.run(["pre", "--unsupported", "fixture"], io.StringIO("{}"), io.StringIO()) == 2


def test_symlink_storage_is_refused_without_writing_the_target(monkeypatch, tmp_path):
    root, target = tmp_path / "state", tmp_path / "other"
    root.mkdir()
    target.mkdir()
    (root / "hook-diagnostics").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("ML_STACK_HOME", str(root))
    text = hook_diagnostics.record(RuntimeError("fixture failure"), "post", "nudge")
    assert "diagnostic storage unavailable" in text
    assert not list(target.iterdir())


def test_inspector_rejects_path_traversal_and_redacts_modified_records(capsys):
    message = hook_diagnostics.record(RuntimeError("fixture failure"), "post", "nudge")
    identifier = re.search(r"diagnostic=([a-f0-9]{32})", message)[1]
    path = hook_diagnostics.directory() / f"{identifier}.json"
    path.write_text(json.dumps({"reason": "Bearer " + "a" * 30}))
    assert hook_diagnostics.main([identifier]) == 0
    assert "a" * 30 not in capsys.readouterr().out
    assert hook_diagnostics.main(["../fixture"]) == 2


def test_nudge_failure_keeps_reference_visible_in_post_context(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise FileNotFoundError("workspace launcher missing")
    monkeypatch.setattr(harnesshook.subprocess, "run", fail)
    answer = harnesshook.post("fixture")
    context = answer["hookSpecificOutput"]["additionalContext"]
    held = _record()
    assert held["stage"] == "nudge" and held["id"] in context
    assert held["id"] in capsys.readouterr().err
