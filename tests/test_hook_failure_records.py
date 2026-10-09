"""Hook failure tracing without daemon, graph or private input."""

import io
import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from poolhouse import harnesshook, hook_bootstrap, hook_diagnostics, windows_private, worktreerules


def _record() -> dict:
    paths = [p for p in hook_diagnostics.directory().glob("*.json") if hook_diagnostics.IDENTIFIER.fullmatch(p.stem)]
    assert len(paths) == 1
    assert paths[0].stat().st_mode & 0o077 == 0
    return json.loads(paths[0].read_text())


def test_failed_event_emits_same_reference_on_both_streams_without_input(monkeypatch, capsys):
    monkeypatch.setenv("PRIVATE_TOKEN", "fixture-private-token")
    out = io.StringIO()
    needle = "fixture-private-token"
    assert harnesshook.run(["post", "--unsupported", needle], io.StringIO("PRIVATE TOOL INPUT"), out) == 0
    stderr = capsys.readouterr().err
    held = _record()
    assert held["stage"] == "options" and held["event"] == "post"
    assert held["id"] in stderr and held["id"] in out.getvalue()
    assert held["runtime"]["version"] and held["runtime"]["package"]
    assert "PRIVATE TOOL INPUT" not in json.dumps(held)
    assert needle not in stderr + out.getvalue() + json.dumps(held)
    assert "frames" in held and all("line" in frame and "code" not in frame for frame in held["frames"])
    assert hook_diagnostics.COMMAND.run([held["id"]]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["id"] == held["id"] and inspected["runtime"] == held["runtime"]


@pytest.mark.parametrize("event, status", [("pre", 2), ("post", 0), ("stop", 2)])
def test_import_failure_is_traced_before_heavy_hook_imports(tmp_path, event, status):
    code = """
import importlib.abc, runpy, sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'poolhouse.harness_claims':
            raise ImportError('missing fixture import sk-' + 'a' * 30)
sys.meta_path.insert(0, Refuse())
sys.argv = ['harnesshook', sys.argv[1]]
runpy.run_module('poolhouse.harnesshook', run_name='__main__')
"""
    environment = dict(os.environ, POOLHOUSE_HOME=str(tmp_path / "state"))
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
    held = json.loads(next(p for p in (tmp_path / "state" / "hook-diagnostics").glob("*.json") if hook_diagnostics.IDENTIFIER.fullmatch(p.stem)).read_text())
    assert held["stage"] == "bootstrap" and held["runtime"]["python"]


def test_unwritable_diagnostics_do_not_change_event_admission(monkeypatch, tmp_path, capsys):
    root = tmp_path / "state"
    root.write_text("fixture")
    monkeypatch.setenv("POOLHOUSE_HOME", str(root))
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
    monkeypatch.setenv("POOLHOUSE_HOME", str(root))
    text = hook_diagnostics.record(RuntimeError("fixture failure"), "post", "nudge")
    assert "diagnostic storage unavailable" in text
    assert not list(target.iterdir())


def test_inspector_rejects_path_traversal_and_redacts_modified_records(capsys):
    message = hook_diagnostics.record(RuntimeError("fixture failure"), "post", "nudge")
    identifier = re.search(r"diagnostic=([a-f0-9]{32})", message)[1]
    path = hook_diagnostics.directory() / f"{identifier}.json"
    path.write_text(json.dumps({"reason": "Bearer " + "a" * 30}))
    assert hook_diagnostics.COMMAND.run([identifier]) == 0
    assert "a" * 30 not in capsys.readouterr().out
    assert hook_diagnostics.COMMAND.run(["../fixture"]) == 2


def test_nudge_failure_keeps_reference_visible_in_post_context(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise FileNotFoundError("workspace launcher missing")
    monkeypatch.setattr(harnesshook, "_reader_run", fail)
    answer = harnesshook.post("fixture")
    context = answer["hookSpecificOutput"]["additionalContext"]
    held = _record()
    assert held["stage"] == "reader" and held["id"] in context
    assert held["id"] in capsys.readouterr().err


def test_occurrence_preserves_first_branch_runtime_and_counts_after_detail_retention(monkeypatch, capsys):
    monkeypatch.setattr(hook_diagnostics, "MAX_RECORDS", 1)
    first = hook_diagnostics.record(RuntimeError("fixture repeated failure /fixture/first/state"), "post", "nudge",
                                    metadata={"checkout": {"branch": "feat/first", "head": "a" * 40}})
    first_id = re.search(r"diagnostic=([a-f0-9]{32})", first)[1]
    second = hook_diagnostics.record(RuntimeError("fixture repeated failure /fixture/second/state"), "post", "nudge",
                                     metadata={"checkout": {"branch": "feat/second", "head": "b" * 40}})
    second_id = re.search(r"diagnostic=([a-f0-9]{32})", second)[1]
    assert not (hook_diagnostics.directory() / f"{first_id}.json").exists()
    assert hook_diagnostics.COMMAND.run([second_id]) == 0
    held = json.loads(capsys.readouterr().out)
    occurrence = held["occurrence"]
    assert occurrence["count"] == 2 and occurrence["first_id"] == first_id
    assert occurrence["first_checkout"]["branch"] == "feat/first"
    assert held["trace"]["checkout"]["branch"] == "feat/second"
    assert occurrence["first_seen"] <= occurrence["last_seen"]
    assert occurrence["first_runtime"]["commit"] == held["runtime"]["commit"]


def test_checkout_metadata_is_read_without_git_and_correlation_ids_are_hashed(tmp_path, monkeypatch):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/feat/fixture")
    (git / "refs" / "heads" / "feat").mkdir()
    (git / "refs" / "heads" / "feat" / "fixture").write_text("a" * 40)
    monkeypatch.setattr(worktreerules.subprocess, "run", lambda *a, **kw: pytest.fail("started Git"))
    held = worktreerules.checkout_metadata(tmp_path)
    assert held["branch"] == "feat/fixture" and held["head"] == "a" * 40
    monkeypatch.setattr(hook_bootstrap, "METADATA", {})
    hook_bootstrap.metadata({"cwd": str(tmp_path), "session_id": "private-session-payload",
                             "tool_use_id": "private-tool-id", "tool_input": "private-tool-input"})
    trace = hook_bootstrap.timings()
    assert trace["checkout"]["branch"] == "feat/fixture"
    assert "private-" not in json.dumps(trace)


def test_post_watchdog_traces_slow_import_and_returns_before_caller_timeout(tmp_path):
    code = """
import importlib.abc, runpy, sys, time
from poolhouse import hook_bootstrap
hook_bootstrap.POST_SECONDS = .25
class Slow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'poolhouse.harness_claims':
            time.sleep(3)
sys.meta_path.insert(0, Slow())
sys.argv = ['harnesshook', 'post']
runpy.run_module('poolhouse.harnesshook', run_name='__main__')
"""
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, input="{}", text=True,
                          env=dict(os.environ, POOLHOUSE_HOME=str(tmp_path / "state")), timeout=5, check=False)
    assert done.returncode == 0 and time.monotonic() - started < 2
    context = json.loads(done.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "diagnostic=" in context[:250] and "stage=imports" in context
    held = json.loads(next(p for p in (tmp_path / "state" / "hook-diagnostics").glob("*.json")
                          if hook_diagnostics.IDENTIFIER.fullmatch(p.stem)).read_text())
    assert held["trace"]["elapsed_seconds"] < 2
    assert "imports" in held["trace"]["stage_started_seconds"]


def test_windows_records_use_existing_acl_helpers_without_posix_only_options(monkeypatch):
    calls = []
    monkeypatch.setattr(hook_diagnostics, "sys", SimpleNamespace(platform="win32", executable=sys.executable))
    namespace = {name: value for name, value in vars(os).items() if name not in ("getuid", "O_NOFOLLOW", "O_DIRECTORY")}
    monkeypatch.setattr(hook_diagnostics, "os", SimpleNamespace(**namespace))
    monkeypatch.setattr(windows_private, "validate", lambda path: calls.append(("validate", path)))
    monkeypatch.setattr(windows_private, "restrict", lambda path: calls.append(("restrict", path)))
    monkeypatch.setattr(windows_private, "problem", lambda path: "")
    text = hook_diagnostics.record(RuntimeError("fixture failure"), "post", "nudge")
    assert "diagnostic storage unavailable" not in text
    assert {operation for operation, _ in calls} == {"validate", "restrict"}


def test_inspector_refuses_symlinked_record_instead_of_reading_target(tmp_path, capsys):
    hook_diagnostics.record(RuntimeError("fixture failure"), "post", "nudge")
    path = next(p for p in hook_diagnostics.directory().glob("*.json") if hook_diagnostics.IDENTIFIER.fullmatch(p.stem))
    private = tmp_path / "private"
    private.write_text('{"fixture": "PRIVATE FILE CONTENT"}')
    path.unlink()
    path.symlink_to(private)
    assert hook_diagnostics.COMMAND.run([path.stem]) == 2
    assert "PRIVATE FILE CONTENT" not in capsys.readouterr().out


def test_nonsticky_writable_ancestor_is_refused(monkeypatch, tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o777)
    parent.chmod(0o777)
    monkeypatch.setenv("POOLHOUSE_HOME", str(parent / "state"))
    result = hook_diagnostics.record(RuntimeError("fixture failure"), "post", "nudge")
    assert "diagnostic storage unavailable" in result
    assert not (parent / "state").exists()


def test_doctor_hooks_forwards_the_inspectable_diagnostic_id(capsys):
    from poolhouse import doctor
    message = hook_diagnostics.record(RuntimeError("fixture failure"), "post", "nudge")
    identifier = re.search(r"diagnostic=([a-f0-9]{32})", message)[1]
    assert doctor.main(["hooks", identifier]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == identifier
