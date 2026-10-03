"""The MCP server as a prompt-injected client sees it: hostile frames, wrong arguments, argument
text that tries to become a command, and tools whose hints claim to only read.

Everything runs against the real tool table with an isolated state directory.
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import os
import subprocess
import textwrap
import types
from pathlib import Path

import pytest

from ml_stack import home, mcp
from ml_stack.fleet import join

ACTING = ("serve_up", "serve_down", "serve_escalate", "models_fetch", "bench_run", "fleet_join",
          "world_make", "speech_say", "conversation_compact", "workspace_send", "workspace_ack",
          "workspace_note_add", "workspace_claim", "workspace_release", "workspace_heartbeat",
          "workspace_scratch_new", "workspace_scratch_rm")
WRITING_CALLS = {"write_text", "write_bytes", "mkdir", "detached", "detach", "unlink", "rmtree",
                 "Popen", "rename", "replace", "touch", "makedirs"}


def tree(root: Path) -> dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in sorted(root.rglob("*"))
            if p.is_file()} if root.exists() else {}


def talk(*lines: str) -> list[dict]:
    out = io.StringIO()
    mcp.serve(io.StringIO("\n".join(lines) + "\n"), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_hostile_json_rpc_gets_an_error_and_runs_nothing(tmp_path):
    before = tree(home.home())
    answers = talk(
        "not json at all", "[1, 2, 3]", "null", '"tools/call"', "{}",
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": 7}}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "../../bin/sh", "arguments": {"a": "b"}}}),
        json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "serve_up", "arguments": ["--exec"]}}),
        json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {"name": "bench_run", "arguments": {"argv": "sweep", "x": "y" * 10_000_000}}}),
        json.dumps({"jsonrpc": "2.0", "id": 6, "method": "__import__('os').system('id')"}),
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": []}),
    )
    by_id = {a["id"]: a for a in answers if a.get("id") is not None}
    assert [a["error"]["code"] for a in answers if a.get("id") is None] == [-32700]
    assert by_id[1]["error"]["code"] == -32602 and by_id[2]["error"]["code"] == -32602
    assert by_id[3]["result"]["isError"] and by_id[4]["result"]["isError"]
    assert by_id[5]["result"]["isError"]
    assert by_id[6]["error"]["code"] == -32601
    assert tree(home.home()) == before
    assert not (home.home() / "mcp").exists()


def test_a_tool_called_with_the_wrong_types_is_refused_before_it_runs(tmp_path):
    before = tree(home.home())
    for tool in mcp.TOOLS:
        named = mcp.call(tool.name, {"__not_a_parameter__": "x"})
        assert named["isError"], tool.name
    for name, arguments in (("serve_up", {"model": {"a": 1}}), ("serve_up", {"model": ["x"]}),
                            ("models_fetch", {"reference": 7}), ("bench_run", {"argv": 7}),
                            ("serve_down", {"port": "x"}),
                            ("speech_say", {"text": "x", "out": {"a": 1}})):
        assert mcp.call(name, arguments)["isError"], (name, arguments)
    assert tree(home.home()) == before


class Spawned:
    """Records every process the tools start instead of starting it."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.shells: list[object] = []

    def __call__(self, command, *args, **kwargs):
        self.commands.append([str(c) for c in command] if isinstance(command, (list, tuple))
                             else [str(command)])
        self.shells.append(kwargs.get("shell", False))
        return types.SimpleNamespace(pid=os.getpid() + 1, poll=lambda: 0)


@pytest.fixture
def spawned(monkeypatch):
    spy = Spawned()
    monkeypatch.setattr(subprocess, "Popen", spy)
    return spy


def test_a_hostile_model_name_is_one_argument_never_a_shell_line(tmp_path, spawned):
    name = "hf:x/y; touch /tmp/pwned; $(touch /tmp/pwned) `id` | cat /etc/passwd > /tmp/x"
    mcp.models_fetch(name)
    mcp.serve_up(name, extra=[name])
    assert len(spawned.commands) == 2 and not any(spawned.shells)
    assert spawned.commands[0][-2:] == ["fetch", name]
    assert spawned.commands[1].count(name) == 2 and "up" in spawned.commands[1]


@pytest.mark.xfail(strict=True, reason="a reference that starts with a dash is parsed as an option")
def test_a_reference_that_starts_with_a_dash_is_not_an_option(tmp_path, spawned):
    answer = mcp.call("models_fetch", {"reference": "--bogus-flag"})
    argv = spawned.commands[0] if spawned.commands else []
    assert answer["isError"] or "--" in argv[argv.index("fetch") + 1:-1]


def calls_in(fn) -> set[str]:
    tree_ = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            for n in ast.walk(tree_) if isinstance(n, ast.Call)}


def test_every_tool_that_changes_something_says_so_in_its_hints():
    by_name = {t.name: t for t in mcp.TOOLS}
    for name in ACTING:
        assert not by_name[name].hints["readOnly"], f"{name} acts and says it only reads"
    for tool in mcp.TOOLS:
        if tool.hints["readOnly"]:
            writes = calls_in(tool.fn) & WRITING_CALLS
            assert not writes, f"{tool.name} says it only reads and calls {sorted(writes)}"


def test_a_destructive_tool_is_not_marked_idempotent_by_accident():
    by_name = {t.name: t for t in mcp.TOOLS}
    assert by_name["conversation_compact"].hints["destructive"]
    assert by_name["serve_down"].hints["destructive"]
    assert by_name["workspace_scratch_rm"].hints["destructive"]


class Spoken:
    duration_s, sample_rate, voice = 0.1, 16000, "v"

    def to_wav(self) -> bytes:
        return b"RIFF-attacker-chosen-bytes"


@pytest.mark.xfail(strict=True, reason="speech_say writes to any path the model names (finding F10)")
def test_speech_say_writes_only_inside_the_state_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_stack.speech.service.say", lambda *a, **k: Spoken())
    outside = tmp_path / "elsewhere" / "authorized_keys"
    mcp.call("speech_say", {"text": "x", "out": str(outside)})
    assert not outside.exists()


@pytest.mark.xfail(strict=True, reason="fleet_join joins a cluster on a model-supplied passphrase "
                                       "with no person present (finding F11)")
def test_fleet_join_needs_a_person():
    source = inspect.getsource(join.join_machine) + inspect.getsource(mcp.fleet_join)
    assert "require_person" in source or "mint(" in source or "HumanRequired" in source
