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

ACTING = ("serve_up", "serve_down", "serve_escalate", "models_fetch", "bench_run",
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
    assert spawned.commands[0][-3:] == ["fetch", "--", name]
    assert spawned.commands[1].count(name) == 2 and "up" in spawned.commands[1]


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


def test_speech_say_writes_only_inside_the_state_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_stack.speech.service.say", lambda *a, **k: Spoken())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "link").symlink_to(home.home())
    for out in (elsewhere / "authorized_keys", "../../../../../tmp/x.wav", "/etc/cron.d/x",
                elsewhere / "link" / "planted.wav", "~/planted.wav"):
        answer = mcp.call("speech_say", {"text": "x", "out": str(out)})
        assert answer["isError"], out
    assert tree(elsewhere) == {}
    assert not (home.home() / "planted.wav").exists()


def test_speech_say_names_its_own_file_under_the_state_directory(monkeypatch):
    monkeypatch.setattr("ml_stack.speech.service.say", lambda *a, **k: Spoken())
    first = json.loads(mcp.call("speech_say", {"text": "x"})["content"][0]["text"])
    second = json.loads(mcp.call("speech_say", {"text": "x"})["content"][0]["text"])
    assert first["out"] != second["out"]
    for made in (first, second):
        target = Path(made["out"]).resolve()
        assert target.is_relative_to(home.home().resolve()) and target.suffix == ".wav"
        assert target.read_bytes() == b"RIFF-attacker-chosen-bytes"


def test_fleet_join_is_not_a_tool_a_model_can_call():
    assert "fleet_join" not in {t.name for t in mcp.TOOLS}
    assert not hasattr(mcp, "fleet_join")
    answer = mcp.call("fleet_join", {"passphrase": "correct horse battery staple"})
    assert answer["isError"]


def test_joining_a_fleet_stays_a_command_a_person_runs():
    with pytest.raises(SystemExit) as left:
        join.main(["join", "--help"])
    assert left.value.code == 0


def test_a_value_that_starts_with_a_dash_is_refused_where_a_command_would_read_it(spawned):
    for tool, arguments in (("models_fetch", {"reference": "--bogus-flag"}),
                            ("models_fetch", {"reference": " -x"}),
                            ("serve_up", {"model": "--exec=id"}),
                            ("serve_up", {"model": "m.gguf", "draft": "-rf"}),
                            ("serve_up", {"model": "m.gguf", "mmproj": "--oops"})):
        assert mcp.call(tool, arguments)["isError"], (tool, arguments)
    assert spawned.commands == []


def test_the_tool_arguments_are_checked_against_the_declared_types(spawned):
    for name, arguments in (("serve_escalate", {"port": "x"}), ("serve_escalate", {"add": 1.5}),
                            ("serve_escalate", {"port": True}), ("serve_down", {"port": "8080"}),
                            ("serve_up", {"model": "m", "extra": "--exec"}),
                            ("serve_up", {"model": "m", "extra": [1]}),
                            ("serve_up", {"model": "m", "escalate": "yes"}),
                            ("serve_up", {}), ("bench_run", {"argv": ["a", 7]}),
                            ("models_find", {"words": "x", "limit": "3"}),
                            ("fleet_peers", {"timeout_s": "2"})):
        answer = mcp.call(name, arguments)
        assert answer["isError"], (name, arguments)
    assert spawned.commands == []
    assert "must be" in mcp.call("serve_escalate", {"port": "x"})["content"][0]["text"]


def test_a_number_may_be_given_to_a_float_argument(tmp_path):
    mcp.checked(mcp.fleet_peers, {"timeout_s": 2})
    mcp.checked(mcp.fleet_peers, {"timeout_s": 2.5})
