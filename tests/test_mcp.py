"""``ml-stack-mcp``: the commands as tools, driven over stdio with a fake transport.

The transport is two in-memory streams carrying newline-delimited JSON-RPC, which is what
the built-in loop reads from a real stdin. What each tool calls is the function the command
calls, so the fakes here stand in for the machine -- a recorded server, a subprocess that
would measure for an hour -- and never for the tool. Nothing reaches the Hub, a GPU or
``~/.ml-stack``: the bench's home is a temporary directory.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from ml_stack import mcp as server

EXPECTED = {"serve_status", "serve_up", "serve_down", "serve_escalate", "models_find",
            "models_files", "models_fetch", "bench_run", "bench_status", "bench_history",
            "bench_show", "fleet_peers", "fleet_join", "world_make", "setup_look", "doctor"}


def rpc(ident, method, **params):
    return {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}


def drive(*messages: dict) -> list[dict]:
    """Feed messages through the loop as a client on stdio would, and read the replies."""
    reader = io.StringIO("".join(json.dumps(m) + "\n" for m in messages))
    writer = io.StringIO()
    server.serve(reader, writer)
    return [json.loads(line) for line in writer.getvalue().splitlines() if line.strip()]


def said(reply: dict):
    """The JSON a tool answered with, out of the tools/call envelope."""
    assert not reply["result"]["isError"], reply["result"]["content"][0]["text"]
    return json.loads(reply["result"]["content"][0]["text"])


@pytest.fixture(autouse=True)
def bench_at_home(tmp_path, monkeypatch):
    from ml_stack.graph import bench

    monkeypatch.setattr(bench, "HOME", tmp_path / "bench")


class TestTheProtocol:
    def test_initialize_and_list_over_a_fake_stdio(self):
        replies = drive(rpc(1, "initialize", protocolVersion="2024-11-05", capabilities={}),
                        {"jsonrpc": "2.0", "method": "notifications/initialized"},
                        rpc(2, "tools/list"), rpc(3, "ping"))
        assert [r["id"] for r in replies] == [1, 2, 3], "a notification gets no reply"
        assert replies[0]["result"]["capabilities"] == {"tools": {}}
        assert replies[0]["result"]["serverInfo"]["name"] == "ml-stack"
        tools = {t["name"]: t for t in replies[1]["result"]["tools"]}
        assert set(tools) == EXPECTED
        for name, tool in tools.items():
            assert tool["description"].strip(), name
            assert tool["inputSchema"]["type"] == "object", name
        assert tools["bench_run"]["inputSchema"]["required"] == ["argv"]
        assert tools["bench_run"]["inputSchema"]["properties"]["argv"] == {
            "type": "array", "items": {"type": "string"}}
        assert tools["serve_status"]["inputSchema"]["properties"]["port"] == {
            "type": "integer", "default": 8080}
        assert tools["fleet_join"]["inputSchema"]["properties"]["persist"]["type"] == "boolean"

    def test_a_bad_line_and_an_unknown_method_are_answered_not_fatal(self):
        reader = io.StringIO('not json\n' + json.dumps(rpc(5, "resources/list")) + "\n")
        writer = io.StringIO()
        server.serve(reader, writer)
        first, second = [json.loads(x) for x in writer.getvalue().splitlines()]
        assert first["error"]["code"] == -32700
        assert second["id"] == 5 and second["error"]["code"] == -32601

    def test_an_unknown_tool_is_an_error_result(self):
        (reply,) = drive(rpc(1, "tools/call", name="serve_everything", arguments={}))
        assert reply["result"]["isError"]
        assert "serve_everything" in reply["result"]["content"][0]["text"]

    def test_every_tool_has_a_schema_the_sdk_would_agree_with(self):
        assert {t.name for t in server.TOOLS} == EXPECTED
        for tool in server.TOOLS:
            schema = tool.schema()
            for name, prop in schema["properties"].items():
                assert prop["type"] in ("string", "integer", "number", "boolean", "array"), \
                    f"{tool.name}.{name}"


class TestTheTools:
    def test_serve_status_calls_the_look_the_command_calls(self, monkeypatch):
        from ml_stack.serve import cli

        monkeypatch.setattr(cli, "recorded_servers", lambda state: {8083: {"model": "x"}})
        monkeypatch.setattr(cli, "look", lambda port, records: cli.Snapshot(
            port=port, base_url=f"http://127.0.0.1:{port}", model="quince-2b.gguf",
            context=8192, slots=1, recorded=port in records) if port == 8083 else None)
        (reply,) = drive(rpc(1, "tools/call", name="serve_status", arguments={"port": 8080}))
        rows = said(reply)
        assert [r["port"] for r in rows] == [8083], "the recorded port and the asked one"
        assert rows[0]["model"] == "quince-2b.gguf" and rows[0]["recorded"] is True

    def test_bench_run_detaches_and_returns_the_handle(self, tmp_path, monkeypatch):
        """The measurement is hours; the call is milliseconds and hands back where to look."""
        from ml_stack.graph.bench import run

        spawned: list[list[str]] = []

        class Child:
            pid = 4242

            def __init__(self, command, **kw):
                spawned.append(list(command))
                assert kw.get("start_new_session") or "creationflags" in kw, \
                    "a measurement owned by this process dies with it"

        monkeypatch.setattr(run.subprocess, "Popen", Child)
        argv = ["sweep", "--serve", "quince-2b.gguf", "--smoke"]
        (reply,) = drive(rpc(1, "tools/call", name="bench_run", arguments={"argv": argv}))
        handle = said(reply)
        assert handle["pid"] == 4242 and handle["argv"] == argv
        assert Path(handle["log"]).is_file(), "the log exists before the call returns"
        assert Path(handle["log"]).parent == tmp_path / "bench" / "logs"
        bench_line = next(c for c in spawned if "ml_stack.graph.bench" in c)
        assert bench_line[-len(argv):] == argv
        assert "--detach" not in bench_line

        (reply,) = drive(rpc(2, "tools/call", name="bench_status", arguments={}))
        state = said(reply)
        assert state["measuring"] is None, "pid 4242 is nobody; the run has ended"
        assert "quince-2b.gguf" in state["text"]

        (reply,) = drive(rpc(3, "tools/call", name="bench_history", arguments={}))
        rows = said(reply)
        assert [r["subcommand"] for r in rows] == ["sweep"]

    def test_bench_show_is_the_command_captured(self):
        (reply,) = drive(rpc(1, "tools/call", name="bench_show", arguments={"args": []}))
        got = said(reply)
        assert got["exit"] == 0 and isinstance(got["output"], str)

    def test_fleet_peers_in_no_cluster_is_empty_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ML_STACK_CLUSTER_KEY", str(tmp_path / "none.key"))
        (reply,) = drive(rpc(1, "tools/call", name="fleet_peers", arguments={"timeout_s": 0.2}))
        assert said(reply) == []

    def test_a_tool_that_raises_answers_with_the_error(self, monkeypatch):
        from ml_stack import hub

        def down(*a, **k):
            raise RuntimeError("the Hub is somebody else's machine")

        monkeypatch.setattr(hub, "find", down)
        (reply,) = drive(rpc(1, "tools/call", name="models_find", arguments={"words": "larch"}))
        assert reply["result"]["isError"]
        assert "somebody else's machine" in reply["result"]["content"][0]["text"]

    def test_detached_writes_the_command_at_the_top_of_its_log(self, tmp_path, monkeypatch):
        class Child:
            pid = 77

            def __init__(self, command, **kw):
                pass

        monkeypatch.setattr(server.subprocess, "Popen", Child)
        got = server.detached("ml_stack.hub", ["fetch", "hf:pellard/larch/larch.gguf"],
                              name="fetch-larch", home=tmp_path / "mcp")
        assert got["pid"] == 77
        first = Path(got["log"]).read_text().splitlines()[0]
        assert first.startswith("command:") and "ml_stack.hub fetch" in first


class TestTheCommand:
    def test_list_prints_every_tool_and_which_transport(self, capsys):
        assert server.main(["--list"]) == 0
        out = capsys.readouterr().out
        for name in EXPECTED:
            assert name in out
        assert "transport:" in out

    @pytest.mark.skipif(not server.sdk_available(), reason="the mcp SDK is not installed")
    def test_the_sdk_server_carries_the_same_tools(self):
        # `asyncio.run` refuses to run inside a live loop, so this went red whenever a
        # neighbour in the same xdist worker left one running -- on the ordering alone, and
        # so only when some unrelated test file was added. `on_a_fresh_loop` is the helper
        # conftest already keeps for exactly that.
        from conftest import on_a_fresh_loop

        app = server.build_sdk_server()
        listed = on_a_fresh_loop(app.list_tools())
        assert {t.name for t in listed} == EXPECTED
        by_name = {t.name: t for t in listed}
        assert by_name["bench_run"].inputSchema["required"] == ["argv"]
