"""``ml-stack-mcp`` -- the commands as MCP tools over stdio, for an agent to drive.

Each tool calls the same function the matching command calls -- `serve.cli.look`,
`hub.find`, `graph.bench.run.detach`, `fleet.join.join_machine`, `setup.look`,
`doctor.look` -- so what an agent is told is what a person at the terminal would be
told, and nothing is reimplemented here. Anything long -- a model load, a download, a
measurement -- never blocks the call: it is started in its own session, owned by no
terminal, and the handle (log path and pid) comes back at once. ``bench_status`` and
``bench_history`` read the same files ``ml-stack-bench status|history`` read.

The protocol is spoken two ways, chosen at startup: the ``mcp`` SDK's ``FastMCP`` when it
is installed (``pip install 'ml-stack[mcp]'``), and otherwise a small JSON-RPC loop over
newline-delimited stdio that answers ``initialize``, ``tools/list``, ``tools/call`` and
``ping`` -- the four an MCP client needs to list and call tools. The tool table and the
functions are shared, so the two differ only in transport; ``--builtin`` forces the loop.

Register it in Claude Code's config as a stdio server::

    claude mcp add ml-stack -- ml-stack-mcp

or in ``.mcp.json``: ``{"mcpServers": {"ml-stack": {"command": "ml-stack-mcp"}}}``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import inspect
import io
import json
import os
import subprocess
import sys
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

__all__ = ["MCP_HOME", "PROTOCOL", "Tool", "TOOLS", "build_sdk_server", "detached",
           "handle", "main", "schema_of", "serve", "sdk_available"]

PROTOCOL = "2024-11-05"
MCP_HOME = Path("~/.ml-stack/mcp")
"""Where detached commands that are not the bench's keep their logs. The bench keeps its
own, under its own home, because ``ml-stack-bench status`` reads them from there."""


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return schema_of(self.fn)

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "inputSchema": self.schema()}


_JSON_TYPES: dict[Any, dict[str, Any]] = {
    str: {"type": "string"}, int: {"type": "integer"}, float: {"type": "number"},
    bool: {"type": "boolean"},
}


def schema_of(fn: Callable[..., Any]) -> dict[str, Any]:
    """A JSON schema for ``fn``'s keyword arguments, read from its type hints.

    ``str``, ``int``, ``float``, ``bool`` and ``list[str]`` are what the tools take; a
    parameter with no default is required. The same hints are what FastMCP reads, so the
    two transports describe every tool identically.
    """
    hints = typing.get_type_hints(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        hint = hints.get(name, str)
        if typing.get_origin(hint) is list:
            inner = typing.get_args(hint)[0] if typing.get_args(hint) else str
            prop: dict[str, Any] = {"type": "array",
                                    "items": dict(_JSON_TYPES.get(inner, {"type": "string"}))}
        else:
            prop = dict(_JSON_TYPES.get(hint, {"type": "string"}))
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default
        props[name] = prop
    out: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


# -- detaching -------------------------------------------------------------------------
def detached(module: str, argv: list[str], *, name: str,
             home: Path | None = None) -> dict[str, Any]:
    """Run ``python -m module argv`` in its own session and return its log and pid.

    The same shape ``ml-stack-bench --detach`` makes, for the commands that have no
    ``--detach`` of their own: a server coming up, a download. The first line of the log
    is the command, so a log found later says what it was.
    """
    logs = (home or MCP_HOME).expanduser() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}-{time.strftime('%Y%m%dT%H%M%S')}.log"
    from ml_stack.platform import process_group_kwargs

    command = [sys.executable, "-m", module, *argv]
    with log.open("ab") as out:
        out.write((f"command: {' '.join(command)}\nstarted: {time.strftime('%FT%T')}\n")
                  .encode("utf-8"))
        out.flush()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                 **process_group_kwargs())
    return {"log": str(log), "pid": child.pid, "command": " ".join(command)}


def _captured(fn: Callable[[], int]) -> dict[str, Any]:
    """Run a command's ``main`` in-process and hand back what it printed and its exit."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = int(fn() or 0)
        except SystemExit as left:
            code = int(left.code or 0) if isinstance(left.code, int) else 1
    return {"exit": code, "output": out.getvalue(), "errors": err.getvalue()}


def _plain(value: Any) -> Any:
    """Dataclasses and paths as JSON."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return value


# -- the tools -------------------------------------------------------------------------
def serve_status(port: int = 8080) -> list[dict[str, Any]]:
    """What is serving on this machine: every recorded server and ``port``, each with its
    model, context, slots and lease -- what ``ml-stack-serve status`` prints."""
    from ml_stack.serve import cli

    records = cli.recorded_servers(cli.STATE_FILE)
    found = []
    for one in sorted({*records, int(port)}):
        snapshot = cli.look(one, records)
        if snapshot is not None:
            found.append(_plain(snapshot))
    return found


def serve_up(model: str, port: int = 8080, context: int = 0, parallel: int = 1,
             draft: str = "", mmproj: str = "", extra: list[str] = []) -> dict[str, Any]:
    """Put ``model`` (a path or ``hf:owner/repo/file.gguf``) up on ``port`` with
    ``ml-stack-serve up``, detached; returns the log and pid, and ``serve_status`` says
    when it is answering. ``draft`` and ``mmproj`` take a path or ``auto``."""
    argv = ["up", model, "--port", str(port), "--parallel", str(parallel)]
    if context:
        argv += ["--context", str(context)]
    if draft:
        argv += ["--draft", draft]
    if mmproj:
        argv += ["--mmproj", mmproj]
    argv += list(extra or [])
    return detached("ml_stack.serve.cli", argv, name=f"serve-{Path(model).name}")


def serve_down(port: int = 8080) -> dict[str, Any]:
    """Stop the server this machine started on ``port`` (``ml-stack-serve down``)."""
    from ml_stack.serve.cli import main as serve_main

    return _captured(lambda: serve_main(["down", "--port", str(port)]))


def models_find(words: str, limit: int = 12, gguf_only: bool = True) -> list[dict[str, Any]]:
    """Search the Hub for repositories matching ``words``, the trusted publishers first
    (``ml-stack-models find``); a model newer than the assistant remembers is found here."""
    from ml_stack.hub import find

    return [{"repo": f.repo, "downloads": f.downloads, "likes": f.likes}
            for f in find(words, gguf=gguf_only, limit=limit)]


def models_files(repo: str, ending: str = ".gguf") -> list[dict[str, Any]]:
    """The files in one Hub repository, weights first and largest first, each with the
    ``hf:`` reference that serves it (``ml-stack-models files``)."""
    from ml_stack.hub import aside, files, ref

    return [{"name": name, "bytes": size, "reference": ref(repo, name),
             "alongside": bool(aside(name))} for name, size in files(repo, ending=ending)]


def models_fetch(reference: str) -> dict[str, Any]:
    """Download an ``hf:owner/repo/file.gguf`` reference -- every shard -- into the cache
    without serving it (``ml-stack-models fetch``), detached; returns the log and pid."""
    return detached("ml_stack.hub", ["fetch", reference],
                    name=f"fetch-{reference.rsplit('/', 1)[-1]}")


def bench_run(argv: list[str]) -> dict[str, Any]:
    """Start ``ml-stack-bench argv`` (e.g. ``["sweep", "--serve", "hf:...", "--smoke"]``)
    detached, exactly as ``--detach`` would; returns the log path, pid and argv, and
    ``bench_status`` follows it."""
    from ml_stack.graph.bench.run import detach, measuring_file

    log = detach(list(argv))
    try:
        held = json.loads(measuring_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        held = {}
    return {"log": str(log), "pid": held.get("pid"), "argv": list(argv),
            "started": held.get("started"), "commit": held.get("commit")}


def bench_status() -> dict[str, Any]:
    """What is measuring right now, its last log line, or that nothing is
    (``ml-stack-bench status``)."""
    from ml_stack.graph.bench.run import measuring, status

    return {"text": status(), "measuring": measuring()}


def bench_history(since: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """Every measurement the bench has run, newest first -- what ran, when, how long, how it
    ended and what it kept (``ml-stack-bench history``); ``since`` is an ISO date or a
    span like ``2d``."""
    from ml_stack.graph.bench import HOME
    from ml_stack.graph.bench.history import _iso, history
    from ml_stack.graph.bench.history import since as since_at

    rows = history(HOME)
    if since:
        floor = _iso(since_at(since))
        rows = [e for e in rows if e.started >= floor]
    return [_plain(e) for e in rows[::-1][: max(1, int(limit))]]


def bench_show(args: list[str] = []) -> dict[str, Any]:
    """The bench's table of kept runs, as ``ml-stack-bench show args`` prints it -- pass
    ``["--rates"]`` for accuracy per second, per 1k tokens and per GB, ``["--rank",
    "FILE.md"]`` to write the ranking."""
    from ml_stack.graph.bench.run import _main

    return _captured(lambda: _main(["show", *args]))


def fleet_peers(timeout_s: float = 2.0) -> list[dict[str, Any]]:
    """Every peer on the LAN holding this machine's cluster key: what each serves, its
    room, whether it is busy or measuring, and its commit (``ml-stack-fleet status``)."""
    from ml_stack.fleet.join import peers
    from ml_stack.fleet.launch import already_running

    me = already_running() or {}
    return peers(timeout_s=timeout_s, self_name=str(me.get("name") or ""))


def fleet_join(passphrase: str = "", group: str = "ml-stack", name: str = "",
               persist: bool = False) -> dict[str, Any]:
    """Make this machine a peer (``ml-stack-fleet join``): the checks, the cluster
    ``passphrase`` joins, the daemon started if none answers, at logon with ``persist``,
    and the peers that answered."""
    from ml_stack.fleet.join import join_machine

    said: list[str] = []
    joined = join_machine(name=name, passphrase=passphrase, group=group, persist=persist,
                          say=said.append)
    return {**joined.public(), "said": said}


def world_make(kind: str = "company", size: str = "small", seed: int = 0,
               out: str = "") -> dict[str, Any]:
    """Invent an organised group as a graph with people who could talk
    (``ml-stack-world make``); ``out`` is the directory written (default: under
    ``~/.ml-stack/worlds``)."""
    from ml_stack.world.cli import main as world_main

    where = out or str(Path("~/.ml-stack/worlds").expanduser() / f"{kind}-{size}-{seed}")
    got = _captured(lambda: world_main(["make", "--kind", kind, "--size", size, "--seed",
                                        str(seed), "--out", where, "--json"]))
    try:
        return {**json.loads(got["output"].strip().splitlines()[-1]), "exit": got["exit"]}
    except (ValueError, IndexError):
        return got


def setup_look() -> list[dict[str, Any]]:
    """The machine facts serving depends on -- memory a model may use and whether that
    survives a reboot, the llama-server and what it reads, what is downloaded -- with the
    fix for each (``ml-stack-setup``, without running any fix)."""
    from ml_stack.setup import look

    return [_plain(f) for f in look()]


def doctor(repos: list[str] = []) -> list[dict[str, Any]]:
    """The checkouts, the bench store and the managed llama.cpp, each finding with its fix
    (``ml-stack-doctor``, without running any fix); ``repos`` picks the checkouts."""
    from ml_stack.doctor import look, repositories

    return [_plain(f) for f in look(repositories(list(repos)) if repos else None)]


TOOLS: list[Tool] = [
    Tool("serve_status", "What is serving on this machine, and what a lease would do.",
         serve_status),
    Tool("serve_up", "Put a model up on a port, detached; returns the log and pid.", serve_up),
    Tool("serve_down", "Stop the server this machine started on a port.", serve_down),
    Tool("models_find", "Search the Hub for a model, the trusted publishers first.",
         models_find),
    Tool("models_files", "The files in one Hub repository, with the hf: reference for each.",
         models_files),
    Tool("models_fetch", "Download an hf: reference into the cache, detached.", models_fetch),
    Tool("bench_run", "Start a measurement in the background; returns its log and pid.",
         bench_run),
    Tool("bench_status", "What is measuring now, or that nothing is.", bench_status),
    Tool("bench_history", "Every measurement run, newest first, with how each ended.",
         bench_history),
    Tool("bench_show", "The table of kept runs, as ml-stack-bench show prints it.", bench_show),
    Tool("fleet_peers", "Every peer on the LAN: serving, room, busy, commit.", fleet_peers),
    Tool("fleet_join", "Make this machine a peer: checks, cluster, daemon, announce.",
         fleet_join),
    Tool("world_make", "Invent an organised group as a graph with people who could talk.",
         world_make),
    Tool("setup_look", "The machine facts serving depends on, with a fix for each.",
         setup_look),
    Tool("doctor", "The checkouts, the bench store and the managed llama.cpp, checked.",
         doctor),
]
_BY_NAME = {t.name: t for t in TOOLS}


# -- the built-in transport ------------------------------------------------------------
def _version() -> str:
    try:
        from importlib.metadata import version

        return version("ml-stack")
    except Exception:  # noqa: BLE001
        return "0"


def call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one tool and wrap what it returned the way ``tools/call`` answers."""
    tool = _BY_NAME.get(name)
    if tool is None:
        return {"content": [{"type": "text", "text": f"no tool called {name!r}"}],
                "isError": True}
    try:
        got = tool.fn(**(arguments or {}))
    except Exception as exc:  # noqa: BLE001 - the error is the answer
        return {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True}
    return {"content": [{"type": "text",
                         "text": json.dumps(_plain(got), indent=1, default=str)}],
            "isError": False}


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """One JSON-RPC message in, one response out -- or None for a notification."""
    method = message.get("method", "")
    ident = message.get("id")
    params = message.get("params") or {}

    def reply(result: Any) -> dict[str, Any] | None:
        return None if ident is None else {"jsonrpc": "2.0", "id": ident, "result": result}

    def refuse(code: int, text: str) -> dict[str, Any] | None:
        return None if ident is None else {"jsonrpc": "2.0", "id": ident,
                                           "error": {"code": code, "message": text}}

    if method == "initialize":
        return reply({"protocolVersion": params.get("protocolVersion") or PROTOCOL,
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "ml-stack", "version": _version()}})
    if method == "ping":
        return reply({})
    if method == "tools/list":
        return reply({"tools": [t.public() for t in TOOLS]})
    if method == "tools/call":
        if not isinstance(params.get("name"), str):
            return refuse(-32602, "tools/call needs a tool name")
        return reply(call(params["name"], params.get("arguments") or {}))
    if method.startswith("notifications/"):
        return None
    return refuse(-32601, f"no such method: {method}")


def serve(reader: TextIO, writer: TextIO) -> int:
    """The loop: one JSON message per line in, one per line out, until EOF."""
    for line in reader:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            writer.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "parse error"}})
                         + "\n")
            writer.flush()
            continue
        answer = handle(message if isinstance(message, dict) else {})
        if answer is not None:
            writer.write(json.dumps(answer, default=str) + "\n")
            writer.flush()
    return 0


# -- the SDK transport -----------------------------------------------------------------
def sdk_available() -> bool:
    try:
        import mcp.server.fastmcp  # noqa: F401
    except ImportError:
        return False
    return True


def build_sdk_server() -> Any:
    """A ``FastMCP`` server carrying the same tools; needs ``pip install 'ml-stack[mcp]'``."""
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("ml-stack")
    for tool in TOOLS:
        app.add_tool(tool.fn, name=tool.name, description=tool.description)
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ml-stack-mcp",
        description="The ml-stack commands as MCP tools over stdio. Register it with "
                    "'claude mcp add ml-stack -- ml-stack-mcp'.")
    ap.add_argument("--list", action="store_true", help="print the tools and exit")
    ap.add_argument("--builtin", action="store_true",
                    help="speak the protocol with the built-in loop even when the mcp SDK "
                         "is installed")
    args = ap.parse_args(argv)
    if args.list:
        for tool in TOOLS:
            required = tool.schema().get("required", [])
            props = tool.schema()["properties"]
            shown = ", ".join(f"{k}{'' if k in required else '?'}" for k in props)
            print(f"{tool.name:<14} ({shown})\n    {tool.description}")
        print(f"\ntransport: {'mcp SDK' if sdk_available() and not args.builtin else 'built-in'}")
        return 0
    if sdk_available() and not args.builtin:
        build_sdk_server().run("stdio")
        return 0
    return serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
