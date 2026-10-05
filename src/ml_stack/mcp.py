"""``ml-stack-mcp`` -- the commands as MCP tools over stdio, for an agent to drive.

Each tool calls the same function the matching command calls -- `serve.cli.look`,
`hub.find`, `bench.underway.detach`, `setup.look`,
`setup.look_checkouts` -- so what an agent is told is what a person at the terminal
would be told, and nothing is reimplemented here. Anything long -- a model load, a download, a
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

import argparse
import contextlib
import dataclasses
import inspect
import io
import json
import re
import secrets
import sys
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from ml_stack.agent import Compaction, Counter, Spill, Transcript, compact, model_summarizer
from ml_stack.client import Client
from ml_stack.decide import router
from ml_stack.home import state
from ml_stack.log import say
from ml_stack.serve import ops, quant_guard
from ml_stack.tool_schema import _JSON_TYPES, schema_of
from ml_stack.workspace import tools as workspace_tools

__all__ = [
    "PROTOCOL",
    "TOOLS",
    "Tool",
    "build_sdk_server",
    "detached",
    "handle",
    "main",
    "mcp_home",
    "schema_of",
    "sdk_available",
    "serve",
]

PROTOCOL = "2024-11-05"
def mcp_home() -> Path:
    """Where a detached command run through MCP keeps its logs."""
    return state("mcp")
"""Where detached commands that are not the bench's keep their logs. The bench keeps its
own, under its own home, because ``ml-stack-bench status`` reads them from there."""


def _hints(*, read_only: bool, destructive: bool = False, idempotent: bool = False,
           open_world: bool = False) -> dict[str, bool]:
    return {"readOnly": read_only, "destructive": destructive, "idempotent": idempotent,
            "openWorld": open_world}


READS = _hints(read_only=True, idempotent=True)
"""Hints for a tool that only looks."""

WRITES = _hints(read_only=False)
"""Hints for a tool that changes something and can be repeated at a cost."""


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    hints: dict[str, bool] = field(default_factory=lambda: dict(WRITES))

    def schema(self) -> dict[str, Any]:
        return schema_of(self.fn)

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "inputSchema": self.schema(), "annotations": self.annotations()}

    def annotations(self) -> dict[str, bool]:
        """The tool's behaviour hints under the names the protocol uses."""
        return {f"{k}Hint": v for k, v in self.hints.items()}


# -- detaching -------------------------------------------------------------------------
def detached(module: str, argv: list[str], *, name: str,
             home: Path | None = None) -> dict[str, Any]:
    """Run ``python -m module argv`` in its own session and return its log and pid.

    The same shape ``ml-stack-bench --detach`` makes, for the commands that have no
    ``--detach`` of their own: a server coming up, a download. The log's header names the
    module and its arguments, so a log found later says what it was.
    """
    from ml_stack import jobs

    log = (home or mcp_home()) / "logs" / f"{name}-{time.strftime('%Y%m%dT%H%M%S')}.log"
    ran = jobs.detach(module, argv, log=log, lines=[f"command: {module} {' '.join(argv)}"])
    return {"log": str(ran.log), "pid": ran.pid, "command": " ".join(ran.command)}


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


def _not_an_option(value: str, what: str) -> str:
    """``value`` unchanged, or a refusal when a command would read it as an option."""
    if value.lstrip().startswith("-"):
        raise ValueError(f"{what} may not start with '-': {value!r}")
    return value


def _confined(path: str) -> Path:
    """``path`` resolved, or ``ValueError`` unless it lies under the MCP home or the working directory."""
    resolved = Path(path).expanduser().resolve()
    for root in (mcp_home().resolve(), Path.cwd().resolve()):
        if resolved == root or root in resolved.parents:
            return resolved
    raise ValueError(f"{path!r} is outside the MCP home and the working directory")


def _local_server(url: str) -> str:
    """``url`` when it is the address of a server this machine's broker holds, else ``ValueError``."""
    held = {ops.base_url_for(port).rstrip("/") for port in ops.recorded_servers(ops.lease_file())}
    if url.rstrip("/") not in held:
        raise ValueError(f"{url!r} is not a server this machine is serving")
    return url


def _check_type(name: str, value: Any, hint: Any) -> None:
    """Raise ``TypeError`` unless ``value`` is what the tool's hint for ``name`` declares."""
    if type(None) in typing.get_args(hint):
        hint = next(part for part in typing.get_args(hint) if part is not type(None))
    if typing.get_origin(hint) is list:
        inner = (typing.get_args(hint) or (str,))[0]
        if not isinstance(value, list) or any(not _is(v, inner) for v in value):
            raise TypeError(f"{name} must be a list of {inner.__name__}")
    elif hint in _JSON_TYPES and not _is(value, hint):
        raise TypeError(f"{name} must be {_JSON_TYPES[hint]['type']}, not {type(value).__name__}")


def _is(value: Any, kind: Any) -> bool:
    if kind is bool:
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) if kind is float else isinstance(value, kind)


def checked(fn: Callable[..., Any], arguments: dict[str, Any]) -> None:
    """Raise ``TypeError`` for an argument ``fn`` does not take, a missing required one, or
    one whose JSON type is not the one its hint declares."""
    params = inspect.signature(fn).parameters
    hints = typing.get_type_hints(fn)
    for name, value in arguments.items():
        if name not in params:
            raise TypeError(f"no argument called {name!r}")
        _check_type(name, value, hints.get(name, str))
    missing = [n for n, p in params.items() if p.default is inspect.Parameter.empty
               and n not in arguments]
    if missing:
        raise TypeError(f"missing: {', '.join(missing)}")


# -- the tools -------------------------------------------------------------------------
def serve_status(port: int = 8080) -> list[dict[str, Any]]:
    """What is serving on this machine: every recorded server and ``port``, each with its
    model, context, slots and lease -- what ``ml-stack-serve status`` prints."""
    records = ops.recorded_servers(ops.lease_file())
    found = []
    for one in sorted({*records, int(port)}):
        snapshot = ops.look(one, records)
        if snapshot is not None:
            found.append(_plain(snapshot))
    return found


_NOT_A_TOOL_ARGUMENT = ("--iq", "--port", "--escalate", "--binary", "--build", "--root")
"""``up`` flags a model may not set through ``extra``: the IQ mode is a person's, the broker
picks the port, and growing a server, the binary and the fleet root are not a tool's to choose."""


def serve_up(model: str, *, context: int = 0, draft: str = "",
             mmproj: str = "", extra: list[str] | None = None) -> dict[str, Any]:
    """Ask for a lease on a server for ``model`` (a path or ``hf:owner/repo/file.gguf``) with
    ``ml-stack-serve up --no-wait``, detached; returns the log and pid, and ``serve_status``
    says when it is answering. The broker picks the port, checks the memory and queues the
    lease when it is short. ``draft`` and ``mmproj`` take a path or ``auto``."""
    if why := quant_guard.blocked_message(model):
        return {"started": False, "blocked": why}
    for one in extra or []:
        flag = str(one).split("=", 1)[0]
        if flag in _NOT_A_TOOL_ARGUMENT:
            raise ValueError(f"{flag} is a person's to set, not a tool argument")
    argv = ["up", _not_an_option(model, "model"), "--no-wait", "--parallel", "1"]
    if context:
        argv += ["--context", str(context)]
    if draft:
        argv += ["--draft", _not_an_option(draft, "draft")]
    if mmproj:
        argv += ["--mmproj", _not_an_option(mmproj, "mmproj")]
    argv += list(extra or [])
    return detached("ml_stack.serve.cli", argv, name=f"serve-{Path(model).name}")


def serve_down(port: int = 8080) -> dict[str, Any]:
    """Stop the server this machine started on ``port`` (``ml-stack-serve down``); says
    whether a process was still running, and why it would not act when it would not."""
    try:
        stopped, fleet = ops.down(int(port))
    except ops.Refused as no:
        return {"port": int(port), "stopped": False, "why": [line.strip() for line in no.lines
                                                             if line.strip()]}
    return {"port": stopped.port, "base_url": stopped.base_url, "pid": stopped.pid,
            "stopped": stopped.was_running,
            **({"fleet": fleet.strip()} if fleet.strip() else {})}


def serve_escalate(port: int = 8080, add: int = 1) -> dict[str, Any]:
    """Grow the slots the server on ``port`` holds by ``add``, keeping every live
    conversation (``ml-stack-serve escalate``), detached; ``serve_status`` says when the
    relaunch has finished."""
    return detached("ml_stack.serve.cli", ["escalate", "--port", str(port), "--add",
                                           str(add)], name=f"escalate-{port}")


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
    return detached("ml_stack.hub", ["fetch", "--", _not_an_option(reference, "reference")],
                    name=f"fetch-{reference.rsplit('/', 1)[-1]}")


_BENCH_COMMANDS = ("sweep", "run")
_BENCH_FLAGS = frozenset({
    "--serve", "--context", "--parallel", "--serve-draft", "--serve-kv", "--serve-kv-unified",
    "--no-serve-kv-unified", "--profile", "--no-profile", "--serve-label", "--no-draft",
    "--label-suffix", "--serve-mlock", "--serve-no-flash-attn", "--serve-mmproj", "--n-max",
    "--reasoning-budget", "--plain-only", "--resume", "--shortlist", "--shortlist-for",
    "--margin", "--smoke", "--sample", "--short", "--trace", "--no-trace", "--temperature",
    "--top-p", "--top-k", "--min-p", "--n-predict", "--card", "--anyway", "--ceiling", "--yes",
    "--no-selfcheck", "--no-smoke", "--per-question", "--no-queue", "--no-prefetch",
})
"""The ``ml-stack-bench`` subcommands and flags a tool may pass: none takes a path to run, a
store or kept file to read or write, or an endpoint to call."""


def _checked_bench_argv(argv: list[str]) -> list[str]:
    """``argv`` unchanged when it is an allowed subcommand with allowed flags; else ``ValueError``."""
    words = [str(a) for a in argv]
    if not words or words[0] not in _BENCH_COMMANDS:
        raise ValueError(f"bench_run takes one of {', '.join(_BENCH_COMMANDS)} first")
    for word in words[1:]:
        flag = word.split("=", 1)[0]
        if word.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", word) and flag not in _BENCH_FLAGS:
            raise ValueError(f"{flag} is not a flag bench_run accepts")
    return words


def bench_run(argv: list[str]) -> dict[str, Any]:
    """Start ``ml-stack-bench argv`` (e.g. ``["sweep", "--serve", "hf:...", "--smoke"]``)
    detached, exactly as ``--detach`` would; returns the log path, pid and argv, and
    ``bench_status`` follows it."""
    from ml_stack.bench.underway import detach, measuring_file

    log = detach(_checked_bench_argv(argv))
    try:
        held = json.loads(measuring_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        held = {}
    return {"log": str(log), "pid": held.get("pid"), "argv": list(argv),
            "started": held.get("started"), "commit": held.get("commit")}


def bench_status() -> dict[str, Any]:
    """What is measuring right now, its last log line, or that nothing is
    (``ml-stack-bench status``)."""
    from ml_stack.bench.progress import status
    from ml_stack.bench.underway import measuring

    return {"text": status(), "measuring": measuring()}


def bench_history(since: str = "", limit: int = 20) -> list[dict[str, Any]]:
    """Every measurement the bench has run, newest first -- what ran, when, how long, how it
    ended and what it kept (``ml-stack-bench history``); ``since`` is an ISO date or a
    span like ``2d``."""
    from ml_stack.bench import home_dir
    from ml_stack.bench.history import _iso, history, since as since_at

    rows = history(home_dir())
    if since:
        floor = _iso(since_at(since))
        rows = [e for e in rows if e.started >= floor]
    return [_plain(e) for e in rows[::-1][: max(1, int(limit))]]


def bench_show(last: int = 0, since: str = "", extract: bool = False,
               speed: bool = False) -> list[dict[str, Any]]:
    """Every benchmark run kept, as records: what was served, how it was asked, what it
    scored and what it cost (``ml-stack-bench show``). ``last`` keeps the newest N and
    ``since`` an ISO time; ``extract`` asks for the extraction runs and ``speed`` for the
    speed grids instead of the answering ones."""
    from ml_stack.bench import home_dir
    from ml_stack.bench.ops import kept_for, summarised

    got = kept_for(home_dir() / "runs.ladybug", last=int(last), since=since)
    return summarised(got.speed if speed else got.extracted if extract else got.answering)


def fleet_peers(timeout_s: float = 2.0) -> list[dict[str, Any]]:
    """Every peer on the LAN holding this machine's cluster key: what each serves, its
    room, whether it is busy or measuring, and its commit (``ml-stack-fleet status``)."""
    from ml_stack.fleet.join import peers
    from ml_stack.fleet.launch import already_running

    me = already_running() or {}
    return peers(timeout_s=timeout_s, self_machine=str(me.get("machine") or ""))


def world_make(kind: str = "company", size: str = "small", seed: int = 0,
               out: str = "") -> dict[str, Any]:
    """Invent an organised group as a graph with people who could talk
    (``ml-stack-world make``); ``out`` is the directory written (default: under
    ``~/.ml-stack/worlds``)."""
    from ml_stack.world.ops import invent

    where = out or str(state("worlds") / f"{kind}-{size}-{seed}")
    return invent(kind=kind, size=size, seed=int(seed), out=where)


def setup_look() -> list[dict[str, Any]]:
    """The machine facts serving depends on -- memory a model may use and whether that
    survives a reboot, the llama-server and what it reads, what is downloaded -- with the
    fix for each (``ml-stack-setup``, without running any fix)."""
    from ml_stack.setup import look

    return [_plain(f) for f in look()]


def speech_providers() -> dict[str, Any]:
    """Every speech engine on this machine -- recognition, synthesis and voice activity --
    with whether it could run and which one ``auto`` picks (``ml-stack-speech providers``)."""
    from ml_stack.speech.service import providers

    return providers()


def speech_transcribe(path: str, provider: str = "", language: str = "") -> dict[str, Any]:
    """An audio file (anything ffmpeg reads) as text with per-segment times
    (``ml-stack-speech transcribe``); ``language`` is guessed when it is not given."""
    from ml_stack.speech.service import transcribe

    return _plain(transcribe(_confined(path), provider=provider or None,
                             language=language or None))


def speech_say(text: str, provider: str = "", voice: str = "") -> dict[str, Any]:
    """Speak ``text`` into a new WAV file under ``~/.ml-stack/mcp/speech``
    (``ml-stack-speech say``); returns the path, how long it is and the voice that said it."""
    from ml_stack.speech.service import say

    spoken = say(text, provider=provider or None, voice=voice or None)
    target = mcp_home() / "speech" / f"say-{time.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(4)}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(spoken.to_wav())
    return {"out": str(target), "duration_s": spoken.duration_s,
            "sample_rate": spoken.sample_rate, "voice": spoken.voice}


def decide(question: str, options: list[str], *, state_text: str = "", backend: str = "auto",
           abstain_below: float = -1.0) -> dict[str, Any]:
    """Choose one of ``options`` (``NAME`` or ``NAME=description``) for ``question`` about
    ``state_text`` and say how sure (``ml-stack-decide ask``); ``backend`` is ``auto``,
    ``logprob``, ``pointer``, ``embed`` or ``rules`` using the configured chat server;
    a positive ``abstain_below`` flags answers under that probability."""
    named = {n.strip(): d.strip() for n, _, d in (o.partition("=") for o in options)}
    got = router.decide(question, state_text, named,
                        abstain_below=abstain_below if abstain_below > 0 else None,
                        config=router.shared(backend, ""))
    return got.public()


def doctor(repos: list[str] | None = None) -> list[dict[str, Any]]:
    """The checkouts, the bench store and the managed llama.cpp, each finding with its fix
    (``ml-stack-doctor``, without running any fix); ``repos`` picks the checkouts."""
    from ml_stack.doctor import look_checkouts, repositories

    return [_plain(f) for f in look_checkouts(repositories(list(repos)) if repos else None)]


def conversation_compact(path: str, budget: int, keep_last: int = 6, url: str = "",
                         write: bool = False) -> dict[str, Any]:
    """Fit the chat messages in a JSON file to ``budget`` tokens: long tool results cut,
    repeated calls dropped, the oldest messages summarised by the model at ``url`` (removed
    outright when there is none). ``write`` replaces the file; removed text is kept under
    ``~/.ml-stack/compaction``."""
    target = _confined(path)
    client = Client(_local_server(url)) if url else None
    messages = json.loads(target.read_text(encoding="utf-8"))
    fitted = compact(messages, budget=budget, count=Counter(client), using=Compaction(
        keep_last=keep_last, summarize=bool(url),
        summarizer=model_summarizer(client) if client else None,
        spill=Spill(Transcript())))
    if write and fitted.strategy_used != "none":
        target.write_text(json.dumps(fitted.messages, indent=2), encoding="utf-8")
    return {"strategy": fitted.strategy_used, "dropped": fitted.dropped_count,
            "tokens_before": fitted.tokens_before, "tokens_after": fitted.tokens_after,
            "notes": list(fitted.notes), "written": write and fitted.strategy_used != "none"}


_TOOLS: list[Tool] = [
    Tool("serve_status", "What is serving on this machine, and what a lease would do.",
         serve_status),
    Tool("serve_up", "Ask the broker for a lease on a server for a model, detached; it picks the port.", serve_up),
    Tool("serve_down", "Stop the server this machine started on a port.", serve_down),
    Tool("serve_escalate", "Grow the slots a running server holds, keeping every live "
                          "conversation.", serve_escalate),
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
    Tool("bench_show", "Every benchmark run kept, as records: what was served, how "
         "it was asked, what it scored and what it cost.", bench_show),
    Tool("fleet_peers", "Every peer on the LAN: serving, room, busy, commit.", fleet_peers),
    Tool("world_make", "Invent an organised group as a graph with people who could talk.",
         world_make),
    Tool("setup_look", "The machine facts serving depends on, with a fix for each.",
         setup_look),
    Tool("doctor", "The checkouts, the bench store and the managed llama.cpp, checked.",
         doctor),
    Tool("conversation_compact", "Fit a chat's messages to a token budget: cut long tool "
                                 "results, drop repeated calls, summarise the oldest.",
         conversation_compact),
    Tool("speech_providers", "Every speech engine here: recognition, synthesis, voice "
                             "activity.", speech_providers),
    Tool("speech_transcribe", "An audio file as text, with the times of each segment.",
         speech_transcribe),
    Tool("speech_say", "Speak text into a new WAV file in the state directory.", speech_say),
    Tool("decide", "Choose one of a named set of options and report a probability for each.",
         decide),
]
_TOOLS += [Tool(name, (getattr(workspace_tools, name).__doc__ or "").split("\n\n")[0].replace("\n", " "),
               getattr(workspace_tools, name),
               _hints(read_only=ro, destructive=bad, idempotent=same))
          for name, (ro, bad, same) in workspace_tools.HINTS.items()]
_HINTS: dict[str, dict[str, bool]] = {
    "serve_status": READS, "models_find": _hints(read_only=True, idempotent=True, open_world=True),
    "models_files": _hints(read_only=True, idempotent=True, open_world=True),
    "bench_status": READS, "bench_history": READS, "bench_show": READS,
    "fleet_peers": READS, "setup_look": READS, "doctor": READS, "speech_providers": READS,
    "speech_transcribe": READS,
    "serve_up": _hints(read_only=False, idempotent=True, open_world=True),
    "serve_down": _hints(read_only=False, destructive=True, idempotent=True),
    "serve_escalate": _hints(read_only=False),
    "models_fetch": _hints(read_only=False, idempotent=True, open_world=True),
    "bench_run": _hints(read_only=False),
    "world_make": _hints(read_only=False, idempotent=True),
    "speech_say": _hints(read_only=False),
    "conversation_compact": _hints(read_only=False, destructive=True),
}
TOOLS: list[Tool] = [dataclasses.replace(t, hints=_HINTS.get(t.name, t.hints)) for t in _TOOLS]
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
        checked(tool.fn, arguments or {})
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
        import mcp.server.mcpserver  # noqa: F401
    except ImportError:
        return False
    return True


def build_sdk_server() -> Any:
    """An ``MCPServer`` carrying the same tools, with their behaviour hints and structured
    results; needs ``pip install 'ml-stack[mcp]'`` (mcp 2.3 or later)."""
    from mcp.server.mcpserver import MCPServer
    from mcp_types import ToolAnnotations

    app = MCPServer("ml-stack")
    for tool in TOOLS:
        hints = tool.hints
        app.add_tool(tool.fn, name=tool.name, description=tool.description,
                     annotations=ToolAnnotations(
                         read_only_hint=hints["readOnly"], destructive_hint=hints["destructive"],
                         idempotent_hint=hints["idempotent"], open_world_hint=hints["openWorld"]))
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ml-stack-mcp",
        description="The ml-stack commands as MCP tools over stdio. Register it with "
                    "'claude mcp add ml-stack -- ml-stack-mcp'.")
    ap.add_argument("--workspace-only", action="store_true", help="expose authenticated workspace messaging tools only")
    ap.add_argument("--list", action="store_true", help="print the tools and exit")
    ap.add_argument("--builtin", action="store_true",
                    help="speak the protocol with the built-in loop even when the mcp SDK "
                         "is installed")
    args = ap.parse_args(argv)
    if args.workspace_only:
        global TOOLS, _BY_NAME
        names = {"workspace_inbox", "workspace_send", "workspace_thread", "workspace_claim",
                 "workspace_who_owns", "workspace_announce", "workspace_ack", "workspace_status",
                 "workspace_reputation"}
        TOOLS = [tool for tool in TOOLS if tool.name in names]
        _BY_NAME = {tool.name: tool for tool in TOOLS}
    if args.list:
        for tool in TOOLS:
            required = tool.schema().get("required", [])
            props = tool.schema()["properties"]
            shown = ", ".join(f"{k}{'' if k in required else '?'}" for k in props)
            say(f"{tool.name:<14} ({shown})\n    {tool.description}")
        say(f"\ntransport: {'mcp SDK' if sdk_available() and not args.builtin else 'built-in'}")
        return 0
    if sdk_available() and not args.builtin:
        build_sdk_server().run("stdio")
        return 0
    return serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
