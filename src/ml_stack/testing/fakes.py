"""Fakes with the real signatures, so what the real thing refuses, the fake refuses too.

A fake client written as ``def __init__(self, base_url, **kwargs)`` accepts every keyword,
so a test that hands it one the real `Client` does not take goes green. That is how a
``--also tight`` flag reached ``Client.__init__`` in production and took an 87G load down
with it: the test that covered the path had faked the client with ``**kwargs``. Every fake
here carries the real signature -- the same names, the same kinds, the same defaults, and
never a ``**kwargs`` the real one lacks -- and `mirrors` diffs them against the real ones
so they cannot drift when the real one changes. ``tests/test_testing_fakes.py`` runs
that diff over every fake in this module.

What is here:

- `FakeClient`: `Client` that reaches no server. Scripted replies, every call recorded.
- `ScriptedModel`: the graph tests' model -- a script of tool calls, then words.
- `FakeServe` / `fake_serve`: `serve()` that starts nothing and yields a real `ServerInfo`.
- `FakeReport` / `FakePreflight`: a preflight that read nothing and passed, or refused.
- `FakeBackend`: a `ServerBackend` that binds no socket and records every spec.
- `Served` / `FakeLlamaServer` / `fake_llama_server`: a llama-server on a real socket.
- `fake_llama_binary` / `fake_binary`: the same as an executable, and a bare `--help` stub.
- `mirrors` / `drift`: does a fake's signature match the real one, and how not.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

from ml_stack.client import families
from ml_stack.client.chat import Client, Reply
from ml_stack.client.counters import Speculative
from ml_stack.client.families import Family
from ml_stack.http import json_body
from ml_stack.serve.backend import (
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
)
from ml_stack.serve.manager import ServerManager, serve
from ml_stack.serve.preflight import Check, Preflight, Report

__all__ = [
    "DRAFTING",
    "LLAMA_SERVER_FLAGS",
    "LLAMA_SERVER_HELP",
    "FakeBackend",
    "FakeClient",
    "FakeLlamaServer",
    "FakePreflight",
    "FakeReport",
    "FakeServe",
    "ScriptedModel",
    "Served",
    "drift",
    "fake_binary",
    "fake_llama_binary",
    "fake_llama_server",
    "fake_serve",
    "metrics_text",
    "mirrors",
    "reply_from",
    "serve_from_argv",
]


# ---------------------------------------------------------------- what a script entry is

def reply_from(entry: Any, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]] | None) -> Reply:
    """One script entry as the `Reply` a server would have sent.

    A `Reply` is itself; a ``str`` is the answer; a ``dict`` is the answer as JSON (what an
    extraction returns); a ``(name, args)`` tuple is one tool call; a callable is asked with
    ``(messages, tools)`` and whatever it returns is read the same way.
    """
    if callable(entry) and not isinstance(entry, type):
        return reply_from(entry(messages, tools), messages, tools)
    if isinstance(entry, Reply):
        return entry
    if entry is None:
        return Reply(content="")
    if isinstance(entry, str):
        return Reply(content=entry)
    if isinstance(entry, dict):
        return Reply(content=json.dumps(entry))
    if _is_call(entry):
        name, args = entry
        return Reply(content="", tool_calls=[{"id": "c1", "function": {
            "name": name, "arguments": json.dumps(args)}}])
    raise TypeError(f"a script entry is a Reply, str, dict, (name, args) or callable, "
                    f"not {type(entry).__name__}")


def _is_call(entry: Any) -> bool:
    """A ``(name, args)`` pair: one tool call."""
    return (isinstance(entry, tuple) and len(entry) == 2
            and isinstance(entry[0], str) and isinstance(entry[1], dict))


def _entries(replies: Any) -> list[Any]:
    """``replies`` as the list it spends: one entry stays one entry, a callable is none."""
    if callable(replies) and not isinstance(replies, type):
        return []
    if isinstance(replies, (str, dict, Reply)) or replies is None or _is_call(replies):
        return [replies]
    return list(replies)


def _offered(tools: list[dict[str, Any]] | None) -> set[str]:
    return {str((t.get("function") or {}).get("name")) for t in (tools or [])}


def _tool_turns(seen: list[list[dict[str, Any]]]) -> str:
    return " ".join(str(m.get("content") or "") for turn in seen for m in turn
                    if m.get("role") == "tool")


# ---------------------------------------------------------------- the client

class FakeClient:
    """`Client` that reaches no server.

    Built exactly as `Client` is built -- the same keywords, no others -- so a keyword the
    real one would refuse is refused here. ``chat`` and ``extract`` answer from ``replies``:
    a list is spent in order and its last entry repeated once it is gone (a script ending in
    words keeps answering them), a callable is asked with ``(messages, tools)`` each time,
    and an empty script answers nothing. Each entry is read by `reply_from`.

    Code that builds its own clients -- ``served()`` does -- is given a *class*, not an
    instance: `FakeClient.scripted(replies)` makes a subclass with those replies and a fresh
    ``built`` list of every instance it constructed, and that is what to monkeypatch in.

    What was seen: ``seen`` is the messages of every ``chat``, ``calls`` is every call to
    any method with its arguments, ``told()`` is what the tools answered as the model saw
    it. ``sampling`` is computed as the real one computes it; ``card`` is ``card_says``.
    """

    replies: Any = ()
    card_says: dict[str, Any] = {}
    built: list[FakeClient] = []

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        slot: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        n_predict: int = 16384,
        spec_draft_max: int | None = None,
        spec_p_min: float | None = None,
        timeout: float = 180.0,
        tries: int = 1,
        api_key: str | None = None,
        family: Family | str | None = None,
        api: str | None = None,
        model: str | None = None,
        context: int | None = None,
        keep_alive: str | int | None = None,
    ) -> None:
        from ml_stack.client.chat import parse_url

        self.base_url, self.api, found = parse_url(base_url, api)
        self.model = model or found
        self.context = context
        self.keep_alive = keep_alive
        self.slot = slot
        self.asked_temperature = temperature
        self.asked_top_p = top_p
        self.asked_top_k = top_k
        self.asked_min_p = min_p
        self.n_predict = n_predict
        self.asked_spec_draft_max = spec_draft_max
        self.asked_spec_p_min = spec_p_min
        self.timeout = timeout
        self.tries = tries
        self.api_key = api_key
        self.pinned_family = families.resolve(family)
        self.seen: list[list[dict[str, Any]]] = []
        self.calls: list[dict[str, Any]] = []
        self.pending: list[Any] = _entries(self.replies)
        self._last: Any = None
        type(self).built.append(self)

    @classmethod
    def scripted(cls, replies: Any, *, card: dict[str, Any] | None = None) -> type[FakeClient]:
        """A subclass answering ``replies``, with its own empty ``built``."""
        # A function stored on a class is a method when read off an instance; kept static
        # so it is asked with ``(messages, tools)`` and nothing else.
        held = staticmethod(replies) if callable(replies) and not isinstance(replies, type) \
            else replies
        return type(cls.__name__, (cls,), {
            "replies": held,
            "card_says": dict(card) if card is not None else dict(cls.card_says),
            "built": [],
        })

    # --- what the real one exposes

    @property
    def family(self) -> Family:
        return self.pinned_family or families.GENERIC

    @property
    def sampling(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in (("temperature", self.asked_temperature),
                            ("top_p", self.asked_top_p), ("top_k", self.asked_top_k),
                            ("min_p", self.asked_min_p)):
            if value is not None:
                out[name] = value
        out.setdefault("temperature", 0.0)
        return out

    @property
    def card(self) -> dict[str, Any]:
        return dict(self.card_says)

    @property
    def temperature(self) -> float:
        return float(self.sampling.get("temperature", 0.0))

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        timeout: float | None = None,
        on_delta: Callable[[str, str], None] | None = None,
        **extra: Any,
    ) -> Reply:
        self.seen.append(list(messages))
        self.calls.append({"method": "chat", "messages": list(messages), "tools": tools,
                           "tool_choice": tool_choice, "timeout": timeout,
                           "on_delta": on_delta, **extra})
        reply = self._next(messages, tools)
        if on_delta is not None:
            if reply.thinking:
                on_delta("thinking", reply.thinking)
            if reply.content:
                on_delta("content", reply.content)
        return reply

    def extract(self, text: str, schema: dict[str, Any], *, instructions: str = "",
                n_predict: int | None = None,
                check: Callable[[dict[str, Any]], list[str]] | None = None,
                tries: int = 2, prompt: str | None = None,
                messages: list[dict[str, Any]] | None = None,
                think: bool = False,
                schema_name: str = "extraction") -> dict[str, Any]:
        if tries < 1:
            raise ValueError(f"tries must be at least 1, got {tries}")
        self.calls.append({"method": "extract", "text": text, "schema": schema,
                           "instructions": instructions, "n_predict": n_predict,
                           "check": check, "tries": tries, "prompt": prompt,
                           "messages": messages, "think": think,
                           "schema_name": schema_name})
        convo = list(messages) if messages is not None else [
            {"role": "system", "content": instructions}, {"role": "user", "content": text}]
        answer = json.loads(self._next(convo, None).content or "null")
        objections = list(check(answer)) if check else []
        if objections and isinstance(answer, dict):
            return dict(answer, _objections=objections)
        return answer

    def told(self) -> str:
        """What the tools answered, as the model saw it."""
        return _tool_turns(self.seen)

    def _next(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Reply:
        if callable(self.replies):
            return reply_from(self.replies, messages, tools)
        if self.pending:
            self._last = self.pending.pop(0)
        return reply_from(self._last, messages, tools)


# ---------------------------------------------------------------- the graph tests' model

class ScriptedModel:
    """Answers with the tool calls it was told to, then with words.

    The tool loop hands a model only the tools it may call this turn, and a model can only
    call what it was offered: the next scripted call is issued when its tool is on offer and
    otherwise the model answers ``answer`` in words without spending it. ``seen`` is the
    messages of every turn; ``told()`` is what the tools answered.

    Not a `Client`: nothing here builds one, the tool loop takes whatever has a ``chat`` --
    but that ``chat`` has `Client.chat`'s signature.
    """

    ANSWER = "Ada and Bea both work on compilers."

    def __init__(self, script: list[tuple[str, dict[str, Any]]] | None = None, *,
                 answer: str = ANSWER) -> None:
        self.script = list(script or [])
        self.answer = answer
        self.seen: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        timeout: float | None = None,
        on_delta: Callable[[str, str], None] | None = None,
        **extra: Any,
    ) -> Reply:
        self.seen.append(list(messages))
        if self.script and self.script[0][0] in _offered(tools):
            return reply_from(self.script.pop(0), messages, tools)
        return Reply(content=self.answer)

    def told(self) -> str:
        """What the tools answered, as the model saw it."""
        return _tool_turns(self.seen)


# ---------------------------------------------------------------- serving

class FakeServe:
    """`serve()` that starts nothing and yields a real `ServerInfo`.

    Called exactly as `serve` is called; the spec is built the way the real one builds it,
    so a keyword `ServerSpec` lacks is refused. ``leased`` is every spec, ``timeouts`` every
    timeout asked, ``released`` every info whose block ended. A model whose name holds one
    of ``refuse`` raises ``raising`` instead of yielding -- the backend refusing a load.
    """

    def __init__(self, *, base_url: str | None = None, pid: int | None = None,
                 backend: str = "fake", load_s: float | None = None,
                 warmup_s: float | None = None, refuse: tuple[str, ...] = (),
                 raising: type[Exception] = ServerFailed) -> None:
        self.base_url = base_url
        self.pid = pid
        self.backend = backend
        self.load_s = load_s
        self.warmup_s = warmup_s
        self.refuse = tuple(refuse)
        self.raising = raising
        self.leased: list[ServerSpec] = []
        self.timeouts: list[float | None] = []
        self.released: list[ServerInfo] = []

    @contextmanager
    def __call__(
        self,
        model: str | Path,
        *,
        port: int | None = None,
        context: int = 4096,
        timeout: float | None = None,
        manager: ServerManager | None = None,
        **spec_kwargs: object,
    ) -> Iterator[ServerInfo]:
        spec = ServerSpec(model=model, port=port if port is not None else 1,
                          context=context, **spec_kwargs)  # type: ignore[arg-type]
        self.leased.append(spec)
        self.timeouts.append(timeout)
        if any(word in str(model) for word in self.refuse):
            raise self.raising(f"FAIL  shards: not on this machine yet: {model}")
        info = ServerInfo(base_url=self.base_url or f"http://127.0.0.1:{spec.port}",
                          port=spec.port, pid=self.pid, backend=self.backend,
                          load_s=self.load_s, warmup_s=self.warmup_s)
        try:
            yield info
        finally:
            self.released.append(info)


@contextmanager
def fake_serve(
    model: str | Path,
    *,
    port: int | None = None,
    context: int = 4096,
    timeout: float | None = None,
    manager: ServerManager | None = None,
    **spec_kwargs: object,
) -> Iterator[ServerInfo]:
    """`serve()` that starts nothing: a `ServerInfo` on the port asked for, or port 1.

    Drop-in for ``monkeypatch.setattr(ml_stack.serve, "serve", fake_serve)`` when nothing
    about the lease needs asserting; `FakeServe` when it does.
    """
    with FakeServe()(model, port=port, context=context, timeout=timeout, manager=manager,
                     **spec_kwargs) as info:
        yield info


# ---------------------------------------------------------------- preflight

class FakeReport(Report):
    """A `Report` from a preflight that read nothing.

    Every check passes, or the shards check fails the way a missing file would when
    ``ok`` is False. ``said()``, ``ok``, ``weights_bytes`` and ``kv_estimate_bytes`` are the
    real ones -- this is a `Report`, not a stand-in for one.
    """

    def __init__(self, *, ok: bool = True, weights_bytes: int = 0,
                 kv_estimate_bytes: int = 0, limit_bytes: int = 0,
                 model: str | Path = "") -> None:
        detail = "complete" if ok else f"missing or empty: {model}"
        super().__init__(checks=[
            Check("shards", ok, detail),
            Check("architecture", True, "gemma4"),
            Check("fit", True, f"{(weights_bytes + kv_estimate_bytes) / 2**30:.1f}G estimated "
                               f"fits under {limit_bytes / 2**30:.1f}G"),
            Check("flags", True, "every flag this spec would emit is one this build accepts"),
        ], weights_bytes=weights_bytes, kv_estimate_bytes=kv_estimate_bytes)


class FakePreflight:
    """`Preflight` that reads nothing: every check passes, unless the model's name holds
    one of ``refuse``. ``seen`` is every spec it was asked about."""

    def __init__(self, *, refuse: tuple[str, ...] = (), weights_bytes: int = 5 * 2**30,
                 kv_estimate_bytes: int = 3 * 2**30) -> None:
        self.refuse = tuple(refuse)
        self.weights_bytes = weights_bytes
        self.kv_estimate_bytes = kv_estimate_bytes
        self.seen: list[ServerSpec] = []

    def __call__(self, spec, *, binary: str | Path, limit_bytes: int = 0) -> Report:
        self.seen.append(spec)
        bad = any(word in str(spec.model) for word in self.refuse)
        return FakeReport(ok=not bad, weights_bytes=self.weights_bytes,
                          kv_estimate_bytes=self.kv_estimate_bytes, limit_bytes=limit_bytes,
                          model=spec.model)


# ---------------------------------------------------------------- a llama-server

LLAMA_SERVER_FLAGS = (
    "-m, --model FNAME", "-c, --ctx-size N", "-ngl, --gpu-layers, --n-gpu-layers N",
    "-fa, --flash-attn [on|off|auto]", "-np, --parallel N", "--host HOST", "--port PORT",
    "--alias NAME", "--jinja", "--metrics", "--embeddings", "--pooling TYPE", "--mlock",
    "--no-mmap", "--no-warmup", "--chat-template-file FNAME", "--cache-reuse N",
    "--cache-ram N", "--cache-idle-slots, --no-cache-idle-slots", "-ctk, --cache-type-k TYPE",
    "-ctv, --cache-type-v TYPE", "-kvu, --kv-unified, --no-kv-unified",
    "--kv-unified-per-slot N", "-sps, --slot-prompt-similarity SIMILARITY",
    "--slot-save-path PATH", "-ot, --override-tensor PATTERN", "--cpu-moe",
    "--n-cpu-moe N", "--reasoning-budget N", "--rope-scale N", "--rope-scaling TYPE",
    "--yarn-orig-ctx N", "--yarn-ext-factor N", "--yarn-attn-factor N",
    "--yarn-beta-fast N", "--yarn-beta-slow N", "--mmproj FILE", "--mmproj-url URL",
    "-hf, --hf-repo REPO", "--hf-file FILE", "-hfd, --hf-repo-draft REPO",
    "-md, --model-draft FNAME", "--spec-type TYPE", "--spec-draft-n-max N",
    "--spec-draft-n-min N", "--spec-draft-ngl N",
    "-ngld, --gpu-layers-draft, --n-gpu-layers-draft N", "--spec-draft-type-k TYPE",
    "--spec-draft-type-v TYPE", "--spec-ngram-mod-n-max N", "--spec-ngram-mod-n-min N",
    "--lookup-cache-static FNAME", "--lookup-cache-dynamic FNAME",
)
"""Every flag `ml_stack.serve.backend.ServerSpec` can put on a command line."""

LLAMA_SERVER_HELP = "".join(f"{flag:<52}what it sets\n" for flag in LLAMA_SERVER_FLAGS)
"""``--help`` in llama-server's shape, listing `LLAMA_SERVER_FLAGS`."""

CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

DRAFTING = Speculative(drafts=412, drafted=1648, accepted=1533,
                       per_position=(402, 380, 341, 300, 110))
"""Counters a server with a draft head reports: four tokens offered a pass, most kept."""

_SPEC_TOTALS = (("llamacpp:spec_decode_num_drafts_total", "drafts"),
                ("llamacpp:spec_decode_num_draft_tokens_total", "drafted"),
                ("llamacpp:spec_decode_num_accepted_tokens_total", "accepted"))
_SPEC_PER_POS = "llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
_PROCESSING = "llamacpp:requests_processing"


def metrics_text(counted: Speculative | None) -> str:
    """``/metrics`` in the Prometheus shape llama.cpp writes; the speculative counters
    appear only where ``counted`` is given, which is where a draft head was loaded."""
    out = [f"# HELP {_PROCESSING} Number of requests processing",
           f"# TYPE {_PROCESSING} gauge",
           f"{_PROCESSING} {0 if counted is None else counted.processing}"]
    if counted is not None:
        for name, part in _SPEC_TOTALS:
            out += [f"# HELP {name} Speculative decoding", f"# TYPE {name} counter",
                    f"{name} {getattr(counted, part)}"]
        out += [f"# HELP {_SPEC_PER_POS} Accepted tokens per draft position",
                f"# TYPE {_SPEC_PER_POS} counter"]
        out += [f'{_SPEC_PER_POS}{{position="{at}"}} {n}'
                for at, n in enumerate(counted.per_position)]
    return "\n".join(out) + "\n"


@dataclass(frozen=True, slots=True)
class Served:
    """What a fake llama-server is holding, and what it says when asked.

    ``counted`` standing in for the draft head llama.cpp reports nothing else about:
    None is a server with no head and no speculative counters on ``/metrics``.
    ``metrics`` False is a server started without ``--metrics``, which answers 501 there.
    ``answer`` is the reply a completion comes back with, or a callable given the request
    body; ``pieces`` is how a streamed one is broken up and ``gap`` the seconds between.
    """

    model: str = "quince-2b.gguf"
    context: int = 32768
    slots: int = 1
    metrics: bool = True
    draft: str = ""
    spec_type: str = ""
    draft_max: int | None = None
    counted: Speculative | None = None
    answer: Any = "hello"
    pieces: tuple[str, ...] = ()
    gap: float = 0.0
    build_info: str = "b7000-fakebuild"

    @property
    def name(self) -> str:
        """The id ``/v1/models`` lists: the weights file's own name."""
        return PurePosixPath(self.model).name or self.model

    def props(self) -> dict[str, Any]:
        """``/props``, with the fields `serving_params` and `drafting_of` read."""
        return {"model_path": self.model, "total_slots": self.slots,
                "endpoint_metrics": self.metrics, "build_info": self.build_info,
                "chat_template": CHAT_TEMPLATE,
                "default_generation_settings": {"n_ctx": self.context,
                                                "model": self.model, "seed": 0xFFFFFFFF}}


class FakeLlamaServer:
    """A llama-server on a real loopback socket: the routes this repo's clients ask for.

    ``requests`` is every ``(method, path, body)`` it took; ``saved`` and ``restored``
    every slot the manager asked it to write out or read back; ``slots`` the rows
    ``/slots`` answers with, which a test may edit. ``refuse`` maps a path -- with its
    query, or without -- to the status it answers there instead. ``disconnected`` is set
    when a reader hangs up mid-stream.
    """

    def __init__(self, served: Served | None = None, *, port: int = 0) -> None:
        self.served = served if served is not None else Served()
        self.requests: list[tuple[str, str, bytes]] = []
        self.saved: list[tuple[int, str]] = []
        self.restored: list[tuple[int, str]] = []
        self.slots = [{"id": n, "n_ctx": self.served.context // max(self.served.slots, 1),
                       "is_processing": False} for n in range(self.served.slots)]
        self.refuse: dict[str, int] = {}
        self.disconnected = threading.Event()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), _routes(self))
        self.port = int(self._httpd.server_address[1])
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop serving and give the port back."""
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    def record(self, *, pid: int = 0) -> dict[str, Any]:
        """This server as a row of `ml_stack.serve.process.every_server`."""
        return {"pid": pid or os.getpid(), "port": self.port, "defunct": False,
                "model": self.served.model, "binary": "llama-server",
                "draft": self.served.draft, "spec_type": self.served.spec_type,
                "draft_max": self.served.draft_max, "rss": 0}

    def sent_to(self, path: str) -> list[dict[str, Any]]:
        """The bodies posted to ``path``."""
        return [json_body(raw) for method, seen, raw in self.requests
                if method == "POST" and seen.partition("?")[0] == path]

    def refused(self, path: str) -> tuple[int, str, bytes] | None:
        """What ``refuse`` says to answer at ``path``, or None where it says nothing."""
        bare, _, _ = path.partition("?")
        status = self.refuse.get(path) or self.refuse.get(bare)
        return _json({"error": f"refusing {path}"}, status=status) if status else None

    def answer_to(self, body: dict[str, Any]) -> str:
        """The text a completion comes back with, for the request ``body``."""
        said = self.served.answer
        return str(said(body) if callable(said) else said)

    def get(self, path: str) -> tuple[int, str, bytes]:
        """``(status, content type, body)`` for a GET of ``path``."""
        if (said := self.refused(path)) is not None:
            return said
        served, bare = self.served, path.partition("?")[0]
        if bare.startswith("/health"):
            return _json({"status": "ok"})
        if bare.startswith("/props"):
            return _json(served.props())
        if bare in ("/v1/models", "/models"):
            return _json({"object": "list", "data": [{"id": served.name}]})
        if bare.startswith("/metrics"):
            if not served.metrics:
                return 501, "text/plain", b"metrics endpoint is disabled\n"
            return 200, "text/plain", metrics_text(served.counted).encode()
        if bare == "/slots":
            return _json(self.slots)
        return _json({"error": f"no such route: {bare}"}, status=404)

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, str, bytes]:
        """``(status, content type, body)`` for a POST of ``body`` to ``path``."""
        if (said := self.refused(path)) is not None:
            return said
        bare, _, query = path.partition("?")
        parts = [one for one in bare.split("/") if one]
        if bare == "/v1/chat/completions":
            return _json({"model": self.served.name, "choices": [
                {"index": 0, "message": {"role": "assistant",
                                         "content": self.answer_to(body)},
                 "finish_reason": "stop"}]})
        if bare in ("/completion", "/completions"):
            return _json({"content": self.answer_to(body), "stopped_limit": False,
                          "truncated": False})
        if bare == "/tokenize":
            return _json({"tokens": list(range(len(str(body.get("content") or "").split()))) })
        if bare == "/detokenize":
            return _json({"content": " ".join(str(n) for n in body.get("tokens") or [])})
        if bare in ("/v1/embeddings", "/embedding", "/embeddings"):
            return _json({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})
        if len(parts) == 2 and parts[0] == "slots" and parts[1].isdigit():
            return self._slot(int(parts[1]), query, body)
        return _json({"error": f"no such route: {bare}"}, status=404)

    def _slot(self, which: int, query: str, body: dict[str, Any]) -> tuple[int, str, bytes]:
        action = dict(one.split("=", 1) for one in query.split("&") if "=" in one).get("action")
        kept = str(body.get("filename") or "")
        if action == "save":
            self.saved.append((which, kept))
        elif action == "restore":
            self.restored.append((which, kept))
        else:
            return _json({"error": f"unknown action: {action}"}, status=400)
        return _json({"id_slot": which, "filename": kept})

    def frames(self, body: dict[str, Any]) -> list[bytes]:
        """The ``data:`` frames a streamed chat completion sends, the final one included."""
        said = self.answer_to(body)
        pieces = self.served.pieces or (said,)
        out = [b"data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {"content": piece}}]}).encode() + b"\n\n"
            for piece in pieces]
        return [*out, b"data: [DONE]\n\n"]


def _json(payload: Any, *, status: int = 200) -> tuple[int, str, bytes]:
    return status, "application/json", json.dumps(payload).encode()


def _routes(fake: FakeLlamaServer) -> type[BaseHTTPRequestHandler]:
    """A handler class answering ``fake``'s routes."""

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            fake.requests.append(("GET", self.path, b""))
            self._answer(*fake.get(self.path))

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            fake.requests.append(("POST", self.path, raw))
            body = json_body(raw)
            if body.get("stream") and self.path.split("?")[0].endswith("/chat/completions"):
                self._stream(fake.frames(body))
                return
            self._answer(*fake.post(self.path, body))

        def _answer(self, status: int, kind: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _stream(self, frames: Iterable[bytes]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for frame in frames:
                    self.wfile.write(frame)
                    self.wfile.flush()
                    if fake.served.gap:
                        time.sleep(fake.served.gap)
            except (BrokenPipeError, ConnectionResetError):
                fake.disconnected.set()

        def log_message(self, *args: object) -> None:
            pass

    return _H


@contextmanager
def fake_llama_server(served: Served | None = None, *, port: int = 0
                      ) -> Iterator[FakeLlamaServer]:
    """A llama-server on a real socket for the block, closed at the end."""
    fake = FakeLlamaServer(served, port=port)
    try:
        yield fake
    finally:
        fake.close()


# ---------------------------------------------------------------- as a binary

def fake_binary(where: Path, *, help_text: str = "-m, --model FNAME  model path\n",
                name: str = "llama-server") -> Path:
    """An executable in ``where`` answering ``--help`` with ``help_text``, exit 0 otherwise."""
    path = where / name
    path.write_text("#!/bin/sh\nif [ \"$1\" = --help ]; then cat <<'HELP'\n"
                    + help_text + "HELP\nexit 0\nfi\nexit 0\n")
    path.chmod(0o755)
    return path


_FLAGS = {"model": ("--model", "-m"), "context": ("--ctx-size", "-c"),
          "slots": ("--parallel", "-np"), "draft": ("--model-draft", "-md"),
          "spec_type": ("--spec-type",), "port": ("--port",)}


def _after(argv: list[str], names: tuple[str, ...]) -> str:
    for at, one in enumerate(argv[:-1]):
        if one in names:
            return argv[at + 1]
    return ""


def serve_from_argv(argv: list[str], *, where: Path) -> int:
    """Serve `fake_llama_server` on the ``--port`` in ``argv`` until the process is killed.

    ``--help`` prints `LLAMA_SERVER_HELP` and returns instead. The argv it was launched
    with is written to ``argv.json`` in ``where``, for a test asserting on a command line.
    """
    if "--help" in argv:
        sys.stdout.write(LLAMA_SERVER_HELP)
        return 0
    (where / "argv.json").write_text(json.dumps(argv))
    context = _after(argv, _FLAGS["context"])
    slots = _after(argv, _FLAGS["slots"])
    draft = _after(argv, _FLAGS["draft"])
    served = Served(model=_after(argv, _FLAGS["model"]) or "model.gguf",
                    context=int(context) if context.isdigit() else 4096,
                    slots=int(slots) if slots.isdigit() else 1,
                    draft=draft, spec_type=_after(argv, _FLAGS["spec_type"]),
                    counted=DRAFTING if draft else None)
    port = _after(argv, _FLAGS["port"])
    FakeLlamaServer(served, port=int(port) if port.isdigit() else 8080)
    threading.Event().wait()
    return 0


def _import_root() -> Path:
    """The directory ``ml_stack`` is imported from, for a subprocess to put on its path."""
    return Path(__file__).resolve().parents[2]


_LAUNCHER = """#!{python}
import os, sys
sys.path.insert(0, {root!r})
# argv[0] is what a process scan matches a llama-server on, so this wears the name.
if not os.environ.get("MLSTACK_FAKE_LLAMA"):
    os.environ["MLSTACK_FAKE_LLAMA"] = "1"
    os.execv(sys.executable, ["llama-server", __file__, *sys.argv[1:]])
from pathlib import Path

from ml_stack.testing.fakes import serve_from_argv

raise SystemExit(serve_from_argv(sys.argv[1:], where=Path(__file__).parent))
"""


def fake_llama_binary(where: Path, *, name: str = "llama-server") -> Path:
    """An executable in ``where`` that answers ``--help`` and then really serves.

    Started for real by ``Popen``, it binds the ``--port`` it was given, presents to a
    process scan as ``llama-server``, and answers everything `FakeLlamaServer` answers
    for the model, context, slots and draft head its command line named.
    """
    path = where / name
    path.write_text(_LAUNCHER.format(python=sys.executable, root=str(_import_root())))
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------- a backend

class FakeBackend(ServerBackend):
    """`ServerBackend` that binds nothing: every spec it was asked to start is in
    ``started``, and the info it hands back names the port the spec asked for."""

    name = "fake"

    def __init__(self, *, pid: int = 90000, load_s: float | None = 1.5,
                 warmup_s: float | None = 0.5) -> None:
        self.started: list[ServerSpec] = []
        self.pid = pid
        self.load_s = load_s
        self.warmup_s = warmup_s

    def command(self, spec: ServerSpec) -> list[str]:
        return ["fake", "--port", str(spec.port), "--model", str(spec.model)]

    def start(self, spec: ServerSpec, *, lease: Any, timeout: float = 300.0,
              check_flags: bool = True, preflight: bool = True,
              warmup_request: bool = True) -> ServerInfo:
        self.started.append(spec)
        return ServerInfo(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                          pid=self.pid + len(self.started), backend=self.name,
                          load_s=self.load_s, warmup_s=self.warmup_s)


# ---------------------------------------------------------------- the diff

_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _params(obj: Any) -> dict[str, inspect.Parameter]:
    """``obj``'s parameters, less a leading ``self`` -- so an unbound method and a plain
    function that stands in for it compare on what a caller passes."""
    params = list(inspect.signature(obj).parameters.values())
    if params and params[0].name == "self":
        params = params[1:]
    return {p.name: p for p in params}


def drift(fake: Any, real: Any) -> list[str]:
    """Every way ``fake``'s signature fails to mirror ``real``'s; empty when it does.

    A fake may leave out an *optional* parameter of the real one. It may not take a name
    the real one lacks, give a shared name a different kind or default, leave out a required
    one, or take ``*args``/``**kwargs`` the real one does not -- that last is the one that
    lets a wrong keyword through, and is the reason this module exists.
    """
    mine = _params(fake)
    theirs = _params(real)
    out: list[str] = []
    for name, p in mine.items():
        if p.kind in _VARIADIC:
            if not any(q.kind is p.kind for q in theirs.values()):
                out.append(f"takes {p} where the real one takes nothing of the kind")
            continue
        if name not in theirs:
            out.append(f"takes {name!r}, which the real one does not")
        elif theirs[name].kind is not p.kind:
            out.append(f"{name!r} is {p.kind.description}; the real one's is "
                       f"{theirs[name].kind.description}")
        elif theirs[name].default != p.default:
            out.append(f"{name!r} defaults to {p.default!r}; the real one's to "
                       f"{theirs[name].default!r}")
    for name, q in theirs.items():
        if name not in mine and q.kind not in _VARIADIC and q.default is inspect.Parameter.empty:
            out.append(f"leaves out {name!r}, which the real one requires")
    return out


def mirrors(fake: Any, real: Any) -> bool:
    """True when ``fake`` takes what ``real`` takes -- see `drift` for how it may not."""
    return not drift(fake, real)


MIRRORED: tuple[tuple[str, Any, Any], ...] = (
    ("FakeClient.__init__", FakeClient.__init__, Client.__init__),
    ("FakeClient.chat", FakeClient.chat, Client.chat),
    ("FakeClient.extract", FakeClient.extract, Client.extract),
    ("ScriptedModel.chat", ScriptedModel.chat, Client.chat),
    ("FakeServe.__call__", FakeServe.__call__, serve),
    ("fake_serve", fake_serve, serve),
    ("FakePreflight.__call__", FakePreflight.__call__, Preflight),
    ("FakeBackend.command", FakeBackend.command, ServerBackend.command),
    ("FakeBackend.start", FakeBackend.start, LlamaServerBackend.start),
)
"""Every fake here beside what it stands in for. The test walks this; a fake added to the
module and not to this table is a fake nothing checks."""
