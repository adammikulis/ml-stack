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
- `mirrors` / `drift`: does a fake's signature match the real one, and how not.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ml_stack.client import families
from ml_stack.client.chat import Client, Reply
from ml_stack.client.families import Family
from ml_stack.serve.backend import ServerFailed, ServerInfo, ServerSpec
from ml_stack.serve.manager import ServerManager, serve
from ml_stack.serve.preflight import Check, Preflight, Report

__all__ = [
    "FakeClient",
    "FakePreflight",
    "FakeReport",
    "FakeServe",
    "ScriptedModel",
    "drift",
    "fake_serve",
    "mirrors",
    "reply_from",
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
    otherwise the model answers ``answer`` in words without spending it -- which is the whole
    point of the last turn taking the searching tools away. ``seen`` is the messages of
    every turn; ``told()`` is what the tools answered.

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
)
"""Every fake here beside what it stands in for. The test walks this; a fake added to the
module and not to this table is a fake nothing checks."""
