"""Interventions: hooks that watch an agent loop and answer Proceed, Deny, Confirm or Guide.

An `Intervention` is a set of hooks. A loop calls `before_tool_call` and its siblings with the
call or the messages in hand and acts on the verdict. `first_deny_wins` and `require_all`
combine several; `guard_tool_call` is the whole before-and-run step for one tool call.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "Allow",
    "Call",
    "Confirm",
    "Context",
    "Deny",
    "Guarded",
    "Guide",
    "Intervention",
    "Proceed",
    "Verdict",
    "ask",
    "first_deny_wins",
    "guard_tool_call",
    "require_all",
    "severity",
]


@dataclass(frozen=True, slots=True)
class Call:
    """One tool call a model asked for."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass(slots=True)
class Context:
    """What a hook may read about the run: the task, the messages so far, the tools offered.

    ``trusted`` holds text the operator vouches for (the user's own request); everything a
    tool returned is untrusted until a hook says otherwise. ``notes`` is scratch space shared
    by the hooks of one run.
    """

    task: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    trusted: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Proceed:
    """Carry on."""


@dataclass(frozen=True, slots=True)
class Deny:
    """Do not do it; ``reason`` is what the model is told."""

    reason: str


@dataclass(frozen=True, slots=True)
class Confirm:
    """Ask a person first; ``details`` is what to show them."""

    question: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Guide:
    """Do not do it as asked; ``message`` steers the model to try again."""

    message: str


Verdict = Proceed | Deny | Confirm | Guide
Allow = Proceed

_RANK = {Proceed: 0, Guide: 1, Confirm: 2, Deny: 3}


def severity(verdict: Verdict) -> int:
    """0 for Proceed, then 1 Guide, 2 Confirm, 3 Deny."""
    return _RANK[type(verdict)]


class Intervention(Protocol):
    """Hooks a loop calls; a hook an object lacks counts as Proceed."""

    def before_invocation(self, context: Context) -> Verdict: ...
    def before_model_call(self, context: Context) -> Verdict: ...
    def after_model_call(self, context: Context, reply: Any) -> Verdict: ...
    def before_tool_call(self, call: Call, context: Context) -> Verdict: ...
    def after_tool_call(self, call: Call, result: str, context: Context) -> Verdict: ...
    def after_invocation(self, context: Context) -> Verdict: ...


class Base:
    """An `Intervention` whose every hook proceeds; subclass and override the ones you need."""

    def before_invocation(self, context: Context) -> Verdict:
        return Proceed()

    def before_model_call(self, context: Context) -> Verdict:
        return Proceed()

    def after_model_call(self, context: Context, reply: Any) -> Verdict:
        return Proceed()

    def before_tool_call(self, call: Call, context: Context) -> Verdict:
        return Proceed()

    def after_tool_call(self, call: Call, result: str, context: Context) -> Verdict:
        return Proceed()

    def after_invocation(self, context: Context) -> Verdict:
        return Proceed()


async def _settle(value: Verdict | Awaitable[Verdict]) -> Verdict:
    return await value if inspect.isawaitable(value) else value


def _hook(one: Intervention, name: str) -> Callable[..., Verdict | Awaitable[Verdict]]:
    return getattr(one, name, None) or (lambda *args: Proceed())


async def ask(hook: str, items: Sequence[Intervention], args: tuple[Any, ...], *,
              collect: bool = False) -> Verdict:
    """The verdict of ``hook`` across ``items``, awaiting any hook that is a coroutine."""
    found: list[Verdict] = []
    for one in items:
        verdict = await _settle(_hook(one, hook)(*args))
        if isinstance(verdict, Proceed):
            continue
        if isinstance(verdict, Deny) and not collect:
            return verdict
        found.append(verdict)
    return merge(found)


class _Combined(Base):
    """Several interventions behind one; ``collect`` runs all of them instead of stopping."""

    def __init__(self, items: Sequence[Intervention], *, collect: bool) -> None:
        self.items = tuple(items)
        self.collect = collect

    def _combine(self, hook: str, args: tuple[Any, ...]) -> Verdict:
        found: list[Verdict] = []
        for one in self.items:
            verdict = _hook(one, hook)(*args)
            if inspect.isawaitable(verdict):
                raise TypeError(f"{hook} is a coroutine; run it through guard_tool_call")
            if isinstance(verdict, Proceed):
                continue
            if isinstance(verdict, Deny) and not self.collect:
                return verdict
            found.append(verdict)
        return merge(found)

    def before_invocation(self, context: Context) -> Verdict:
        return self._combine("before_invocation", (context,))

    def before_model_call(self, context: Context) -> Verdict:
        return self._combine("before_model_call", (context,))

    def after_model_call(self, context: Context, reply: Any) -> Verdict:
        return self._combine("after_model_call", (context, reply))

    def before_tool_call(self, call: Call, context: Context) -> Verdict:
        return self._combine("before_tool_call", (call, context))

    def after_tool_call(self, call: Call, result: str, context: Context) -> Verdict:
        return self._combine("after_tool_call", (call, result, context))

    def after_invocation(self, context: Context) -> Verdict:
        return self._combine("after_invocation", (context,))


def merge(found: Sequence[Verdict]) -> Verdict:
    """The most severe verdict; several of one kind join their reasons, questions or messages."""
    if not found:
        return Proceed()
    worst = max(severity(v) for v in found)
    same = [v for v in found if severity(v) == worst]
    first = same[0]
    if isinstance(first, Deny):
        return Deny("; ".join(dict.fromkeys(v.reason for v in same if isinstance(v, Deny))))
    if isinstance(first, Confirm):
        asks = [v for v in same if isinstance(v, Confirm)]
        details: dict[str, Any] = {}
        for ask in asks:
            details.update(ask.details)
        return Confirm("; ".join(dict.fromkeys(a.question for a in asks)), details)
    return Guide(" ".join(dict.fromkeys(v.message for v in same if isinstance(v, Guide))))


def first_deny_wins(*items: Intervention) -> Intervention:
    """Run in order; the first Deny is the answer, else the most severe of the rest."""
    return _Combined(items, collect=False)


def require_all(*items: Intervention) -> Intervention:
    """Run every one; Proceed only when all do, else the most severe with every reason joined."""
    return _Combined(items, collect=True)


@dataclass(frozen=True, slots=True)
class Guarded:
    """What happened to one tool call: ``ran`` says whether it executed.

    ``text`` is what goes back to the model as the tool result, ``verdict`` the verdict that
    decided it, and ``confirmed`` records an answered Confirm.
    """

    ran: bool
    text: str
    verdict: Verdict
    confirmed: bool | None = None


HOOK_FAILURES = (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError)

Executor = Callable[[Call], str | Awaitable[str]]
Asker = Callable[[Confirm], bool | Awaitable[bool]]


async def guard_tool_call(call: Call, context: Context, execute: Executor,
                          interventions: Sequence[Intervention] = (), *,
                          confirm: Asker | None = None) -> Guarded:
    """Run ``call`` through ``before_tool_call`` and ``after_tool_call`` of each intervention.

    A Deny or Guide stops the call and its text goes back to the model. A Confirm calls
    ``confirm`` and runs the call only when it answers True; with no ``confirm`` it is a
    Deny. A hook that raises is a Deny: a guard that fails does not let the call through.
    """
    gate: Verdict = Proceed()
    try:
        gate = await ask("before_tool_call", interventions, (call, context))
    except HOOK_FAILURES as exc:
        gate = Deny(f"a guard failed ({type(exc).__name__}); the call was not made")
    asked: bool | None = None
    if isinstance(gate, Confirm):
        asked = bool(await _settle_bool(confirm(gate))) if confirm else False
        if not asked:
            return Guarded(False, f"Denied: a person did not confirm: {gate.question}", gate, asked)
    elif isinstance(gate, Deny):
        return Guarded(False, f"Denied: {gate.reason}", gate)
    elif isinstance(gate, Guide):
        return Guarded(False, gate.message, gate)
    out = await _settle_text(execute(call))
    after: Verdict = Proceed()
    try:
        after = await ask("after_tool_call", interventions, (call, out, context))
    except HOOK_FAILURES as exc:
        after = Deny(f"a guard failed ({type(exc).__name__}); the result was withheld")
    if isinstance(after, Deny):
        return Guarded(True, f"Withheld: {after.reason}", after, asked)
    if isinstance(after, Guide):
        return Guarded(True, f"{out}\n\n{after.message}", after, asked)
    return Guarded(True, out, gate, asked)


async def _settle_bool(value: bool | Awaitable[bool]) -> bool:
    return await value if inspect.isawaitable(value) else value


async def _settle_text(value: str | Awaitable[str]) -> str:
    return await value if inspect.isawaitable(value) else value
