"""Interventions: hooks that watch an agent loop and answer Proceed, Deny, Confirm, Guide or Rewrite.

An `Intervention` is any object with some of ``before_invocation``, ``before_model_call``,
``after_model_call``, ``before_tool_call``, ``after_tool_call`` and ``after_invocation``. The
guard's rails, a decision model's tool-call check and an application's own policy are all this
one thing. A loop owns one `Run`, which asks the interventions, resolves a `Confirm` with the
person, collects `Guide` messages and tracks whether text from outside the person has been read.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "Allow",
    "Base",
    "Call",
    "Confirm",
    "Context",
    "Deny",
    "Gate",
    "Guarded",
    "Guide",
    "Intervention",
    "Proceed",
    "Rewrite",
    "Run",
    "Screened",
    "Verdict",
    "first_deny_wins",
    "guard_tool_call",
    "merge",
    "require_all",
    "severity",
]

logger = logging.getLogger("ml_stack.guard")
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class Call:
    """One tool call a model asked for. ``arguments`` is None when what the model sent was not
    a JSON object; ``raw`` is what it sent."""

    name: str
    arguments: dict[str, Any] | None = field(default_factory=dict)
    id: str = ""
    raw: str = ""


@dataclass(slots=True)
class Context:
    """What a hook may read about the run.

    ``task`` is the request the person made; ``messages`` the conversation so far; ``tools`` the
    tools offered; ``trusted`` text the operator vouches for. ``tainted`` is set once text from
    outside the person has entered the conversation. ``notes`` is scratch space shared by the
    hooks of one run.
    """

    task: str = ""
    messages: Sequence[Mapping[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    trusted: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    tool_calls: int = 0
    tainted: bool = False


@dataclass(frozen=True, slots=True)
class Proceed:
    """Carry on."""


@dataclass(frozen=True, slots=True)
class Deny:
    """Do not; ``reason`` is what the model is told. ``by`` names the intervention."""

    reason: str
    by: str = ""


@dataclass(frozen=True, slots=True)
class Confirm:
    """Ask the person first: ``question``, and the ``details`` to show with it."""

    question: str
    details: Mapping[str, Any] = field(default_factory=dict)
    by: str = ""


@dataclass(frozen=True, slots=True)
class Guide:
    """Carry on, and tell the model ``message`` on its next turn."""

    message: str
    by: str = ""


@dataclass(frozen=True, slots=True)
class Rewrite:
    """Carry on with ``text`` in place of the text asked about; ``tainted`` marks text that
    carries instructions from outside the person."""

    text: str
    reason: str = ""
    tainted: bool = False
    by: str = ""


Verdict = Proceed | Deny | Confirm | Guide | Rewrite
Allow = Proceed

_RANK = {Proceed: 0, Rewrite: 1, Guide: 2, Confirm: 3, Deny: 4}


def severity(verdict: Verdict) -> int:
    """0 for Proceed, then 1 Rewrite, 2 Guide, 3 Confirm, 4 Deny."""
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

    name = ""

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


def _name(one: Any) -> str:
    return str(getattr(one, "name", "") or type(one).__name__)


def _hook(one: Any, name: str) -> Callable[..., Verdict | Awaitable[Verdict]]:
    return getattr(one, name, None) or (lambda *args: Proceed())


def merge(found: Sequence[Verdict]) -> Verdict:
    """The most severe verdict; several of one kind join their reasons, questions or messages."""
    if not found:
        return Proceed()
    worst = max(severity(v) for v in found)
    same = [v for v in found if severity(v) == worst]
    first = same[0]
    by = ", ".join(dict.fromkeys(v.by for v in same if getattr(v, "by", "")))
    if isinstance(first, Deny):
        return Deny("; ".join(dict.fromkeys(v.reason for v in same if isinstance(v, Deny))), by)
    if isinstance(first, Confirm):
        asks = [v for v in same if isinstance(v, Confirm)]
        details: dict[str, Any] = {}
        for one in asks:
            details.update(one.details)
        return Confirm("; ".join(dict.fromkeys(a.question for a in asks)), details, by)
    if isinstance(first, Rewrite):
        return first
    return Guide(" ".join(dict.fromkeys(v.message for v in same if isinstance(v, Guide))), by)


class _Combined(Base):
    """Several interventions behind one; ``collect`` runs all of them instead of stopping."""

    def __init__(self, items: Sequence[Any], *, collect: bool) -> None:
        self.items = tuple(items)
        self.collect = collect

    def _combine(self, hook: str, args: tuple[Any, ...]) -> Verdict:
        found: list[Verdict] = []
        for one in self.items:
            verdict = _hook(one, hook)(*args)
            if inspect.isawaitable(verdict):
                raise TypeError(f"{hook} is a coroutine; run it through a Run")
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


def first_deny_wins(*items: Any) -> Intervention:
    """Run in order; the first Deny is the answer, else the most severe of the rest."""
    return _Combined(items, collect=False)


def require_all(*items: Any) -> Intervention:
    """Run every one; Proceed only when all do, else the most severe with every reason joined."""
    return _Combined(items, collect=True)


Asker = Callable[[Confirm, Call | None], bool | Awaitable[bool]]
"""``confirm(question, call)`` -> whether the person agrees; a coroutine function may wait."""
Notify = Callable[[Confirm, Call | None], Any]


@dataclass(frozen=True, slots=True)
class Gate:
    """The outcome of the checks before a step: ``allowed``, and when it is not, the text the
    model is told. ``verdict`` decided it; ``confirmed`` records an answered Confirm."""

    allowed: bool
    text: str = ""
    verdict: Verdict = field(default_factory=Proceed)
    confirmed: bool | None = None


@dataclass(frozen=True, slots=True)
class Screened:
    """Text after the checks: ``text`` is what the model sees, ``withheld`` is set when it was
    replaced, and ``verdict`` is the one that decided."""

    text: str
    withheld: bool = False
    verdict: Verdict = field(default_factory=Proceed)


@dataclass(frozen=True, slots=True)
class Guarded:
    """What happened to one tool call: ``ran`` says whether it executed. ``text`` goes back to
    the model as the tool result."""

    ran: bool
    text: str
    verdict: Verdict
    confirmed: bool | None = None


HOOK_FAILURES = (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError)


class Run:
    """The interventions of one run, the person's answer to a `Confirm`, the guidance waiting for
    the model's next turn, and whether outside text has been read.

    Every question has an async form and a plain one; hooks may be plain functions (run on a
    worker thread when asked from async code) or coroutine functions. A hook that raises, or
    answers with anything else, is a `Deny`. With no ``confirm`` a `Confirm` is a refusal.
    """

    def __init__(self, items: Sequence[Any] = (), *, context: Context | None = None,
                 confirm: Asker | None = None, notify: Notify | None = None) -> None:
        self.items = list(items)
        self.context = context or Context()
        self.confirm = confirm
        self.notify = notify
        self.guides: list[str] = []
        self.events: list[tuple[str, str, Verdict]] = []

    @property
    def tainted(self) -> bool:
        return self.context.tainted

    def approve(self, text: str) -> None:
        """Pass ``text``, which the person has read and agreed to, to the interventions that
        keep a list of what is approved."""
        for one in self.items:
            fn = getattr(one, "approve", None)
            if fn is not None:
                fn(text)

    async def _answer(self, one: Any, hook: str, *args: Any) -> Verdict:
        fn = getattr(one, hook)
        try:
            got = fn(*args) if inspect.iscoroutinefunction(fn) \
                else await asyncio.to_thread(fn, *args)
            if inspect.isawaitable(got):
                got = await got
        except HOOK_FAILURES as exc:
            got = Deny(f"{_name(one)}.{hook} raised {type(exc).__name__}: {exc}", _name(one))
        if not isinstance(got, Verdict):
            got = Deny(f"{_name(one)}.{hook} answered {got!r}", _name(one))
        if not isinstance(got, Proceed):
            self.events.append((hook, _name(one), got))
            logger.log(logging.WARNING if isinstance(got, Deny | Confirm) else logging.DEBUG,
                       "%s: %s at %s", _name(one), type(got).__name__, hook)
        return got

    async def ask(self, hook: str, *args: Any) -> list[Verdict]:
        """The verdict of each intervention's ``hook`` that is not Proceed, in order, ending at
        the first Deny."""
        found: list[Verdict] = []
        for one in self.items:
            if getattr(one, hook, None) is None:
                continue
            verdict = await self._answer(one, hook, *args)
            if not isinstance(verdict, Proceed):
                found.append(verdict)
            if isinstance(verdict, Deny):
                break
        return found

    async def _agreed(self, ask: Confirm, call: Call | None) -> bool:
        if self.notify is not None:
            self.notify(ask, call)
        if self.confirm is None:
            return False
        got = self.confirm(ask, call)
        return bool(await got if inspect.isawaitable(got) else got)

    async def decide(self, hook: str, *args: Any, call: Call | None = None) -> Gate:
        """Ask ``hook`` and settle it: a Deny refuses, a Confirm refuses unless the person
        agrees, a Guide is kept for the model's next turn."""
        confirmed: bool | None = None
        for verdict in await self.ask(hook, *args):
            if isinstance(verdict, Deny):
                return Gate(False, f"Denied: {verdict.reason}", verdict)
            if isinstance(verdict, Confirm):
                confirmed = await self._agreed(verdict, call)
                if not confirmed:
                    return Gate(False, f"Denied: a person did not confirm: {verdict.question}",
                                Deny(f"the person declined: {verdict.question}", verdict.by),
                                confirmed)
            elif isinstance(verdict, Guide):
                self.guides.append(verdict.message)
        return Gate(True, "", Proceed(), confirmed)

    async def before_tool(self, call: Call) -> Gate:
        """Whether ``call`` may run."""
        return await self.decide("before_tool_call", call, self.context, call=call)

    async def after_tool(self, call: Call, text: str) -> Screened:
        """``text``, a tool's result, as the model may see it. Each intervention sees what the
        ones before it left; a Rewrite replaces the text, and one marked tainted taints the run."""
        return await self._chain("after_tool_call", text, call, lambda t: (call, t, self.context))

    async def model_text(self, reply: Any) -> Screened:
        """Text the model produced or is about to show, after ``after_model_call``."""
        text = reply if isinstance(reply, str) else str(getattr(reply, "content", "") or "")
        return await self._chain("after_model_call", text, None, lambda t: (self.context, t))

    async def _chain(self, hook: str, text: str, call: Call | None,
                     args: Callable[[str], tuple[Any, ...]]) -> Screened:
        current, last = text, Proceed()
        for one in self.items:
            if getattr(one, hook, None) is None:
                continue
            verdict = await self._answer(one, hook, *args(current))
            if isinstance(verdict, Rewrite):
                current, last = verdict.text, verdict
                self.context.tainted = self.context.tainted or verdict.tainted
            elif isinstance(verdict, Guide):
                self.guides.append(verdict.message)
            elif isinstance(verdict, Deny):
                return Screened(f"[withheld by the {verdict.by or 'guard'} rail: "
                                f"{verdict.reason}]", True, verdict)
            elif isinstance(verdict, Confirm) and not await self._agreed(verdict, call):
                return Screened(f"[withheld: the person declined: {verdict.question}]", True,
                                Deny(verdict.question, verdict.by))
        return Screened(current, False, last)

    async def guarded(self, call: Call, execute: Callable[[Call], Any]) -> Guarded:
        """The whole step for one call: check it, run it if allowed, screen what came back."""
        gate = await self.before_tool(call)
        if not gate.allowed:
            return Guarded(False, gate.text, gate.verdict, gate.confirmed)
        got = execute(call)
        out = await got if inspect.isawaitable(got) else got
        shown = await self.after_tool(call, out)
        text = shown.text
        if self.guides:
            text = f"{text}\n\n{' '.join(self.guides)}"
            self.guides = []
        return Guarded(True, text, shown.verdict if shown.withheld else gate.verdict,
                       gate.confirmed)

    def check_call(self, call: Call) -> Gate:
        """`before_tool`, from code that is not async."""
        return _sync(self.before_tool(call))

    def screen_result(self, call: Call, text: str) -> Screened:
        """`after_tool`, from code that is not async."""
        return _sync(self.after_tool(call, text))

    def screen_model(self, reply: Any) -> Screened:
        """`model_text`, from code that is not async."""
        return _sync(self.model_text(reply))


def _sync(coro: Awaitable[Any]) -> Any:
    """Run ``coro`` to completion from plain code, on a worker thread when a loop is running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]


async def guard_tool_call(call: Call, context: Context, execute: Callable[[Call], Any],
                          interventions: Sequence[Any] = (), *,
                          confirm: Callable[[Confirm], Any] | None = None) -> Guarded:
    """Run ``call`` through every intervention's ``before_tool_call`` and ``after_tool_call``.

    A Deny stops the call; a Confirm calls ``confirm`` and runs the call only when it answers
    True (without ``confirm`` it is a Deny); a Guide's message is added to the result.
    """
    asker: Asker | None = (lambda ask, _call: confirm(ask)) if confirm else None
    return await Run(interventions, context=context, confirm=asker).guarded(call, execute)
