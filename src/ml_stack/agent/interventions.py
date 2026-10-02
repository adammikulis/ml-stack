"""What an intervention may answer before the agent acts, and how the agent asks.

An intervention is any object with some of ``before_invocation(context)``,
``before_model_call(context)`` and ``before_tool_call(call, context)``, each returning
`Proceed`, `Deny`, `Confirm` or `Guide` (it may be a coroutine function). The agent stops at
the first `Deny`. A hook that raises, or answers with anything else,
is a `Deny`.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["Confirm", "Decision", "Deny", "Guide", "Intervention", "InterventionContext",
           "Proceed", "ask"]


@dataclass(frozen=True, slots=True)
class Proceed:
    """Carry on."""


@dataclass(frozen=True, slots=True)
class Deny:
    """Do not; ``reason`` is what the model is told."""

    reason: str


@dataclass(frozen=True, slots=True)
class Confirm:
    """Ask the person first: ``question``, and the ``details`` to show with it."""

    question: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Guide:
    """Carry on, and tell the model ``message`` on its next turn."""

    message: str


Decision = Proceed | Deny | Confirm | Guide


@dataclass(frozen=True, slots=True)
class InterventionContext:
    """What an intervention sees: the conversation so far, the model turn, and the tool
    calls already made."""

    messages: Sequence[Mapping[str, Any]]
    step: int
    tool_calls: int


class Intervention(Protocol):
    """Any subset of these three methods."""

    def before_invocation(self, context: InterventionContext) -> Decision: ...

    def before_model_call(self, context: InterventionContext) -> Decision: ...

    def before_tool_call(self, call: Mapping[str, Any],
                         context: InterventionContext) -> Decision: ...


async def _call(fn: Any, *args: Any) -> Any:
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    got = await asyncio.to_thread(fn, *args)
    return await got if inspect.isawaitable(got) else got


async def ask(hooks: Sequence[Any], method: str, *args: Any) -> AsyncIterator[Decision]:
    """The answer of each hook that has ``method``, one at a time; stop reading at a `Deny`
    and the later hooks are not called."""
    for hook in hooks:
        fn = getattr(hook, method, None)
        if fn is None:
            continue
        got = (await asyncio.gather(_call(fn, *args), return_exceptions=True))[0]
        name = f"{type(hook).__name__}.{method}"
        if isinstance(got, BaseException):
            got = Deny(f"{name} raised {type(got).__name__}: {got}")
        elif not isinstance(got, Decision):
            got = Deny(f"{name} answered {got!r}")
        yield got
