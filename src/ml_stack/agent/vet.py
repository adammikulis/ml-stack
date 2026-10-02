"""Putting an agent's interventions to work: asking them, waiting on the person, and
carrying their guidance into the next model turn."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml_stack.agent.events import ConfirmRequest
from ml_stack.agent.interventions import Confirm, Deny, Guide, ask

__all__ = ["Confirmer", "Verdict", "Vetter"]

Confirmer = Callable[[Confirm, Mapping[str, Any]], bool | Awaitable[bool]]
"""``confirm(question, call)`` -> whether the person agrees; a coroutine function may wait."""


@dataclass(slots=True)
class Verdict:
    """Why a step was refused, or empty."""

    denied: str = ""


class Vetter:
    """The interventions of one agent, the person's answer to a `Confirm`, and the guidance
    waiting for the model's next turn. With no ``confirm``, a `Confirm` is a refusal."""

    def __init__(self, hooks: Sequence[Any]) -> None:
        self.hooks = list(hooks)
        self.confirm: Confirmer | None = None
        self.guides: list[str] = []

    async def check(self, method: str, args: tuple[Any, ...], verdict: Verdict, *,
                    ident: str = "", name: str = "") -> AsyncIterator[ConfirmRequest]:
        """Ask every hook's ``method``; a `Deny` (or a declined `Confirm`) is put in
        ``verdict`` and ends the asking."""
        async for decision in ask(self.hooks, method, *args):
            if isinstance(decision, Guide):
                self.guides.append(decision.message)
            elif isinstance(decision, Deny):
                verdict.denied = decision.reason
                return
            elif isinstance(decision, Confirm):
                yield ConfirmRequest(ident, name, decision.question, dict(decision.details))
                if not await self._agreed(decision, args[0] if ident else {}):
                    verdict.denied = f"the person declined: {decision.question}"
                    return

    async def _agreed(self, decision: Confirm, call: Mapping[str, Any]) -> bool:
        if self.confirm is None:
            return False
        answer = self.confirm(decision, call)
        return bool(await answer if inspect.isawaitable(answer) else answer)

    def inject(self, messages: list[dict[str, Any]]) -> None:
        """Put the waiting guidance into ``messages`` as part of the next user turn."""
        if not self.guides:
            return
        text = "[Guidance]\n" + "\n".join(self.guides)
        self.guides = []
        if messages and messages[-1].get("role") == "user":
            last = messages[-1]
            messages[-1] = {**last, "content": f"{last.get('content') or ''}\n\n{text}"}
        else:
            messages.append({"role": "user", "content": text})
