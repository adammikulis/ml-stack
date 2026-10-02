"""Rails between a model and the world: input, output and tool-call checks that return
allow, deny or modify with a reason.

The default :class:`Guard` holds the built-in rails and needs nothing installed. Turning one
off is :func:`rails` with ``without=`` and a ``because``; it is logged and printed. Another
library joins as one more :class:`Rail` (`ml_stack.guard.nemo`).

    from ml_stack.guard import Guard

    guard = Guard.default()
    guard.input(tool_result_text, "tool:models_find").text
    guard.tool_call(ToolCall("serve_up", {"model": "x.gguf"})).denied
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from ml_stack import log
from ml_stack.guard.policy import ToolPolicyRail, tool_schemas
from ml_stack.guard.secrets import SecretRail
from ml_stack.guard.untrusted import NOTICE, UntrustedRail
from ml_stack.guard.verdict import Rail, ToolCall, Verdict, allow

__all__ = ["BUILTIN", "NOTICE", "Guard", "Rail", "ToolCall", "Verdict", "rails"]

logger = logging.getLogger("ml_stack.guard")
logger.addHandler(logging.NullHandler())

BUILTIN = ("untrusted", "secrets", "tool-policy")


def rails(*, without: Iterable[str] = (), because: str = "") -> list[Rail]:
    """The built-in rails, minus the named ones. Dropping any needs a ``because``, and the
    drop is logged and printed."""
    dropped = tuple(without)
    unknown = [n for n in dropped if n not in BUILTIN]
    if unknown:
        raise ValueError(f"no built-in rail named {unknown}; they are {list(BUILTIN)}")
    if dropped:
        if not because.strip():
            raise ValueError("turning a rail off needs a because=")
        message = f"guard: {', '.join(dropped)} turned off: {because.strip()}"
        logger.warning(message)
        log.warn(message)
    made: dict[str, Rail] = {"untrusted": UntrustedRail(), "secrets": SecretRail(),
                             "tool-policy": ToolPolicyRail()}
    return [made[n] for n in BUILTIN if n not in dropped]


class Guard:
    """An ordered list of rails. The first deny wins; modifications chain."""

    def __init__(self, rail_list: Sequence[Rail]) -> None:
        self.rails = list(rail_list)
        self.tainted = False
        self.events: list[Verdict] = []

    @classmethod
    def default(cls, extra: Sequence[Rail] = ()) -> Guard:
        """The built-in rails, then ``extra`` ones such as a NeMo rail."""
        return cls([*rails(), *extra])

    @classmethod
    def off(cls, because: str) -> Guard:
        """No rails at all; needs a ``because`` and is logged and printed."""
        return cls(rails(without=BUILTIN, because=because))

    def bind(self, offered: Sequence[Mapping[str, Any]],
             confirm: Callable[[ToolCall], bool] | None = None) -> Guard:
        """Tell the rails which tools a run offers and who confirms a tainted call."""
        schemas = tool_schemas(offered)
        for rail in self.rails:
            if isinstance(rail, ToolPolicyRail):
                rail.bind(schemas)
                if confirm is not None:
                    rail.confirm = confirm
        return self

    def approve(self, text: str) -> None:
        """The person has read ``text`` and agreed: the sensitive tools it names may run."""
        for rail in self.rails:
            if isinstance(rail, ToolPolicyRail):
                rail.approve(text)

    def input(self, text: str, source: str) -> Verdict:
        """Text entering the model's context from ``source`` (``person`` or ``tool:<name>``)."""
        return self._chain(text, source, "on_input")

    def output(self, text: str, source: str = "model") -> Verdict:
        """Text the model produced."""
        return self._chain(text, source, "on_output")

    def tool_call(self, call: ToolCall) -> Verdict:
        """A call the model asked for, before it runs."""
        call = replace(call, tainted=self.tainted)
        for rail in self.rails:
            verdict = rail.on_tool_call(call)
            if verdict.denied:
                return self._note(verdict, f"tool:{call.name}")
        return allow()

    def _chain(self, text: str, source: str, hook: str) -> Verdict:
        current, last = text, allow()
        for rail in self.rails:
            verdict: Verdict = getattr(rail, hook)(current, source)
            if verdict.denied:
                return self._note(verdict, source)
            if verdict.action == "modify":
                current, last = verdict.text, verdict
                self.tainted = self.tainted or verdict.tainted
                self._note(verdict, source)
        return replace(last, text=current)

    def _note(self, verdict: Verdict, source: str) -> Verdict:
        level = logging.WARNING if verdict.denied or verdict.tainted else logging.DEBUG
        logger.log(level, "%s: %s %s (%s)", verdict.rail, verdict.action, verdict.reason, source)
        if level == logging.WARNING:
            self.events.append(verdict)
        return verdict
