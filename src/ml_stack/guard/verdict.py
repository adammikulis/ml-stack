"""What a rail answers, the call it is asked about, and the rail protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = ["ALLOW", "Rail", "ToolCall", "Verdict", "allow", "deny", "modify"]

Action = Literal["allow", "deny", "modify"]


@dataclass(frozen=True)
class Verdict:
    """``allow`` passes the text through, ``modify`` replaces it with ``text``, ``deny`` stops it.

    ``tainted`` marks text a rail judged to carry instructions from outside the person.
    """

    action: Action = "allow"
    text: str = ""
    reason: str = ""
    rail: str = ""
    tainted: bool = False

    @property
    def denied(self) -> bool:
        return self.action == "deny"


ALLOW = Verdict()


def allow(rail: str = "") -> Verdict:
    return Verdict(rail=rail)


def deny(rail: str, reason: str) -> Verdict:
    return Verdict("deny", reason=reason, rail=rail)


def modify(rail: str, text: str, reason: str, *, tainted: bool = False) -> Verdict:
    return Verdict("modify", text=text, reason=reason, rail=rail, tainted=tainted)


@dataclass(frozen=True)
class ToolCall:
    """One call the model asked for. ``arguments`` is None when the model's arguments were
    not a JSON object; ``raw`` is what it sent; ``tainted`` is set by the guard when text from
    outside the person is already in the model's context."""

    name: str
    arguments: dict[str, Any] | None
    raw: str = ""
    tainted: bool = False


@runtime_checkable
class Rail(Protocol):
    """One check. ``on_input`` sees text entering the model's context, ``on_output`` sees text
    the model produced, ``on_tool_call`` sees a call before it runs."""

    name: str

    def on_input(self, text: str, source: str) -> Verdict: ...

    def on_output(self, text: str, source: str) -> Verdict: ...

    def on_tool_call(self, call: ToolCall) -> Verdict: ...
