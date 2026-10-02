"""What an agent run yields: text and thinking deltas, tool calls and their results,
rejected calls, the context's fill, compactions, and the end."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Compacted", "Context", "Done", "Event", "Repair", "Text", "Thinking", "ToolCall",
           "ToolResult"]


@dataclass(frozen=True, slots=True)
class Text:
    delta: str


@dataclass(frozen=True, slots=True)
class Thinking:
    delta: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    id: str
    name: str
    text: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Repair:
    """A call that was not dispatched, and what the model is told is wrong with it."""

    id: str
    name: str
    errors: list[str]


@dataclass(frozen=True, slots=True)
class Context:
    """How full the context is before a request: ``used`` of ``limit`` tokens."""

    used: int
    limit: int
    fraction: float


@dataclass(frozen=True, slots=True)
class Compacted:
    """A compaction that ran: the context fraction before and after, the messages dropped,
    the stages that did it, and where the removed text was written."""

    before: float
    after: float
    dropped: int
    strategy: str
    tokens_before: int
    tokens_after: int
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"type": "compact", "before": self.before, "after": self.after,
                "dropped": self.dropped, "strategy": self.strategy,
                "tokens_before": self.tokens_before, "tokens_after": self.tokens_after,
                "notes": list(self.notes)}


@dataclass(frozen=True, slots=True)
class Done:
    """Why the run ended (``answer``, ``max_steps``, ``max_tool_calls``, ``max_tokens`` or
    ``repairs_exhausted``), the final text, and the counts."""

    reason: str
    text: str = ""
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list, repr=False)


Event = Text | Thinking | ToolCall | ToolResult | Repair | Context | Compacted | Done
