"""One question as the page sent it, and the conversation that goes back with it."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["Ask", "History"]


class History(list):
    """What goes back with a question: the window as messages, and what goes ahead of it.

    A list of ``{"role", "content"}`` -- the last ``WINDOW`` turns, chosen by recency
    alone. On it, for ``converse(..., summary=, recalled=)``: ``summary`` is the latest
    summary ``Turn`` or None, ``recalled`` the earlier turns found for this question,
    oldest first. ``as_dict`` is the same three things by name.
    """

    __slots__ = ("recalled", "summary")

    def __init__(self, turns: Sequence[Mapping[str, str]] = (), *, summary: Any = None,
                 recalled: Sequence[Any] = ()) -> None:
        super().__init__(turns)
        self.summary = summary
        self.recalled = list(recalled)

    def as_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "recalled": list(self.recalled), "turns": list(self)}


class Ask:
    """One question as the page sent it, after the body was checked and history resolved."""

    __slots__ = ("began", "body", "highlighted", "question", "sent", "thread", "turns")

    def __init__(self, body: Mapping[str, Any]) -> None:
        self.body = dict(body)
        self.question = str(body.get("question") or "").strip()
        self.sent = body.get("turns") if isinstance(body.get("turns"), list) else []
        self.turns: list = list(self.sent)
        self.thread = str(body.get("thread") or "")[:64]
        lit = body.get("highlighted")
        self.highlighted = (list(lit) if isinstance(lit, list)
                            and all(isinstance(h, str) for h in lit) else [])
        self.began = time.time()

    @property
    def took_s(self) -> float:
        return round(time.time() - self.began, 1)

    @property
    def summary(self) -> Any:
        """The latest summary of the thread, when ``history`` found one."""
        return getattr(self.turns, "summary", None)

    @property
    def recalled(self) -> list:
        """The earlier turns recalled for this question, oldest first."""
        return list(getattr(self.turns, "recalled", ()) or ())
