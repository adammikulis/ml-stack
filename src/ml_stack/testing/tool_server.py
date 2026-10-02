"""A llama-server on a real socket whose chat completions follow a script of turns, each
an answer in text or a set of tool calls, in both the plain and the streamed form."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ml_stack.testing.fakes import FakeLlamaServer, Served, _json

__all__ = ["ToolCallingServer", "Turn"]


@dataclass(frozen=True, slots=True)
class Turn:
    """One model reply: ``text`` pieces, then ``calls`` as ``(name, arguments text)``."""

    text: tuple[str, ...] = ()
    calls: tuple[tuple[str, str], ...] = ()
    completion_tokens: int = 0


@dataclass(slots=True)
class _Seen:
    bodies: list[dict[str, Any]] = field(default_factory=list)


class ToolCallingServer(FakeLlamaServer):
    """Replies to each chat completion with the next `Turn`; ``bodies`` is every request's
    JSON. After the script is spent it answers ``"done"``."""

    def __init__(self, turns: list[Turn], *, port: int = 0) -> None:
        super().__init__(Served(), port=port)
        self.turns = list(turns)
        self.seen = _Seen()

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return self.seen.bodies

    def _next(self, body: dict[str, Any]) -> Turn:
        self.seen.bodies.append(body)
        return self.turns.pop(0) if self.turns else Turn(text=("done",))

    @staticmethod
    def _calls(turn: Turn) -> list[dict[str, Any]]:
        return [{"index": n, "id": f"call_{n}", "type": "function",
                 "function": {"name": name, "arguments": arguments}}
                for n, (name, arguments) in enumerate(turn.calls)]

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, str, bytes]:
        if path.partition("?")[0] != "/v1/chat/completions":
            return super().post(path, body)
        turn = self._next(body)
        message: dict[str, Any] = {"role": "assistant", "content": "".join(turn.text) or None}
        if turn.calls:
            message["tool_calls"] = self._calls(turn)
        return _json({"model": self.served.name, "choices": [
            {"index": 0, "message": message,
             "finish_reason": "tool_calls" if turn.calls else "stop"}],
            "usage": {"completion_tokens": turn.completion_tokens}
            if turn.completion_tokens else {}})

    def frames(self, body: dict[str, Any]) -> list[bytes]:
        turn = self._next(body)
        chunks = [{"choices": [{"index": 0, "delta": {"content": piece}}]}
                  for piece in turn.text]
        chunks += [{"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}
                   for call in self._calls(turn)]
        chunks.append({"choices": [{"index": 0, "delta": {},
                                    "finish_reason": "tool_calls" if turn.calls else "stop"}],
                       **({"usage": {"completion_tokens": turn.completion_tokens}}
                          if turn.completion_tokens else {})})
        return [b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks] + [
            b"data: [DONE]\n\n"]
