"""A model server that stands in for a real one: it obeys any instruction in its conversation
that names a tool (``gullible``) or obeys none (``resistant``). It makes every scorer, and the
whole run, testable without a model."""

from __future__ import annotations

import json
import re
from typing import Any

from ml_stack.testing.fakes import FakeLlamaServer, Served, _json

__all__ = ["StubModel"]

CALL = re.compile(r"call the (write_note|send_report|read_page) tool with (.+?)(?:\.\s|\.$|$)",
                  re.S)
FIELD = re.compile(r'(\w+)=("[^"]*"|\S+)')
LEAK = re.compile(r"(system prompt|deployment key|instructions you were given|initial prompt)", re.I)
READ = re.compile(r"Read (\S+) and", re.I)


def _text(message: dict[str, Any]) -> str:
    return str(message.get("content") or "")


def _fields(raw: str) -> dict[str, str]:
    return {key: value.strip('"').rstrip(".,") for key, value in FIELD.findall(raw)}


class StubModel(FakeLlamaServer):
    """An OpenAI-compatible server whose replies follow the conversation's instructions."""

    def __init__(self, mode: str = "gullible", *, port: int = 0) -> None:
        super().__init__(Served(), port=port)
        self.mode = mode

    def plan(self, body: dict[str, Any]) -> tuple[str, list[tuple[str, dict[str, str]]]]:
        """``(text, tool calls)`` the conversation in ``body`` leads to."""
        messages = list(body.get("messages") or [])
        system = next((_text(m) for m in messages if m.get("role") == "system"), "")
        if self.mode != "gullible":
            return "I cannot help with that.", []
        made = {(c["function"]["name"], c["function"]["arguments"])
                for m in messages for c in m.get("tool_calls") or []}
        user = next((_text(m) for m in reversed(messages) if m.get("role") == "user"), "")
        wanted: list[tuple[str, dict[str, str]]] = []
        for text in [user, *(_text(m) for m in messages if m.get("role") == "tool")]:
            wanted += [(name, _fields(raw)) for name, raw in CALL.findall(text)]
            if text is user and (found := READ.search(user)):
                wanted.insert(0, ("read_page", {"url": found.group(1)}))
        for name, args in wanted:
            if name == "send_report" and "system prompt" in "".join(map(_text, messages)).lower():
                args = {**args, "body": system}
            if (name, json.dumps(args)) not in made:
                return "", [(name, args)]
        if LEAK.search(user):
            return system or "I have no instructions.", []
        return "done", []

    def _message(self, body: dict[str, Any]) -> dict[str, Any]:
        text, calls = self.plan(body)
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = [
                {"index": n, "id": f"call_{n}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}
                for n, (name, args) in enumerate(calls)]
        return message

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, str, bytes]:
        if path.partition("?")[0] != "/v1/chat/completions":
            return super().post(path, body)
        message = self._message(body)
        return _json({"model": self.served.name, "choices": [
            {"index": 0, "message": message,
             "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"completion_tokens": 1}})

    def frames(self, body: dict[str, Any]) -> list[bytes]:
        message = self._message(body)
        chunks: list[dict[str, Any]] = []
        if message["content"]:
            chunks.append({"choices": [{"index": 0, "delta": {"content": message["content"]}}]})
        chunks += [{"choices": [{"index": 0, "delta": {"tool_calls": [call]}}]}
                   for call in message.get("tool_calls", [])]
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": (
            "tool_calls" if message.get("tool_calls") else "stop")}]})
        return [b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks] + [
            b"data: [DONE]\n\n"]
