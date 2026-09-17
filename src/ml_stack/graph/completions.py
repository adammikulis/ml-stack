"""OpenAI-compatible chat completions over an in-process model, for any handler to mix in.

``POST /v1/chat/completions`` streams or answers whole; the reasoning comes back as
``reasoning_content`` and the answer as ``content``, with llama.cpp's ``timings`` beside the
usage. ``GET /health``, ``/v1/models`` and ``/props`` say what is loaded. Every route is a
404 until the handler has a ``completer``.

A completer is ``name``, ``context``, ``program`` (what ``/props`` says serves it) and
``complete(messages, options, on_text, stop)``,
which returns a mapping: ``content``, ``reasoning``, ``finish``, ``prompt_tokens``,
``cached_tokens``, ``completion_tokens``, ``reasoning_tokens`` and ``timings``.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Mapping
from typing import Any

from ml_stack.graph.payloads import sse
from ml_stack.log import warn

__all__ = ["CompletionRoutes", "options_of"]

#: request fields passed through to the completer as they are, by the name they take there
FIELDS = {"max_tokens": int, "max_completion_tokens": int, "temperature": float,
          "top_p": float, "top_k": int, "min_p": float, "seed": int,
          "reasoning_effort": str, "accept": str}


def options_of(body: Mapping[str, Any]) -> dict[str, Any]:
    """The completer's options from a request body. Raises ValueError on a malformed field."""
    messages = body.get("messages")
    if not (isinstance(messages, list) and messages
            and all(isinstance(m, dict) and "role" in m for m in messages)):
        raise ValueError('"messages" must be a non-empty list of {"role": ..., "content": ...}')
    out: dict[str, Any] = {}
    for key, kind in FIELDS.items():
        if body.get(key) is not None:
            try:
                out[key] = kind(body[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key}: {exc}") from exc
    if "max_completion_tokens" in out:
        out["max_tokens"] = out.pop("max_completion_tokens")
    kwargs = body.get("chat_template_kwargs") or {}
    thinking = body.get("enable_thinking", kwargs.get("enable_thinking",
                                                      body.get("thinking")))
    if thinking is not None:
        out["thinking"] = bool(thinking)
    if "reasoning_effort" not in out and kwargs.get("reasoning_effort"):
        out["reasoning_effort"] = str(kwargs["reasoning_effort"])
    if body.get("n_predict") is not None and "max_tokens" not in out:
        out["max_tokens"] = int(body["n_predict"])
    tools = body.get("tools")
    if tools:
        if not (isinstance(tools, list) and all(isinstance(t, dict) for t in tools)):
            raise ValueError('"tools" must be a list of {"type": "function", "function": ...}')
        out["tools"] = tools
    return out


class CompletionRoutes:
    """The routes; the handler that mixes them in supplies ``send_json`` and ``completer``."""

    completer: Any = None

    def handle_health(self) -> None:
        if self.completer is None:
            self.send_error(404)
            return
        self.send_json(200, {"status": "ok", "model": self.completer.name})

    def handle_models(self) -> None:
        if self.completer is None:
            self.send_error(404)
            return
        self.send_json(200, {"object": "list", "data": [
            {"id": self.completer.name, "object": "model", "owned_by": "local"}]})

    def handle_props(self) -> None:
        if self.completer is None:
            self.send_error(404)
            return
        self.send_json(200, {"model_path": self.completer.name, "total_slots": 1,
                             "default_generation_settings": {"n_ctx": self.completer.context},
                             **dict(self.completer.program)})

    def handle_completion(self, body: Mapping[str, Any] | None) -> None:
        if self.completer is None:
            self.send_error(404)
            return
        try:
            options = options_of(body or {})
        except ValueError as exc:
            self.send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return
        head = {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "created": int(time.time()),
                "model": self.completer.name}
        messages = list((body or {})["messages"])
        if (body or {}).get("stream"):
            self._stream(messages, options, head)
            return
        try:
            done = self.completer.complete(messages, options, None, None)
        except (RuntimeError, ValueError) as exc:
            warn(f"{time.strftime('%FT%T')} completion failed: {exc}")
            self.send_json(500, {"error": {"message": str(exc)[:200], "type": "server_error"}})
            return
        message = {"role": "assistant", "content": done["content"]}
        if done.get("reasoning"):
            message["reasoning_content"] = done["reasoning"]
        if done.get("tool_calls"):
            message["tool_calls"] = list(done["tool_calls"])
        self.send_json(200, {**head, "object": "chat.completion",
                             "choices": [{"index": 0, "message": message,
                                          "finish_reason": _finish(done)}],
                             **_tail(done)})

    def _stream(self, messages: list, options: Mapping[str, Any], head: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        gone = [False]

        def chunk(delta: Mapping[str, Any], finish: str | None = None, **more: Any) -> None:
            if gone[0]:
                return
            try:
                sse(self.wfile, {**head, "object": "chat.completion.chunk",
                                 "choices": [{"index": 0, "delta": dict(delta),
                                              "finish_reason": finish}], **more})
            except OSError:
                gone[0] = True

        chunk({"role": "assistant", "content": ""})
        try:
            done = self.completer.complete(
                messages, options,
                lambda piece, thinking: chunk({"reasoning_content" if thinking else "content": piece}),
                lambda: gone[0])
            if done.get("tool_calls"):
                chunk({"tool_calls": [{"index": i, **call}
                                      for i, call in enumerate(done["tool_calls"])]})
            chunk({}, _finish(done), **_tail(done))
        except (RuntimeError, ValueError) as exc:
            warn(f"{time.strftime('%FT%T')} completion failed: {exc}")
            if not gone[0]:
                with contextlib.suppress(OSError):
                    sse(self.wfile, {"error": {"message": str(exc)[:200], "type": "server_error"}})
        if not gone[0]:
            with contextlib.suppress(OSError):
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        self.close_connection = True


def _finish(done: Mapping[str, Any]) -> str:
    if done.get("tool_calls"):
        return "tool_calls"
    return "length" if done.get("finish") == "length" else "stop"


def _tail(done: Mapping[str, Any]) -> dict[str, Any]:
    prompt, written = int(done.get("prompt_tokens") or 0), int(done.get("completion_tokens") or 0)
    return {"usage": {"prompt_tokens": prompt, "completion_tokens": written,
                      "total_tokens": prompt + written,
                      "prompt_tokens_details": {"cached_tokens": int(done.get("cached_tokens") or 0)},
                      "completion_tokens_details": {
                          "reasoning_tokens": int(done.get("reasoning_tokens") or 0)}},
            "timings": dict(done.get("timings") or {})}
