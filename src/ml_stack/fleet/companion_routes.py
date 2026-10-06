"""TLS Android status and streamed chat with scoped expiring sessions."""
from __future__ import annotations

import ipaddress
import json
import select
import ssl
from typing import Any

from ml_stack.gate import QueueTimeout, turn

from .chat import ChatError, reply_text, targets

try:
    from . import sdk_chat
except ImportError:
    sdk_chat = None


class _Control:
    def __init__(self, ui: Any, handler: Any, token: str):
        self.ui, self.handler, self.token = ui, handler, token
        self.cancelled = self

    def is_set(self) -> bool:
        try:
            self.ui.invitations.authorize(self.token)
            return bool(select.select([self.handler.connection], [], [], 0)[0])
        except (ValueError, OSError):
            return True


def answer(ui: Any, handler: Any, raw: bytes | None = None) -> bool:
    path = handler.path.split("?", 1)[0]
    if not path.startswith("/companion/"):
        return False
    try:
        source = ipaddress.ip_address(handler.client_address[0])
        if (ui is None or not isinstance(handler.connection, ssl.SSLSocket)
                or not source.is_private or source.is_unspecified or source.is_multicast):
            raise ValueError("Android connections require local network TLS")
        auth = handler.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise ValueError("Android session required")
        token = auth[7:]
        ui.invitations.authorize(token)
    except ValueError as exc:
        handler._send(403, {"error": str(exc)}, headers={"Cache-Control": "no-store"})
        return True
    available = targets([], ui.serving)
    if path == "/companion/v1/status" and handler.command == "GET":
        handler._send(200, {"name": ui.name, "models": [target.model for target in available]},
                      headers={"Cache-Control": "no-store"})
    elif path == "/companion/v1/chat" and handler.command == "POST":
        _chat(ui, handler, raw, token, available)
    else:
        handler._send(404, {"error": "Android route unavailable"})
    return True


def _chat(ui: Any, handler: Any, raw: bytes | None, token: str, available: list) -> None:
    try:
        if raw is None or not 0 < len(raw) <= 65536:
            raise ValueError("chat request is too large or empty")
        body = json.loads(raw)
        if not isinstance(body, dict) or set(body) != {"model", "messages"}:
            raise ValueError("chat requires a model and one user message")
        messages = body["messages"]
        if (not isinstance(messages, list) or len(messages) != 1
                or not isinstance(messages[0], dict) or set(messages[0]) != {"role", "content"}
                or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str)
                or not 0 < len(messages[0]["content"].strip()) <= 16384):
            raise ValueError("chat requires one bounded user message")
        target = next((t for t in available if t.model == body["model"]), None)
        if target is None:
            raise ValueError("choose an available local chat model")
        ui.invitations.permit(handler.client_address[0])
        if sdk_chat is None:
            raise ImportError("install ml-stack[agents] for Android chat")
    except (ValueError, ImportError) as exc:
        handler._send(400, {"error": str(exc)})
        return
    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    size = 0
    try:
        control = _Control(ui, handler, token)
        with turn(target.url, cancelled=control.is_set):
            ui.invitations.authorize(token)
            with_stream = sdk_chat.stream(target, {"model": target.model, "messages": messages,
                                         "stream": True}, control=control)
            try:
                for frame in with_stream:
                    ui.invitations.authorize(token)
                    delta = reply_text(frame)
                    if not delta:
                        continue
                    piece = json.dumps({"delta": delta}, ensure_ascii=False).encode() + b"\n"
                    size += len(piece)
                    if size > (1 << 20) - 1024:
                        raise ValueError("chat response is too large")
                    _chunk(handler, piece)
                ui.invitations.authorize(token)
                _chunk(handler, b'{"done":true}\n')
            finally:
                with_stream.close()
    except (ChatError, ValueError, QueueTimeout):
        _chunk(handler, b'{"error":"Chat ended; check the model and Android connection."}\n')
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True
        return
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()


def _chunk(handler: Any, raw: bytes) -> None:
    handler.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
    handler.wfile.flush()
