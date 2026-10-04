"""``/ui/conversations`` and ``/ui/chat``: the kept conversations, and a question put to whatever is serving a model."""

from __future__ import annotations

from typing import Any

from .discovery import derive_token, load_cluster_key

__all__ = ["ChatRoutes"]


class ChatRoutes:
    """The kept conversations, and a question put to whatever is serving a model."""

    def route(self) -> bool:
        if self.path.startswith("/ui/conversations"):
            return self._conversations()
        if self.path == "/ui/chat":
            return self._chat()
        return super().route()

    def _conversations(self) -> bool:
        ui = self.ui
        if ui.conversations is None:
            self.send(501, {"error": "no chat store on this daemon"})
            return True
        rest = self.path[len("/ui/conversations"):].strip("/")
        if rest:
            return self._one_conversation(rest)
        if self.method == "GET":
            self.send(200, {"conversations": [
                c.public(full=False) for c in ui.conversations.search(self.asked("q"))]})
            return True
        if self.method == "POST":
            req = self.body()
            made = ui.conversations.start(model=str(req.get("model") or ""),
                                          title=str(req.get("title") or ""))
            self.send(201, made.public())
            return True
        return super().route()

    def _one_conversation(self, rest: str) -> bool:
        ui = self.ui
        found = ui.conversations.get(rest)
        if found is None:
            self.send(404, {"error": "no such chat"})
            return True
        if self.method == "GET":
            self.send(200, found.public())
            return True
        if self.method == "DELETE":
            ui.conversations.remove(rest)
            self.send(200, {"removed": rest})
            return True
        if self.method == "POST":
            renamed = ui.conversations.rename(rest, str(self.body().get("title") or ""))
            self.send(200, renamed.public(full=False))
            return True
        return super().route()

    def _chat(self) -> bool:
        from .chat import targets
        ui = self.ui
        key = load_cluster_key(ui.cluster_key_path)
        available = targets(ui.peers() if key is not None else [], ui.serving,
                            derive_token(key) if key else "")
        if self.method == "GET":
            self.send(200, {"models": [t.public() for t in available]})
            return True
        if self.method == "POST":
            return self._say(available)
        return super().route()

    def _say(self, available: list) -> bool:
        from .chat import ChatError, find, reply_text, stream
        req = self.body()
        target = find(available, str(req.get("model") or ""))
        if target is None:
            self.send(503, {"error": "no machine on this network is serving a model"})
            return True
        messages = [m for m in (req.get("messages") or [])
                    if isinstance(m, dict) and m.get("content")]
        if not messages:
            self.send(400, {"error": "nothing to send"})
            return True
        payload = {"model": target.model, "messages": messages, "stream": True}
        if req.get("temperature") is not None:
            payload["temperature"] = float(req["temperature"])
        try:
            pieces = stream(target, payload)
            first = next(pieces, b"")
        except ChatError as exc:
            self.send(502, {"error": str(exc)})
            return True
        cid = str(req.get("conversation") or "")
        if self.ui.conversations is not None and cid:
            self.ui.conversations.append(cid, "user", str(messages[-1]["content"]))
        said = self._relay(target, first, pieces)
        if self.ui.conversations is not None and cid:
            spoken = reply_text(said)
            if spoken:
                self.ui.conversations.append(cid, "assistant", spoken)
        return True

    def _relay(self, target: Any, first: bytes, pieces: Any) -> bytes:
        """Stream the answer to the caller as it arrives, and return all of it."""
        handler = self.handler
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-ML-Stack-Peer", target.peer or "")
        handler.send_header("X-ML-Stack-Model", target.model)
        handler.send_header("Connection", "close")
        handler.end_headers()
        said = bytearray(first)
        try:
            handler.wfile.write(first)
            handler.wfile.flush()
            for block in pieces:
                said += block
                handler.wfile.write(block)
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        return bytes(said)
