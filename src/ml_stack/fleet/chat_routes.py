"""``/ui/conversations`` and ``/ui/chat``: the kept conversations, and a question put to whatever is serving a model."""

from __future__ import annotations

from ml_stack import agent_dependency

from .chat import find, reply_parts, targets
from .conversation_routes import ConversationRoutes
from .discovery import derive_token, load_cluster_key

__all__ = ["ChatRoutes"]


class ChatRoutes(ConversationRoutes):
    """The kept conversations, and a question put to whatever is serving a model."""

    def route(self) -> bool:
        if self.path == "/ui/chat":
            return self._chat()
        return super().route()

    def _chat(self) -> bool:
        ui = self.ui
        key = load_cluster_key(ui.cluster_key_path)
        available = targets(ui.peers() if key is not None else [], ui.serving,
                            derive_token(key) if key else "")
        if self.method == "GET":
            problem = agent_dependency.problem()
            self.send(200, {"models": [t.public() for t in available], "runtime_ready": not problem,
                            "runtime_error": problem})
            return True
        if self.method == "POST":
            return self._say(available)
        return super().route()

    def _say(self, available: list) -> bool:
        from .chat_stream import Transfer, frame, release, reserve
        try:
            from .sdk_chat import stream
        except ImportError:
            self.send(503, {"error": "chat requires the agents extra: pip install ml-stack[agents]"})
            return True
        req = self.body()
        target = find(available, str(req.get("model") or ""))
        if target is None:
            self.send(503, {"error": "no machine on this network is serving a model"})
            return True
        messages = [m for m in (req.get("messages") or [])
                    if isinstance(m, dict) and m.get("content")]
        if not messages or messages[-1].get("role") != "user":
            self.send(400, {"error": "send a user message"})
            return True
        cid = str(req.get("conversation") or "")
        store = self.ui.conversations
        found = store.get(cid) if store is not None and cid else None
        if cid and found is None:
            self.send(404, {"error": "no such chat"})
            return True
        if cid and not reserve(store, cid):
            self.send(409, {"error": "this conversation already has a pending response"})
            return True
        try:
            if found is not None:
                messages = [{"role": m.role, "content": m.content}
                            for m in found.messages if m.content] + [messages[-1]]
            payload = self._chat_payload(req, target, messages)
            if payload is None:
                return True
            if found is not None:
                store.append(cid, "user", str(messages[-1]["content"]))
            handler = self.handler
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("X-ML-Stack-Peer", target.peer or "")
            handler.send_header("X-ML-Stack-Model", target.model)
            handler.send_header("Connection", "close")
            handler.end_headers()
            try:
                handler.wfile.write(frame({"ml_stack": {"state": "queued", "conversation": cid,
                                                       "message": "Waiting for the model slot"}}))
                handler.wfile.flush()
            except OSError:
                return True
            said, status = Transfer(target, payload, cid, source=stream).relay(handler, defer_done=True)
            spoken, reasoning = reply_parts(said)
            if found is not None and (spoken or reasoning):
                store.append(cid, "assistant", spoken, reasoning=reasoning, status=status)
            if status != "cancelled":
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
        finally:
            if cid:
                release(store, cid)
        return True

    def _chat_payload(self, req, target, messages):
        payload = {"model": target.alias or target.model, "messages": messages, "stream": True}
        if req.get("max_output_tokens") is not None:
            output_tokens = req["max_output_tokens"]
            if type(output_tokens) is not int or output_tokens < 1:
                self.send(400, {"error": "Maximum output tokens must be a positive integer"})
                return None
            payload["max_output_tokens"] = output_tokens
        if req.get("temperature") is not None:
            try:
                payload["temperature"] = float(req["temperature"])
            except (TypeError, ValueError):
                self.send(400, {"error": "invalid temperature"})
                return None
        return payload
