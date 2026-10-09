"""Chat reasoning, queue status, and cancellation."""

import io
import threading
from contextlib import nullcontext
from types import SimpleNamespace

from poolhouse.fleet.chat import Target, reply_parts
from poolhouse.fleet.chat_stream import Transfer, frame, release, reserve
from poolhouse.fleet.conversations import Conversations


def test_reasoning_survives_conversation_reload(tmp_path):
    raw = frame({"choices": [{"delta": {"reasoning_content": "think"}}]})
    raw += frame({"choices": [{"delta": {"content": "answer"}}]})
    content, reasoning = reply_parts(raw)
    assert (content, reasoning) == ("answer", "think")
    store = Conversations(tmp_path)
    chat = store.start()
    store.append(chat.id, "assistant", content, reasoning=reasoning, status="cancelled")
    message = Conversations(tmp_path).get(chat.id).messages[0]
    assert (message.content, message.reasoning, message.status) == ("answer", "think", "cancelled")


def test_pending_conversation_is_reserved_once():
    store = object()
    assert reserve(store, "chat")
    assert not reserve(store, "chat")
    release(store, "chat")
    assert reserve(store, "chat")
    release(store, "chat")


def test_upstream_error_is_an_sse_error(monkeypatch):
    from poolhouse.fleet import chat_stream
    from poolhouse.fleet.chat import ChatError
    monkeypatch.setattr(chat_stream, "turn", lambda *a, **k: nullcontext())
    def fail(*args, **kwargs):
        raise ChatError("upstream failed")
    monkeypatch.setattr(chat_stream, "stream", fail)
    handler = SimpleNamespace(wfile=io.BytesIO())
    raw, status = Transfer(Target("model", "http://test/error"), {}, "chat").relay(handler)
    assert status == "failed"
    assert b'"state": "rebuilding"' in raw
    assert b'"error": {"message": "upstream failed"}' in raw
    assert raw.endswith(b"data: [DONE]\n\n")


def test_disconnect_cancels_worker_and_frees_model_slot(monkeypatch):
    from poolhouse.fleet import chat_stream
    monkeypatch.setattr(chat_stream, "turn", lambda *a, **k: nullcontext())
    stopped = threading.Event()
    def waiting(target, payload, *, control):
        control.cancelled.wait(3)
        stopped.set()
        yield b"data: [DONE]\n\n"
    monkeypatch.setattr(chat_stream, "stream", waiting)
    class Broken:
        def write(self, block):
            raise BrokenPipeError
    transfer = Transfer(Target("model", "http://test/cancel"), {}, "chat")
    _, status = transfer.relay(SimpleNamespace(wfile=Broken()))
    assert status == "cancelled"
    assert not transfer.worker.is_alive()
    assert transfer.cancelled.is_set()
