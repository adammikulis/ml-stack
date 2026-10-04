"""Queued chat streams and caller cancellation."""

from __future__ import annotations

import contextlib
import json
import queue
import socket
import threading
from typing import Any

from ml_stack.gate import QueueTimeout, turn
from ml_stack.http import ServerError

from .chat import stream

_LOCK = threading.Lock()
_TARGETS: dict[str, threading.Lock] = {}
_ACTIVE: set[tuple[int, str]] = set()


def frame(value: dict) -> bytes:
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def reserve(store: Any, cid: str) -> bool:
    with _LOCK:
        key = (id(store), cid)
        if key in _ACTIVE:
            return False
        _ACTIVE.add(key)
        return True


def release(store: Any, cid: str) -> None:
    with _LOCK:
        _ACTIVE.discard((id(store), cid))


class Transfer:
    """One cancellable upstream chat request."""

    def __init__(self, target: Any, payload: dict, cid: str) -> None:
        self.target, self.payload, self.cid = target, payload, cid
        self.cancelled = threading.Event()
        self.events: queue.Queue = queue.Queue(maxsize=64)
        self.response = None
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self.run, daemon=True)

    def bind(self, response: Any) -> None:
        with self.lock:
            self.response = response
        if self.cancelled.is_set():
            self.cancel()

    def cancel(self) -> None:
        self.cancelled.set()
        with self.lock:
            response = self.response
        if response is not None:
            with contextlib.suppress(AttributeError, OSError):
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR)

    def emit(self, block: bytes | None) -> None:
        while not self.cancelled.is_set():
            try:
                self.events.put(block, timeout=0.1)
                return
            except queue.Full:
                continue

    def state(self, state: str, message: str) -> None:
        self.emit(frame({"ml_stack": {"state": state, "conversation": self.cid,
                                       "message": message}}))

    def run(self) -> None:
        with _LOCK:
            lock = _TARGETS.setdefault(self.target.url, threading.Lock())
        acquired = False
        try:
            while not self.cancelled.is_set():
                if lock.acquire(timeout=0.1):
                    acquired = True
                    break
            if not acquired:
                return
            with turn(self.target.url, cancelled=self.cancelled.is_set):
                if self.cancelled.is_set():
                    return
                self.state("rebuilding", "Rebuilding model context from the conversation transcript")
                self.state("loading", "Waiting for the model to begin generating")
                for block in stream(self.target, self.payload, control=self):
                    self.emit(block)
                    if self.cancelled.is_set():
                        break
        except (ServerError, QueueTimeout, OSError, ValueError) as exc:
            if not self.cancelled.is_set():
                self.emit(frame({"error": {"message": str(exc)}}))
                self.emit(b"data: [DONE]\n\n")
        finally:
            if acquired:
                lock.release()
            self.emit(None)

    def relay(self, handler: Any) -> tuple[bytes, str]:
        said = bytearray()
        status = "complete"
        self.worker.start()
        try:
            while True:
                try:
                    block = self.events.get(timeout=0.5)
                except queue.Empty:
                    block = b": keep-alive\n\n"
                if block is None:
                    break
                said.extend(block)
                handler.wfile.write(block)
                handler.wfile.flush()
                if b'"error"' in block:
                    status = "failed"
        except (BrokenPipeError, ConnectionResetError, OSError):
            status = "cancelled"
        finally:
            self.cancel()
            self.worker.join(timeout=2)
        return bytes(said), status
