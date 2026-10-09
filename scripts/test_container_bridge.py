"""Bounded stdio transport for the supervisor admission socket."""
from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import errno
import json
import os
import selectors
import socket
import sys
import threading
from pathlib import Path

PAYLOAD = 65536
FRAME = 88000
CHANNELS = 128
BUFFER = FRAME * 4


def identity(path):
    from test_kernel_endpoint import socket_identity
    return socket_identity(str(path))


def encode(frame):
    data = json.dumps(frame, separators=(",", ":")).encode() + b"\n"
    if len(data) > FRAME:
        raise ValueError("bridge frame exceeds bound")
    return data


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate bridge field")
        result[key] = value
    return result


def decode(raw):
    try:
        frame = json.loads(raw, object_pairs_hook=unique_fields)
    except RecursionError as exc:
        raise ValueError("bridge frame nesting exceeds bound") from exc
    if not isinstance(frame, dict) or not isinstance(frame.get("op"), str) or frame.get("op") not in {"ready", "open", "data", "close"}:
        raise ValueError("invalid bridge frame")
    op = frame["op"]
    keys = {"ready": {"op", "path", "identity"}, "open": {"op", "id"},
            "data": {"op", "id", "data"}, "close": {"op", "id"}}[op]
    if set(frame) != keys:
        raise ValueError("invalid bridge frame fields")
    if op == "ready":
        values = frame["identity"]
        if (not isinstance(frame["path"], str) or not Path(frame["path"]).is_absolute() or len(frame["path"]) > 1024 or "\x00" in frame["path"]
                or not isinstance(values, list) or len(values) != 3
                or any(type(v) is not int or v < 0 for v in values)):
            raise ValueError("invalid endpoint handshake")
    elif type(frame["id"]) is not int or not 0 < frame["id"] < 2**63:
        raise ValueError("invalid channel identity")
    if op == "data":
        if not isinstance(frame["data"], str) or len(frame["data"]) > 4 * ((PAYLOAD + 2) // 3):
            raise ValueError("channel data exceeds bound")
        try:
            data = base64.b64decode(frame["data"], validate=True)
        except (TypeError, ValueError, binascii.Error) as exc:
            raise ValueError("invalid channel data") from exc
        if not data or len(data) > PAYLOAD:
            raise ValueError("channel data exceeds bound")
        frame["data"] = data
    return frame


class Pump:
    def __init__(self, reader, writer, *, target=None, listener=None, ready=None):
        self.reader, self.writer = reader, writer
        self.endpoint, self.expected = target or (None, None)
        self.listener, self.ready = listener, ready
        self.selector = selectors.DefaultSelector()
        self.channels = {}
        self.pending = {}
        self.closing = set()
        self.connecting = set()
        self.incoming = bytearray()
        self.outgoing = bytearray()
        self.last_id = 0
        self.stop = threading.Event()
        self.failure = None
        self.wake_read, self.wake_write = socket.socketpair()
        for fd in (reader.fileno(), writer.fileno()):
            os.set_blocking(fd, False)
        self.selector.register(reader, selectors.EVENT_READ, "input")
        self.selector.register(self.wake_read, selectors.EVENT_READ, "wake")
        if listener:
            listener.setblocking(False)
            self.selector.register(listener, selectors.EVENT_READ, "accept")

    def emit(self, frame):
        data = encode(frame)
        if len(self.outgoing) + len(data) > BUFFER:
            raise ValueError("bridge output backpressure exceeded")
        self.outgoing.extend(data)
        try:
            self.selector.modify(self.writer, selectors.EVENT_WRITE, "output")
        except KeyError:
            self.selector.register(self.writer, selectors.EVENT_WRITE, "output")

    def add(self, channel, conn):
        if len(self.channels) >= CHANNELS:
            conn.close()
            raise ValueError("bridge channel limit exceeded")
        conn.setblocking(False)
        self.channels[channel] = conn
        self.pending[channel] = bytearray()
        self.selector.register(conn, selectors.EVENT_READ, channel)

    def drop(self, channel):
        conn = self.channels.pop(channel)
        self.pending.pop(channel)
        self.closing.discard(channel)
        self.connecting.discard(channel)
        self.selector.unregister(conn)
        conn.close()

    def handle(self, frame):
        op = frame["op"]
        if op == "ready":
            if self.listener or self.ready is None:
                raise ValueError("unexpected endpoint handshake")
            callback, self.ready = self.ready, None
            callback(frame)
            return
        if self.ready is not None:
            raise ValueError("endpoint handshake missing")
        channel = frame["id"]
        if op == "open":
            if self.listener or channel != self.last_id + 1 or len(self.channels) >= CHANNELS:
                raise ValueError("invalid channel open")
            self.last_id = channel
            if identity(self.endpoint) != self.expected:
                raise ValueError("host admission endpoint changed")
            conn = socket.socket(socket.AF_UNIX)
            try:
                conn.setblocking(False)
                code = conn.connect_ex(str(self.endpoint))
                if code not in (0, errno.EINPROGRESS):
                    raise OSError(code, "admission connection failed")
                self.add(channel, conn)
                self.connecting.add(channel)
                self.selector.modify(conn, selectors.EVENT_WRITE, channel)
            except BaseException:
                conn.close()
                raise
            return
        if channel not in self.channels:
            if op == "close" and channel <= self.last_id:
                return
            raise ValueError("unknown bridge channel")
        if channel in self.closing:
            raise ValueError("closed bridge channel")
        if op == "close":
            if self.pending[channel]:
                self.closing.add(channel)
                self.selector.modify(self.channels[channel], selectors.EVENT_WRITE, channel)
            else:
                self.drop(channel)
        else:
            pending = self.pending[channel]
            if len(pending) + len(frame["data"]) > BUFFER:
                raise ValueError("channel backpressure exceeded")
            pending.extend(frame["data"])
            self.selector.modify(self.channels[channel], selectors.EVENT_READ | selectors.EVENT_WRITE, channel)

    def input(self):
        chunk = os.read(self.reader.fileno(), PAYLOAD)
        if not chunk:
            return False
        self.incoming.extend(chunk)
        while b"\n" in self.incoming:
            if self.stop.is_set():
                return False
            raw, _, remainder = self.incoming.partition(b"\n")
            if len(raw) + 1 > FRAME:
                raise ValueError("bridge frame exceeds bound")
            self.incoming[:] = remainder
            self.handle(decode(raw))
        if len(self.incoming) >= FRAME:
            raise ValueError("bridge frame exceeds bound")
        return True

    def channel_event(self, kind, events):
        try:
            conn = self.channels[kind]
            if kind in self.connecting:
                if conn.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR):
                    raise ConnectionError("admission connection failed")
                if identity(self.endpoint) != self.expected:
                    raise ValueError("host admission endpoint changed")
                self.connecting.remove(kind)
                self.selector.modify(conn, selectors.EVENT_READ | (selectors.EVENT_WRITE if self.pending[kind] else 0), kind)
            if events & selectors.EVENT_WRITE:
                pending = self.pending[kind]
                count = conn.send(pending)
                del pending[:count]
                if not pending:
                    if kind in self.closing:
                        self.drop(kind)
                        return
                    self.selector.modify(conn, selectors.EVENT_READ, kind)
            if events & selectors.EVENT_READ:
                data = conn.recv(PAYLOAD)
                if data:
                    self.emit({"op": "data", "id": kind,
                               "data": base64.b64encode(data).decode("ascii")})
                else:
                    self.emit({"op": "close", "id": kind})
                    self.drop(kind)
        except (BrokenPipeError, ConnectionResetError):
            self.emit({"op": "close", "id": kind})
            self.drop(kind)

    def run(self):
        try:
            while not self.stop.is_set():
                for key, events in self.selector.select():
                    if self.stop.is_set():
                        return
                    kind = key.data
                    if kind == "wake":
                        return
                    if kind == "input":
                        if not self.input():
                            return
                    elif kind == "output":
                        count = os.write(self.writer.fileno(), self.outgoing)
                        del self.outgoing[:count]
                        if not self.outgoing:
                            self.selector.unregister(self.writer)
                    elif kind == "accept":
                        conn, _ = self.listener.accept()
                        self.last_id += 1
                        self.add(self.last_id, conn)
                        self.emit({"op": "open", "id": self.last_id})
                    elif kind in self.channels:
                        self.channel_event(kind, events)
        except (OSError, ValueError, TypeError) as exc:
            self.failure = type(exc).__name__
        finally:
            for channel in list(self.channels):
                self.drop(channel)
            self.selector.close()
            self.wake_read.close()
            self.wake_write.close()

    def close(self):
        self.stop.set()
        with contextlib.suppress(OSError):
            self.wake_write.send(b"x")


class HostBridge:
    def __init__(self, process, endpoint, expected, expected_guest):
        self.process = process
        self.closed = False
        self.handshake = threading.Event()
        self.guest = None
        self.expected_guest = expected_guest
        self.pump = Pump(process.stdout, process.stdin, target=(Path(endpoint), tuple(expected)), ready=self.receive_ready)
        self.thread = threading.Thread(target=self.pump.run, daemon=True)

    def receive_ready(self, frame):
        if (frame["path"], frame["identity"][2]) != self.expected_guest:
            raise ValueError("guest admission endpoint changed")
        self.guest = (frame["path"], tuple(frame["identity"]))
        self.handshake.set()

    def start(self, timeout=10):
        self.thread.start()
        if not self.handshake.wait(timeout) or self.pump.failure:
            self.close()
            raise RuntimeError("container admission bridge handshake failed")
        return self.guest

    def close(self):
        if self.closed:
            return
        self.pump.close()
        if self.thread.ident:
            self.thread.join(2)
            if self.thread.is_alive():
                raise RuntimeError("container admission bridge did not stop")
        else:
            self.pump.run()
        self.closed = True
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()


def guest(directory):
    directory = Path(directory)
    if not directory.is_absolute() or directory.parent.resolve() != directory.parent:
        raise ValueError("guest socket directory is not normalised")
    directory.mkdir(mode=0o700)
    endpoint = directory / "admission.sock"
    listener = socket.socket(socket.AF_UNIX)
    try:
        listener.bind(str(endpoint))
        endpoint.chmod(0o600)
        listener.listen(CHANNELS)
        pump = Pump(sys.stdin.buffer, sys.stdout.buffer, listener=listener)
        pump.emit({"op": "ready", "path": str(endpoint), "identity": list(identity(endpoint))})
        pump.run()
        return 1 if pump.failure else 0
    finally:
        listener.close()
        endpoint.unlink(missing_ok=True)
        directory.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--guest", action="store_true", required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    raise SystemExit(guest(args.directory))
