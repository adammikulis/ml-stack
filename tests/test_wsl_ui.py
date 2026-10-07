"""Owned loopback UI relays retain byte streams and release their processes."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ml_stack.fleet import wsl_ui


def _helper(monkeypatch, port):
    environment = dict(os.environ, PYTHONPATH=str(Path(wsl_ui.__file__).resolve().parents[2]))
    original = subprocess.Popen
    children = []

    def spawn(argv, **kwargs):
        assert argv == [sys.executable, "-m", "ml_stack.fleet.wsl_ui", str(port)]
        assert not kwargs.get("shell")
        child = original(argv, env=environment, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(wsl_ui, "start_process", spawn)
    return [sys.executable, "-m", "ml_stack.fleet.wsl_ui", str(port)], children


def test_local_ui_relay_preserves_bytes_and_actual_loopback_peer(monkeypatch):
    with socket.socket() as upstream:
        upstream.bind(("127.0.0.1", 0))
        upstream.listen()
        upstream.settimeout(10)
        received = []

        def respond():
            client, address = upstream.accept()
            with client:
                data = bytearray()
                while chunk := client.recv(65536):
                    data.extend(chunk)
                received.append((address, bytes(data)))
                client.sendall(bytes(data)[::-1])

        server = threading.Thread(target=respond, daemon=True)
        server.start()
        argv, children = _helper(monkeypatch, upstream.getsockname()[1])
        relay = wsl_ui.LocalUIBridge(argv, 0).start()
        try:
            listener = relay.listeners[0].getsockname()
            assert listener[0] == "127.0.0.1"
            payload = b"POST /ui/setup/local-session HTTP/1.1\r\nHost: localhost\r\n\r\n\x00\xff; $(touch injected)" * 1000
            with socket.create_connection(listener, timeout=10) as client:
                client.sendall(payload)
                client.shutdown(socket.SHUT_WR)
                reply = bytearray()
                while chunk := client.recv(65536):
                    reply.extend(chunk)
            assert bytes(reply) == payload[::-1]
            server.join(10)
            assert received == [(("127.0.0.1", received[0][0][1]), payload)]
        finally:
            relay.close()
        for child in children:
            assert child.wait(timeout=10) == 0
        with socket.socket() as replacement:
            replacement.bind(listener)


def test_local_ui_close_terminates_idle_helper_and_releases_slot(monkeypatch):
    with socket.socket() as upstream:
        upstream.bind(("127.0.0.1", 0))
        upstream.listen()
        upstream.settimeout(10)
        argv, children = _helper(monkeypatch, upstream.getsockname()[1])
        relay = wsl_ui.LocalUIBridge(argv, 0).start()
        with socket.create_connection(relay.listeners[0].getsockname(), timeout=10) as client:
            with upstream.accept()[0]:
                relay.close()
                assert client.recv(1) == b""
            for child in children:
                child.wait(timeout=10)
                assert child.poll() is not None
        deadline = time.monotonic() + 10
        while relay.clients and time.monotonic() < deadline:
            time.sleep(.01)
        assert not relay.clients
        assert not relay.processes


def test_local_ui_rejects_connections_over_admission_limit(monkeypatch):
    relay = wsl_ui.LocalUIBridge(["unstarted-helper"], 0)
    for _ in range(32):
        assert relay.slots.acquire(blocking=False)
    monkeypatch.setattr(wsl_ui, "start_process", lambda *_a, **_k: pytest.fail("helper exceeded admission"))
    relay.start()
    try:
        with socket.create_connection(relay.listeners[0].getsockname(), timeout=5) as client:
            assert client.recv(1) == b""
    finally:
        relay.close()
        for _ in range(32):
            relay.slots.release()


def test_local_ui_transport_keeps_silent_connections_unbounded(monkeypatch):
    import io
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    monkeypatch.setattr(wsl_ui.threading, "Thread", lambda target, args=(), **_kwargs:
                        SimpleNamespace(start=lambda: target(*args), join=lambda _timeout: None))
    upstream = MagicMock()
    upstream.__enter__.return_value = upstream
    upstream.recv.side_effect = [b"delayed response", b""]
    monkeypatch.setattr(wsl_ui.socket, "create_connection", lambda *_args, **_kwargs: upstream)
    output = io.BytesIO()
    monkeypatch.setattr(wsl_ui.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"request")))
    monkeypatch.setattr(wsl_ui.sys, "stdout", SimpleNamespace(buffer=output))
    wsl_ui.stdio(12345)
    upstream.settimeout.assert_called_once_with(None)
    assert output.getvalue() == b"delayed response"

    client = MagicMock()
    client.recv.return_value = b""
    child = MagicMock(stdin=io.BytesIO(), stdout=io.BytesIO(b"delayed response"))
    child.wait.return_value = 0
    monkeypatch.setattr(wsl_ui, "start_process", lambda *_args, **_kwargs: child)
    relay = wsl_ui.LocalUIBridge(["owned-helper"], 0)
    relay._relay(client)
    client.settimeout.assert_called_once_with(None)
    client.sendall.assert_called_once_with(b"delayed response")
