"""A busy test admission server must not kill the runs waiting for it.

With dozens of runs queued, the server's accept queue filled and a connect timed out after five
seconds, which pytest reported as an internal error and aborted the whole run (observed with 36 runs
ahead). The queue is now large, and a client waits for a busy server, but only for a timeout."""

from __future__ import annotations

import importlib.util
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "scripts"))


endpoint = load("test_kernel_endpoint")


def saturated_listener():
    """A listener nobody accepts from, filled until the next connect times out (as the real one did)."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    held = []
    for _ in range(8):
        probe = socket.socket()
        probe.settimeout(0.3)
        try:
            probe.connect(listener.getsockname())
            held.append(probe)
        except OSError:
            probe.close()
            return listener, held
    raise AssertionError("the accept queue never filled, so no connect timed out")


def address(listener) -> str:
    host, port = listener.getsockname()
    return f"{host}:{port}"


def test_a_full_accept_queue_is_a_timeout_that_a_patient_client_outwaits():
    listener, held = saturated_listener()
    try:
        with pytest.raises(TimeoutError):
            endpoint.connection(address(listener), attempt=0.3, patience=0)

        def drain_later():
            time.sleep(1.5)
            listener.settimeout(0.5)
            for _ in range(len(held) + 12):
                try:
                    listener.accept()[0].close()
                except OSError:
                    return

        threading.Thread(target=drain_later, daemon=True).start()
        began = time.monotonic()
        stream = endpoint.connection(address(listener), attempt=0.3, patience=30)
        stream.close()
        assert time.monotonic() - began < 25
    finally:
        for probe in held:
            probe.close()
        listener.close()


def test_a_server_that_never_drains_is_given_up_on_with_a_clear_message():
    listener, held = saturated_listener()
    try:
        with pytest.raises(TimeoutError, match="accepted no connection in 2 s"):
            endpoint.connection(address(listener), attempt=0.3, patience=2)
    finally:
        for probe in held:
            probe.close()
        listener.close()


def test_a_refused_connection_fails_at_once_and_is_not_waited_on():
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    began = time.monotonic()
    with pytest.raises(ConnectionRefusedError):
        endpoint.connection(f"127.0.0.1:{port}", patience=60)
    assert time.monotonic() - began < 30


def test_a_bad_endpoint_identity_is_refused_at_once_and_is_not_waited_on():
    began = time.monotonic()
    with pytest.raises(PermissionError, match="identity is missing"):
        endpoint.connection("unix:/nonexistent/socket", None, patience=60)
    with pytest.raises(PermissionError, match="invalid endpoint identity"):
        endpoint.connection("unix:/nonexistent/socket", "[1, 2]", patience=60)
    assert time.monotonic() - began < 30


def test_the_server_queues_more_connections_than_socketserver_defaults_to():
    admission = load("testslots_rpc")
    assert admission.Admission.request_queue_size >= 128
    assert admission.UnixAdmission.request_queue_size >= 128
