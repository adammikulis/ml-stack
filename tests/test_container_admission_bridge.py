"""Stdio admission preserves socket bytes and closes failed channels."""
from __future__ import annotations

import base64
import errno
import json
import os
import selectors
import socket
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_container_bridge as bridge


@pytest.fixture
def tmp_path():
    """A directory short enough for an AF_UNIX path, whatever the worktree or worker path is."""
    import shutil
    import tempfile
    folder = Path(tempfile.mkdtemp(prefix="cb", dir="/tmp")).resolve()
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.parametrize("frame", [
    {"op": "open", "id": True}, {"op": "open", "id": 0},
    {"op": "open", "id": 1, "target": "/other"},
    {"op": "data", "id": 1, "data": "!"},
    {"op": "data", "id": 1, "data": base64.b64encode(b"x" * 65537).decode()},
    {"op": "ready", "path": "relative", "identity": [1, 2, 3]},
])
def test_frames_reject_invalid_bounds_types_and_targets(frame):
    with pytest.raises(ValueError):
        bridge.decode(json.dumps(frame).encode())


def test_payload_boundary_is_accepted():
    payload = b"x" * bridge.PAYLOAD
    frame = {"op": "data", "id": 1, "data": base64.b64encode(payload).decode()}
    assert bridge.decode(bridge.encode(frame))["data"] == payload


def test_unknown_repeated_and_excess_channel_opens_fail_closed(tmp_path):
    reader_read, reader_write = os.pipe()
    writer_read, writer_write = os.pipe()
    streams = [os.fdopen(fd, "rb" if index in (0, 2) else "wb", buffering=0)
               for index, fd in enumerate((reader_read, reader_write, writer_read, writer_write))]
    pump = bridge.Pump(streams[0], streams[3], target=(tmp_path / "none", (1, 2, 3)))
    try:
        with pytest.raises(ValueError, match="unknown"):
            pump.handle({"op": "close", "id": 1})
        pump.last_id = 3
        with pytest.raises(ValueError, match="open"):
            pump.handle({"op": "open", "id": 3})
        pump.channels = dict.fromkeys(range(bridge.CHANNELS))
        with pytest.raises(ValueError, match="open"):
            pump.handle({"op": "open", "id": 4})
        pump.channels.clear()
    finally:
        pump.selector.close()
        pump.wake_read.close()
        pump.wake_write.close()
        for stream in streams:
            stream.close()


def test_guest_roundtrip_preserves_bytes_lifetime_and_eof_cleanup(tmp_path):
    tmp_path = tmp_path.resolve()
    endpoint = tmp_path / "host.sock"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(endpoint))
    endpoint.chmod(0o600)
    server.listen(1)
    server.settimeout(5)
    process = subprocess.Popen([sys.executable, str(Path(bridge.__file__)), "--guest",
                                "--directory", str(tmp_path / "guest")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    host = bridge.HostBridge(process, endpoint, bridge.identity(endpoint),
                             (str(tmp_path / "guest" / "admission.sock"), os.getuid()))
    guest = None
    accepted = None
    try:
        path, expected = host.start()
        assert bridge.identity(path) == expected
        assert (Path(path).parent.stat().st_mode & 0o777) == 0o700
        guest = socket.socket(socket.AF_UNIX)
        guest.settimeout(5)
        guest.connect(path)
        accepted, _ = server.accept()
        accepted.settimeout(5)
        payload = b'{"token":"opaque-test-token","operation":"acquire"}\n'
        guest.sendall(payload)
        assert accepted.recv(65536) == payload
        accepted.sendall(b'{"granted":true}\n')
        assert guest.recv(65536) == b'{"granted":true}\n'
        accepted.settimeout(.05)
        with pytest.raises(TimeoutError):
            accepted.recv(1)
        guest.close()
        guest = None
        accepted.settimeout(5)
        assert accepted.recv(1) == b""
        host.close()
        process.wait(timeout=5)
        assert process.returncode == 0
        assert not Path(path).exists()
    finally:
        if guest:
            guest.close()
        if accepted:
            accepted.close()
        server.close()
        host.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stderr.close()


def test_host_endpoint_replacement_is_refused(tmp_path):
    tmp_path = tmp_path.resolve()
    endpoint = tmp_path / "host.sock"
    old = socket.socket(socket.AF_UNIX)
    new = socket.socket(socket.AF_UNIX)
    old.bind(str(endpoint))
    endpoint.chmod(0o600)
    expected = bridge.identity(endpoint)
    endpoint.unlink()
    new.bind(str(endpoint))
    endpoint.chmod(0o600)
    new.listen(1)
    reader_read, reader_write = os.pipe()
    writer_read, writer_write = os.pipe()
    streams = [os.fdopen(fd, "rb" if index in (0, 2) else "wb", buffering=0)
               for index, fd in enumerate((reader_read, reader_write, writer_read, writer_write))]
    pump = bridge.Pump(streams[0], streams[3], target=(endpoint, expected))
    try:
        with pytest.raises(ValueError, match="endpoint changed"):
            pump.handle({"op": "open", "id": 1})
        assert not pump.channels
    finally:
        pump.selector.close()
        pump.wake_read.close()
        pump.wake_write.close()
        for stream in streams:
            stream.close()
        old.close()
        new.close()


def test_nested_json_and_duplicate_fields_are_refused():
    with pytest.raises(ValueError):
        bridge.decode(b"[" * 2000 + b"0" + b"]" * 2000)
    with pytest.raises(ValueError, match="duplicate"):
        bridge.decode(b'{"op":"open","id":1,"id":2}')


def test_simultaneous_close_is_idempotent_and_data_after_close_is_refused():
    reader_read, reader_write = os.pipe()
    writer_read, writer_write = os.pipe()
    streams = [os.fdopen(fd, "rb" if index in (0, 2) else "wb", buffering=0)
               for index, fd in enumerate((reader_read, reader_write, writer_read, writer_write))]
    pump = bridge.Pump(streams[0], streams[3])
    channel, peer = socket.socketpair()
    try:
        pump.last_id = 1
        pump.add(1, channel)
        pump.drop(1)
        pump.handle({"op": "close", "id": 1})
        with pytest.raises(ValueError, match="unknown"):
            pump.handle({"op": "data", "id": 1, "data": b"x"})
        with pytest.raises(ValueError, match="unknown"):
            pump.handle({"op": "close", "id": 2})
    finally:
        pump.selector.close()
        pump.wake_read.close()
        pump.wake_write.close()
        peer.close()
        for stream in streams:
            stream.close()


def test_backlog_refusal_closes_unconnected_channel(tmp_path, monkeypatch):
    tmp_path = tmp_path.resolve()
    endpoint = tmp_path / "host.sock"
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(endpoint))
    endpoint.chmod(0o600)
    reader_read, reader_write = os.pipe()
    writer_read, writer_write = os.pipe()
    streams = [os.fdopen(fd, "rb" if index in (0, 2) else "wb", buffering=0)
               for index, fd in enumerate((reader_read, reader_write, writer_read, writer_write))]
    pump = bridge.Pump(streams[0], streams[3], target=(endpoint, bridge.identity(endpoint)))

    class Refused:
        closed = False

        def setblocking(self, value):
            assert value is False

        def connect_ex(self, path):
            assert path == str(endpoint)
            return errno.EAGAIN

        def close(self):
            self.closed = True

    refused = Refused()
    monkeypatch.setattr(bridge.socket, "socket", lambda *args: refused)
    try:
        with pytest.raises(OSError):
            pump.handle({"op": "open", "id": 1})
        assert refused.closed and not pump.channels
    finally:
        pump.selector.close()
        pump.wake_read.close()
        pump.wake_write.close()
        server.close()
        for stream in streams:
            stream.close()


def test_client_cancel_preserves_other_live_channels():
    reader_read, reader_write = os.pipe()
    writer_read, writer_write = os.pipe()
    streams = [os.fdopen(fd, "rb" if index in (0, 2) else "wb", buffering=0)
               for index, fd in enumerate((reader_read, reader_write, writer_read, writer_write))]
    pump = bridge.Pump(streams[0], streams[3])
    first, first_peer = socket.socketpair()
    second, second_peer = socket.socketpair()
    try:
        pump.add(1, first)
        pump.add(2, second)
        first_peer.close()
        pump.pending[1].extend(b"response")
        pump.channel_event(1, selectors.EVENT_WRITE)
        assert 1 not in pump.channels and 2 in pump.channels
        second_peer.sendall(b"still live")
        pump.channel_event(2, selectors.EVENT_READ)
        assert b'"id":2' in pump.outgoing
    finally:
        for channel in list(pump.channels):
            pump.drop(channel)
        pump.selector.close()
        pump.wake_read.close()
        pump.wake_write.close()
        second_peer.close()
        for stream in streams:
            stream.close()
