"""WSL discovery transport, LAN relay and rejected bridge requests."""

from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ml_stack.fleet import discovery, wsl, wsl_network


def native_socket(*, bind=None, **_options):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if bind is not None:
        sock.bind(bind)
    return sock


@pytest.fixture
def bridge(monkeypatch):
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.2", 0))
        target = reserved.getsockname()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    with wsl_network.NetworkBridge("127.0.0.1", target, discovery.DEFAULT_GROUP, port, native_socket) as service:
        monkeypatch.setenv(wsl_network.ENV, service.config)
        yield service


def test_udp_bridge_preserves_payload_and_peer_address(bridge):
    payload = b"MLD3\x00\xff encrypted beacon"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as peer:
        peer.bind(("127.0.0.1", bridge.port))
        peer.settimeout(2)
        with discovery._socket(broadcast=True, bind=("", 0)) as sock:
            assert sock.sendto(payload, peer.getsockname()) == len(payload)
            request, address = peer.recvfrom(65535)
            assert request == payload
            peer.sendto(payload[::-1], address)
            assert sock.recvfrom(65535) == (payload[::-1], peer.getsockname())


@pytest.mark.parametrize("address", [("8.8.8.8", 53), ("127.0.0.1", 22), ("192.168.2.1", 8771)])
def test_udp_bridge_refuses_non_discovery_destinations(bridge, address):
    udp = MagicMock()
    udp.__enter__.return_value = udp
    udp.sendto.return_value = 7
    bridge.factory = lambda **_options: udp
    with discovery._socket(bind=("", 0)) as sock, pytest.raises(OSError, match="Only LAN discovery"):
        sock.sendto(b"hostile", address)
    udp.sendto.assert_not_called()


def test_udp_bridge_only_replies_to_received_private_peers(bridge):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as peer:
        peer.bind(("127.0.0.1", 0))
        with discovery._socket(bind=("", bridge.port)) as sock:
            with pytest.raises(OSError):
                sock.sendto(b"before", peer.getsockname())
            peer.sendto(b"query", ("127.0.0.1", bridge.port))
            assert sock.recvfrom(65535) == (b"query", peer.getsockname())
            sock.sendto(b"reply", peer.getsockname())
            peer.settimeout(2)
            assert peer.recvfrom(65535)[0] == b"reply"


def test_udp_bridge_preserves_timeout(bridge):
    with discovery._socket(bind=("", 0)) as sock:
        sock.settimeout(0.01)
        with pytest.raises(TimeoutError):
            sock.recvfrom(65535)


@pytest.mark.parametrize("override", [{"token": "wrong"}, {"options": {"bind": ["", 22]}},
                                      {"options": {"group": "239.1.2.3"}}])
def test_bridge_refuses_authentication_and_socket_scope(bridge, override):
    config = json.loads(bridge.config)
    with socket.create_connection(tuple(config["address"]), timeout=2) as client:
        request = {"operation": "open", "token": config["token"], "options": {}, **override}
        client.sendall(json.dumps(request).encode() + b"\n")
        assert client.recv(100) == b""


def test_bridge_caps_messages_and_datagrams(bridge):
    with discovery._socket(bind=("", 0)) as sock, pytest.raises(OSError, match="Only LAN discovery"):
        sock.sendto(b"x" * 65508, ("127.0.0.1", bridge.port))
    config = json.loads(bridge.config)
    with socket.create_connection(tuple(config["address"]), timeout=2) as client:
        client.sendall(b"x" * (wsl_network.LIMIT + 1))
        assert client.recv(100) == b""


def test_lan_relay_preserves_stream_half_close_and_releases_ports(bridge):
    payload = (b"request\x00\xff" * 20000)
    with socket.socket() as upstream:
        upstream.bind(bridge.target)
        upstream.listen()

        def respond():
            connection, _ = upstream.accept()
            with connection:
                data = bytearray()
                while chunk := connection.recv(65536):
                    data.extend(chunk)
                connection.sendall(data[::-1])

        worker = threading.Thread(target=respond)
        worker.start()
        with socket.create_connection((bridge.host, bridge.target[1]), timeout=3) as client:
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            received = bytearray()
            while chunk := client.recv(65536):
                received.extend(chunk)
        worker.join(3)
        assert not worker.is_alive()
        assert received == payload[::-1]
    bridge.close()
    with socket.socket() as released:
        released.bind((bridge.host, bridge.target[1]))


def test_windows_discovery_ignores_udp_connection_reset(monkeypatch):
    sock = MagicMock()
    monkeypatch.setattr(discovery, "os", SimpleNamespace(name="nt", environ={}))
    monkeypatch.setattr(discovery.socket, "socket", lambda *_args: sock)
    disable = MagicMock()
    monkeypatch.setattr(wsl_network, "disable_udp_reset", disable)
    assert discovery._socket() is sock
    disable.assert_called_once_with(sock)


def test_windows_udp_reset_uses_winsock_control_and_reports_failure(monkeypatch):
    winsock = MagicMock()
    winsock.WSAIoctl.return_value = 0
    monkeypatch.setattr(wsl_network.ctypes, "windll", SimpleNamespace(ws2_32=winsock), raising=False)
    sock = MagicMock()
    sock.fileno.return_value = 123
    wsl_network.disable_udp_reset(sock)
    args = winsock.WSAIoctl.call_args.args
    assert args[0].value == 123
    assert args[1].value == 0x9800000C
    assert args[2]._obj.value == 0
    winsock.WSAIoctl.return_value = -1
    winsock.WSAGetLastError.return_value = 10022
    with pytest.raises(OSError, match="10022"):
        wsl_network.disable_udp_reset(sock)


def test_udp_bridge_rejects_public_reply_address(bridge):
    udp = MagicMock()
    udp.recvfrom.return_value = (b"untrusted", ("8.8.8.8", 8771))
    with pytest.raises(OSError, match="outside the local network"):
        bridge._operate(udp, {"operation": "receive", "timeout": .1, "size": 10}, set())
    udp.sendto.assert_not_called()


def test_udp_bridge_rejects_unknown_operation(bridge):
    with pytest.raises(OSError, match="Unknown discovery operation"):
        bridge._operate(MagicMock(), {"operation": "execute", "command": "hostile"}, set())


def test_wsl_launcher_cleans_bridge_when_process_creation_fails(monkeypatch):
    bridge = MagicMock()
    bridge.config = "authenticated socket configuration"
    monkeypatch.setattr(wsl, "_bridge", lambda *_args: bridge)
    monkeypatch.setattr(wsl, "command", lambda *args: list(args))
    monkeypatch.delenv("ML_STACK_HOME", raising=False)
    monkeypatch.delenv("ML_STACK_CACHE", raising=False)

    def fail(_argv, **_kwargs):
        raise OSError("child creation failed")

    monkeypatch.setattr(wsl, "launch", fail)
    with pytest.raises(OSError, match="child creation failed"):
        wsl.start([], executable="linux-python")
    bridge.close.assert_called_once()


def test_wsl_offline_runtime_does_not_require_lan_listener(monkeypatch):
    monkeypatch.setattr(discovery, "primary_ip", lambda: "")
    read = MagicMock()
    monkeypatch.setattr(wsl, "_read", read)
    assert wsl._bridge("linux-python", []) is None
    read.assert_not_called()


def test_windows_bridge_factory_uses_native_socket_with_inherited_proxy(monkeypatch):
    monkeypatch.setenv(wsl_network.ENV, "inherited Linux proxy configuration")
    monkeypatch.setattr(discovery, "primary_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(wsl, "_read", lambda *_args: "127.0.0.2")
    captured = []

    def factory(*args):
        captured.append(args[-1])
        return MagicMock()

    monkeypatch.setattr(wsl_network, "NetworkBridge", factory)
    wsl._bridge("linux-python", [])
    assert captured == [discovery._native_socket]
