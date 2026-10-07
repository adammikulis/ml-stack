"""Windows LAN discovery sockets and TCP forwarding for a WSL daemon."""

from __future__ import annotations

import base64
import contextlib
import ctypes
import hmac
import ipaddress
import json
import os
import secrets
import select
import socket
import threading
from collections.abc import Callable
from typing import Any

from .onboard.lan import require_local

ENV = "ML_STACK_WSL_NETWORK"
LIMIT = 100_000


def disable_udp_reset(sock: socket.socket) -> None:
    """Disable Windows UDP resets from unreachable discovery destinations."""
    value, returned = ctypes.c_ulong(0), ctypes.c_ulong(0)
    winsock = ctypes.windll.ws2_32
    result = winsock.WSAIoctl(ctypes.c_size_t(sock.fileno()), ctypes.c_ulong(0x9800000C),
                             ctypes.byref(value), ctypes.sizeof(value), None, 0,
                             ctypes.byref(returned), None, None)
    if result:
        raise OSError(winsock.WSAGetLastError(), "Windows could not disable UDP connection resets")


def _write(stream: Any, value: dict) -> None:
    stream.write(json.dumps(value).encode() + b"\n")
    stream.flush()


def _read(stream: Any) -> dict:
    raw = stream.readline(LIMIT + 1)
    if not raw or len(raw) > LIMIT:
        raise OSError("WSL LAN bridge closed or exceeded its message limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise OSError("Invalid WSL LAN bridge message")
    return value


class DiscoverySocket:
    """A discovery UDP socket on the Windows LAN interface."""

    def __init__(self, options: dict) -> None:
        config = json.loads(os.environ[ENV])
        address = tuple(config["address"])
        require_local(*address)
        self.connection = socket.create_connection(address, timeout=5)
        self.stream = self.connection.makefile("rwb")
        self.lock = threading.Lock()
        self.timeout = 2.0
        self.closed = False
        try:
            self._call("open", token=config["token"], options=options)
        except (OSError, ValueError):
            self.close()
            raise

    def _call(self, operation: str, **values: Any) -> dict:
        with self.lock:
            _write(self.stream, {"operation": operation, **values})
            reply = _read(self.stream)
        if reply.get("timeout"):
            raise TimeoutError("Discovery receive timed out")
        if reply.get("error"):
            raise OSError(reply["error"])
        return reply

    def sendto(self, data: bytes, address: tuple[str, int]) -> int:
        return int(self._call("send", data=base64.b64encode(data).decode(), address=address)["sent"])

    def recvfrom(self, size: int) -> tuple[bytes, tuple[str, int]]:
        reply = self._call("receive", size=size, timeout=self.timeout)
        return base64.b64decode(reply["data"]), (str(reply["address"][0]), int(reply["address"][1]))

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def setsockopt(self, level: int, option: int, _value: Any) -> None:
        if (level, option) != (socket.IPPROTO_IP, socket.IP_MULTICAST_IF):
            raise OSError("Only the Windows LAN multicast interface is supported")

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)
            self.stream.close()
            self.connection.close()

    def __enter__(self) -> DiscoverySocket:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _forward(client: socket.socket, target: tuple[str, int], stop: threading.Event) -> None:
    with client:
        try:
            require_local(*target)
            with socket.create_connection(target, timeout=3) as upstream:
                sockets = [client, upstream]
                while sockets and not stop.is_set():
                    ready, _, _ = select.select(sockets, [], [], 0.25)
                    for source in ready:
                        destination = upstream if source is client else client
                        data = source.recv(65536)
                        if data:
                            destination.sendall(data)
                        else:
                            sockets.remove(source)
                            destination.shutdown(socket.SHUT_WR)
        except OSError:
            return


class NetworkBridge:
    """Discovery and daemon listeners held by the Windows launcher."""

    def __init__(self, host: str, target: tuple[str, int], group: str, port: int,
                 factory: Callable[..., socket.socket | DiscoverySocket]) -> None:
        self.host, self.target, self.group, self.port = host, target, group, port
        self.factory = factory
        self.gateway = ""
        self.token = secrets.token_hex(32)
        self.stop = threading.Event()
        self.listeners: list[socket.socket] = []
        self.clients: set[socket.socket] = set()
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(32)
        self.threads: list[threading.Thread] = []

    def start(self) -> NetworkBridge:
        try:
            control = self._listen(0, self._discovery)
            def forward(client):
                _forward(client, self.target, self.stop)
            self._listen(self.target[1], forward)
            if self.gateway and self.gateway != self.host:
                self._listen(self.target[1], forward, host=self.gateway)
            self.config = json.dumps({"address": control.getsockname(), "token": self.token})
            return self
        except OSError:
            self.close()
            raise

    def _listen(self, port: int, handler: Callable[[socket.socket], None], *, host: str = "") -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listeners.append(listener)
        listener.bind((host or self.host, port))
        listener.listen(32)
        listener.settimeout(0.25)
        thread = threading.Thread(target=self._accept, args=(listener, handler), daemon=True)
        self.threads.append(thread)
        thread.start()
        return listener

    def _accept(self, listener: socket.socket, handler: Callable[[socket.socket], None]) -> None:
        while not self.stop.is_set():
            try:
                client, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if not self.slots.acquire(blocking=False):
                client.close()
                continue
            with self.lock:
                self.clients.add(client)
            threading.Thread(target=self._handle, args=(client, handler), daemon=True).start()

    def _handle(self, client: socket.socket, handler: Callable[[socket.socket], None]) -> None:
        try:
            handler(client)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        finally:
            client.close()
            with self.lock:
                self.clients.discard(client)
            self.slots.release()

    def _discovery(self, client: socket.socket) -> None:
        client.settimeout(5)
        with client.makefile("rwb") as stream:
            request = _read(stream)
            if request.get("operation") != "open" or not hmac.compare_digest(
                    str(request.get("token", "")), self.token):
                return
            options = request["options"]
            bind = tuple(options["bind"]) if options.get("bind") is not None else None
            if bind is not None and (bind[0] != "" or bind[1] not in {0, self.port}):
                return
            if options.get("group") not in {None, self.group}:
                return
            client.settimeout(None)
            with self.factory(**{**options, "bind": bind}) as udp:
                udp.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(self.host))
                allowed = dict.fromkeys(((self.group, self.port), ("255.255.255.255", self.port), ("127.0.0.1", self.port)))
                _write(stream, {})
                while not self.stop.is_set():
                    request = _read(stream)
                    try:
                        reply = self._operate(udp, request, allowed)
                    except TimeoutError:
                        reply = {"timeout": True}
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        reply = {"error": str(exc)}
                    _write(stream, reply)

    def _operate(self, udp: socket.socket | DiscoverySocket, request: dict, allowed: dict) -> dict:
        if request["operation"] == "send":
            address = tuple(request["address"])
            data = base64.b64decode(request["data"], validate=True)
            if address not in allowed or len(data) > 65507:
                raise OSError("Only LAN discovery destinations and replies are allowed")
            return {"sent": udp.sendto(data, address)}
        if request["operation"] != "receive":
            raise OSError("Unknown discovery operation")
        udp.settimeout(max(0.001, min(float(request["timeout"]), 2)))
        data, address = udp.recvfrom(max(1, min(int(request["size"]), 65535)))
        if not ipaddress.ip_address(address[0]).is_private:
            raise OSError("Discovery reply is outside the local network")
        fixed = {(self.group, self.port), ("255.255.255.255", self.port), ("127.0.0.1", self.port)}
        if address not in fixed:
            allowed.pop(address, None)
            allowed[address] = None
            if len(allowed) > 128:
                oldest = next(peer for peer in allowed if peer not in fixed)
                del allowed[oldest]
        return {"data": base64.b64encode(data).decode(), "address": address}

    def close(self) -> None:
        self.stop.set()
        for listener in self.listeners:
            listener.close()
        with self.lock:
            for client in self.clients:
                with contextlib.suppress(OSError):
                    client.shutdown(socket.SHUT_RDWR)
        for thread in self.threads:
            thread.join(1)

    def __enter__(self) -> NetworkBridge:
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()
