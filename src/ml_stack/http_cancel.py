"""Cancellable sockets for maintained urllib HTTP requests."""
from __future__ import annotations

import http.client
import socket
import threading
import urllib.request
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import Any

_CURRENT: ContextVar[Cancellation | None] = ContextVar('http_cancellation', default=None)


class Cancellation:
    """Close request sockets when their consumer cancels."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.sockets: list[socket.socket] = []

    def is_set(self) -> bool:
        return self.event.is_set()

    def bind(self, sock: socket.socket) -> None:
        with self.lock:
            self.sockets.append(sock)
            if self.is_set():
                self._close(sock)
                raise OSError('HTTP request cancelled')

    def set(self) -> None:
        with self.lock:
            self.event.set()
            for sock in self.sockets:
                self._close(sock)
            self.sockets.clear()

    @staticmethod
    def _close(sock: socket.socket) -> None:
        with suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        with suppress(OSError):
            sock.close()

    def connect(self, address: tuple[str, int], timeout: Any = socket._GLOBAL_DEFAULT_TIMEOUT,
                source_address: Any = None) -> socket.socket:
        for family, kind, protocol, _, target in socket.getaddrinfo(
                address[0], address[1], 0, socket.SOCK_STREAM):
            if self.is_set():
                raise OSError('HTTP request cancelled')
            sock = socket.socket(family, kind, protocol)
            self.bind(sock)
            try:
                if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                    sock.settimeout(timeout)
                if source_address:
                    sock.bind(source_address)
                sock.connect(target)
                if self.is_set():
                    raise OSError('HTTP request cancelled')
                return sock
            except OSError as exc:
                sock.close()
                last = exc
        raise last if 'last' in locals() else OSError('no addresses for HTTP endpoint')


@contextmanager
def scope(control: Cancellation):
    """Bind one request-opening thread to its cancellation handle."""
    token = _CURRENT.set(control)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def active() -> Cancellation | None:
    return _CURRENT.get()


def handlers(control: Cancellation, context: Any) -> tuple[Any, Any]:
    """HTTP handlers retaining urllib routing and the caller's TLS context."""
    class HTTP(http.client.HTTPConnection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._create_connection = control.connect

    class HTTPS(http.client.HTTPSConnection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._create_connection = control.connect

        def connect(self) -> None:
            http.client.HTTPConnection.connect(self)
            self.sock = self._context.wrap_socket(self.sock,
                server_hostname=self._tunnel_host or self.host, do_handshake_on_connect=False)
            control.bind(self.sock)
            self.sock.do_handshake()

    class Plain(urllib.request.HTTPHandler):
        def http_open(self, request: Any) -> Any:
            return self.do_open(HTTP, request)

    class Secure(urllib.request.HTTPSHandler):
        def https_open(self, request: Any) -> Any:
            return self.do_open(HTTPS, request, context=self._context)

    return Plain(), Secure(context=context)
