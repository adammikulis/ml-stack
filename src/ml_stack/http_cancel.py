"""Cancellable sockets for maintained urllib HTTP requests."""
from __future__ import annotations

import http.client
import ipaddress
import queue
import socket
import threading
import time
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_CURRENT: ContextVar[Cancellation | None] = ContextVar('http_cancellation', default=None)

_DNS_WORKERS = 2
_DNS_JOBS: queue.Queue[_Resolution] = queue.Queue(maxsize=2)
_DNS_START = threading.Lock()
_DNS_STARTED = False


@dataclass
class _Resolution:
    address: tuple[str, int]
    resolver: Callable[..., Any]
    cancelled: Callable[[], bool]
    ready: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Exception | None = None


def _resolve_worker() -> None:
    while True:
        job = _DNS_JOBS.get()
        try:
            if not job.cancelled():
                job.result = job.resolver(*job.address, 0, socket.SOCK_STREAM)
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            job.error = error
        finally:
            job.ready.set()
            _DNS_JOBS.task_done()


def _resolve(control: Cancellation, address: tuple[str, int], timeout: float | None) -> Any:
    global _DNS_STARTED
    if control.is_set():
        raise OSError('HTTP request cancelled')
    try:
        ipaddress.ip_address(address[0].split('%', 1)[0])
    except ValueError:
        pass
    else:
        return socket.getaddrinfo(*address, 0, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST)
    with _DNS_START:
        if not _DNS_STARTED:
            for index in range(_DNS_WORKERS):
                threading.Thread(target=_resolve_worker, name=f'ml-stack-dns-{index}', daemon=True).start()
            _DNS_STARTED = True
    deadline = None if timeout is None else time.monotonic() + timeout
    def wait_interval() -> float:
        if deadline is None:
            return 0.05
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('HTTP hostname resolution timed out')
        return min(0.05, remaining)
    job = _Resolution(address, socket.getaddrinfo,
        lambda: control.is_set() or (deadline is not None and time.monotonic() >= deadline))
    while not control.is_set():
        try:
            _DNS_JOBS.put(job, timeout=wait_interval())
            break
        except queue.Full:
            continue
    while not control.is_set():
        if job.ready.wait(wait_interval()):
            wait_interval()
            if job.error is not None:
                raise job.error
            return job.result
    raise OSError('HTTP request cancelled')


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
        resolver_timeout = socket.getdefaulttimeout() if timeout is socket._GLOBAL_DEFAULT_TIMEOUT else timeout
        for family, kind, protocol, _, target in _resolve(self, address, resolver_timeout):
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
