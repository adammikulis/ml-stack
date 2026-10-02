"""Finding machines that are willing to be paired with, before there is any trust.

A machine with pairing open announces itself: name, hostname, model, the port of its pairing
listener and the SHA-256 fingerprint of its certificate. Nothing in an announcement is
authenticated -- the machine asking has no key yet -- so it is treated as a hint and nothing
more: the address is the datagram's source, never a claimed one; every string is cleaned; the
list is bounded and entries age out. A forged announcement can put a wrong name in the list;
it cannot get anyone paired, because pairing is decided by the code and the certificate the
exchange binds (`pairing.py`).

The wire is a `Transport`. `UdpTransport` is the real one (multicast and broadcast, both
stdlib); `MemoryHub` stands in for it in tests that need no network at all. An mDNS/DNS-SD
transport (the `zeroconf` package) fits the same two methods; see docs/onboarding.md for why
the stdlib transport is the default.
"""

from __future__ import annotations

import contextlib
import json
import re
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .events import BUS, Bus
from .requests import clean

__all__ = ["Announcer", "Browser", "MemoryHub", "Nearby", "Transport", "UdpTransport",
           "DEFAULT_GROUP", "DEFAULT_PORT"]

DEFAULT_GROUP = "239.255.77.70"
DEFAULT_PORT = 8773
MOST_DATAGRAM = 1200
KIND = "pair-open"
FINGERPRINT = re.compile(r"[0-9a-f]{64}")


class Transport(Protocol):
    """Datagrams out to everyone listening, and in from anyone, each with its source."""

    def send(self, data: bytes) -> None: ...

    def receive(self, timeout_s: float) -> tuple[bytes, str] | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Nearby:
    name: str
    hostname: str
    model: str
    address: str
    port: int
    fingerprint: str
    seen: float

    def public(self) -> dict[str, object]:
        return {"name": self.name, "hostname": self.hostname, "model": self.model,
                "address": self.address, "port": self.port, "fingerprint": self.fingerprint,
                "fingerprint_short": " ".join(self.fingerprint[i:i + 4] for i in range(0, 16, 4))}


def encode(*, name: str, hostname: str, model: str, port: int, fingerprint: str) -> bytes:
    return json.dumps({"v": 1, "kind": KIND, "name": name, "hostname": hostname,
                       "model": model, "port": port, "fingerprint": fingerprint},
                      sort_keys=True, separators=(",", ":")).encode()


def decode(data: bytes, source: str, now: float) -> Nearby | None:
    """The announcement in ``data`` from ``source``, or None if it is anything else."""
    if len(data) > MOST_DATAGRAM:
        return None
    try:
        msg = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(msg, dict) or msg.get("v") != 1 or msg.get("kind") != KIND:
        return None
    port, fingerprint = msg.get("port"), msg.get("fingerprint")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        return None
    if not isinstance(fingerprint, str) or not FINGERPRINT.fullmatch(fingerprint):
        return None
    return Nearby(name=clean(msg.get("name")) or "unnamed", hostname=clean(msg.get("hostname")),
                  model=clean(msg.get("model")), address=source, port=port,
                  fingerprint=fingerprint, seen=now)


class Announcer:
    """Says "I will pair" every ``interval_s`` seconds until stopped."""

    def __init__(self, transport: Transport, *, name: str, hostname: str, model: str, port: int,
                 fingerprint: str, interval_s: float = 5.0) -> None:
        self.transport, self.interval_s = transport, interval_s
        self.data = encode(name=name, hostname=hostname, model=model, port=port,
                           fingerprint=fingerprint)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def announce(self) -> None:
        self.transport.send(self.data)

    def start(self) -> Announcer:
        def loop() -> None:
            while not self._stop.is_set():
                with contextlib.suppress(OSError):
                    self.announce()
                self._stop.wait(self.interval_s)
        self._thread = threading.Thread(target=loop, daemon=True, name="onboard-announce")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class Browser:
    """The list of machines heard, bounded and aging."""

    def __init__(self, transport: Transport, *, ttl_s: float = 30.0, most: int = 64,
                 per_source: int = 4, ignore: frozenset[str] = frozenset(), bus: Bus = BUS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.transport, self.ttl_s, self.most, self.per_source = transport, ttl_s, most, per_source
        self.ignore, self.bus, self.clock = ignore, bus, clock
        self._found: dict[str, Nearby] = {}

    def _admit(self, found: Nearby) -> None:
        if found.fingerprint in self.ignore:
            return
        mine = [n for n in self._found.values() if n.address == found.address
                and n.fingerprint != found.fingerprint]
        if len(mine) >= self.per_source:
            self.bus.emit("onboard.nearby.flood", "notice", "", address=found.address)
            return
        if found.fingerprint not in self._found and len(self._found) >= self.most:
            return
        self._found[found.fingerprint] = found

    def listen(self, timeout_s: float) -> list[Nearby]:
        """Hear announcements for ``timeout_s`` seconds; returns who is still current."""
        end = time.monotonic() + timeout_s
        while (left := end - time.monotonic()) > 0:
            got = self.transport.receive(left)
            if got is None:
                continue
            found = decode(got[0], got[1], self.clock())
            if found is not None:
                self._admit(found)
        return self.current()

    def current(self) -> list[Nearby]:
        now = self.clock()
        self._found = {k: v for k, v in self._found.items() if now - v.seen <= self.ttl_s}
        return sorted(self._found.values(), key=lambda n: (n.name, n.address))


# -- transports ---------------------------------------------------------------------------
class MemoryHub:
    """Endpoints that hear each other's datagrams, with no network. An endpoint hears every
    other's sends, not its own, as multicast loopback is switched off for a listener."""

    def __init__(self) -> None:
        self._ends: list[_Memory] = []

    def endpoint(self, address: str) -> Transport:
        end = _Memory(self, address)
        self._ends.append(end)
        return end


class _Memory:
    def __init__(self, hub: MemoryHub, address: str) -> None:
        self.hub, self.address = hub, address
        self._queue: deque[tuple[bytes, str]] = deque()
        self._ready = threading.Condition()

    def send(self, data: bytes) -> None:
        for end in self.hub._ends:
            if end is not self:
                with end._ready:
                    end._queue.append((data, self.address))
                    end._ready.notify()

    def receive(self, timeout_s: float) -> tuple[bytes, str] | None:
        with self._ready:
            if not self._queue:
                self._ready.wait(timeout_s)
            return self._queue.popleft() if self._queue else None

    def close(self) -> None:
        return None


class UdpTransport:
    """UDP on one port. ``destinations`` are where `send` goes: by default the multicast group
    and the broadcast address; a test names loopback addresses instead. With ``group`` the
    socket joins that multicast group to hear it. TTL is 1: it stays on the link."""

    def __init__(self, *, bind: str = "", port: int = DEFAULT_PORT,
                 destinations: list[tuple[str, int]] | None = None,
                 group: str | None = None) -> None:
        self.destinations = destinations if destinations is not None else [
            (DEFAULT_GROUP, port), ("255.255.255.255", port)]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        sock.bind((bind, port))
        if group is not None:
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                                struct.pack("4sl", socket.inet_aton(group), socket.INADDR_ANY))
        self.sock = sock

    @property
    def port(self) -> int:
        return int(self.sock.getsockname()[1])

    def send(self, data: bytes) -> None:
        for dest in self.destinations:
            with contextlib.suppress(OSError):
                self.sock.sendto(data, dest)

    def receive(self, timeout_s: float) -> tuple[bytes, str] | None:
        try:
            self.sock.settimeout(max(0.01, timeout_s))
            data, addr = self.sock.recvfrom(65535)
        except (TimeoutError, OSError):
            return None
        return data, addr[0]

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()
