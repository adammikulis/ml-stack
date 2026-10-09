"""The join request: a stranger's ask, the owner's answer, and the limits around the ask.

States: pending -> accepted (a person said yes; the code exists and counts down) -> paired,
or declined, expired, failed (too many wrong codes). The store is a file, so the process that
listens and the command a person types (accept, decline) see one state. Limits, each a test:
one active request per fingerprint and per address; a window per address and a ceiling on
pending requests; a declined fingerprint waits, three declines in a row block it an hour; an
unanswered request expires, an untyped code sooner; three tries at the code; a revoked device
cannot ask again until the owner allows it.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from poolhouse.desktop import clean
from poolhouse.files import read_json, write_json
from poolhouse.lock import only_one
from poolhouse.platform import private_file

from .events import BUS, Bus
from .lan import in_tailnet

__all__ = ["Device", "Devices", "Limits", "Refused", "Request", "Requests", "State", "clean"]

SCHEMA_VERSION = 1
FINGERPRINT = re.compile(r"[0-9a-f]{64}")
CODE_DIGITS = 6


class State(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"
    PAIRED = "paired"
    FAILED = "failed"


ACTIVE = (State.PENDING, State.ACCEPTED)


class Refused(Exception):
    """A request turned away: the HTTP status to answer with and a reason safe to show."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


@dataclass(frozen=True, slots=True)
class Limits:
    pending_ttl_s: float = 300.0
    code_ttl_s: float = 120.0
    max_pending: int = 8
    window_s: float = 600.0
    per_address: int = 5
    attempts: int = 3
    decline_wait_s: float = 60.0
    strikes: int = 3
    block_s: float = 3600.0
    failed_wait_s: float = 900.0
    keep_s: float = 3600.0
    keep_most: int = 256


@dataclass(slots=True)
class Request:
    id: str
    name: str
    hostname: str
    model: str
    address: str
    fingerprint: str
    nonce: str
    created: float
    state: State = State.PENDING
    code: str = ""
    code_expires: float = 0.0
    attempts: int = 0
    decided: float = 0.0
    note: str = ""
    mine: bool | None = None
    """Whether the owner said the asking device is theirs; set when they accept."""

    def public(self) -> dict[str, Any]:
        """What a person sees, and what an API may return: never the code."""
        out = asdict(self)
        out.pop("code")
        out["state"] = self.state.value
        out["fingerprint_short"] = short(self.fingerprint)
        return out


def short(fingerprint: str) -> str:
    """A fingerprint a person can compare: the first 16 hex digits in groups of four."""
    return " ".join(fingerprint[i:i + 4] for i in range(0, 16, 4))


@dataclass(slots=True)
class Device:
    fingerprint: str
    name: str
    hostname: str
    address: str
    paired: float
    status: str = "active"
    revoked: float = 0.0
    shared_cluster_key: bool = False
    mine: bool = False
    """Whether the owner said this device is theirs, as against another person's."""
    secret: str = ""
    """The key this device signs its file requests with (urlsafe base64); private file."""
    tailnet_address: str = ""
    """Where this device is on the tailnet, learned only from the pairing exchange or from a
    Tailscale peer whose certificate matched this device's (`routes.learn`)."""


AUTHENTICATED_SOURCES = frozenset({"pairing", "tailscale-verified"})


@dataclass(slots=True)
class _Ledger:
    requests: list[Request] = field(default_factory=list)
    asks: dict[str, list[float]] = field(default_factory=dict)
    declines: dict[str, list[float]] = field(default_factory=dict)
    waits: dict[str, float] = field(default_factory=dict)


class Devices:
    """The machines this one has paired with, and the ones the owner has revoked."""

    def __init__(self, path: Path, *, bus: Bus = BUS,
                 clock: Callable[[], float] = time.time) -> None:
        self.path, self.bus, self.clock = Path(path), bus, clock
        self._lock = threading.RLock()

    def _read(self) -> list[Device]:
        raw = read_json(self.path, {})
        rows = raw.get("devices", []) if isinstance(raw, dict) else []
        out = []
        for row in rows:
            try:
                out.append(Device(**row))
            except TypeError:
                continue
        return out

    def _write(self, rows: list[Device]) -> None:
        write_json(self.path, {"schema_version": SCHEMA_VERSION,
                               "devices": [asdict(d) for d in rows]})
        private_file(self.path)

    def all(self) -> list[Device]:
        with self._lock:
            return self._read()

    def add(self, request: Request, *, shared_cluster_key: bool, secret: str = "") -> Device:
        with self._lock, only_one(self.path.with_suffix(".lock"), announce=lambda _m: None):
            rows = [d for d in self._read() if d.fingerprint != request.fingerprint]
            device = Device(request.fingerprint, request.name, request.hostname,
                            request.address, self.clock(),
                            shared_cluster_key=shared_cluster_key, mine=bool(request.mine),
                            secret=secret,
                            tailnet_address=request.address if in_tailnet(request.address) else "")
            rows.append(device)
            self._write(rows)
        return device

    def add_accepter(self, fingerprint: str, name: str, address: str, *, secret: str,  # noqa: PLR0913 - all keywords
                     mine: bool = False, shared_cluster_key: bool = False) -> Device:
        """Record the machine that accepted this one (the asking side's record of a pairing).
        ``secret`` is the key that machine signs its file requests with; ``mine`` only if the
        person at this machine said the accepting one is theirs."""
        with self._lock, only_one(self.path.with_suffix(".lock"), announce=lambda _m: None):
            rows = [d for d in self._read() if d.fingerprint != fingerprint]
            device = Device(fingerprint, clean(name) or "unnamed", "", address, self.clock(),
                            shared_cluster_key=shared_cluster_key, mine=mine, secret=secret,
                            tailnet_address=address if in_tailnet(address) else "")
            rows.append(device)
            self._write(rows)
        return device

    def learn_tailnet(self, fingerprint: str, address: str, *, source: str) -> Device:
        """Record ``address`` as the tailnet address of a paired device. ``source`` must be an
        authenticated one (``pairing``, ``tailscale-verified``); an announcement is refused."""
        if source not in AUTHENTICATED_SOURCES:
            raise ValueError(f"a tailnet address is not taken from {source!r}")
        if not in_tailnet(address):
            raise ValueError(f"{address!r} is not a tailnet address")
        with self._lock, only_one(self.path.with_suffix(".lock"), announce=lambda _m: None):
            rows = self._read()
            hit = next((d for d in rows if d.fingerprint == fingerprint
                        and d.status == "active"), None)
            if hit is None:
                raise KeyError(fingerprint)
            hit.tailnet_address = address
            self._write(rows)
        return hit

    def revoked(self, fingerprint: str) -> bool:
        return any(d.fingerprint == fingerprint and d.status == "revoked" for d in self.all())

    def revoke(self, which: str) -> Device:
        """Revoke by fingerprint (any prefix of four or more hex digits) or by name. Raises
        `KeyError` if nothing matches and `ValueError` if the words match several."""
        which = which.strip().lower().replace(" ", "")
        with self._lock, only_one(self.path.with_suffix(".lock"), announce=lambda _m: None):
            rows = self._read()
            hits = [d for d in rows if (len(which) >= 4 and d.fingerprint.startswith(which))
                    or d.name.lower() == which]
            if not hits:
                raise KeyError(which)
            if len(hits) > 1:
                raise ValueError(f"{which!r} matches {len(hits)} devices; give more of the "
                                 "fingerprint")
            hit = hits[0]
            hit.status, hit.revoked = "revoked", self.clock()
            self._write(rows)
        self.bus.emit("onboard.revoked", "notice", f"device:{hit.fingerprint}",
                      name=hit.name, held_cluster_key=hit.shared_cluster_key)
        return hit

    def allow_again(self, fingerprint: str) -> bool:
        with self._lock, only_one(self.path.with_suffix(".lock"), announce=lambda _m: None):
            rows = self._read()
            kept = [d for d in rows if d.fingerprint != fingerprint]
            if len(kept) == len(rows):
                return False
            self._write(kept)
            return True


class Requests:
    """The requests this machine has received, in a file, under one set of `Limits`."""

    def __init__(self, path: Path, *, devices: Devices | None = None,
                 limits: Limits | None = None, bus: Bus = BUS,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.limits = limits or Limits()
        self.devices = devices or Devices(self.path.with_name("devices.json"), bus=bus,
                                          clock=clock)
        self.bus, self.clock = bus, clock
        self._lock = threading.RLock()

    # -- storage --
    def _load(self) -> _Ledger:
        raw = read_json(self.path, {})
        ledger = _Ledger()
        if not isinstance(raw, dict):
            return ledger
        for row in raw.get("requests", []):
            try:
                ledger.requests.append(Request(**{**row, "state": State(row["state"])}))
            except (TypeError, ValueError, KeyError):
                continue
        for name in ("asks", "declines", "waits"):
            got = raw.get(name)
            if isinstance(got, dict):
                setattr(ledger, name, got)
        return ledger

    def _save(self, ledger: _Ledger) -> None:
        now, lim = self.clock(), self.limits
        live = [r for r in ledger.requests
                if r.state in ACTIVE or now - max(r.decided, r.created) < lim.keep_s]
        ledger.requests = live[-lim.keep_most:]
        ledger.asks = {a: kept for a, ts in ledger.asks.items()
                       if (kept := [t for t in ts if now - t < lim.window_s])}
        ledger.declines = {f: kept for f, ts in ledger.declines.items()
                           if (kept := [t for t in ts if now - t < lim.block_s])}
        ledger.waits = {f: t for f, t in ledger.waits.items() if t > now}
        write_json(self.path, {"schema_version": SCHEMA_VERSION,
                               "requests": [{**asdict(r), "state": r.state.value}
                                            for r in ledger.requests],
                               "asks": ledger.asks, "declines": ledger.declines,
                               "waits": ledger.waits})
        private_file(self.path)

    def _transaction(self) -> Any:
        return only_one(self.path.with_suffix(".lock"), announce=lambda _m: None)

    def _expire(self, ledger: _Ledger) -> bool:
        """Close requests whose time has run out; whether any was."""
        now, lim, changed = self.clock(), self.limits, False
        for r in ledger.requests:
            late = (r.state is State.PENDING and now - r.created > lim.pending_ttl_s) or \
                   (r.state is State.ACCEPTED and now > r.code_expires)
            if late:
                r.state, r.code, r.decided = State.EXPIRED, "", now
                changed = True
                self.bus.emit("onboard.request.expired", "info", f"request:{r.id}",
                              fingerprint=r.fingerprint[:16], address=r.address)
        return changed

    # -- the asking side --
    def submit(self, info: dict[str, Any], address: str) -> Request:
        """A new request from ``address`` about the machine ``info`` describes, or `Refused`."""
        fingerprint = str(info.get("fingerprint", "")).lower()
        nonce = str(info.get("nonce", ""))
        if not FINGERPRINT.fullmatch(fingerprint) or not re.fullmatch(r"[0-9a-f]{16,64}", nonce):
            raise Refused(400, "a request names a certificate fingerprint and a nonce")
        lim, now = self.limits, self.clock()
        with self._lock, self._transaction():
            ledger = self._load()
            self._expire(ledger)
            try:
                self._admit(ledger, fingerprint, address, now)
            except Refused as why:
                self._save(ledger)
                self.bus.emit("onboard.request.refused", "notice", f"device:{fingerprint}",
                              address=address, status=why.status, reason=why.reason)
                raise
            ledger.asks.setdefault(address, []).append(now)
            request = Request(id=secrets.token_hex(16), name=clean(info.get("name")) or "unnamed",
                              hostname=clean(info.get("hostname")), model=clean(info.get("model")),
                              address=address, fingerprint=fingerprint, nonce=nonce, created=now)
            ledger.requests.append(request)
            self._save(ledger)
        self.bus.emit("onboard.request.received", "notice", f"request:{request.id}",
                      name=request.name, address=address, fingerprint=fingerprint[:16],
                      expires_in_s=lim.pending_ttl_s)
        return request

    def _admit(self, ledger: _Ledger, fingerprint: str, address: str, now: float) -> None:
        lim = self.limits
        if self.devices.revoked(fingerprint):
            raise Refused(403, "this device was revoked by the owner")
        until = ledger.waits.get(fingerprint, 0.0)
        if until > now:
            raise Refused(429, f"try again in {int(until - now) + 1} seconds")
        recent = [t for t in ledger.asks.get(address, []) if now - t < lim.window_s]
        if len(recent) >= lim.per_address:
            raise Refused(429, "too many requests from this address")
        active = [r for r in ledger.requests if r.state in ACTIVE]
        if any(r.fingerprint == fingerprint for r in active):
            raise Refused(409, "this device already has a request waiting")
        if any(r.address == address for r in active):
            raise Refused(409, "this address already has a request waiting")
        if len([r for r in active if r.state is State.PENDING]) >= lim.max_pending:
            raise Refused(503, "too many requests are waiting for an answer")

    # -- the answering side --
    def get(self, request_id: str) -> Request | None:
        with self._lock, self._transaction():
            ledger = self._load()
            if self._expire(ledger):
                self._save(ledger)
            return next((r for r in ledger.requests if r.id == request_id), None)

    def pending(self) -> list[Request]:
        with self._lock, self._transaction():
            ledger = self._load()
            if self._expire(ledger):
                self._save(ledger)
            return [r for r in ledger.requests if r.state in ACTIVE]

    def find(self, prefix: str) -> Request:
        """The one active request whose id starts with ``prefix``; `KeyError` if none,
        `ValueError` if several."""
        hits = [r for r in self.pending() if prefix and r.id.startswith(prefix)]
        if not hits:
            raise KeyError(prefix)
        if len(hits) > 1:
            raise ValueError(f"{prefix!r} matches {len(hits)} requests")
        return hits[0]

    def accept(self, request_id: str, *, mine: bool) -> Request:
        """A person says yes, and whether the device is theirs or another person's: the
        request gets a code that lasts `Limits.code_ttl_s`.
        Raises `KeyError` for an unknown request and `Refused` (409) for one that is not
        waiting, so accepting twice does not mint a second code."""
        with self._lock, self._transaction():
            ledger = self._load()
            self._expire(ledger)
            request = next((r for r in ledger.requests if r.id == request_id), None)
            if request is None:
                raise KeyError(request_id)
            if request.state is not State.PENDING:
                self._save(ledger)
                raise Refused(409, f"the request is {request.state.value}")
            request.state, request.mine = State.ACCEPTED, mine
            request.code = f"{secrets.randbelow(10 ** CODE_DIGITS):0{CODE_DIGITS}d}"
            request.decided = self.clock()
            request.code_expires = request.decided + self.limits.code_ttl_s
            self._save(ledger)
        self.bus.emit("onboard.request.accepted", "notice", f"request:{request.id}",
                      fingerprint=request.fingerprint[:16], address=request.address)
        return request

    def decline(self, request_id: str) -> Request:
        with self._lock, self._transaction():
            ledger = self._load()
            self._expire(ledger)
            request = next((r for r in ledger.requests if r.id == request_id), None)
            if request is None:
                raise KeyError(request_id)
            if request.state not in ACTIVE:
                self._save(ledger)
                raise Refused(409, f"the request is {request.state.value}")
            now, lim = self.clock(), self.limits
            request.state, request.code, request.decided = State.DECLINED, "", now
            strikes = ledger.declines.setdefault(request.fingerprint, [])
            strikes.append(now)
            recent = [t for t in strikes if now - t < lim.block_s]
            ledger.waits[request.fingerprint] = now + (
                lim.block_s if len(recent) >= lim.strikes else lim.decline_wait_s)
            self._save(ledger)
        self.bus.emit("onboard.request.declined", "info", f"request:{request.id}",
                      fingerprint=request.fingerprint[:16], address=request.address)
        return request

    # -- the pairing step --
    def take_attempt(self, request_id: str) -> Request:
        """Spend one try at the code. `Refused` unless the request is accepted and unexpired;
        the try that goes past `Limits.attempts` kills the request and makes the fingerprint
        wait. The caller then runs the exchange with ``request.code``."""
        with self._lock, self._transaction():
            ledger = self._load()
            self._expire(ledger)
            request = next((r for r in ledger.requests if r.id == request_id), None)
            if request is None or request.state is not State.ACCEPTED:
                self._save(ledger)
                raise Refused(409, "the request is not waiting for a code")
            request.attempts += 1
            if request.attempts > self.limits.attempts:
                self._fail(ledger, request, "too many wrong codes")
                self._save(ledger)
                raise Refused(429, "too many tries; the request is closed")
            self._save(ledger)
            return request

    def wrong(self, request_id: str) -> int:
        """The exchange ended with a wrong confirmation. Returns the tries left; at zero the
        request is closed at once rather than on the next try."""
        with self._lock, self._transaction():
            ledger = self._load()
            request = next((r for r in ledger.requests if r.id == request_id), None)
            if request is None or request.state is not State.ACCEPTED:
                return 0
            left = self.limits.attempts - request.attempts
            self.bus.emit("onboard.pair.wrong_code", "warning", f"request:{request.id}",
                          fingerprint=request.fingerprint[:16], address=request.address,
                          tries_left=max(0, left))
            if left <= 0:
                self._fail(ledger, request, "too many wrong codes")
            self._save(ledger)
            return max(0, left)

    def _fail(self, ledger: _Ledger, request: Request, why: str) -> None:
        now = self.clock()
        request.state, request.code, request.decided, request.note = \
            State.FAILED, "", now, why
        ledger.waits[request.fingerprint] = now + self.limits.failed_wait_s
        self.bus.emit("onboard.pair.locked", "critical", f"request:{request.id}",
                      fingerprint=request.fingerprint[:16], address=request.address,
                      reason=why)

    def paired(self, request_id: str, *, shared_cluster_key: bool, secret: str = "") -> Device:
        with self._lock, self._transaction():
            ledger = self._load()
            request = next((r for r in ledger.requests if r.id == request_id), None)
            if request is None or request.state is not State.ACCEPTED:
                raise Refused(409, "the request is not waiting for a code")
            request.state, request.code, request.decided = State.PAIRED, "", self.clock()
            self._save(ledger)
        device = self.devices.add(request, shared_cluster_key=shared_cluster_key, secret=secret)
        self.bus.emit("onboard.pair.succeeded", "notice", f"device:{request.fingerprint}",
                      name=request.name, address=request.address,
                      shared_cluster_key=shared_cluster_key, mine=bool(request.mine))
        return device
