"""The inbox on disk: one encrypted file per user, edited under a lock so the first answer wins."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from poolhouse import files, home, keystore, lock
from poolhouse.requests.model import (
    APPROVING,
    KINDS,
    PENDING,
    VIAS,
    Origin,
    Request,
    make_choices,
    shown,
)

__all__ = ["DAY", "MAX_PENDING", "MAX_ROWS", "PURPOSE", "TTL_S", "Ask", "Inbox", "Refused", "Unavailable"]

_PLAIN_FAILURES = (OSError, RuntimeError, ValueError, TypeError, KeyError)


def failures() -> tuple[type[BaseException], ...]:
    """What reading, decrypting or writing the file can raise; each becomes `Unavailable`.

    ``InvalidTag`` can only be raised once ``cryptography`` has been imported (by the keystore,
    which decrypts), so it is looked up in ``sys.modules`` rather than imported: importing this
    package (through ``poolhouse.serve`` and ``poolhouse.client``) must load the standard library
    and the core dependencies only."""
    module = sys.modules.get("cryptography.exceptions")
    return (*_PLAIN_FAILURES, module.InvalidTag) if module is not None else _PLAIN_FAILURES
PURPOSE = "requests"
_AAD = b"ml-stack/requests/v1"
_MAGIC = b"MLR1"
SCHEMA_VERSION = 1
DAY = 86400.0
TTL_S = 300.0
"""How long a request waits when its raiser names no expiry."""
MAX_PENDING = 100
MAX_ROWS = 400
RETAIN_S = 7 * DAY
LOCK_WAIT_S = 10.0


class Unavailable(RuntimeError):
    """The store cannot be read or written, so nothing is raised or answered through it."""


class Refused(ValueError):
    """An answer or change that is not taken; ``why`` is ``unknown``, ``resolved``, ``changed``,
    ``choice``, ``via``, ``full`` or ``secret``."""

    def __init__(self, why: str, message: str) -> None:
        super().__init__(message)
        self.why = why


@dataclass(frozen=True, slots=True)
class Row:
    """A request as stored, with the fingerprint taken when it was raised and the hash of the
    raiser's secret (what lets the raiser, and nothing else, withdraw it)."""

    request: Request
    fp: str
    secret: str

    def to_json(self) -> dict[str, Any]:
        return {"request": self.request.to_json(), "fp": self.fp, "secret": self.secret}

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> Row:
        return cls(Request.from_json(row["request"]), str(row["fp"]), str(row["secret"]))


def hashed(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


class Inbox:
    """The requests of one user. ``directory`` defaults to ``requests`` under the state root and
    ``key`` to the ``requests`` subkey of the keystore; a store that cannot be opened is
    `Unavailable`. With ``memory`` the rows are kept in this process only, never on disk."""

    def __init__(self, directory: Path | None = None, *, key: Callable[[], bytes] | None = None,
                 clock: Callable[[], float] | None = None, memory: bool = False) -> None:
        self.memory = memory
        self._held = threading.RLock()
        self.directory = Path(directory) if directory else home.state("requests")
        self.path = self.directory / "requests.enc"
        self._key_from = key or self._subkey
        self._key: bytes | None = None
        self._now = clock or time.time
        self._seen: tuple[int, int, int] | None = None
        self._rows: list[Row] = []

    def _subkey(self) -> bytes:
        return keystore.default().salted_subkey(PURPOSE, self.directory)

    def _cipher(self) -> Any:
        if self._key is None:
            self._key = self._key_from()
        return keystore.aead(self._key)

    def ready(self) -> bool:
        """Whether the key can be had here, so a raised request can be stored."""
        if self.memory:
            return True
        try:
            self._cipher()
        except failures():
            return False
        return True

    # -- the file ------------------------------------------------------------------
    def _load(self) -> list[Row]:
        if self.memory:
            return list(self._rows)
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._seen, self._rows = None, []
            return []
        mark = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        if mark == self._seen:
            return list(self._rows)
        try:
            blob = self.path.read_bytes()
            if not blob.startswith(_MAGIC):
                raise Unavailable("not a requests file")
            plain = self._cipher().decrypt(blob[4:16], blob[16:], _AAD)
            doc = json.loads(plain)
            rows = [Row.from_json(r) for r in doc["rows"]]
        except Unavailable:
            raise
        except failures() as exc:
            raise Unavailable(f"the requests file does not open: {type(exc).__name__}") from exc
        self._seen, self._rows = mark, rows
        return list(rows)

    def _save(self, rows: list[Row]) -> None:
        if self.memory:
            self._rows = rows
            return
        body = json.dumps({"schema_version": SCHEMA_VERSION, "rows": [r.to_json() for r in rows]},
                          sort_keys=True, ensure_ascii=True).encode()
        nonce = os.urandom(12)
        blob = _MAGIC + nonce + self._cipher().encrypt(nonce, body, _AAD)
        with files.writing(self.path) as tmp:
            tmp.write_bytes(blob)
            tmp.chmod(0o600)
        self._seen = None

    @contextlib.contextmanager
    def _edit(self) -> Iterator[list[Row]]:
        """The rows under the lock; a change made in the block is expired, trimmed and written."""
        try:
            with self._held if self.memory else lock.only_one(
                    self.directory / "requests.lock", timeout=LOCK_WAIT_S, announce=lambda _m: None):
                rows = self._load()
                yield rows
                self._save(_trim(_expire(rows, self._now()), self._now()))
        except (Unavailable, Refused):
            raise
        except failures() as exc:
            raise Unavailable(f"the requests file cannot be edited: {type(exc).__name__}") from exc

    # -- raising -------------------------------------------------------------------
    def add(self, request: Request, secret: str) -> Request:
        """Store a new pending request; an older pending one with the same raiser, kind and key
        is superseded. `Refused` (``full``) when too many are pending."""
        with self._edit() as rows:
            _expire(rows, self._now())
            pending = [r for r in rows if r.request.state == PENDING]
            if len(pending) >= MAX_PENDING:
                raise Refused("full", f"{MAX_PENDING} requests are already waiting")
            for at, row in enumerate(rows):
                same = (row.request.state == PENDING and request.key
                        and (row.request.key, row.request.kind, row.request.raised_by) ==
                        (request.key, request.kind, request.raised_by))
                if same:
                    rows[at] = Row(replace(row.request, state="superseded", answered_at=self._now()),
                                   row.fp, row.secret)
            rows.append(Row(request, request.fingerprint, hashed(secret)))
        return request

    # -- reading -------------------------------------------------------------------
    def get(self, ident: str) -> Request | None:
        """The request called ``ident``, with expiry applied to what is returned."""
        now = self._now()
        row = next((r for r in self._load() if r.request.id == ident), None)
        return _aged(row.request, now) if row else None

    def list(self, *, state: str = "", agent: str = "", project: str = "", kind: str = "",
             limit: int = 200) -> list[Request]:
        """Pending requests first (oldest first), then the resolved, newest first."""
        now = self._now()
        found = [_aged(r.request, now) for r in self._load()]
        keep = [r for r in found if (not state or r.state == state) and (not agent or r.raised_by.agent == agent)
                and (not project or r.raised_by.project == project) and (not kind or r.kind == kind)]
        pending = sorted((r for r in keep if r.state == PENDING), key=lambda r: (r.created, r.id))
        done = sorted((r for r in keep if r.state != PENDING), key=lambda r: (-r.answered_at, r.id))
        return [*pending, *done][:limit]

    def pending(self) -> list[Request]:
        return self.list(state=PENDING)

    # -- answering -----------------------------------------------------------------
    def answer(self, ident: str, choice: str, fingerprint: str, via: str) -> Request:
        """Record the answer: the first one wins. `Refused` when the request is unknown, already
        resolved, was changed since ``fingerprint`` was shown, or has no such choice."""
        if via not in VIAS:
            raise Refused("via", f"{via!r} is not a way a person answers")
        with self._edit() as rows:
            at, row = _find(rows, ident)
            now = self._now()
            _expire(rows, now)
            row = rows[at]
            current = row.request
            if current.state != PENDING:
                raise Refused("resolved", f"already resolved: {current.state}"
                              + (f" by {current.answered_by}" if current.answered_by else ""))
            if not (fingerprint and fingerprint == current.fingerprint == row.fp):
                raise Refused("changed", "the request is not what was shown; look again")
            picked = current.choice(choice)
            if picked is None:
                raise Refused("choice", f"{choice!r} is not one of this request's choices")
            state = "approved" if choice in APPROVING else "denied"
            done = replace(current, state=state, answered_by=via, answered_at=now, answer=choice)
            rows[at] = Row(done, row.fp, row.secret)
        return done

    def withdraw(self, ident: str, secret: str) -> bool:
        """Cancel a pending request for the raiser that holds ``secret``; false when it is no
        longer pending."""
        with self._edit() as rows:
            at, row = _find(rows, ident)
            if not secrets.compare_digest(row.secret, hashed(secret)):
                raise Refused("secret", "only the raiser withdraws a request")
            _expire(rows, self._now())
            row = rows[at]
            if row.request.state != PENDING:
                return False
            rows[at] = Row(replace(row.request, state="cancelled", answered_at=self._now()),
                           row.fp, row.secret)
            return True


def _find(rows: list[Row], ident: str) -> tuple[int, Row]:
    for at, row in enumerate(rows):
        if row.request.id == ident:
            return at, row
    raise Refused("unknown", "no such request")


def _aged(request: Request, now: float) -> Request:
    if request.state == PENDING and request.expires <= now:
        return replace(request, state="expired", answered_at=request.expires)
    return request


def _expire(rows: list[Row], now: float) -> list[Row]:
    """Pending rows past their expiry become ``expired`` (a denial) in place."""
    for at, row in enumerate(rows):
        aged = _aged(row.request, now)
        if aged is not row.request:
            rows[at] = Row(aged, row.fp, row.secret)
    return rows


def _trim(rows: list[Row], now: float) -> list[Row]:
    """Drop resolved rows past retention, then the oldest resolved rows beyond `MAX_ROWS`."""
    kept = [r for r in rows if r.request.state == PENDING or now - r.request.answered_at <= RETAIN_S]
    extra = len(kept) - MAX_ROWS
    if extra > 0:
        done = sorted((r for r in kept if r.request.state != PENDING), key=lambda r: r.request.answered_at)
        drop = {id(r) for r in done[:extra]}
        kept = [r for r in kept if id(r) not in drop]
    return kept


@dataclass(frozen=True, slots=True)
class Ask:
    """What a component asks: the ``kind``, what it is about (``subject``), why (``reason``), the
    ``choices`` offered, who raised it, how long it waits and a ``key`` that makes a newer
    request of the same raiser supersede an older pending one."""

    kind: str
    subject: str
    reason: str
    choices: tuple[str, ...]
    origin: Origin = field(default_factory=Origin)
    ttl: float = TTL_S
    key: str = ""
    extra: tuple[tuple[str, str], ...] = ()


def build(ask: Ask, created: float) -> Request:
    """A pending `Request` with every text escaped and cut to its bound."""
    if ask.kind not in KINDS:
        raise ValueError(f"no kind {ask.kind!r}: the kinds are {', '.join(KINDS)}")
    ident = "rq_" + base64.b32encode(os.urandom(8)).decode().lower().rstrip("=")
    origin = ask.origin
    return Request(
        id=ident, kind=ask.kind,
        raised_by=Origin(shown(origin.agent, "name"), shown(origin.project, "name"),
                         shown(origin.session, "name"), shown(origin.model, "model"),
                         shown(origin.model_state, "label")),
        subject=shown(ask.subject, "subject"), reason=shown(ask.reason, "reason"),
        choices=make_choices(ask.choices, destructive=KINDS[ask.kind].destructive),
        created=created, expires=created + max(1.0, ask.ttl), key=shown(ask.key, "name"),
        extra={shown(k, "name"): shown(v, "subject") for k, v in ask.extra[:8]})
