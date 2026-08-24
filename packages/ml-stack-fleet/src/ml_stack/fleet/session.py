"""Browser sessions for the fleet UI, and the throttle that makes login safe to expose.

**The cluster key never enters the browser.** If the derived token sat in
``localStorage``, any script that ever runs on this origin -- an XSS, an extension, a
page the user was tricked into -- would hold permanent, unrevocable remote code execution
on every machine in the cluster. A session id is a random opaque string that means
nothing anywhere else, dies when the daemon restarts, and can be revoked.

**Logging in with the passphrase is the point.** ``check_passphrase`` re-derives and
compares, so someone can type the words they already know instead of pasting a
43-character token. The words are verified and discarded, never stored.

**And that is why the throttle exists.** Verifying a passphrase costs one scrypt: ~64MB
and up to a second or two. The login route is unauthenticated by necessity, and the
daemon is a ``ThreadingHTTPServer`` with no connection cap -- twenty concurrent guesses
would be over a gigabyte of allocation on a box whose entire job is to have memory free
for training. The parameters are right and must not be weakened, so the HTTP path around
them is what has to hold: one derivation at a time, a short queue, and a fast refusal
past that. Refusing quickly is a better failure than swapping.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

__all__ = ["Sessions", "Throttle", "COOKIE", "parse_cookie"]

COOKIE = "ml_stack_ui"
TTL_S = 12 * 3600
TICKET_TTL_S = 60.0


def parse_cookie(header: str, name: str = COOKIE) -> str:
    """One cookie's value out of a ``Cookie:`` header, or empty."""
    for part in (header or "").split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value.strip()
    return ""


@dataclass
class Session:
    sid: str
    created_at: float
    expires_at: float
    who: str = ""


class Sessions:
    """Live sessions and single-use tickets, in memory only.

    In memory is deliberate: a restart should end every session. A credential persisted
    to disk to save people a login is a credential that outlives the process holding it,
    on a box that runs whatever it is sent.
    """

    def __init__(self, *, ttl_s: float = TTL_S) -> None:
        self.ttl_s = ttl_s
        self._sessions: dict[str, Session] = {}
        self._tickets: dict[str, float] = {}
        self._lock = threading.Lock()

    def _reap(self) -> None:
        now = time.time()
        for sid in [s for s, v in self._sessions.items() if v.expires_at <= now]:
            self._sessions.pop(sid, None)
        for t in [t for t, exp in self._tickets.items() if exp <= now]:
            self._tickets.pop(t, None)

    def open(self, who: str = "") -> Session:
        now = time.time()
        session = Session(sid=secrets.token_urlsafe(32), created_at=now,
                          expires_at=now + self.ttl_s, who=who)
        with self._lock:
            self._reap()
            self._sessions[session.sid] = session
        return session

    def get(self, sid: str) -> Session | None:
        if not sid:
            return None
        with self._lock:
            self._reap()
            return self._sessions.get(sid)

    def close(self, sid: str) -> bool:
        with self._lock:
            return self._sessions.pop(sid, None) is not None

    def mint_ticket(self) -> tuple[str, float]:
        """A one-shot credential for handing a browser a session without typing.

        Short-lived and single-use because it travels in a URL, and a URL ends up in
        shell history, in the address bar, and in whatever the browser syncs.
        """
        ticket = secrets.token_urlsafe(24)
        expires = time.time() + TICKET_TTL_S
        with self._lock:
            self._reap()
            self._tickets[ticket] = expires
        return ticket, expires

    def spend_ticket(self, ticket: str) -> bool:
        with self._lock:
            self._reap()
            expires = self._tickets.pop(ticket, None)
        return bool(expires and expires > time.time())

    def cookie_header(self, session: Session, *, secure: bool = False) -> str:
        parts = [f"{COOKIE}={session.sid}", "HttpOnly", "SameSite=Strict", "Path=/ui",
                 f"Max-Age={int(self.ttl_s)}"]
        if secure:
            parts.append("Secure")
        return "; ".join(parts)

    def clear_header(self) -> str:
        return f"{COOKIE}=; HttpOnly; SameSite=Strict; Path=/ui; Max-Age=0"

    def __len__(self) -> int:
        with self._lock:
            self._reap()
            return len(self._sessions)


@dataclass
class Throttle:
    """Serialises expensive derivations and backs off a source that keeps guessing.

    ``slots`` is one on purpose. People log in rarely, so serialising costs nothing that
    matters, and it turns an unbounded memory multiplier into a single 64MB allocation.
    """

    slots: int = 1
    wait_s: float = 2.0
    free_attempts: int = 3
    """Wrong guesses before any delay is imposed. Not one: people mistype, and someone
    who fumbles a passphrase once and is then told to wait has been punished for being
    the legitimate user. Three is the same allowance the fleet gives a peer before it
    quarantines it, and for the same reason."""
    base_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    _sem: threading.Semaphore = field(init=False)
    _fails: dict[str, tuple[int, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._sem = threading.Semaphore(self.slots)

    def blocked_for(self, source: str) -> float:
        """Seconds this source must still wait, or 0.

        Rounded up, never down: reporting "wait 0s" while refusing the request is the
        worst of both -- it reads as a bug, because from the outside it is one.
        """
        with self._lock:
            _count, until = self._fails.get(source, (0, 0.0))
        left = until - time.time()
        return 0.0 if left <= 0 else max(1.0, left)

    def acquire(self) -> bool:
        """A derivation slot, or False if the queue is already too deep.

        False must become a fast 503. Queueing instead simply moves the exhaustion from
        scrypt's arenas to thread stacks, which fails later and less legibly.
        """
        return self._sem.acquire(timeout=self.wait_s)

    def release(self) -> None:
        self._sem.release()

    def failed(self, source: str) -> float:
        """Record a wrong guess. Returns how long this source is now held off."""
        with self._lock:
            count, _ = self._fails.get(source, (0, 0.0))
            count += 1
            over = count - self.free_attempts
            delay = 0.0 if over <= 0 else min(
                self.max_backoff_s, self.base_backoff_s * 2 ** (over - 1))
            self._fails[source] = (count, time.time() + delay)
        return delay

    def succeeded(self, source: str) -> None:
        with self._lock:
            self._fails.pop(source, None)
