"""Authenticating a request with a MAC instead of a bearer token.

A client holding a secret signs each request: its method, target, host, body, a timestamp
and a nonce. The server recomputes the MAC, compares it in constant time, refuses a
timestamp outside its window and a nonce it has seen inside it, and locks out a client
address that keeps failing. The secret itself never crosses the wire, so a captured request
can be sent again to nobody.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
import urllib.parse
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

__all__ = ["PREFIX", "SCHEME", "Authenticator", "Lockout", "Stamp", "Verdict", "derive",
           "sign", "unwrap"]

SCHEME = "ML-Stack-MAC"
PREFIX = "mlsk1."
"""What a MAC secret starts with, so a caller holding a ``token`` string knows to sign."""
WINDOW_S = 120.0
MOST_NONCES = 100_000
INFO = b"ml-stack-request-mac-v1"
EMPTY = hashlib.sha256(b"").hexdigest()


def derive(key: bytes) -> str:
    """The request secret two ends compute from the same cluster key (HKDF-SHA256)."""
    prk = hmac.new(b"ml-stack-hkdf-salt", key, hashlib.sha256).digest()
    okm = hmac.new(prk, INFO + b"\x01", hashlib.sha256).digest()
    return PREFIX + base64.urlsafe_b64encode(okm).decode().rstrip("=")


def unwrap(authorization: str) -> str:
    """The MAC secret a ``Bearer mlsk1.…`` header value carries, else an empty string."""
    bearer = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
    return bearer if bearer.startswith(PREFIX) else ""


def key_id(secret: str) -> str:
    """A short public name for a secret, to tell which of several signed a request."""
    return hashlib.sha256(secret.encode()).hexdigest()[:8]


def _target(url: str) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url)
    return parts.netloc, (parts.path or "/") + (f"?{parts.query}" if parts.query else "")


@dataclass(frozen=True, slots=True)
class Stamp:
    """When a request was signed and the nonce that makes it unique."""

    at: float
    nonce: str

    @classmethod
    def now(cls) -> Stamp:
        return cls(time.time(), secrets.token_hex(12))


def _canonical(method: str, url: str, body: bytes | None, stamp: Stamp) -> bytes:
    host, target = _target(url)
    body_hash = hashlib.sha256(body).hexdigest() if body else EMPTY
    return "\n".join((SCHEME, method.upper(), target, host, body_hash, f"{stamp.at:.0f}",
                      stamp.nonce)).encode()


def sign(secret: str, method: str, url: str, body: bytes | None,
         stamp: Stamp | None = None) -> dict[str, str]:
    """The ``Authorization`` header that signs this request."""
    stamp = stamp or Stamp.now()
    mac = hmac.new(secret.encode(), _canonical(method, url, body, stamp),
                   hashlib.sha256).hexdigest()
    return {"Authorization":
            f"{SCHEME} k={key_id(secret)},t={stamp.at:.0f},n={stamp.nonce},s={mac}"}


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a request is authentic, and when it is not, a reason safe to show."""

    ok: bool
    reason: str = ""
    locked: bool = False


class Lockout:
    """Counts failures per client address and refuses an address that fails too often."""

    def __init__(self, *, failures: int = 10, window_s: float = 60.0, lock_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic, most: int = 4096) -> None:
        self.failures, self.window_s, self.lock_s = failures, window_s, lock_s
        self.clock, self.most = clock, most
        self._seen: OrderedDict[str, list[float]] = OrderedDict()
        self._until: dict[str, float] = {}
        self._lock = threading.Lock()

    def locked(self, who: str) -> bool:
        """Whether ``who`` is refused right now."""
        with self._lock:
            until = self._until.get(who)
            if until is None:
                return False
            if self.clock() >= until:
                self._until.pop(who, None)
                self._seen.pop(who, None)
                return False
            return True

    def failed(self, who: str) -> None:
        """Count one failure, locking ``who`` when there have been too many in the window."""
        with self._lock:
            now = self.clock()
            recent = [t for t in self._seen.pop(who, []) if now - t < self.window_s] + [now]
            self._seen[who] = recent
            while len(self._seen) > self.most:
                self._seen.popitem(last=False)
            if len(recent) >= self.failures:
                self._until[who] = now + self.lock_s

    def passed(self, who: str) -> None:
        """Forget ``who``'s failures."""
        with self._lock:
            self._seen.pop(who, None)


class Authenticator:
    """Checks signed requests against a set of secrets, once each."""

    def __init__(self, secrets_: Callable[[], Iterable[str]], *, window_s: float = WINDOW_S,
                 lockout: Lockout | None = None, clock: Callable[[], float] = time.time,
                 most: int = MOST_NONCES) -> None:
        self.secrets, self.window_s, self.clock, self.most = secrets_, window_s, clock, most
        self.lockout = lockout or Lockout()
        self._nonces: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None,
              who: str = "") -> Verdict:
        """Whether the request is signed by one of the secrets, fresh, and not seen before.

        ``url`` is the request target as received, ``who`` the client's address, and
        ``headers`` need ``Authorization`` and ``Host``.
        """
        if who and self.lockout.locked(who):
            return Verdict(False, "too many failures from this address; try again later",
                           locked=True)
        verdict = self._verify(method, url, headers, body)
        if who:
            (self.lockout.passed if verdict.ok else self.lockout.failed)(who)
        return verdict

    def _verify(self, method: str, url: str, headers: Mapping[str, str],
                body: bytes | None) -> Verdict:
        given = headers.get("Authorization", "")
        if not given.startswith(SCHEME + " "):
            return Verdict(False, "request is not signed")
        fields = dict(part.split("=", 1) for part in given[len(SCHEME) + 1:].split(",")
                      if "=" in part)
        stamp, nonce, mac, kid = (fields.get(k, "") for k in ("t", "n", "s", "k"))
        if not (stamp.isdigit() and 8 <= len(nonce) <= 64 and len(mac) == 64):
            return Verdict(False, "request is not signed")
        if abs(self.clock() - int(stamp)) > self.window_s:
            return Verdict(False, "request time is outside the window; check the clocks")
        _, target = _target(url)
        wanted = _canonical(method, f"//{headers.get('Host', '')}{target}", body,
                            Stamp(float(stamp), nonce))
        match = False
        for secret in self.secrets():
            if secret and hmac.compare_digest(kid, key_id(secret)):
                good = hmac.new(secret.encode(), wanted, hashlib.sha256).hexdigest()
                match = hmac.compare_digest(mac, good) or match
        if not match:
            return Verdict(False, "request is not signed")
        if not self._fresh(nonce):
            return Verdict(False, "request was already seen")
        return Verdict(True)

    def _fresh(self, nonce: str) -> bool:
        with self._lock:
            now = self.clock()
            while self._nonces:
                oldest, until = next(iter(self._nonces.items()))
                if until > now and len(self._nonces) < self.most:
                    break
                self._nonces.pop(oldest)
            if nonce in self._nonces:
                return False
            self._nonces[nonce] = now + 2 * self.window_s
            return True
