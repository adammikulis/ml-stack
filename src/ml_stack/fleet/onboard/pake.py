"""Pairing by a short code: SPAKE2 from the `spake2` package, with confirmations on top.

A six digit code is twenty bits, so a MAC over it on the wire could be tried against every code
offline. In SPAKE2 nothing sent lets a listener test a guess, and a side's confirmation is sent
only after the other's verified, so each guess costs one conversation. No group arithmetic is
written here: `spake2` (MIT, python-spake2, used by magic-wormhole) does it; this adds framing
and HMAC-SHA256 confirmations. Both certificate fingerprints are its two identities and the
request id and nonce go in with the code. The library is pure Python, not constant time: the
exchange is interactive, three tries, 120 s (docs/onboarding.md).
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

__all__ = ["INSTALL", "Bad", "PakeUnavailable", "Session", "require", "start_initiator",
           "start_responder"]

INSTALL = "pip install 'ml-stack[fleet-onboard]' (adds the spake2 package)"
MOST_MESSAGE = 64


class Bad(ValueError):
    """A message that is not what the protocol allows."""


class PakeUnavailable(RuntimeError):
    """The `spake2` package is not installed."""


def require() -> Any:
    """The `spake2` module, or `PakeUnavailable` saying how to get it."""
    try:
        import spake2
    except ImportError as exc:
        raise PakeUnavailable(f"pairing needs the spake2 package: {INSTALL}") from exc
    return spake2


def _lp(*parts: bytes) -> bytes:
    return b"".join(len(p).to_bytes(4, "big") + p for p in parts)


class Session:
    """One end of one exchange. Send `message`, hand the other end's to `receive`, then trade
    confirmations: the initiator's first (`confirmation`), checked by the responder with
    `check`, and only then the responder's."""

    def __init__(self, role: str, code: str, context: bytes, mine: str, theirs: str) -> None:
        lib = require()
        self.role, self._context, self._lib = role, context, lib
        ids = (mine, theirs) if role == "A" else (theirs, mine)
        make = lib.SPAKE2_A if role == "A" else lib.SPAKE2_B
        self._pake = make(context + b"\x00" + code.encode(), idA=ids[0].encode(),
                          idB=ids[1].encode())
        self._ids = ids
        self._out = self._pake.start()
        self.message = self._out.hex()
        self._tt: bytes | None = None
        self._root = b""

    def receive(self, other: str) -> None:
        """Take the other end's message and work out the shared key."""
        if not isinstance(other, str) or not 2 <= len(other) <= 2 * MOST_MESSAGE:
            raise Bad("not a message")
        try:
            raw = bytes.fromhex(other)
            key = self._pake.finish(raw)
        except (ValueError, TypeError, self._lib.SPAKEError) as exc:
            raise Bad(f"bad message: {exc}") from None
        first, second = (self._out, raw) if self.role == "A" else (raw, self._out)
        self._tt = _lp(self._context, self._ids[0].encode(), self._ids[1].encode(), first, second)
        self._root = hashlib.sha256(key).digest()

    def _mac(self, name: str) -> str:
        if self._tt is None:
            raise Bad("no exchange yet")
        sub = hmac.new(self._root, name.encode(), hashlib.sha256).digest()
        return hmac.new(sub, self._tt, hashlib.sha256).hexdigest()

    def confirmation(self) -> str:
        """What this end sends to say it holds the code."""
        return self._mac(f"confirm-{self.role}")

    def check(self, theirs: str) -> bool:
        """Whether the other end's confirmation matches (constant time)."""
        other = "B" if self.role == "A" else "A"
        return isinstance(theirs, str) and hmac.compare_digest(theirs, self._mac(f"confirm-{other}"))

    def key(self, label: str) -> bytes:
        """A 32-byte key for ``label`` that both ends hold once the exchange is done."""
        if self._tt is None:
            raise Bad("no exchange yet")
        return hmac.new(self._root, b"key-" + label.encode(), hashlib.sha256).digest()

    def seal(self, payload: bytes) -> str:
        """A tag over ``payload`` under the exchange's payload key."""
        sub = hmac.new(self._root, b"payload", hashlib.sha256).digest()
        return hmac.new(sub, payload, hashlib.sha256).hexdigest()

    def open(self, payload: bytes, tag: str) -> bool:
        return isinstance(tag, str) and hmac.compare_digest(tag, self.seal(payload))


def start_initiator(code: str, *, context: bytes, mine: str, theirs: str) -> Session:
    """The side that asks to join. ``mine`` and ``theirs`` are certificate fingerprints: its
    own, and the one it saw the accepting machine present."""
    return Session("A", code, context, mine, theirs)


def start_responder(code: str, *, context: bytes, mine: str, theirs: str) -> Session:
    """The side that accepts. ``mine`` is its own certificate's fingerprint, ``theirs`` the one
    the asking machine said it has."""
    return Session("B", code, context, mine, theirs)
