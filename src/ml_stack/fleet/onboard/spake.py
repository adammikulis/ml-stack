"""SPAKE2 over NIST P-256: two machines that share a short code agree on a key, and nothing
sent lets a listener test guesses at the code offline.

A confirmation like HMAC(code, transcript) on the wire would let anyone try all million six
digit codes in a second. Here a side's confirmation is sent only after the other's has
verified, so each guess costs one conversation. M and N are hashed to the curve, so nobody
knows their discrete logarithm; this is not byte-compatible with RFC 9382 and need not be.
The two ends are identified by certificate fingerprints in the transcript, so a relay that
terminates TLS on both sides fails the confirmation. Not constant time (Python integers):
one guess per conversation, rate-limited, bounds what a timer collects. See docs/onboarding.md.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass

__all__ = [
    "Bad",
    "Party",
    "Session",
    "decode",
    "encode",
    "start_initiator",
    "start_responder",
    "word_from_code",
]

P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
A = P - 3
B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5

Point = tuple[int, int] | None
"""An affine point; None is the point at infinity."""

LABEL = b"ml-stack/pair/spake2-p256/v1"


class Bad(ValueError):
    """A message that is not what the protocol allows, or a confirmation that does not match."""


# -- the curve -----------------------------------------------------------------------
def on_curve(point: Point) -> bool:
    if point is None:
        return False
    x, y = point
    return 0 <= x < P and 0 <= y < P and (y * y - (x * x * x + A * x + B)) % P == 0


def _add(p: Point, q: Point) -> Point:
    if p is None:
        return q
    if q is None:
        return p
    (x1, y1), (x2, y2) = p, q
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return None
        slope = (3 * x1 * x1 + A) * pow(2 * y1, -1, P) % P
    else:
        slope = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def _neg(p: Point) -> Point:
    return None if p is None else (p[0], -p[1] % P)


def multiply(k: int, p: Point) -> Point:
    """``k * p``. Always does the addition, keeping it only for a set bit, so the work does
    not depend on the bits (the field arithmetic around it still can)."""
    k %= N
    result: Point = None
    for bit in range(255, -1, -1):
        result = _add(result, result)
        added = _add(result, p)
        if (k >> bit) & 1:
            result = added
    return result


G: Point = (GX, GY)


def _hash_to_curve(label: bytes) -> Point:
    """A point nobody knows the discrete logarithm of: try x = H(label, counter) until it is
    on the curve (p = 3 mod 4, so the square root is one exponentiation)."""
    counter = 0
    while True:
        x = int.from_bytes(hashlib.sha256(label + counter.to_bytes(4, "big")).digest(),
                           "big") % P
        rhs = (x * x * x + A * x + B) % P
        y = pow(rhs, (P + 1) // 4, P)
        if y * y % P == rhs:
            return x, y if y % 2 == 0 else P - y
        counter += 1


M: Point = _hash_to_curve(LABEL + b"/M")
NN: Point = _hash_to_curve(LABEL + b"/N")


def encode(point: Point) -> str:
    if point is None:
        raise Bad("the point at infinity is not a message")
    return "04" + point[0].to_bytes(32, "big").hex() + point[1].to_bytes(32, "big").hex()


def decode(text: str) -> Point:
    """A point from its uncompressed hex form; `Bad` unless it is exactly that and on the
    curve (an off-curve point is how an invalid-curve attack starts)."""
    if not isinstance(text, str) or len(text) != 130 or not text.startswith("04"):
        raise Bad("not an uncompressed P-256 point")
    try:
        raw = bytes.fromhex(text[2:])
    except ValueError:
        raise Bad("not hexadecimal") from None
    point = (int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
    if not on_curve(point):
        raise Bad("the point is not on the curve")
    return point


# -- the exchange ----------------------------------------------------------------------
def word_from_code(code: str, context: bytes) -> int:
    """The scalar a code stands for, in the context of this one request."""
    digest = hashlib.sha256(LABEL + b"/w\x00" + context + b"\x00" + code.encode()).digest()
    return int.from_bytes(digest, "big") % (N - 1) + 1


def _lp(*parts: bytes) -> bytes:
    return b"".join(len(p).to_bytes(4, "big") + p for p in parts)


@dataclass(frozen=True, slots=True)
class Party:
    """Who one end is: ``role`` 'A' (the machine that asks) or 'B' (the one that accepts)
    and the SHA-256 fingerprint of the certificate it is seen to present."""

    role: str
    fingerprint: str


class Session:
    """One end of one exchange. Make it with `start_initiator` / `start_responder`, send
    `message`, hand the other end's to `receive`, then trade confirmations: the initiator's
    first (`confirmation`), checked by the responder with `check`, and only then the
    responder's."""

    def __init__(self, role: str, scalars: tuple[int, int], message: Point, context: bytes,
                 ends: tuple[Party, Party]) -> None:
        self.role, (self._x, self._w) = role, scalars    # the secret and the code's scalar
        self.message_point = message
        self._context, (self.me, self.peer) = context, ends
        self._tt: bytes | None = None
        self._keys: dict[str, bytes] = {}

    @property
    def message(self) -> str:
        return encode(self.message_point)

    def receive(self, other: str) -> None:
        """Take the other end's message and work out the shared key."""
        theirs = decode(other)
        blind = NN if self.role == "A" else M
        k = multiply(self._x, _add(theirs, _neg(multiply(self._w, blind))))
        if k is None:
            raise Bad("degenerate exchange")
        x_msg, y_msg = ((self.message_point, theirs) if self.role == "A"
                        else (theirs, self.message_point))
        a, b = (self.me, self.peer) if self.role == "A" else (self.peer, self.me)
        self._tt = _lp(self._context, a.fingerprint.encode(), b.fingerprint.encode(),
                       encode(x_msg).encode(), encode(y_msg).encode(),
                       k[0].to_bytes(32, "big"), self._w.to_bytes(32, "big"))
        root = hashlib.sha256(self._tt).digest()
        self._keys = {name: hmac.new(root, name.encode(), hashlib.sha256).digest()
                      for name in ("confirm-A", "confirm-B", "payload")}

    def _mac(self, name: str) -> str:
        if self._tt is None:
            raise Bad("no exchange yet")
        return hmac.new(self._keys[name], self._tt, hashlib.sha256).hexdigest()

    def confirmation(self) -> str:
        """What this end sends to say it holds the code."""
        return self._mac(f"confirm-{self.role}")

    def check(self, theirs: str) -> bool:
        """Whether the other end's confirmation matches (constant time)."""
        other = "B" if self.role == "A" else "A"
        return isinstance(theirs, str) and hmac.compare_digest(theirs, self._mac(f"confirm-{other}"))

    def seal(self, payload: bytes) -> str:
        """A tag over ``payload`` under the exchange's payload key, so a joiner can tell what
        arrived is what the accepting machine sent after proving the code."""
        return hmac.new(self._keys["payload"], payload, hashlib.sha256).hexdigest()

    def open(self, payload: bytes, tag: str) -> bool:
        return isinstance(tag, str) and hmac.compare_digest(tag, self.seal(payload))


def _start(role: str, code: str, context: bytes, me: Party, peer: Party) -> Session:
    word = word_from_code(code, context)
    x = secrets.randbelow(N - 1) + 1
    blind = M if role == "A" else NN
    message = _add(multiply(x, G), multiply(word, blind))
    return Session(role, (x, word), message, context, (me, peer))


def start_initiator(code: str, *, context: bytes, mine: str, theirs: str) -> Session:
    """The side that asks to join. ``mine`` and ``theirs`` are certificate fingerprints:
    its own, and the one it saw the accepting machine present."""
    return _start("A", code, context, Party("A", mine), Party("B", theirs))


def start_responder(code: str, *, context: bytes, mine: str, theirs: str) -> Session:
    """The side that accepts. ``mine`` is its own certificate's fingerprint, ``theirs`` the
    one the asking machine said it has."""
    return _start("B", code, context, Party("B", mine), Party("A", theirs))
