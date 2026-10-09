"""Authenticated encryption of fleet traffic: AES-256-GCM under a key derived from a cluster secret.

A sealed message is a 12 byte random nonce followed by the ciphertext and its tag. The
associated data is authenticated and not sent: for a request it is the method, target, host,
timestamp and nonce the signature covers, for a response the request's nonce and the status.
"""

from __future__ import annotations

import secrets

from poolhouse import keystore

__all__ = ["HEADER", "SealError", "box_key", "open_", "request_data", "response_data", "seal"]

HEADER = "X-Poolhouse-Sealed"
"""Sent on a request whose body is sealed and whose answer should be, and on an answer that is."""
NONCE_BYTES = 12
INFO = b"poolhouse-seal-v1"


class SealError(ValueError):
    """A message that does not authenticate, or is too short to be one."""


def box_key(secret: str) -> bytes:
    """The encryption key two ends derive from the same request secret; never the MAC key."""
    return keystore.hkdf(secret.encode(), INFO)


def request_data(method: str, target: str, host: str, at: str, nonce: str) -> bytes:
    """The associated data of a request body."""
    return "\n".join(("poolhouse-request", method.upper(), target, host, at, nonce)).encode()


def response_data(nonce: str, status: int) -> bytes:
    """The associated data of a response body: tied to the request it answers."""
    return f"poolhouse-response\n{nonce}\n{status}".encode()


def seal(key: bytes, plaintext: bytes, data: bytes) -> bytes:
    """``plaintext`` encrypted and authenticated together with ``data``."""
    nonce = secrets.token_bytes(NONCE_BYTES)
    return nonce + keystore.aead(key).encrypt(nonce, plaintext, data)


def open_(key: bytes, blob: bytes, data: bytes) -> bytes:
    """The plaintext of ``blob``, or `SealError` when it was altered or sealed for other data."""
    if len(blob) < NONCE_BYTES + 16:
        raise SealError("the message is too short to be sealed")
    from cryptography.exceptions import InvalidTag

    try:
        return keystore.aead(key).decrypt(blob[:NONCE_BYTES], blob[NONCE_BYTES:], data)
    except InvalidTag as exc:
        raise SealError("the message did not authenticate") from exc
