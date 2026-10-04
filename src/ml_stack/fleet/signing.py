"""The release signing key and a verifier for ssh-keygen signatures (SSHSIG)."""

from __future__ import annotations

import base64
import hashlib
import struct
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

__all__ = ["NAMESPACE", "RELEASE_KEY", "SignatureError", "verify_file"]

RELEASE_KEY = ""
"""The public key releases and tracked commits are signed with, as one ``ssh-ed25519 AAAA...``
line. Empty until it is set; nothing verifies against an empty key."""

NAMESPACE = "ml-stack-release"
"""The ``ssh-keygen -Y sign -n`` namespace a release asset is signed under."""


class SignatureError(ValueError):
    pass


def _strings(blob: bytes) -> list[bytes]:
    out, at = [], 0
    while at < len(blob):
        if at + 4 > len(blob):
            raise SignatureError("the signature is truncated")
        (n,) = struct.unpack(">I", blob[at:at + 4])
        if at + 4 + n > len(blob):
            raise SignatureError("the signature is truncated")
        out.append(blob[at + 4:at + 4 + n])
        at += 4 + n
    return out


def _wire(data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + data


def _key_bytes(line: str) -> bytes:
    parts = line.split()
    if len(parts) < 2 or parts[0] != "ssh-ed25519":
        raise SignatureError("no release key is set")
    try:
        kind, raw = _strings(base64.b64decode(parts[1]))
    except (ValueError, TypeError):
        raise SignatureError("the release key is not an ssh-ed25519 key") from None
    if kind != b"ssh-ed25519" or len(raw) != 32:
        raise SignatureError("the release key is not an ssh-ed25519 key")
    return raw


def verify_file(path: Path | str, signature: str | bytes, key: str | None = None) -> None:
    """Return if ``signature`` (an ``ssh-keygen -Y sign`` file) is the key's over ``path``."""
    pinned = _key_bytes(RELEASE_KEY if key is None else key)
    text = signature.decode("utf-8", "replace") if isinstance(signature, bytes) else signature
    lines = [ln.strip() for ln in text.strip().splitlines()]
    if (len(lines) < 3 or lines[0] != "-----BEGIN SSH SIGNATURE-----"
            or lines[-1] != "-----END SSH SIGNATURE-----"):
        raise SignatureError("the signature file is not an ssh signature")
    try:
        blob = base64.b64decode("".join(lines[1:-1]), validate=True)
    except ValueError:
        raise SignatureError("the signature file is not an ssh signature") from None
    if blob[:6] != b"SSHSIG" or blob[6:10] != b"\x00\x00\x00\x01":
        raise SignatureError("the signature file is not an ssh signature")
    fields = _strings(blob[10:])
    if len(fields) != 5:
        raise SignatureError("the signature file is not an ssh signature")
    signer, namespace, reserved, algorithm, sig_blob = fields
    if _key_bytes("ssh-ed25519 " + base64.b64encode(signer).decode()) != pinned:
        raise SignatureError("the signature is from a different key")
    if namespace != NAMESPACE.encode():
        raise SignatureError("the signature is for something else")
    if algorithm not in (b"sha256", b"sha512"):
        raise SignatureError("the signature uses an unsupported hash")
    kind, raw_sig = _strings(sig_blob)
    if kind != b"ssh-ed25519":
        raise SignatureError("the signature is not ed25519")
    digest = hashlib.new(algorithm.decode())
    with Path(path).open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    signed = b"SSHSIG" + _wire(namespace) + _wire(reserved) + _wire(algorithm) + _wire(digest.digest())
    try:
        Ed25519PublicKey.from_public_bytes(pinned).verify(raw_sig, signed)
    except (InvalidSignature, ValueError):
        raise SignatureError("the signature does not match the download") from None
