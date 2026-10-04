"""The release signing key and an Ed25519 verifier for ssh-keygen signatures (SSHSIG)."""

from __future__ import annotations

import base64
import hashlib
import struct
from pathlib import Path

__all__ = ["NAMESPACE", "RELEASE_KEY", "SignatureError", "verify_file"]

RELEASE_KEY = ""
"""The public key releases and tracked commits are signed with, as one ``ssh-ed25519 AAAA...``
line. Empty until it is set; nothing verifies against an empty key."""

NAMESPACE = "ml-stack-release"
"""The ``ssh-keygen -Y sign -n`` namespace a release asset is signed under."""

_P = 2**255 - 19
_Q = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_I = pow(2, (_P - 1) // 4, _P)
_BY = 4 * pow(5, _P - 2, _P) % _P


class SignatureError(ValueError):
    pass


def _recover_x(y: int, sign: int) -> int | None:
    xx = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P:
        x = x * _I % _P
    if (x * x - xx) % _P:
        return None
    if x & 1 != sign:
        x = _P - x
    return x


_BX = _recover_x(_BY, 0) or 0
_BASE = (_BX, _BY, 1, _BX * _BY % _P)
_ZERO = (0, 1, 1, 0)


def _add(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = a
    x2, y2, z2, t2 = b
    pa = (y1 - x1) * (y2 - x2) % _P
    pb = (y1 + x1) * (y2 + x2) % _P
    pc = 2 * t1 * t2 * _D % _P
    pd = 2 * z1 * z2 % _P
    e, f, g, h = pb - pa, pd - pc, pd + pc, pb + pa
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _mul(k: int, point: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    out = _ZERO
    while k:
        if k & 1:
            out = _add(out, point)
        point = _add(point, point)
        k >>= 1
    return out


def _decode(raw: bytes) -> tuple[int, int, int, int] | None:
    if len(raw) != 32:
        return None
    y = int.from_bytes(raw, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    if y >= _P:
        return None
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _same(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _ed25519_ok(key: bytes, message: bytes, signature: bytes) -> bool:
    if len(signature) != 64:
        return False
    a, r = _decode(key), _decode(signature[:32])
    s = int.from_bytes(signature[32:], "little")
    if a is None or r is None or s >= _Q:
        return False
    k = int.from_bytes(hashlib.sha512(signature[:32] + key + message).digest(), "little") % _Q
    return _same(_mul(s, _BASE), _add(r, _mul(k, a)))


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
    if not _ed25519_ok(pinned, signed, raw_sig):
        raise SignatureError("the signature does not match the download")
