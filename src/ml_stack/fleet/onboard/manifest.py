"""The signed list of what a cluster hands to a new machine.

A manifest names files (the ml-stack wheel, a source archive, model files) with their sizes,
SHA-256 digests and per-chunk digests, and says which may be handed on to other machines. It
is signed with an Ed25519 key that belongs to the cluster; a new machine learns the matching
public key during pairing (`Grant.signing_key`) or from the bootstrap offer, and from then on
accepts file lists only if they verify under it.

Why a signature when the transport is already authenticated TLS: because the bytes may come
from any peer, a cache or a USB stick. The signature is what lets a file travel through a
machine that is not trusted to be honest. It is checked before any chunk is requested, and
every chunk is checked against the digests it lists.

The signer signs only what it has verified itself: the controller checks a wheel against the
digest the package index or release page publishes (`httpguard`), and a model file through the
guarded download pipeline and the scan/quarantine staging, before listing it. A signature
therefore says "the cluster's controller vouches for these bytes", not "these bytes are safe".

Key handling is in docs/onboarding.md ("Signing keys"). Needs the ``cryptography`` package
(extra ``fleet-tls``), which the pinned TLS already requires.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.safenames import Unsafe, safe_filename

__all__ = ["Entry", "Manifest", "ManifestError", "Signer", "SCHEMA_VERSION", "chunk_digests",
           "key_fingerprint", "load_signer", "verify"]

SCHEMA_VERSION = 1
MIN_CHUNK = 64 * 1024
MAX_CHUNK = 64 * 1024 * 1024
DEFAULT_CHUNK = 4 * 1024 * 1024
MOST_ENTRIES = 256
MOST_BYTES = 1 << 40
HEX64 = re.compile(r"[0-9a-f]{64}")
KINDS = ("wheel", "sdist", "model", "other")


class ManifestError(ValueError):
    """A manifest that is not signed by the pinned key, is out of date, or is malformed."""


@dataclass(frozen=True, slots=True)
class Entry:
    name: str
    size: int
    sha256: str
    chunk_size: int
    chunks: tuple[str, ...]
    kind: str = "other"
    shareable: bool = True
    licence: str = ""
    source: str = ""
    """Where the file comes from when it may not be shared (a gated model): the new machine
    fetches it from there with its own credentials, through the guarded pipeline."""

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256,
                "chunk_size": self.chunk_size, "chunks": list(self.chunks), "kind": self.kind,
                "shareable": self.shareable, "licence": self.licence, "source": self.source}

    @classmethod
    def from_json(cls, row: Any) -> Entry:
        if not isinstance(row, dict):
            raise ManifestError("an entry is an object")
        try:
            name = safe_filename(str(row["name"]))
        except (Unsafe, KeyError) as exc:
            raise ManifestError(f"entry name: {exc}") from None
        size, chunk = row.get("size"), row.get("chunk_size")
        if not (isinstance(size, int) and not isinstance(size, bool) and 0 <= size <= MOST_BYTES):
            raise ManifestError(f"{name}: size is not a sensible number")
        if not (isinstance(chunk, int) and not isinstance(chunk, bool)
                and MIN_CHUNK <= chunk <= MAX_CHUNK):
            raise ManifestError(f"{name}: chunk size is outside {MIN_CHUNK}..{MAX_CHUNK}")
        digest, chunks = row.get("sha256"), row.get("chunks")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise ManifestError(f"{name}: sha256 is not a digest")
        if not isinstance(chunks, list) or len(chunks) != math.ceil(size / chunk) or \
                not all(isinstance(c, str) and HEX64.fullmatch(c) for c in chunks):
            raise ManifestError(f"{name}: chunk digests do not match the size")
        kind = str(row.get("kind", "other"))
        if kind not in KINDS:
            raise ManifestError(f"{name}: unknown kind {kind!r}")
        shareable = row.get("shareable", True)
        if not isinstance(shareable, bool):
            raise ManifestError(f"{name}: shareable is true or false")
        return cls(name, size, digest, chunk, tuple(chunks), kind, shareable,
                   str(row.get("licence", ""))[:200], str(row.get("source", ""))[:500])


@dataclass(frozen=True, slots=True)
class Manifest:
    serial: int
    issued: float
    expires: float
    key_id: str
    entries: tuple[Entry, ...] = field(default_factory=tuple)

    def entry(self, name: str) -> Entry:
        for e in self.entries:
            if e.name == name:
                return e
        raise KeyError(name)

    def body(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "serial": self.serial, "issued": self.issued,
                "expires": self.expires, "key_id": self.key_id,
                "entries": [e.to_json() for e in self.entries]}


def _canonical(body: dict[str, Any]) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def key_fingerprint(public: bytes) -> str:
    """The short name a person compares for a public key: SHA-256 of its 32 raw bytes."""
    return hashlib.sha256(public).hexdigest()


def chunk_digests(path: Path, chunk_size: int) -> tuple[str, tuple[str, ...]]:
    """``(sha256 of the file, sha256 of each chunk)``, read once."""
    whole, parts = hashlib.sha256(), []
    with path.open("rb") as fh:
        while piece := fh.read(chunk_size):
            whole.update(piece)
            parts.append(hashlib.sha256(piece).hexdigest())
    return whole.hexdigest(), tuple(parts)


class Signer:
    """The cluster's signing key: makes entries and signs manifests."""

    def __init__(self, private: Any) -> None:
        self._private = private

    @classmethod
    def generate(cls) -> Signer:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        return cls(Ed25519PrivateKey.generate())

    @property
    def public(self) -> bytes:
        from cryptography.hazmat.primitives import serialization
        return self._private.public_key().public_bytes(serialization.Encoding.Raw,
                                                       serialization.PublicFormat.Raw)

    @property
    def key_id(self) -> str:
        return key_fingerprint(self.public)

    def save(self, path: Path) -> None:
        """Write the private key, raw and base64, mode 0600 from the first byte."""
        from cryptography.hazmat.primitives import serialization
        raw = self._private.private_bytes(serialization.Encoding.Raw,
                                          serialization.PrivateFormat.Raw,
                                          serialization.NoEncryption())
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(base64.b64encode(raw))

    def entry(self, path: Path, *, kind: str = "other", shareable: bool = True,
              licence: str = "", source: str = "", chunk_size: int = DEFAULT_CHUNK) -> Entry:
        whole, parts = chunk_digests(path, chunk_size)
        return Entry(safe_filename(path.name), path.stat().st_size, whole, chunk_size, parts,
                     kind, shareable, licence, source)

    def sign(self, entries: Iterable[Entry], *, serial: int, valid_s: float = 7 * 86400,
             now: float | None = None) -> bytes:
        """The signed manifest as bytes to store and serve."""
        now = time.time() if now is None else now
        manifest = Manifest(serial, now, now + valid_s, self.key_id, tuple(entries))
        body = manifest.body()
        signature = self._private.sign(_canonical(body))
        return json.dumps({"manifest": body, "signature": base64.b64encode(signature).decode()},
                          sort_keys=True).encode()


def load_signer(path: Path) -> Signer:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    return Signer(Ed25519PrivateKey.from_private_bytes(base64.b64decode(path.read_bytes())))


def verify(raw: bytes, pinned: bytes, *, now: float | None = None, min_serial: int = 0,
           revoked_keys: Iterable[str] = ()) -> Manifest:
    """The manifest in ``raw`` if ``pinned`` (a raw Ed25519 public key) signed it, it has not
    expired, its serial is at least ``min_serial`` (so an old list cannot be served again),
    and every entry is well formed. `ManifestError` otherwise, naming which."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    now = time.time() if now is None else now
    if len(raw) > 8 * 1024 * 1024:
        raise ManifestError("the manifest is too large")
    try:
        outer = json.loads(raw)
        body, signature = outer["manifest"], base64.b64decode(outer["signature"], validate=True)
    except (ValueError, KeyError, TypeError):
        raise ManifestError("not a signed manifest") from None
    if not isinstance(body, dict) or body.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("a manifest format this version does not know")
    if key_fingerprint(pinned) in set(revoked_keys):
        raise ManifestError("the pinned signing key was revoked")
    try:
        Ed25519PublicKey.from_public_bytes(pinned).verify(signature, _canonical(body))
    except (InvalidSignature, ValueError):
        raise ManifestError("the signature is not by the pinned key") from None
    serial, issued, expires = body.get("serial"), body.get("issued"), body.get("expires")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in (serial, issued, expires)):
        raise ManifestError("serial and dates are numbers")
    if body.get("key_id") != key_fingerprint(pinned):
        raise ManifestError("the manifest names a different key than signed it")
    if now > expires:
        raise ManifestError("the manifest has expired")
    if serial < min_serial:
        raise ManifestError(f"the manifest is older (serial {serial}) than one already seen "
                            f"({min_serial})")
    rows = body.get("entries")
    if not isinstance(rows, list) or len(rows) > MOST_ENTRIES:
        raise ManifestError("entries are a list of at most %d" % MOST_ENTRIES)
    entries = tuple(Entry.from_json(r) for r in rows)
    if len({e.name for e in entries}) != len(entries):
        raise ManifestError("two entries have one name")
    return Manifest(int(serial), float(issued), float(expires), str(body["key_id"]), entries)


Verifier = Callable[[bytes], Manifest]
