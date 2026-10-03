"""The signed list of what a cluster hands to a new machine.

A manifest names files (the wheel, a source archive, model files) with sizes, SHA-256
digests and per-chunk digests, and says which may be handed on to other machines. It is
signed with an Ed25519 key that belongs to the cluster; a new machine learns the public key
during pairing and from then on accepts file lists only if they verify under it. The
signature is what lets a file travel through a machine that is not trusted to be honest. The
signer lists only what it has checked itself (a wheel against the index's digest, a model
through the guarded download and scan), so a signature says "the controller vouches for these
bytes", not "these bytes are safe". Key handling: docs/onboarding.md. Needs ``cryptography``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.safenames import Unsafe, safe_filename

__all__ = [
    "SCHEMA_VERSION",
    "SHARING_LEVELS",
    "VALID_S",
    "Carried",
    "Entry",
    "Manifest",
    "ManifestError",
    "RotationAnnounced",
    "Signer",
    "chunk_digests",
    "key_fingerprint",
    "verify",
]

SCHEMA_VERSION = 1
MIN_CHUNK = 64 * 1024
MAX_CHUNK = 64 * 1024 * 1024
DEFAULT_CHUNK = 4 * 1024 * 1024
MOST_ENTRIES = 256
VALID_S = 3 * 86400
MOST_BYTES = 1 << 40
HEX64 = re.compile(r"[0-9a-f]{64}")
KINDS = ("wheel", "sdist", "model", "other")
SHARING_LEVELS = ("open", "owner", "never")


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
    sharing: str = "open"
    """``open``, ``owner`` or ``never``; see `sharing.py`."""
    licence: str = ""
    licence_url: str = ""
    source: str = ""
    """Where the file comes from when it may not be shared (a gated model): the new machine
    fetches it from there with its own credentials, through the guarded pipeline."""
    repo: str = ""
    """The repository a model file came from (``owner/name``), for the people reading a listing
    and the record a downloading machine keeps; never a path and never trusted for anything."""

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256,
                "chunk_size": self.chunk_size, "chunks": list(self.chunks), "kind": self.kind,
                "sharing": self.sharing, "licence": self.licence,
                "licence_url": self.licence_url, "source": self.source, "repo": self.repo}

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
        sharing = row.get("sharing", "owner")       # not said: treat as restricted
        if sharing not in SHARING_LEVELS:
            raise ManifestError(f"{name}: sharing is one of {SHARING_LEVELS}")
        return cls(name, size, digest, chunk, tuple(chunks), kind, sharing,
                   str(row.get("licence", ""))[:200], str(row.get("licence_url", ""))[:500],
                   str(row.get("source", ""))[:500], str(row.get("repo", ""))[:200])


@dataclass(frozen=True, slots=True)
class Manifest:
    serial: int
    issued: float
    expires: float
    key_id: str
    entries: tuple[Entry, ...] = field(default_factory=tuple)
    revoked_keys: tuple[str, ...] = ()
    """Signing keys the signer says are no longer to be trusted."""

    def entry(self, name: str) -> Entry:
        for e in self.entries:
            if e.name == name:
                return e
        raise KeyError(name)

    def body(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "serial": self.serial, "issued": self.issued,
                "expires": self.expires, "key_id": self.key_id,
                "entries": [e.to_json() for e in self.entries]}


def _ssh_string(raw: bytes) -> bytes:
    return len(raw).to_bytes(4, "big") + raw


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


@dataclass(frozen=True, slots=True)
class Carried:
    """What a manifest carries besides its files: key-change announcements and revocations."""

    rotations: tuple[dict[str, Any], ...] = ()
    revoked: tuple[str, ...] = ()


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

    def private_raw(self) -> bytes:
        """The 32 secret bytes. For the key store only: callers must not write them anywhere
        unencrypted (`signing.py` is the one place that stores them)."""
        from cryptography.hazmat.primitives import serialization
        return self._private.private_bytes(serialization.Encoding.Raw,
                                           serialization.PrivateFormat.Raw,
                                           serialization.NoEncryption())

    @classmethod
    def from_raw(cls, raw: bytes) -> Signer:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        return cls(Ed25519PrivateKey.from_private_bytes(raw))

    def ssh_public_line(self, name: str = "ml-stack") -> str:
        """The key as an OpenSSH ``allowed_signers`` line, so ``ssh-keygen -Y verify`` (on a
        machine that has no Python packages yet) can check what this key signed."""
        blob = _ssh_string(b"ssh-ed25519") + _ssh_string(self.public)
        return f"{name} ssh-ed25519 {base64.b64encode(blob).decode()}"

    def sshsig(self, data: bytes, namespace: str) -> str:
        """An OpenSSH signature (PROTOCOL.sshsig) over ``data`` by this key: the framing is
        written here, the Ed25519 signature is `cryptography`'s, and the check on the other
        side is OpenSSH's own ``ssh-keygen -Y verify``."""
        hashed = hashlib.sha512(data).digest()
        signed = (b"SSHSIG" + _ssh_string(namespace.encode()) + _ssh_string(b"")
                  + _ssh_string(b"sha512") + _ssh_string(hashed))
        signature = _ssh_string(b"ssh-ed25519") + _ssh_string(self._private.sign(signed))
        blob = (b"SSHSIG" + (1).to_bytes(4, "big")
                + _ssh_string(_ssh_string(b"ssh-ed25519") + _ssh_string(self.public))
                + _ssh_string(namespace.encode()) + _ssh_string(b"") + _ssh_string(b"sha512")
                + _ssh_string(signature))
        text = base64.b64encode(blob).decode()
        lines = [text[i:i + 70] for i in range(0, len(text), 70)]
        return "-----BEGIN SSH SIGNATURE-----\n" + "\n".join(lines) + "\n-----END SSH SIGNATURE-----\n"

    def announce_rotation(self, new: Signer, *, now: float | None = None) -> dict[str, Any]:
        """A statement, signed by this (old) key, that ``new`` takes over. Members that pinned
        this key can see the change announced; they adopt it only after a person agrees."""
        statement = {"v": 1, "old": self.key_id, "new": base64.b64encode(new.public).decode(),
                     "issued": time.time() if now is None else now}
        return {"statement": statement,
                "signature": base64.b64encode(self._private.sign(_canonical(statement))).decode()}

    @staticmethod
    def entry_for(path: Path, *, chunk_size: int = DEFAULT_CHUNK, **terms: Any) -> Entry:
        """The entry for the file at ``path`` (hashing needs no key); ``terms`` are `Entry`'s
        ``kind``, ``sharing``, ``licence``, ``licence_url`` and ``source``."""
        whole, parts = chunk_digests(path, chunk_size)
        return Entry(safe_filename(path.name), path.stat().st_size, whole, chunk_size, parts,
                     **terms)

    def entry(self, path: Path, *, chunk_size: int = DEFAULT_CHUNK, **terms: Any) -> Entry:
        return self.entry_for(path, chunk_size=chunk_size, **terms)

    def sign(self, entries: Iterable[Entry], *, serial: int, valid_s: float = VALID_S,
             now: float | None = None, carried: Carried | None = None) -> bytes:
        """The signed manifest as bytes to store and serve. It lasts ``valid_s`` (days, not
        months: a copy that leaks stops working soon). ``carried`` holds announcements of earlier key
        changes and revoked key ids, so members that were behind catch up."""
        carried = carried or Carried()
        now = time.time() if now is None else now
        manifest = Manifest(serial, now, now + valid_s, self.key_id, tuple(entries))
        body = manifest.body()
        body["public_key"] = base64.b64encode(self.public).decode()
        body["rotations"] = list(carried.rotations)
        body["revoked_keys"] = sorted(set(carried.revoked))
        signature = self._private.sign(_canonical(body))
        return json.dumps({"manifest": body, "signature": base64.b64encode(signature).decode()},
                          sort_keys=True).encode()


class RotationAnnounced(ManifestError):
    """The manifest is signed by a key the pinned one handed over to. ``new_public`` is that
    key; nothing is trusted until a person pins it."""

    def __init__(self, new_public: bytes) -> None:
        super().__init__(f"the signing key was rotated to {key_fingerprint(new_public)}; a "
                         "person must accept it (ml-stack fleet signing accept)")
        self.new_public = new_public


def _follow(pinned: bytes, rotations: Any) -> bytes:
    """The key ``pinned`` handed over to along a chain of statements, each signed by the one
    before; ``pinned`` itself if none applies."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    current = pinned
    for row in rotations if isinstance(rotations, list) else []:
        try:
            statement, signature = row["statement"], base64.b64decode(row["signature"],
                                                                      validate=True)
            if statement["old"] != key_fingerprint(current):
                continue
            Ed25519PublicKey.from_public_bytes(current).verify(signature, _canonical(statement))
            current = base64.b64decode(statement["new"], validate=True)
        except (InvalidSignature, ValueError, KeyError, TypeError):
            continue
    return current


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
        successor = _follow(pinned, body.get("rotations"))
        if successor != pinned:
            try:
                Ed25519PublicKey.from_public_bytes(successor).verify(signature, _canonical(body))
            except (InvalidSignature, ValueError):
                pass
            else:
                raise RotationAnnounced(successor) from None
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
        raise ManifestError(f"entries are a list of at most {MOST_ENTRIES}")
    entries = tuple(Entry.from_json(r) for r in rows)
    if len({e.name for e in entries}) != len(entries):
        raise ManifestError("two entries have one name")
    revoked = body.get("revoked_keys", [])
    if not isinstance(revoked, list) or not all(isinstance(k, str) for k in revoked):
        raise ManifestError("revoked keys are a list of key ids")
    return Manifest(int(serial), float(issued), float(expires), str(body["key_id"]), entries,
                    tuple(revoked))

