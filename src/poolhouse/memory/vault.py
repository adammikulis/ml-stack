"""Encryption at rest for the memory store: AES-256-GCM, a subkey of the user's master key in the OS keystore (or,
when the person asks for it, derived from a passphrase), and the identity a store belongs to."""

from __future__ import annotations

import base64
import contextlib
import getpass
import hashlib
import hmac
import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from poolhouse import files, keystore, legacy, lock

__all__ = [
    "KEYS_ENV",
    "MODES",
    "PASSPHRASE_ENV",
    "PROFILE_ENV",
    "BadSeal",
    "KeyUnavailable",
    "Keys",
    "KeystoreKeys",
    "PassphraseKeys",
    "default_keys",
    "identity",
    "open_blob",
    "seal_blob",
    "valid_profile",
]

SERVICE = legacy.MEMORY_SERVICE
PURPOSE = "memory"
KEYS_ENV = "POOLHOUSE_MEMORY_KEYS"
PASSPHRASE_ENV = "POOLHOUSE_MEMORY_PASSPHRASE"  # noqa: S105 - the name of a variable
PROFILE_ENV = "POOLHOUSE_MEMORY_PROFILE"
MAGIC = b"MLM1"
MODES = {"keystore": 0, "passphrase": 1}
SALT, KID, NONCE = 16, 8, 12
HEAD = len(MAGIC) + 1 + SALT + KID
SCRYPT_N = 1 << 15
_PROFILE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")


class KeyUnavailable(RuntimeError):
    """The key cannot be had, so the store is neither read nor written."""


class BadSeal(RuntimeError):
    """The bytes do not authenticate under any key offered."""


def identity() -> str:
    """The OS account this process runs as: its real uid and login name, never a request field."""
    try:
        import pwd
        uid = os.getuid()
        try:
            name = pwd.getpwuid(uid).pw_name
        except KeyError:
            name = ""
        return f"{uid}:{name}"
    except ImportError:
        return "0:" + getpass.getuser()


def valid_profile(name: str) -> str:
    """``name`` as a profile, or ``ValueError`` when it is not 1-32 of a-z, 0-9, ``_``, ``-``."""
    if not _PROFILE.fullmatch(name):
        raise ValueError("a profile is 1-32 characters of a-z, 0-9, _ and -")
    return name


class Keys(Protocol):
    """Where a store's keys come from. ``keys`` returns the key to write with first, then any
    that older bytes may still be under; it raises ``KeyUnavailable`` rather than guessing."""

    mode: str

    def keys(self, salt: bytes, *, create: bool = False) -> list[bytes]: ...

    def rotate(self, salt: bytes, new: bytes | None = None) -> bytes: ...

    def settle(self) -> None: ...


def _locked(exc: keystore.KeystoreError) -> KeyUnavailable:
    if isinstance(exc, keystore.KeystoreUnavailable):
        return KeyUnavailable(f"{exc}, so the memory store stays locked. For a person at a "
                              f"terminal: set {KEYS_ENV}=passphrase")
    return KeyUnavailable(str(exc))


class KeystoreKeys:
    """The store key: a subkey of the user's master key (`poolhouse.keystore`), cut for this
    account and the store's own salt, so a new salt is a new key."""

    mode = "keystore"

    def __init__(self, account: str, *, owner: str = "", files: tuple[Path, ...] = (),
                 purpose: str = PURPOSE, ks: keystore.Keystore | None = None) -> None:
        self.account, self.owner, self.files, self.purpose = account, owner, files, purpose
        self._ks = ks

    @property
    def ks(self) -> keystore.Keystore:
        return self._ks or keystore.default()

    def _derive(self, salt: bytes, *, create: bool) -> bytes:
        try:
            return self.ks.subkey(self.purpose, self.account, context=salt, create=create)
        except keystore.KeystoreMissing:
            return b""
        except keystore.KeystoreError as exc:
            raise _locked(exc) from exc

    def keys(self, salt: bytes, *, create: bool = False) -> list[bytes]:
        older = self._migrate()
        key = self._derive(salt, create=create)
        return [*([key] if key else []), *older]

    def rotate(self, salt: bytes, new: bytes | None = None) -> bytes:
        """The key for the fresh ``salt`` the store chose."""
        return new or self._derive(salt, create=True)

    def settle(self) -> None:
        return None

    def _migrate(self) -> list[bytes]:
        """Re-wrap a store kept under the random key an older version left in the keystore,
        then delete that item once every file reads back under the new key. Returns the old
        keys while the bytes a caller already holds may still be under them."""
        ks = self.ks
        if ks.migrated(SERVICE, self.account) or not self.files:
            return []
        try:
            held = ks.legacy_get(SERVICE, self.account, self.purpose)
            if held is None:
                ks.mark_migrated(SERVICE, self.account)
                return []
            doc = json.loads(held)
            old = [base64.b64decode(doc[k]) for k in ("current", "previous") if doc.get(k)]
            if not all(self._rewrap(path, old) for path in self.files):
                return old
            with contextlib.suppress(keystore.KeystoreError):
                ks.legacy_drop(SERVICE, self.account, self.purpose)
                ks.mark_migrated(SERVICE, self.account)
            return old
        except keystore.KeystoreError as exc:
            raise _locked(exc) from exc
        except (ValueError, KeyError, TypeError):
            return []

    def _rewrap(self, path: Path, old: list[bytes]) -> bool:
        """Whether ``path`` is absent or now opens under the new key."""
        with lock.only_one(path.parent / "store.lock", timeout=10, announce=lambda _m: None):
            try:
                blob = path.read_bytes()
            except FileNotFoundError:
                return True
            mode, salt = header(blob)
            if mode != "keystore":
                return True
            fresh = self._derive(salt, create=True)
            try:
                open_blob(blob, [fresh], owner=self.owner)
                return True
            except BadSeal:
                pass
            try:
                plain, _ = open_blob(blob, old, owner=self.owner)
            except BadSeal:
                return False
            with files.writing(path) as tmp:
                tmp.write_bytes(seal_blob(plain, fresh, mode="keystore", salt=salt, owner=self.owner))
                tmp.chmod(0o600)
            try:
                return open_blob(path.read_bytes(), [fresh], owner=self.owner)[0] == plain
            except BadSeal:
                return False


class PassphraseKeys:
    """A key derived from a passphrase with scrypt; ``ask`` gets the prompt."""

    mode = "passphrase"

    def __init__(self, ask: Callable[[str], str]) -> None:
        self.ask, self._next = ask, None

    def _derive(self, salt: bytes, prompt: str) -> bytes:
        return keystore.scrypt_key(self.ask(prompt), salt, SCRYPT_N)

    def keys(self, salt: bytes, *, create: bool = False) -> list[bytes]:
        return [self._derive(salt, "memory passphrase: ")]

    def rotate(self, salt: bytes, new: bytes | None = None) -> bytes:
        return new or self._derive(salt, "new memory passphrase: ")

    def settle(self) -> None:
        return None


def _ask(prompt: str) -> str:
    told = os.environ.get(PASSPHRASE_ENV, "")
    if told:
        return told
    if sys.stdin.isatty():
        return getpass.getpass(prompt)
    raise KeyUnavailable(f"the memory store is under a passphrase: set {PASSPHRASE_ENV} or run in a terminal")


def default_keys(user: str, profile: str, directory: str, stored: tuple[Path, ...] = (),
                 purpose: str = PURPOSE) -> Keys:
    """The keys this process uses: the OS keystore, or a passphrase when ``$POOLHOUSE_MEMORY_KEYS``
    says ``passphrase``."""
    want = os.environ.get(KEYS_ENV, "keystore") or "keystore"
    if want == "passphrase":
        return PassphraseKeys(_ask)
    if want != "keystore":
        raise KeyUnavailable(f"{KEYS_ENV} is keystore or passphrase")
    where = hashlib.sha256(directory.encode()).hexdigest()[:12]
    return KeystoreKeys(f"{user}/{profile}/{where}", owner=f"{user}|{profile}", files=tuple(stored),
                        purpose=purpose)


def _aead(key: bytes) -> Any:
    try:
        return keystore.aead(key)
    except keystore.KeystoreUnavailable as exc:
        raise KeyUnavailable("the memory store needs `pip install cryptography`") from exc


def _kid(key: bytes) -> bytes:
    return hmac.new(key, b"ml-stack-memory-key-id", hashlib.sha256).digest()[:KID]


def seal_blob(plain: bytes, key: bytes, *, mode: str, salt: bytes, owner: str) -> bytes:
    """``plain`` encrypted and authenticated; the header and ``owner`` are authenticated too."""
    head = MAGIC + bytes([MODES[mode]]) + salt + _kid(key)
    nonce = os.urandom(NONCE)
    return head + nonce + _aead(key).encrypt(nonce, plain, head + b"\0" + owner.encode())


def header(blob: bytes) -> tuple[str, bytes]:
    """``(mode, salt)`` from the front of ``blob``; ``BadSeal`` when it is not one."""
    if len(blob) < HEAD + NONCE + 16 or not blob.startswith(MAGIC) or blob[4] not in MODES.values():
        raise BadSeal("not a memory store file")
    return next(m for m, n in MODES.items() if n == blob[4]), blob[5:5 + SALT]


def open_blob(blob: bytes, candidates: list[bytes], *, owner: str) -> tuple[bytes, bytes]:
    """``(plain, key)``: the bytes decrypted under the first of ``candidates`` that
    authenticates, else ``BadSeal``."""
    header(blob)
    head, nonce, body = blob[:HEAD], blob[HEAD:HEAD + NONCE], blob[HEAD + NONCE:]
    from cryptography.exceptions import InvalidTag
    for key in candidates:
        if _kid(key) != head[-KID:]:
            continue
        try:
            return _aead(key).decrypt(nonce, body, head + b"\0" + owner.encode()), key
        except InvalidTag:
            break
    raise BadSeal("the store failed authentication")
