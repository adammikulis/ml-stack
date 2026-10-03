"""Encryption at rest for the memory store: AES-256-GCM, a per-user key in the OS keystore (or,
when the person asks for it, derived from a passphrase), and the identity a store belongs to."""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import json
import os
import re
import sys
from collections.abc import Callable
from typing import Any, Protocol

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

SERVICE = "ml-stack-memory"
KEYS_ENV = "ML_STACK_MEMORY_KEYS"
PASSPHRASE_ENV = "ML_STACK_MEMORY_PASSPHRASE"  # noqa: S105 - the name of a variable
PROFILE_ENV = "ML_STACK_MEMORY_PROFILE"
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


def _keyring() -> Any:
    try:
        import keyring
    except ImportError as exc:
        raise KeyUnavailable("the memory store needs `pip install keyring cryptography` "
                             f"({KEYS_ENV}=passphrase uses a passphrase instead)") from exc
    try:
        usable = float(getattr(keyring.get_keyring(), "priority", 0)) > 0
    except (keyring.errors.KeyringError, AttributeError, TypeError, ValueError, OSError):
        usable = False
    if not usable:
        raise KeyUnavailable("this machine has no usable OS keystore (macOS Keychain, Linux "
                             "Secret Service), so the memory store stays locked. For a person at "
                             f"a terminal: set {KEYS_ENV}=passphrase, and the store is encrypted "
                             "under a passphrase asked for each time")
    return keyring


class KeystoreKeys:
    """The key kept in the OS keystore under ``account``, with the one before it during a rekey."""

    mode = "keystore"

    def __init__(self, account: str) -> None:
        self.account = account

    def _read(self) -> dict[str, str]:
        ring = _keyring()
        try:
            held = ring.get_password(SERVICE, self.account)
        except ring.errors.KeyringError as exc:
            raise KeyUnavailable(f"the OS keystore refused the memory key ({type(exc).__name__})") from exc
        try:
            doc = json.loads(held) if held else {}
        except ValueError:
            doc = {}
        return doc if isinstance(doc, dict) else {}

    def _write(self, doc: dict[str, str]) -> None:
        ring = _keyring()
        try:
            ring.set_password(SERVICE, self.account, json.dumps(doc))
        except ring.errors.KeyringError as exc:
            raise KeyUnavailable(f"the OS keystore would not keep the memory key ({type(exc).__name__})") from exc

    def keys(self, salt: bytes, *, create: bool = False) -> list[bytes]:
        doc = self._read()
        if not doc.get("current"):
            if not create:
                return []
            self._write({"current": base64.b64encode(os.urandom(32)).decode()})
            doc = self._read()
        return [base64.b64decode(doc[k]) for k in ("current", "previous") if doc.get(k)]

    def rotate(self, salt: bytes, new: bytes | None = None) -> bytes:
        """Make a new key current, remembering the old one until ``settle``."""
        doc = self._read()
        key = new or os.urandom(32)
        self._write({"current": base64.b64encode(key).decode(),
                     **({"previous": doc["current"]} if doc.get("current") else {})})
        return key

    def settle(self) -> None:
        """Forget the previous key, once everything is written under the current one."""
        doc = self._read()
        doc.pop("previous", None)
        self._write(doc)


class PassphraseKeys:
    """A key derived from a passphrase with scrypt; ``ask`` gets the prompt."""

    mode = "passphrase"

    def __init__(self, ask: Callable[[str], str]) -> None:
        self.ask, self._next = ask, None

    def _derive(self, salt: bytes, prompt: str) -> bytes:
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        return Scrypt(salt=salt, length=32, n=SCRYPT_N, r=8, p=1).derive(self.ask(prompt).encode())

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


def default_keys(user: str, profile: str, directory: str) -> Keys:
    """The keys this process uses: the OS keystore, or a passphrase when ``$ML_STACK_MEMORY_KEYS``
    says ``passphrase``."""
    want = os.environ.get(KEYS_ENV, "keystore") or "keystore"
    if want == "passphrase":
        return PassphraseKeys(_ask)
    if want != "keystore":
        raise KeyUnavailable(f"{KEYS_ENV} is keystore or passphrase")
    where = hashlib.sha256(directory.encode()).hexdigest()[:12]
    return KeystoreKeys(f"{user}/{profile}/{where}")


def _aead(key: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise KeyUnavailable("the memory store needs `pip install cryptography`") from exc
    return AESGCM(key)


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
