"""The only module that talks to the operating system's keystore.

One item per OS user holds a random 32-byte master key. Each purpose (memory vault, fleet
signing key, wrapped credentials) gets a subkey: HKDF-SHA256 over the master with the purpose,
owner and context in ``info``; `wrap` and `unwrap` bind purpose and owner as AAD.

Nothing touches the keystore at import, construction or `status`. The master is read once per
process. Every backend call passes a gate: denial latch, hourly ceiling, cross-process lock. Reads
of an existing item and the operations that make or remove one (creates, deletes, retries after a
refusal) have separate hourly ceilings. A background process never creates the master and reads it
only after ``ml-stack-security unlock``. Each backend call is a sentinel event (purpose, outcome),
never a value.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import random
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from ml_stack import home, sentinel
from ml_stack.files import read_json, write_json
from ml_stack.lock import Busy, only_one
from ml_stack.log import warn
from ml_stack.sentinel import human
from ml_stack.sentinel.events import Event, Severity

__all__ = ["COOLDOWN_S", "ENV_NONINTERACTIVE", "ENV_NO_REAL", "NOTICE", "RATE_WINDOW_S", "READ_CEILING",
           "UNLOCK_COMMAND", "WRITE_CEILING", "Keystore", "KeystoreBusy", "KeystoreDenied", "KeystoreError",
           "KeystoreLocked", "KeystoreMissing", "KeystoreUnavailable", "Wires", "aead", "default", "hkdf",
           "interactive", "is_real", "os_user", "scrypt_key"]

SERVICE = "ml-stack"
ENV_NONINTERACTIVE = "ML_STACK_NONINTERACTIVE"
ENV_NO_REAL = "ML_STACK_NO_REAL_KEYSTORE"
"""Set by the test suite for itself and every process it starts: the machine's own keystore is
treated as absent, so a test child can never store or prompt for a Keychain item."""
REAL_BACKENDS = ("keyring.backends.macOS", "keyring.backends.SecretService",
                 "keyring.backends.Windows", "keyring.backends.kwallet")
UNLOCK_COMMAND = "ml-stack-security unlock"
READ_CEILING = 600
WRITE_CEILING = 5
NEAR_CEILING = 0.8
RATE_WINDOW_S = 3600.0
COOLDOWN_S = 600.0
FLIGHT_WAIT_S = 180.0
STATE_WAIT_S = 10.0
SCRYPT_N = 1 << 15
NOTICE = ("ml-stack will ask your computer to store one encryption key so your memory and fleet "
          "identity stay private. You will see one Keychain prompt.")
_SALT = b"ml-stack/keystore/v1"
_MAGIC = b"MKW1"
_VERSION = 1

_JITTER = random.SystemRandom()
logger = logging.getLogger("ml_stack.keystore")
logger.addHandler(logging.NullHandler())


def is_real(ring: Any) -> bool:
    """Whether ``ring`` (or any backend a chainer holds) talks to the machine's own keystore."""
    parts = getattr(ring, "backends", None) or [ring]
    return any(type(part).__module__.startswith(REAL_BACKENDS) for part in parts)


class KeystoreError(RuntimeError):
    """The keystore cannot be used, so whatever needed a key stays locked."""


class KeystoreDenied(KeystoreError):
    """The operating system refused, or the person declined the prompt."""


class KeystoreBusy(KeystoreError):
    """Too many keystore operations this hour, or another process holds the one prompt."""


class KeystoreLocked(KeystoreError):
    """A background process asked before a person ran the unlock command."""


class KeystoreMissing(KeystoreError):
    """The keystore holds no master key and the caller did not ask for one to be made."""


class KeystoreUnavailable(KeystoreError):
    """This machine has no usable keystore, or a library the keystore needs is missing."""


def os_user() -> str:
    """The OS account this process runs as: its uid and login name."""
    try:
        import pwd
        uid = os.getuid()
        try:
            return f"{uid}:{pwd.getpwuid(uid).pw_name}"
        except KeyError:
            return f"{uid}:"
    except ImportError:
        import getpass
        return "0:" + getpass.getuser()


def hkdf(key: bytes, info: bytes, *, salt: bytes = _SALT, length: int = 32) -> bytes:
    """HKDF-SHA256 (RFC 5869) of ``key``."""
    prk = hmac.new(salt or b"\0" * 32, key, hashlib.sha256).digest()
    out, block = b"", b""
    for counter in range(1, -(-length // 32) + 1):
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
    return out[:length]


def scrypt_key(passphrase: str, salt: bytes, n: int = SCRYPT_N) -> bytes:
    """A 32-byte key from a passphrase, for the explicit passphrase fallback."""
    try:
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError as exc:
        raise KeystoreUnavailable("this needs `pip install cryptography`") from exc
    return Scrypt(salt=salt, length=32, n=n, r=8, p=1).derive(passphrase.encode())


def _desktop() -> bool:
    if sys.platform == "darwin":
        return not os.environ.get("SSH_CONNECTION") and os.environ.get("XPC_SERVICE_NAME", "0") in ("0", "")
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def interactive() -> bool:
    """Whether a person could answer an OS prompt: a terminal or a desktop session, and
    ``ML_STACK_NONINTERACTIVE`` unset."""
    if os.environ.get(ENV_NONINTERACTIVE):
        return False
    try:
        if sys.stdin.isatty():
            return True
    except (AttributeError, ValueError, OSError):
        pass
    return _desktop()


def _fields(*parts: bytes) -> bytes:
    """``parts`` joined so that no two different tuples give the same bytes."""
    return b"".join(len(part).to_bytes(4, "big") + part for part in parts)


def _aad(purpose: str, owner: str) -> bytes:
    return _fields(b"ml-stack/keystore/v1", purpose.encode(), owner.encode())


class Wires(NamedTuple):
    """What a `Keystore` calls out to: ``clock``, ``say`` for the first-use sentence, ``bus``
    for audit events (None: the sentinel's), ``interactive`` (None: the module's), ``sleep`` for
    the backoff near the read ceiling."""

    clock: Callable[[], float] = time.time
    say: Callable[[str], None] = warn
    bus: Any = None
    interactive: Callable[[], bool] | None = None
    sleep: Callable[[float], None] = time.sleep


class Keystore:
    """The master key of one OS user, and the subkeys cut from it."""

    def __init__(self, *, user: str | None = None, directory: Path | None = None,
                 ceiling: int = READ_CEILING, write_ceiling: int = WRITE_CEILING,
                 wires: Wires | None = None) -> None:
        self.user = user or os_user()
        self.account = "master/" + self.user
        self._fixed, self._ceiling, self._write_ceiling = directory, ceiling, write_ceiling
        self._clock, self._say, self._bus, self._interactive, self._sleep = wires or Wires()
        self._key: bytearray | None = None
        self._legacy: dict[tuple[str, str], str | None] = {}
        self._denied = ""
        self._noticed = False
        self.created = False
        self._told: set[tuple[str, str]] = set()
        self._mutex = threading.RLock()

    # -- state files; none is read or made until something needs it --
    @property
    def directory(self) -> Path:
        return self._fixed or home.state("keystore")

    def _path(self, name: str) -> Path:
        return self.directory / name

    def _ensure_dir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        human.protect(self.directory)

    def _doc(self, name: str) -> dict[str, Any]:
        doc = read_json(self._path(name), {})
        return doc if isinstance(doc, dict) else {}

    def _put_doc(self, name: str, doc: dict[str, Any]) -> None:
        self._ensure_dir()
        write_json(self._path(name), {"schema_version": _VERSION, **doc})
        self._path(name).chmod(0o600)

    @contextmanager
    def _held(self, name: str, wait_s: float) -> Iterator[None]:
        self._ensure_dir()
        try:
            with only_one(self._path(name), timeout=wait_s, announce=lambda _m: None):
                yield
        except Busy as exc:
            raise KeystoreBusy("another ml-stack process is talking to the keystore; try again "
                               "in a moment") from exc

    def _is_interactive(self) -> bool:
        return (self._interactive or interactive)()

    # -- what is known without touching the keystore --
    def status(self) -> dict[str, Any]:
        """Provisioning, denial and rate state from the state files; never a backend call."""
        now = self._clock()
        reads, writes = self._window(now)
        until = float(self._doc("denied.json").get("until", 0))
        return {"account": self.account, "provisioned": bool(self._doc("provisioned.json").get("at")),
                "cached": self._key is not None, "denied_for_s": max(0, round(until - now)) if until > now else 0,
                "denied_in_process": bool(self._denied), "reads_this_hour": len(reads),
                "writes_this_hour": len(writes), "read_ceiling": self._ceiling,
                "write_ceiling": self._write_ceiling, "interactive": self._is_interactive()}

    def _window(self, now: float) -> tuple[list[float], list[float]]:
        """The read and write times inside the last hour; an older file's single list counts as reads."""
        doc = self._doc("rate.json")
        pick = [float(t) for t in doc.get("reads", doc.get("ops", [])) if now - float(t) < RATE_WINDOW_S]
        writes = [float(t) for t in doc.get("writes", []) if now - float(t) < RATE_WINDOW_S]
        return pick, writes

    def pending_notice(self) -> str:
        """The first-use sentence when a prompt may still come and the person has not been
        told, else an empty string. For a chat or tool reply."""
        return "" if self._noticed or self._path("noticed.json").exists() else NOTICE

    def available(self) -> bool:
        """Whether this machine has a usable keystore backend (no item is touched)."""
        try:
            self._ring()
        except KeystoreUnavailable:
            return False
        return True

    def _ring(self) -> Any:
        try:
            import keyring
            import keyring.errors
        except ImportError as exc:
            raise KeystoreUnavailable("the keystore needs `pip install keyring cryptography`") from exc
        try:
            usable = float(getattr(keyring.get_keyring(), "priority", 0)) > 0
        except (keyring.errors.KeyringError, AttributeError, TypeError, ValueError, OSError):
            usable = False
        if usable and os.environ.get(ENV_NO_REAL) and is_real(keyring.get_keyring()):
            raise KeystoreUnavailable(f"the machine's own keystore is switched off ({ENV_NO_REAL})")
        if not usable:
            raise KeystoreUnavailable("this machine has no usable OS keystore (macOS Keychain, "
                                      "Linux Secret Service)")
        return keyring

    # -- the gate every backend call passes --
    def _event(self, kind: str, purpose: str, outcome: str,
               severity: Severity = Severity.INFO) -> None:
        if kind == "refused":
            if (purpose, outcome) in self._told:
                return
            self._told.add((purpose, outcome))
        try:
            bus = self._bus or sentinel.default().bus
            bus.emit(Event(f"keystore.{kind}", severity, "keystore", f"purpose:{purpose}",
                           {"outcome": outcome}, self._clock()))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("keystore event not recorded: %s", type(exc).__name__)

    def _denial(self) -> KeystoreDenied | None:
        if self._denied:
            return KeystoreDenied(self._denied)
        until = float(self._doc("denied.json").get("until", 0))
        if until > self._clock():
            return KeystoreDenied(_refused(round(until - self._clock())))
        return None

    def _latch(self, purpose: str, cause: BaseException) -> KeystoreDenied:
        self._denied = _refused(None)
        self._put_doc("denied.json", {"until": self._clock() + COOLDOWN_S, "cause": type(cause).__name__})
        self._event("denied", purpose, type(cause).__name__, Severity.WARNING)
        return KeystoreDenied(self._denied)

    def _gate(self, purpose: str, *, person: bool = False) -> None:
        denied = self._denial()
        if denied is not None:
            self._event("refused", purpose, "denied", Severity.WARNING)
            raise denied
        if person or self._is_interactive():
            return
        if self._doc("provisioned.json").get("at"):
            return
        self._event("refused", purpose, "noninteractive", Severity.NOTICE)
        raise KeystoreLocked("ml-stack has no encryption key for this background process. A person "
                             f"runs `{UNLOCK_COMMAND}` once in a terminal to create it.")

    def _spend(self, kind: str, purpose: str) -> None:
        """Count one backend call against its class; a read after an earlier refusal is a retry
        and counts with the writes."""
        write = kind != "read" or self._path("denied.json").exists()
        with self._held("state.lock", STATE_WAIT_S):
            now = self._clock()
            reads, writes = self._window(now)
            used, ceiling = (writes, self._write_ceiling) if write else (reads, self._ceiling)
            if len(used) >= ceiling:
                self._event("refused", purpose, "busy", Severity.WARNING)
                what = "changed or retried" if write else "read"
                raise KeystoreBusy(f"ml-stack has already {what} the keystore {ceiling} times in "
                                   "the last hour and will not ask again until that passes")
            used.append(now)
            self._put_doc("rate.json", {"reads": reads, "writes": writes})
            near = len(used) / ceiling >= NEAR_CEILING and not write
        if near:
            self._sleep(_JITTER.uniform(0.05, 0.25))

    def _tell(self) -> None:
        if self._noticed:
            return
        self._noticed = True
        if not self._path("noticed.json").exists():
            self._say(NOTICE)
            self._put_doc("noticed.json", {"at": self._clock()})

    def _call(self, kind: str, purpose: str, work: Callable[[Any], Any]) -> Any:
        """One backend call, gated, counted, audited. A refusal latches and raises."""
        ring = self._ring()
        self._spend(kind, purpose)
        self._tell()
        try:
            out = work(ring)
        except ring.errors.PasswordDeleteError:
            self._event(kind, purpose, "absent")
            return None
        except (ring.errors.KeyringError, OSError) as exc:
            raise self._latch(purpose, exc) from exc
        self._event(kind, purpose, "absent" if kind == "read" and out is None else "ok")
        if kind == "read":
            self._path("denied.json").unlink(missing_ok=True)
        return out

    @contextmanager
    def _flight(self, purpose: str, *, person: bool = False) -> Iterator[None]:
        self._ring()
        with self._mutex:
            self._gate(purpose, person=person)
            with self._held("flight.lock", FLIGHT_WAIT_S):
                self._gate(purpose, person=person)
                yield

    # -- the master key --
    def _master(self, purpose: str, *, create: bool, person: bool = False) -> bytes:
        with self._mutex:
            if self._key is not None:
                return bytes(self._key)
        with self._flight(purpose, person=person):
            if self._key is not None:
                return bytes(self._key)
            found = self._call("read", purpose, lambda r: r.get_password(SERVICE, self.account))
            key = _parse(found)
            if key is None:
                if not create:
                    raise KeystoreMissing("the OS keystore holds no ml-stack key")
                if not (person or self._is_interactive()):
                    raise KeystoreLocked("ml-stack has no encryption key for this background process. "
                                         f"A person runs `{UNLOCK_COMMAND}` once in a terminal.")
                key = os.urandom(32)
                stored = "v1:" + base64.b64encode(key).decode()
                self._call("create", purpose, lambda r: r.set_password(SERVICE, self.account, stored))
                self.created = True
            self._key = bytearray(key)
            if not self._doc("provisioned.json").get("at"):
                self._put_doc("provisioned.json", {"at": self._clock(), "account": self.account})
            return key

    def provision(self) -> bool:
        """Make the master key if there is none (for the unlock command, a person present);
        whether it was made now."""
        with self._mutex:
            self._denied, self.created = "", False
        self._path("denied.json").unlink(missing_ok=True)
        self._master("unlock", create=True, person=True)
        return self.created

    def subkey(self, purpose: str, owner: str = "", *, context: bytes = b"", create: bool = True) -> bytes:
        """32 bytes derived from the master key for ``purpose`` and ``owner``; another purpose
        or owner gives a different key. ``create=False`` raises `KeystoreMissing` rather than
        making a master."""
        master = self._master(purpose, create=create)
        info = _fields(purpose.encode(), owner.encode(), context)
        return hkdf(master, info)

    def wrap(self, purpose: str, owner: str, plain: bytes) -> bytes:
        """``plain`` encrypted under the subkey of ``purpose`` and ``owner``."""
        cipher = aead(self.subkey(purpose, owner))
        nonce = os.urandom(12)
        return _MAGIC + nonce + cipher.encrypt(nonce, plain, _aad(purpose, owner))

    def unwrap(self, purpose: str, owner: str, blob: bytes) -> bytes:
        """What `wrap` sealed for the same purpose and owner, else `KeystoreError`."""
        if not blob.startswith(_MAGIC) or len(blob) < len(_MAGIC) + 12 + 16:
            raise KeystoreError("not a wrapped value")
        from cryptography.exceptions import InvalidTag
        cipher = aead(self.subkey(purpose, owner, create=False))
        try:
            return cipher.decrypt(blob[4:16], blob[16:], _aad(purpose, owner))
        except InvalidTag:
            raise KeystoreError("the wrapped value does not open under this purpose and owner") from None

    def lock(self) -> None:
        """Wipe the master key from memory; the next use reads the keystore again."""
        with self._mutex:
            if self._key is not None:
                self._key[:] = b"\0" * len(self._key)
            self._key = None

    def reset(self) -> None:
        """Delete the master item and the state files. Everything wrapped under it is lost."""
        self.lock()
        with self._flight("reset", person=True):
            self._call("delete", "reset", lambda r: r.delete_password(SERVICE, self.account))
        for name in ("provisioned.json", "denied.json", "noticed.json", "legacy.json"):
            self._path(name).unlink(missing_ok=True)
        self._denied, self._noticed = "", False

    # -- items older versions kept one per feature --
    def legacy_get(self, service: str, account: str, purpose: str) -> str | None:
        """The value of an old per-feature item, read once per process."""
        item = (service, account)
        if item not in self._legacy:
            with self._flight(purpose):
                self._legacy[item] = self._call("read", purpose, lambda r: r.get_password(service, account))
        return self._legacy[item]

    def legacy_drop(self, service: str, account: str, purpose: str) -> None:
        """Delete an old per-feature item."""
        with self._flight(purpose):
            self._call("delete", purpose, lambda r: r.delete_password(service, account))
        self._legacy[(service, account)] = None

    def migrated(self, service: str, account: str) -> bool:
        """Whether the old item is known to be gone or moved."""
        return _item(service, account) in self._doc("legacy.json").get("done", [])

    def mark_migrated(self, service: str, account: str) -> None:
        with self._held("state.lock", STATE_WAIT_S):
            done = {*self._doc("legacy.json").get("done", []), _item(service, account)}
            self._put_doc("legacy.json", {"done": sorted(done)})


def _item(service: str, account: str) -> str:
    return hashlib.sha256(f"{service}\0{account}".encode()).hexdigest()[:16]


def _refused(wait_s: int | None) -> str:
    when = "for a few minutes" if wait_s is None else f"for another {max(1, wait_s // 60)} minutes"
    return ("Your computer's keystore declined ml-stack's request, so encrypted data stays locked. "
            f"ml-stack will not ask again {when}; to try again run `{UNLOCK_COMMAND}` in a terminal.")


def _parse(held: Any) -> bytes | None:
    if not isinstance(held, str) or not held.startswith("v1:"):
        return None
    try:
        key = base64.b64decode(held[3:], validate=True)
    except ValueError:
        return None
    return key if len(key) == 32 else None


def aead(key: bytes) -> Any:
    """AES-256-GCM under ``key``."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:
        raise KeystoreUnavailable("the keystore needs `pip install cryptography`") from exc
    return AESGCM(key)


_DEFAULTS: dict[tuple[str, int], Keystore] = {}
_LOCK = threading.Lock()


def default() -> Keystore:
    """The process's keystore for the state root and keyring backend in force now."""
    try:
        import keyring
        backend = id(keyring.get_keyring())
    except ImportError:
        backend = 0
    key = (str(home.state("keystore")), backend)
    with _LOCK:
        if key not in _DEFAULTS:
            _DEFAULTS[key] = Keystore()
        return _DEFAULTS[key]
