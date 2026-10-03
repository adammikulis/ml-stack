"""Where the cluster's Ed25519 signing key lives, and who may do what with it.

The key is generated on the controller and is a different key from the cluster key, so losing
one is not losing the other. It is kept in the operating system's keystore through `keyring`
(macOS Keychain, Windows Credential Manager, Linux Secret Service). With no usable keystore it
is kept in a file encrypted under a passphrase (scrypt, ChaCha20-Poly1305, from `cryptography`,
mode 0600) with a warning; there is no plaintext path. Signing is automatic. Exporting,
rotating or revoking a key, and turning on confirm-before-signing, need a `HumanGrant`.

What this protects: the key leaking through a file copy, a backup or a repository. What it does
not: code running as you can ask the keystore for the key, unless the operating system prompts.
A hardware key (a signing device that never releases the key) is the designed next step.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ml_stack.credentials import keychain as keystore
from ml_stack.files import promote, read_json, write_json
from ml_stack.log import warn
from ml_stack.platform import private_file

from .events import BUS, Bus
from .human import HumanGrant
from .manifest import VALID_S, Carried, Entry, Signer

__all__ = ["UNLOCK_ENV", "KeyStoreError", "SigningKeys", "keyring_usable", "open_file",
           "seal_file"]

SERVICE = "ml-stack"
UNLOCK_ENV = "ML_STACK_SIGNING_PASSPHRASE"
SCHEMA_VERSION = 1
SCRYPT_N = 1 << 15


class KeyStoreError(RuntimeError):
    """The signing key cannot be stored, found or used."""


def keyring_usable() -> bool:
    """Whether `keyring` has a real backend here (not the null or failing one)."""
    try:
        import keyring
    except ImportError:
        return False
    try:
        return float(getattr(keyring.get_keyring(), "priority", 0)) > 0
    except (keyring.errors.KeyringError, AttributeError, TypeError, ValueError, OSError):
        return False


def _kdf(passphrase: str, salt: bytes, n: int) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    return Scrypt(salt=salt, length=32, n=n, r=8, p=1).derive(passphrase.encode())


def seal_file(path: Path, raw: bytes, passphrase: str) -> None:
    """Write ``raw`` to ``path`` encrypted under ``passphrase``; never overwrites."""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    salt, nonce = os.urandom(16), os.urandom(12)
    box = ChaCha20Poly1305(_kdf(passphrase, salt, SCRYPT_N)).encrypt(nonce, raw, b"ml-stack/v1")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump({"version": SCHEMA_VERSION, "kdf": "scrypt", "n": SCRYPT_N,
                   "salt": base64.b64encode(salt).decode(),
                   "nonce": base64.b64encode(nonce).decode(),
                   "ciphertext": base64.b64encode(box).decode()}, out)


def open_file(path: Path, passphrase: str) -> bytes:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    try:
        doc = json.loads(path.read_text())
        n = int(doc["n"])
        if doc["kdf"] != "scrypt" or not 1 << 12 <= n <= 1 << 20:
            raise ValueError
        key = _kdf(passphrase, base64.b64decode(doc["salt"]), n)
        return ChaCha20Poly1305(key).decrypt(base64.b64decode(doc["nonce"]),
                                             base64.b64decode(doc["ciphertext"]), b"ml-stack/v1")
    except InvalidTag:
        raise KeyStoreError("wrong passphrase for the signing key file") from None
    except (OSError, ValueError, KeyError, TypeError):
        raise KeyStoreError(f"{path} is not a signing key file this version reads") from None


def _ask_passphrase(prompt: str) -> str:
    told = os.environ.get(UNLOCK_ENV, "")
    if told:
        return told
    import getpass
    import sys
    if sys.stdin.isatty():
        return getpass.getpass(prompt)
    raise KeyStoreError("this machine has no OS keystore, so the signing key is kept in an "
                        f"encrypted file and needs a passphrase: set {UNLOCK_ENV} or run "
                        "in a terminal")


class SigningKeys:
    """The signing key of the state directory ``directory``."""

    def __init__(self, directory: Path, *, passphrase: Callable[[str], str] = _ask_passphrase,
                 bus: Bus = BUS, say: Callable[[str], None] = warn) -> None:
        self.directory, self.passphrase, self.bus, self.say = Path(directory), passphrase, bus, say
        self.meta_path = self.directory / "signing.json"
        self.file_path = self.directory / "signing.key.enc"
        self.account = "onboard-signing-" + hashlib.sha256(
            str(self.directory.resolve()).encode()).hexdigest()[:12]

    # -- the public record --
    def meta(self) -> dict[str, Any]:
        """What is known without the secret, creating the key on first use."""
        return self.peek() or self._create()

    def peek(self) -> dict[str, Any]:
        """Return existing public metadata without accessing or creating a secret."""
        doc = read_json(self.meta_path, {})
        return doc if isinstance(doc, dict) and doc.get("key_id") else {}

    @property
    def public(self) -> bytes:
        return base64.b64decode(self.meta()["public"])

    @property
    def key_id(self) -> str:
        return str(self.meta()["key_id"])

    def _write(self, doc: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_json(self.meta_path, {"schema_version": SCHEMA_VERSION, **doc})
        private_file(self.meta_path)

    # -- the secret --
    def _put(self, signer: Signer) -> str:
        raw = signer.private_raw()
        if keyring_usable():
            import keyring
            try:
                keystore.perform(keyring, "set", self.account, base64.b64encode(raw).decode())
            except keystore.KeychainError as exc:
                raise KeyStoreError(str(exc)) from None
            return "keyring"
        seal_file(self.file_path, raw, self.passphrase("passphrase for the new signing key: "))
        message = (f"no OS keystore here: the signing key is encrypted with your passphrase in "
                   f"{self.file_path}; anyone with that file and the passphrase can sign")
        self.say(message)
        self.bus.emit("onboard.signing.file_fallback", "warning", f"key:{signer.key_id}")
        return "file"

    def _get(self, store: str) -> Signer:
        if store == "keyring":
            if not keyring_usable():
                raise KeyStoreError("the signing key is in the OS keystore, which is not "
                                    "available in this session")
            import keyring
            try:
                held = keystore.perform(keyring, "get", self.account)
            except keystore.KeychainError as exc:
                raise KeyStoreError(str(exc)) from None
            if not held:
                raise KeyStoreError("the OS keystore has no signing key for this directory")
            return Signer.from_raw(base64.b64decode(held))
        return Signer.from_raw(open_file(self.file_path, self.passphrase("signing key passphrase: ")))

    def _drop(self, store: str) -> None:
        if store == "keyring" and keyring_usable():
            import keyring
            try:
                keystore.perform(keyring, "delete", self.account)
            except keystore.KeychainError as exc:
                raise KeyStoreError(str(exc)) from None
        elif store == "file":
            self.file_path.unlink(missing_ok=True)

    def _create(self) -> dict[str, Any]:
        signer = Signer.generate()
        store = self._put(signer)
        doc = {"store": store, "key_id": signer.key_id, "public": base64.b64encode(signer.public).decode(),
               "created": time.time(), "rotations": [], "revoked": [], "confirm": False}
        self._write(doc)
        self.bus.emit("onboard.signing.created", "notice", f"key:{signer.key_id}", store=store)
        return doc

    # -- routine use: no prompts --
    def sign(self, entries: Iterable[Entry], *, serial: int,
             confirm: Callable[[str], bool] | None = None) -> bytes:
        """A signed manifest of ``entries``. With confirm-before-signing on, ``confirm`` is
        asked first and must say yes."""
        doc = self.meta()
        listed = list(entries)
        if doc.get("confirm"):
            summary = f"sign a manifest of {len(listed)} files with key {doc['key_id'][:16]}"
            if confirm is None or not confirm(summary):
                raise KeyStoreError("signing needs a person's confirmation and did not get it")
        signer = self._get(doc["store"])
        carried = Carried(tuple(doc.get("rotations", [])), tuple(doc.get("revoked", [])))
        raw = signer.sign(listed, serial=serial, valid_s=VALID_S, carried=carried)
        self.bus.emit("onboard.signing.signed", "info", f"key:{doc['key_id']}",
                      files=len(listed), serial=serial)
        return raw

    def attest(self, entries: Iterable[Entry], *, serial: int, namespace: str,
               confirm: Callable[[str], bool] | None = None) -> dict[str, Any]:
        """A signed manifest plus an OpenSSH signature over those bytes, for a machine that can
        only run ``ssh-keygen -Y verify``. The key never leaves this class."""
        raw = self.sign(entries, serial=serial, confirm=confirm)
        signer = self._get(self.meta()["store"])
        return {"manifest": raw, "sshsig": signer.sshsig(raw, namespace),
                "allowed_signers": signer.ssh_public_line() + "\n", "key_id": signer.key_id}

    def entry(self, path: Path, **terms: Any) -> Entry:
        """An entry for a file; hashing needs no secret, so this never touches the key."""
        return Signer.entry_for(path, **terms)

    # -- only a person --
    def export(self, grant: HumanGrant, path: Path, passphrase: str) -> Path:
        """The key written to ``path`` encrypted under ``passphrase``."""
        doc = self.meta()
        grant.check("export", doc["key_id"])
        seal_file(path, self._get(doc["store"]).private_raw(), passphrase)
        self.bus.emit("onboard.signing.exported", "warning", f"key:{doc['key_id']}")
        return path

    def rotate(self, grant: HumanGrant) -> dict[str, Any]:
        """A new key takes over; the old one signs the announcement and is then destroyed."""
        doc = self.meta()
        grant.check("rotate", doc["key_id"])
        old = self._get(doc["store"])
        new = Signer.generate()
        statement = old.announce_rotation(new)
        store = self._put_after(doc["store"], new)
        fresh = {**doc, "store": store, "key_id": new.key_id,
                 "public": base64.b64encode(new.public).decode(),
                 "rotations": [*doc.get("rotations", []), statement][-8:]}
        self._write(fresh)
        self.bus.emit("onboard.signing.rotated", "warning", f"key:{new.key_id}", old=old.key_id)
        return fresh

    def _put_after(self, previous: str, new: Signer) -> str:
        """Store ``new`` where the old key was (replacing it)."""
        if previous == "keyring":
            if not keyring_usable():
                raise KeyStoreError("The existing signing key's OS keystore is unavailable")
            return self._put(new)
        with tempfile.TemporaryDirectory(dir=self.directory) as directory:
            target = Path(directory) / "signing.key.enc"
            seal_file(target, new.private_raw(), self.passphrase("signing key passphrase: "))
            promote(target, self.file_path)
        return "file"

    def revoke(self, grant: HumanGrant, key_id: str) -> list[str]:
        """Name a key as no longer trusted; manifests carry the list from then on."""
        doc = self.meta()
        grant.check("revoke", key_id)
        revoked = sorted({*doc.get("revoked", []), key_id})
        self._write({**doc, "revoked": revoked})
        self.bus.emit("onboard.signing.revoked", "warning", f"key:{key_id}")
        return revoked

    def require_confirmation(self, grant: HumanGrant, on: bool) -> None:
        """High-assurance mode: signing asks a person each time."""
        doc = self.meta()
        grant.check("confirm-signing", doc["key_id"])
        self._write({**doc, "confirm": bool(on)})
