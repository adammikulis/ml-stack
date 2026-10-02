"""A JSON file sealed with an HMAC under a key kept beside it, so an edit made outside the
program is noticed."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.files import write_text

__all__ = ["Loaded", "SealedFile"]

VERSION = 1


@dataclass(frozen=True, slots=True)
class Loaded:
    """What was read: ``status`` is ``fresh`` (no file yet), ``ok``, ``recovered`` (the
    main file failed its seal and the previous copy held) or ``tampered`` (neither held)."""

    payload: dict[str, Any]
    status: str


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode()


class SealedFile:
    """``path`` holds ``{"version", "payload", "mac"}``; ``path.prev`` the copy before the
    last write; ``path.key`` the 32-byte key, mode 0600."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.prev = self.path.with_name(self.path.name + ".prev")
        self.keyfile = self.path.with_name(self.path.name + ".key")

    def _key(self) -> bytes:
        try:
            key = self.keyfile.read_bytes()
        except FileNotFoundError:
            key = b""
        if len(key) == 32:
            return key
        self.keyfile.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = secrets.token_bytes(32)
        fd = os.open(self.keyfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, key)
        finally:
            os.close(fd)
        return key

    def mac(self, data: bytes) -> str:
        """An HMAC-SHA256 of ``data`` under this file's key."""
        return hmac.new(self._key(), data, hashlib.sha256).hexdigest()

    def _seal(self, payload: dict[str, Any]) -> str:
        return hmac.new(self._key(), _canonical(payload), hashlib.sha256).hexdigest()

    def _valid(self, path: Path) -> dict[str, Any] | None:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            payload, mac = doc["payload"], str(doc["mac"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not isinstance(payload, dict) or not hmac.compare_digest(mac, self._seal(payload)):
            return None
        return payload

    def load(self) -> Loaded:
        """The payload, the previous copy's when the main file fails its seal, or an empty
        one marked ``tampered`` when no copy holds."""
        if not self.path.exists() and not self.prev.exists():
            return Loaded({}, "fresh")
        main = self._valid(self.path) if self.path.exists() else None
        if main is not None:
            return Loaded(main, "ok")
        earlier = self._valid(self.prev) if self.prev.exists() else None
        if earlier is not None:
            return Loaded(earlier, "recovered")
        return Loaded({}, "tampered")

    def save(self, payload: dict[str, Any]) -> None:
        """Seal and write ``payload``, keeping the file it replaces as ``.prev`` when that
        one holds."""
        if self.path.exists() and self._valid(self.path) is not None:
            write_text(self.prev, self.path.read_text(encoding="utf-8"))
        doc = {"version": VERSION, "payload": payload, "mac": self._seal(payload)}
        write_text(self.path, json.dumps(doc, sort_keys=True, indent=1))
