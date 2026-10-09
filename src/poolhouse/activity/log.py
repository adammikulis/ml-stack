"""The encrypted, hash-chained file the activity records live in."""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poolhouse import keystore
from poolhouse.activity.schema import Entry
from poolhouse.sentinel.events import Chain, EventLog

__all__ = ["PURPOSE", "ActivityLog", "Limits", "Unreadable"]

PURPOSE = "activity"
_AAD = b"ml-stack/activity/v1"
DAY = 86400.0


class Unreadable(ValueError):
    """A record that does not open under this user's key."""


@dataclass(frozen=True, slots=True)
class Limits:
    """How much the log keeps: bytes per file, files, the age at which the current file is
    rotated, and the age past which a rotated file is removed (0: never)."""

    max_bytes: int = 1_000_000
    keep: int = 8
    max_age_s: float = 7 * DAY
    retention_s: float = 0.0


class ActivityLog(EventLog):
    """Records as JSON lines whose only plaintext fields are the time and the chain: the rest
    is AES-256-GCM under the ``activity`` subkey of the user's master key. Rotated by size
    and by age; files older than ``retention_s`` are removed from the old end with the chain
    carried across."""

    def __init__(self, directory: Path, *, key: Callable[[], bytes] | None = None,
                 limits: Limits | None = None, clock: Callable[[], float] | None = None) -> None:
        self.directory = Path(directory)
        limits = limits or Limits()
        super().__init__(self.directory / "activity.log", max_bytes=limits.max_bytes,
                         keep=limits.keep)
        self._key_from = key or self._subkey
        self._key: bytes | None = None
        self.max_age_s, self.retention_s = limits.max_age_s, limits.retention_s
        self._now = clock

    def _subkey(self) -> bytes:
        return keystore.default().salted_subkey(PURPOSE, self.directory)

    def _cipher(self) -> Any:
        if self._key is None:
            self._key = self._key_from()
        return keystore.aead(self._key)

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Encrypt ``payload`` and chain it onto the log; returns the line as written."""
        nonce = os.urandom(12)
        body = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
        ts = float(payload["ts"])
        sealed = self._cipher().encrypt(nonce, body, _AAD + repr(ts).encode())
        return self.append_record({"v": 1, "ts": ts, "ct": base64.b64encode(nonce + sealed).decode()})

    def _open(self, line: dict[str, Any]) -> dict[str, Any]:
        from cryptography.exceptions import InvalidTag
        cipher = self._cipher()
        try:
            raw = base64.b64decode(str(line["ct"]))
            plain = cipher.decrypt(raw[:12], raw[12:], _AAD + repr(float(line["ts"])).encode())
            payload = json.loads(plain)
        except (KeyError, TypeError, ValueError) as exc:
            raise Unreadable("not a record") from exc
        except InvalidTag as exc:
            raise Unreadable("does not open under this key") from exc
        if not isinstance(payload, dict):
            raise Unreadable("not a record")
        return payload

    def entries(self) -> Iterator[Entry | Unreadable]:
        """Every record in order; one that does not open comes back as the `Unreadable` that
        says so, in its place."""
        for file in self.files():
            for number, text in enumerate(file.read_text(encoding="utf-8", errors="replace")
                                          .splitlines(), 1):
                try:
                    line = json.loads(text)
                    yield Entry.from_payload(self._open(line), int(line["seq"]), str(line["hash"]))
                except Unreadable as exc:
                    yield Unreadable(f"{file.name}:{number}: {exc}")
                except (ValueError, KeyError, TypeError):
                    yield Unreadable(f"{file.name}:{number}: not a chained record")

    def _due(self) -> bool:
        if super()._due():
            return True
        if not self.path.exists() or not self.max_age_s:
            return False
        try:
            with self.path.open(encoding="utf-8") as handle:
                first = float(json.loads(handle.readline())["ts"])
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return self._clock() - first > self.max_age_s

    def _clock(self) -> float:
        return self._now() if self._now else time.time()

    def _rotate(self, chain: Chain) -> Chain:
        return self._expire(super()._rotate(chain))

    def _expire(self, chain: Chain) -> Chain:
        """Remove rotated files whose newest record is past retention, oldest first."""
        if not self.retention_s:
            return chain
        base, horizon = chain.base, self._clock() - self.retention_s
        for number in range(self.keep - 1, 0, -1):
            old = self._numbered(number)
            if not old.exists():
                continue
            try:
                last = json.loads(old.read_text(encoding="utf-8").splitlines()[-1])
                stamp, digest = float(last["ts"]), str(last["hash"])
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                break
            if stamp >= horizon:
                break
            base = digest
            self._head.save({"count": chain.count, "last": chain.last, "base": base})
            old.unlink()
        return Chain(chain.count, chain.last, base)
