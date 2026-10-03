"""Pinning files by digest and finding the ones that are no longer what was pinned."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.files import sha256_file
from ml_stack.sentinel.events import Severity
from ml_stack.sentinel.findings import HIGH, Finding, finding
from ml_stack.sentinel.sealed import SealedFile

__all__ = ["FILE_KINDS", "Manifest", "Pin", "check_file"]

FILE_KINDS = ("model", "binary", "artifact", "config")


@dataclass(frozen=True, slots=True)
class Pin:
    """What a file was when it was pinned."""

    path: str
    kind: str
    sha256: str
    bytes: int
    mtime_ns: int
    inode: int
    link: str
    source: str
    pinned_at: float
    origin: str = ""
    """Where a pulled file came from (its URL, or ``peer:<name>``); empty for other pins."""
    digest_from: str = ""
    """``expected`` when the digest is one the download was verified against (the Hub's, the
    release's, a signed manifest's); ``computed`` when it is that of the bytes as they arrived."""

    def to_json(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__slots__}


def _state(path: Path) -> tuple[int, int, int, str]:
    info = path.stat()
    link = str(path.readlink()) if path.is_symlink() else ""
    return info.st_size, info.st_mtime_ns, info.st_ino, link


class Manifest:
    """The pinned files of this machine, sealed on disk."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self._file = SealedFile(path)
        self.clock = clock

    def pins(self) -> dict[str, Pin]:
        return {p: Pin(**v) for p, v in self._file.load().payload.get("pins", {}).items()}

    def _save(self, pins: dict[str, Pin]) -> None:
        self._file.save({"pins": {p: v.to_json() for p, v in sorted(pins.items())}})

    def pin(self, path: Path | str, kind: str, source: str = "") -> Pin:
        """Record the digest of ``path`` as it is now."""
        if kind not in FILE_KINDS:
            raise ValueError(f"cannot pin a {kind!r}")
        where = Path(path).expanduser()
        size, mtime, inode, link = _state(where)
        made = Pin(str(where), kind, sha256_file(where), size, mtime, inode, link, source,
                   self.clock())
        pins = self.pins()
        pins[str(where)] = made
        self._save(pins)
        return made

    def pin_verified(self, path: Path | str, kind: str, sha256: str, source: str,
                     **notes: str) -> Pin:
        """Record ``sha256`` for ``path`` without hashing it again: for a file whose digest a
        download already checked (or, with ``digest_from="computed"``, that the caller hashed
        itself). ``notes`` are ``origin`` and ``digest_from``."""
        if kind not in FILE_KINDS:
            raise ValueError(f"cannot pin a {kind!r}")
        where = Path(path).expanduser()
        size, mtime, inode, link = _state(where)
        made = Pin(str(where), kind, sha256.lower(), size, mtime, inode, link, source,
                   self.clock(), **notes)
        pins = self.pins()
        pins[str(where)] = made
        self._save(pins)
        return made

    def unpin(self, path: Path | str) -> bool:
        pins = self.pins()
        gone = pins.pop(str(Path(path).expanduser()), None)
        if gone is not None:
            self._save(pins)
        return gone is not None

    def refresh_stat(self, pin: Pin) -> None:
        """Record a new mtime and inode for a file whose digest still matches."""
        size, mtime, inode, link = _state(Path(pin.path))
        pins = self.pins()
        pins[pin.path] = Pin(pin.path, pin.kind, pin.sha256, size, mtime, inode, link,
                             pin.source, pin.pinned_at, pin.origin, pin.digest_from)
        self._save(pins)


def _changed(pin: Pin, what: str, now: dict[str, Any]) -> Finding:
    return finding(f"integrity.{what}", Severity.CRITICAL, (pin.kind, pin.path), HIGH, {"pinned_sha256": pin.sha256, "pinned_bytes": pin.bytes, **now},
                   path=pin.path, move="link" if what == "link_retargeted" else "file")


def check_file(pin: Pin, *, deep: bool = True) -> Finding | None:
    """A finding when the file at ``pin.path`` is not what was pinned, else None.

    A missing file, a symlink that now points elsewhere, or a digest that differs is a
    finding. With ``deep=False`` a file whose size, mtime and inode are unchanged is not
    hashed again; with ``deep=True`` every file is.
    """
    path = Path(pin.path)
    if not os.path.lexists(path) or not path.exists():
        return _changed(pin, "missing", {})
    try:
        size, mtime, inode, link = _state(path)
    except OSError as exc:
        return _changed(pin, "unreadable", {"why": str(exc)})
    if link != pin.link:
        return _changed(pin, "link_retargeted", {"link": link, "pinned_link": pin.link})
    if not deep and (size, mtime, inode) == (pin.bytes, pin.mtime_ns, pin.inode):
        return None
    digest = sha256_file(path)
    if digest != pin.sha256:
        return _changed(pin, "content_changed", {"sha256": digest, "bytes": size})
    return None
