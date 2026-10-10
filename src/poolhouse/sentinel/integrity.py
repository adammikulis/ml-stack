"""Pinning files by digest and finding the ones that are no longer what was pinned."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from poolhouse.files import sha256_file
from poolhouse.lock import only_one
from poolhouse.sentinel.events import Severity
from poolhouse.sentinel.findings import HIGH, Finding, finding
from poolhouse.sentinel.sealed import SealedFile

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

    def pin_of(self, path: Path | str) -> Pin | None:
        """The pin of the file at ``path``. A link is looked up as itself and then as the file it
        points at: the Hub cache keeps a model's bytes in ``blobs/<digest>`` and names them with a
        link in ``snapshots/``, and a pull pinned the blob."""
        where = Path(path).expanduser()
        pins = self.pins()
        found = pins.get(str(where))
        if found is None and where.is_symlink():
            found = pins.get(str(where.resolve()))
        return found

    def _save(self, pins: dict[str, Pin]) -> None:
        self._file.save({"pins": {p: v.to_json() for p, v in sorted(pins.items())}})

    def _record(self, made: Pin) -> Pin:
        with only_one(self._file.path.with_suffix(".lock")):
            pins = self.pins()
            pins[made.path] = made
            self._save(pins)
        return made

    def pin(self, path: Path | str, kind: str, source: str = "") -> Pin:
        """Record the digest of ``path`` as it is now."""
        if kind not in FILE_KINDS:
            raise ValueError(f"cannot pin a {kind!r}")
        where = Path(path).expanduser()
        size, mtime, inode, link = _state(where)
        made = Pin(str(where), kind, sha256_file(where), size, mtime, inode, link, source,
                   self.clock())
        return self._record(made)

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
        return self._record(made)

    def unpin(self, path: Path | str) -> bool:
        key = str(Path(path).expanduser())
        if key not in self.pins():
            return False
        with only_one(self._file.path.with_suffix(".lock")):
            pins = self.pins()
            gone = pins.pop(key, None)
            if gone is not None:
                self._save(pins)
        return gone is not None

    def refresh_stat(self, pin: Pin) -> None:
        """Record changed file metadata while preserving a concurrent pin update or removal."""
        size, mtime, inode, link = _state(Path(pin.path))
        refreshed = replace(pin, bytes=size, mtime_ns=mtime, inode=inode, link=link)
        if refreshed == pin or self.pins().get(pin.path) != pin:
            return
        with only_one(self._file.path.with_suffix(".lock")):
            pins = self.pins()
            if pins.get(pin.path) != pin:
                return
            pins[pin.path] = refreshed
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
