"""Files between daemons: their digests, the paths they may be written to, the ranges a
GET asks for, and the `Fetcher` that pulls one from a peer."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .discovery import derive_token, discover
from .jobs import DaemonError
from .remote import Peer, sha256_file

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
FILE_CHUNK = 1 << 20
DIGEST_HEADER = "X-ML-Stack-SHA256"

_DIGESTS: dict[tuple[str, int, int], str] = {}
_DIGEST_LOCK = threading.Lock()
_DIGEST_CACHE_MAX = 256


def file_digest(path: Path) -> str:
    """sha256 of a file, read in fixed pieces and cached by (path, size, mtime)."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    with _DIGEST_LOCK:
        hit = _DIGESTS.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(FILE_CHUNK)
            if not block:
                break
            h.update(block)
    digest = h.hexdigest()
    with _DIGEST_LOCK:
        if len(_DIGESTS) >= _DIGEST_CACHE_MAX:
            _DIGESTS.pop(next(iter(_DIGESTS)))
        _DIGESTS[key] = digest
    return digest


def remember_digest(path: Path, digest: str) -> None:
    """Record a digest already known, so the next download need not recompute it."""
    try:
        st = path.stat()
    except OSError:
        return
    with _DIGEST_LOCK:
        if len(_DIGESTS) >= _DIGEST_CACHE_MAX:
            _DIGESTS.pop(next(iter(_DIGESTS)))
        _DIGESTS[(str(path), st.st_size, st.st_mtime_ns)] = digest


def safe_relpath(root: Path, relpath: str) -> Path:
    """Resolve ``relpath`` under ``root``, refusing anything that escapes it."""
    relpath = urllib.parse.unquote(relpath)
    if not relpath:
        raise DaemonError("empty path")
    if relpath.startswith("/") or (len(relpath) > 1 and relpath[1] == ":"):
        raise DaemonError(f"absolute path not allowed: {relpath!r}")
    for seg in Path(relpath).parts:
        if seg in ("..", "/") or not _SAFE_SEGMENT.match(seg):
            raise DaemonError(f"unsafe path segment: {seg!r}")
    target = (root / relpath).resolve()
    root_resolved = root.resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise DaemonError("path escapes the file root")
    return target


def byte_range(header: str) -> tuple[int, int | None]:
    """Start and end from a Range header, or (0, None)."""
    if not header.startswith("bytes="):
        return 0, None
    head, _, tail = header.split("=", 1)[1].split(",")[0].strip().partition("-")
    try:
        return int(head or 0), (int(tail) if tail else None)
    except ValueError:
        return 0, None


@dataclass
class Fetch:
    """One peer-to-peer transfer in flight."""

    id: str
    source: str
    relpath: str
    to: str
    state: str = "fetching"        # fetching | done | failed
    done: int = 0
    total: int = 0
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "source": self.source, "relpath": self.relpath,
                "to": self.to, "state": self.state, "done": self.done,
                "total": self.total, "error": self.error,
                "started_at": self.started_at, "finished_at": self.finished_at}


class Fetcher:
    """Pulls files from other daemons, without going through the job queue."""

    def __init__(self, files_root: Path, key: bytes | None, *, slots: int = 2,
                 discover_timeout_s: float = 2.0) -> None:
        self.files_root = files_root
        self.key = key
        self.slots = slots
        self.discover_timeout_s = discover_timeout_s
        self.fetches: dict[str, Fetch] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(slots)
        self._peers: dict[str, tuple[float, str]] = {}

    def resolve(self, name: str) -> str:
        """A peer name to a base URL, via signed discovery. Never from the caller."""
        if self.key is None:
            raise DaemonError(
                "this daemon holds no cluster key, so it cannot authenticate to a peer "
                "-- run 'ml-stack-peers init' and copy the key here")
        cached = self._peers.get(name)
        if cached and cached[0] > time.time():
            return cached[1]
        for beacon in discover(self.key, timeout_s=self.discover_timeout_s):
            self._peers[beacon.name] = (time.time() + 30.0, beacon.base_url)
        found = self._peers.get(name)
        if not found:
            raise DaemonError(f"no peer named {name!r} answered on this LAN")
        return found[1]

    def start(self, *, source: str, relpath: str, to: str,
              sha256: str = "") -> Fetch:
        target = safe_relpath(self.files_root, to)
        fetch = Fetch(id=f"{int(time.time())}-{secrets.token_hex(3)}",
                      source=source, relpath=relpath, to=to)
        with self._lock:
            self.fetches[fetch.id] = fetch
        threading.Thread(target=self._run, args=(fetch, target, sha256),
                         daemon=True, name=f"fetch-{fetch.id}").start()
        return fetch

    def _run(self, fetch: Fetch, target: Path, sha256: str) -> None:
        with self._sem:
            try:
                base_url = self.resolve(fetch.source)
                peer = Peer(base_url, derive_token(self.key))  # type: ignore[arg-type]

                def progress(done: int, total: int) -> None:
                    fetch.done, fetch.total = done, total

                target.parent.mkdir(parents=True, exist_ok=True)
                peer.pull(fetch.relpath, target, on_progress=progress)
                if sha256:
                    got = sha256_file(target)
                    if not hmac.compare_digest(got, sha256.strip().lower()):
                        target.unlink(missing_ok=True)
                        raise DaemonError(
                            f"fetched {fetch.relpath} but its digest is {got[:16]}..., "
                            f"not the {sha256[:16]}... that was asked for")
                fetch.state = "done"
                fetch.done = fetch.total = target.stat().st_size
            except Exception as exc:                  # noqa: BLE001
                fetch.state = "failed"
                fetch.error = str(exc)
            finally:
                fetch.finished_at = time.time()

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = [f for f in self.fetches.values() if f.state == "fetching"]
            return {"slots": self.slots, "fetching": len(active)}
