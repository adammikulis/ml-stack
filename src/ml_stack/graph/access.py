"""Who may open a store, and when. The one owner of opening, caching, locking and leasing.

The locking itself is the database's, and it works: a second process trying to open a store
somebody is writing gets `IO exception: Could not set lock on file`, immediately. What is not
the database's is everything around that — who is holding it, whether to wait, and whether a
handle nobody is using should keep holding the file. That bookkeeping is here.

The compatibility matrix, measured by driving real processes rather than assumed:

    holder      open writable     open read-only
    ------      -------------     --------------
    writable    blocked           blocked
    read-only   blocked           fine

Two consequences shape everything here. A writable handle blocks *every* other process,
readers included, so any process that keeps one open for its lifetime wedges the store for
everyone; writers take a short lease and give it back. Read-only handles compose, so readers
share a cached handle and run concurrently across as many processes as you like.

The lock is the kernel's, so it dies with its holder — a dead process cannot hold one. What
survives a crash is the sidecar beside the store recording *who* held the lease, which is what
turns an opaque IO error into "pid 123 has held this for 4 seconds". A dead owner's record is
cleared on the way in; a live owner's never is, because it is doing legitimate work.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

if os.name == "nt":  # pragma: no cover - windows
    import msvcrt

    fcntl = None
else:
    import fcntl

    msvcrt = None

# A cached reader keeps the file locked against writers in other processes, so an idle handle
# is closed rather than parked forever. Reopening is cheap.
READER_IDLE_TTL_S = 30.0
WRITE_LEASE_TIMEOUT_S = 30.0
READER_YIELD_TIMEOUT_S = 30.0
_WRITE_POLL_S = 0.25
_READER_POLL_S = 0.05
_REAPER_INTERVAL_S = 0.5


class LockError(RuntimeError):
    """The store could not be taken, or could not be opened because someone else has it."""


@dataclass(frozen=True)
class Holder:
    """The process recorded as holding the write lease."""

    pid: int
    host: str
    since: float
    alive: bool

    def describe(self) -> str:
        state = "alive" if self.alive else "dead"
        return f"pid={self.pid} host={self.host} ({state}, held {max(0.0, time.time() - self.since):.1f}s)"


def pid_alive(pid: int) -> bool:
    """Whether a process is running here. Asked, never assumed."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # it exists; it belongs to somebody else
    except OSError:
        return False
    return True


def lock_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved.with_suffix(resolved.suffix + ".lock")


def holder(path: str | Path) -> Holder | None:
    """Who holds the write lease, as recorded. Read without locking, so a blocked caller can
    name whoever is in its way."""
    try:
        raw = lock_path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not raw:
        return None
    try:
        record = json.loads(raw)
        pid = int(record["pid"])
    except (ValueError, KeyError, TypeError):
        return None
    host = str(record.get("host", ""))
    # a pid from another host says nothing about what is alive here
    alive = pid_alive(pid) if host == socket.gethostname() else True
    return Holder(pid=pid, host=host, since=float(record.get("since", 0.0)), alive=alive)


def recover_stale(path: str | Path) -> bool:
    """Clear a lease record left by a dead process. Never touches a live one."""
    who = holder(path)
    if who is None or who.alive:
        return False
    try:
        with lock_path(path).open("r+", encoding="utf-8") as handle:
            handle.truncate(0)
    except OSError:
        return False
    logger.warning("cleared a stale lock record on %s (%s)", path, who.describe())
    return True


def _take(handle) -> bool:
    try:
        if msvcrt is not None:  # pragma: no cover - windows
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        return False


def _give_back(handle) -> None:
    try:
        if msvcrt is not None:  # pragma: no cover - windows
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


_state = threading.local()


def _mine() -> dict[str, int]:
    """Locks this thread already holds, so taking one twice is not a deadlock."""
    if not hasattr(_state, "locks"):
        _state.locks = {}
    return _state.locks


@contextmanager
def write_lock(path: str | Path, *, timeout_s: float = WRITE_LEASE_TIMEOUT_S) -> Iterator[Path]:
    """The exclusive turn on a store, across processes, without opening it.

    Re-entrant within a thread. Raises LockError on timeout, naming the holder when one is
    recorded.
    """
    key = str(Path(path).expanduser())
    held = _mine()
    if key in held:
        held[key] += 1
        try:
            yield lock_path(key)
        finally:
            held[key] -= 1
            if held[key] <= 0:
                del held[key]
        return

    Path(key).parent.mkdir(parents=True, exist_ok=True)
    # a crashed writer's pid must not masquerade as the holder in our error messages
    recover_stale(key)
    deadline = time.monotonic() + max(0.0, timeout_s)
    # "a+" and not "w": opening for write would truncate the owner record before the lock is
    # held, erasing a live holder's pid
    with lock_path(key).open("a+", encoding="utf-8") as handle:
        while not _take(handle):
            if time.monotonic() >= deadline:
                who = holder(key)
                raise LockError(f"timed out after {timeout_s:.1f}s waiting for the write lock "
                                f"on {key}" + (f", held by {who.describe()}" if who else ""))
            time.sleep(_WRITE_POLL_S)
        held[key] = 1
        try:
            handle.seek(0)
            handle.truncate(0)
            handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                     "since": time.time()}))
            handle.flush()
            yield lock_path(key)
        finally:
            held.pop(key, None)
            try:
                handle.seek(0)
                handle.truncate(0)
                handle.flush()
            except OSError:
                pass
            _give_back(handle)


@dataclass
class _Cached:
    store: Any
    refs: int = 0
    last_used: float = field(default_factory=time.monotonic)
    evicted: bool = False


_readers: dict[str, _Cached] = {}
_guard = threading.Lock()
_reaper: threading.Thread | None = None


def _close(key: str, entry: _Cached) -> None:
    _readers.pop(key, None)
    try:
        entry.store.close()
    except Exception:  # noqa: BLE001 - closing twice is not worth an error
        pass


def _expire(now: float) -> None:
    for key, entry in list(_readers.items()):
        idle = entry.refs == 0 and now - entry.last_used >= READER_IDLE_TTL_S
        if idle or (entry.refs == 0 and entry.evicted):
            _close(key, entry)


def _writer_waiting(key: str) -> bool:
    who = holder(key)
    return who is not None and who.alive and who.pid != os.getpid()


def _reap() -> None:
    while True:
        time.sleep(_REAPER_INTERVAL_S)
        with _guard:
            if not _readers:
                globals()["_reaper"] = None
                return
            _expire(time.monotonic())
            for key, entry in list(_readers.items()):
                # a writer is waiting on a file this process is only holding out of habit
                if _writer_waiting(key):
                    entry.evicted = True
                    if entry.refs == 0:
                        _close(key, entry)


def _ensure_reaper() -> None:
    global _reaper
    if _reaper is None or not _reaper.is_alive():
        _reaper = threading.Thread(target=_reap, name="ml-stack-store-reaper", daemon=True)
        _reaper.start()


def _yield_to_writer(key: str, *, timeout_s: float = READER_YIELD_TIMEOUT_S) -> None:
    """Wait while somebody else is writing, rather than failing into their lock."""
    deadline = time.monotonic() + timeout_s
    while _writer_waiting(key) and time.monotonic() < deadline:
        with _guard:
            entry = _readers.get(key)
            if entry is not None and entry.refs == 0:
                _close(key, entry)
        time.sleep(_READER_POLL_S)


@contextmanager
def reading(path: str | Path, opener: Callable[[Path], Any]) -> Iterator[Any]:
    """A read-only handle for the length of the block.

    Handles are cached and reference counted, so readers in one process share one and it is
    closed once idle — a parked handle blocks writers in other processes for no benefit.
    """
    key = str(Path(path).expanduser())
    if not Path(key).exists():
        # opening read-only cannot create a store, and silently creating one would hand back an
        # empty graph instead of a bad path
        raise FileNotFoundError(f"no store at {key}")
    _yield_to_writer(key)
    with _guard:
        _expire(time.monotonic())
        entry = _readers.get(key)
        if entry is None:
            try:
                entry = _Cached(store=opener(Path(key)))
            except Exception as exc:  # noqa: BLE001 - whatever the opener raises
                who = holder(key)
                raise LockError(f"could not open {key} read-only" +
                                (f": {who.describe()} holds it" if who else f": {exc}")) from exc
            _readers[key] = entry
            _ensure_reaper()
        entry.refs += 1
    try:
        yield entry.store
    finally:
        with _guard:
            entry.refs -= 1
            entry.last_used = time.monotonic()
            now_close = entry.evicted and entry.refs == 0
        if now_close:
            _close(key, entry)


@contextmanager
def writing(path: str | Path, opener: Callable[[Path], Any], *,
            timeout_s: float = WRITE_LEASE_TIMEOUT_S,
            before: Callable[[Path], Any] | None = None) -> Iterator[Any]:
    """The exclusive turn, and a writable handle, for the length of the block.

    ``before`` runs inside the lock and before the store is opened — where a snapshot goes, so
    that what is about to change is recoverable.
    """
    key = Path(path).expanduser()
    with write_lock(key, timeout_s=timeout_s):
        with _guard:                       # our own readers hold the file against us
            for cached_key, entry in list(_readers.items()):
                if cached_key == str(key):
                    entry.evicted = True
                    if entry.refs == 0:
                        _close(cached_key, entry)
        if before is not None:
            before(key)
        store = opener(key)
        try:
            yield store
        finally:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass


def release_all() -> list[str]:
    """Close every cached reader this process holds. Returns what was closed."""
    with _guard:
        keys = list(_readers)
        for key in keys:
            _close(key, _readers[key])
    return keys
