"""Who may open a store, and when. The one owner of opening, caching, locking and leasing.

The engine lets readers open while a writer holds a store, and refuses a second writer. What
it does not survive is a read-only handle used after another handle wrote and checkpointed
the file under it (`tests/test_graph_engine_contract.py` measures both). So the rules here:

- a writer holds the exclusive lock on the sidecar beside the store for the whole lease;
- a reader holds the shared lock for the length of each read, so no write lands mid-read;
- a cached read-only handle is reused only while the store's files are as they were when it
  opened, and is reopened otherwise.

The lock is the kernel's, so it dies with its holder. What survives a crash is the record in
the sidecar of *who* held the lease, which turns a wait into "pid 123 has held this for 4
seconds". A dead owner's record is cleared on the way in; a live owner's never is.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.files import promote
from ml_stack.graph.snapshots import WAL_SUFFIX
from ml_stack.lock import release, take

logger = logging.getLogger(__name__)

# a parked reader holds memory, so an idle one is closed; reopening is cheap
READER_IDLE_TTL_S = 30.0
WRITE_LEASE_TIMEOUT_S = 30.0
READ_TIMEOUT_S = 30.0
_POLL_S = 0.05
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


_state = threading.local()


def _mine(kind: str) -> dict[str, Any]:
    """What this thread already holds, by store: ``locks``, ``shared`` or ``leases``."""
    if not hasattr(_state, kind):
        setattr(_state, kind, {})
    return getattr(_state, kind)


def _waited(key: str, timeout_s: float, doing: str) -> LockError:
    who = holder(key)
    return LockError(f"timed out after {timeout_s:.1f}s waiting to {doing} {key}"
                     + (f", held by {who.describe()}" if who else ""))


@contextmanager
def write_lock(path: str | Path, *, timeout_s: float = WRITE_LEASE_TIMEOUT_S) -> Iterator[Path]:
    """The exclusive turn on a store, across processes, without opening it.

    Re-entrant within a thread. Raises LockError on timeout, naming the holder when one is
    recorded, and at once when this thread is reading the same store.
    """
    key = str(Path(path).expanduser())
    held = _mine("locks")
    if key in held:
        held[key] += 1
        try:
            yield lock_path(key)
        finally:
            held[key] -= 1
            if held[key] <= 0:
                del held[key]
        return
    if key in _mine("shared"):
        raise LockError(f"a write to {key} was asked for inside a read of it on the same "
                        "thread, which would wait for itself")

    Path(key).parent.mkdir(parents=True, exist_ok=True)
    # a crashed writer's pid must not masquerade as the holder in our error messages
    recover_stale(key)
    deadline = time.monotonic() + max(0.0, timeout_s)
    # "a+" and not "w": opening for write would truncate the owner record before the lock is
    # held, erasing a live holder's pid
    with lock_path(key).open("a+", encoding="utf-8") as handle:
        while not take(handle):
            if time.monotonic() >= deadline:
                raise _waited(key, timeout_s, "write")
            time.sleep(_POLL_S)
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
            release(handle)


@contextmanager
def read_lock(path: str | Path, *, timeout_s: float = READ_TIMEOUT_S) -> Iterator[None]:
    """A turn to read a store alongside other readers, and never alongside a writer.

    Re-entrant within a thread, and free inside this thread's own write lock.
    """
    key = str(Path(path).expanduser())
    shared = _mine("shared")
    if key in _mine("locks") or key in shared:
        shared[key] = shared.get(key, 0) + 1
        try:
            yield
        finally:
            shared[key] -= 1
            if shared[key] <= 0:
                del shared[key]
        return
    Path(key).parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, timeout_s)
    with lock_path(key).open("a+", encoding="utf-8") as handle:
        while not take(handle, shared=True):
            if time.monotonic() >= deadline:
                raise _waited(key, timeout_s, "read")
            time.sleep(_POLL_S)
        shared[key] = 1
        try:
            yield
        finally:
            shared.pop(key, None)
            release(handle)


def on_disk(path: str | Path) -> tuple[tuple[int, int] | None, ...]:
    """The size and modification time of a store and its log: what a cached reader opened on."""
    out = []
    for part in (Path(path).expanduser(), Path(str(Path(path).expanduser()) + WAL_SUFFIX)):
        try:
            stat = part.stat()
        except OSError:
            out.append(None)
            continue
        out.append((stat.st_mtime_ns, stat.st_size))
    return tuple(out)


def _open_waiting(opener: Callable[[Path], Any], key: Path, deadline: float) -> Any:
    """Open a store, waiting while an engine lock held outside these leases refuses it."""
    while True:
        try:
            return opener(key)
        except RuntimeError as exc:
            if "lock" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
            time.sleep(_POLL_S)


@dataclass
class _Cached:
    store: Any
    seen: tuple = ()
    refs: int = 0
    last_used: float = field(default_factory=time.monotonic)
    evicted: bool = False


class ReaderCache:
    """The cached read-only handles of one process: the cache, its guard, and the reaper
    thread that closes idle handles. Timings are instance attributes."""

    def __init__(self, *, idle_ttl_s: float = READER_IDLE_TTL_S,
                 write_timeout_s: float = WRITE_LEASE_TIMEOUT_S,
                 read_timeout_s: float = READ_TIMEOUT_S,
                 reap_interval_s: float = _REAPER_INTERVAL_S) -> None:
        self.idle_ttl_s = idle_ttl_s
        self.write_timeout_s = write_timeout_s
        self.read_timeout_s = read_timeout_s
        self.reap_interval_s = reap_interval_s
        self._readers: dict[str, _Cached] = {}
        self._guard = threading.Lock()
        self._reaper: threading.Thread | None = None

    def _close(self, key: str, entry: _Cached) -> None:
        if self._readers.get(key) is entry:
            del self._readers[key]
        with suppress(Exception):  # closing twice is not worth an error
            entry.store.close()

    def _drop(self, key: str) -> None:
        """Take a cached handle out of use; it closes now, or when its last reader leaves.

        Caller holds the guard.
        """
        entry = self._readers.pop(key, None)
        if entry is None:
            return
        entry.evicted = True
        if entry.refs == 0:
            self._close(key, entry)

    def _expire(self, now: float) -> None:
        for key, entry in list(self._readers.items()):
            if entry.refs == 0 and now - entry.last_used >= self.idle_ttl_s:
                self._close(key, entry)

    def _reap(self) -> None:
        while True:
            time.sleep(self.reap_interval_s)
            with self._guard:
                if not self._readers:
                    self._reaper = None
                    return
                self._expire(time.monotonic())

    def _ensure_reaper(self) -> None:
        if self._reaper is None or not self._reaper.is_alive():
            self._reaper = threading.Thread(target=self._reap, name="ml-stack-store-reaper",
                                            daemon=True)
            self._reaper.start()

    @contextmanager
    def reading(self, path: str | Path, opener: Callable[[Path], Any]) -> Iterator[Any]:
        """A read-only handle for the length of the block, under the shared lock.

        Handles are cached and reference counted, so readers in one process share one; it is
        reopened when the store changed on disk since it opened, and closed once idle.
        """
        key = str(Path(path).expanduser())
        if not Path(key).exists():
            # opening read-only cannot create a store, and silently creating one would hand
            # back an empty graph instead of a bad path
            raise FileNotFoundError(f"no store at {key}")
        with read_lock(key, timeout_s=self.read_timeout_s):
            with self._guard:
                self._expire(time.monotonic())
                entry = self._readers.get(key)
                now = on_disk(key)
                if entry is not None and entry.seen != now:
                    self._drop(key)
                    entry = None
                if entry is None:
                    try:
                        entry = _Cached(store=opener(Path(key)), seen=now)
                    except Exception as exc:
                        raise LockError(f"could not open {key} read-only: {exc}") from exc
                    self._readers[key] = entry
                    self._ensure_reaper()
                entry.refs += 1
            try:
                yield entry.store
            finally:
                with self._guard:
                    entry.refs -= 1
                    entry.last_used = time.monotonic()
                    now_close = entry.evicted and entry.refs == 0
                if now_close:
                    self._close(key, entry)

    @contextmanager
    def writing(self, path: str | Path, opener: Callable[[Path], Any], *,
                timeout_s: float | None = None,
                before: Callable[[Path], Any] | None = None) -> Iterator[Any]:
        """The exclusive turn, and a writable handle, for the length of the block.

        Re-entrant within a thread: a lease inside a lease on the same store is the same
        handle. ``before`` runs inside the lock and before the store is opened — where a
        snapshot goes, so that what is about to change is recoverable.
        """
        key = Path(path).expanduser()
        leases = _mine("leases")
        if str(key) in leases:
            yield leases[str(key)]
            return
        wait = self.write_timeout_s if timeout_s is None else timeout_s
        with write_lock(key, timeout_s=wait):
            deadline = time.monotonic() + max(0.0, wait)
            self.evict(key)
            if before is not None:
                before(key)
            store = _open_waiting(opener, key, deadline)
            leases[str(key)] = store
            try:
                yield store
            finally:
                leases.pop(str(key), None)
                with suppress(Exception):
                    store.close()
                self.evict(key)

    def evict(self, path: str | Path) -> None:
        """Take this process's cached reader of a store out of use."""
        with self._guard:
            self._drop(str(Path(path).expanduser()))

    def release_all(self) -> list[str]:
        """Close every cached reader this process holds. Returns what was closed."""
        with self._guard:
            keys = list(self._readers)
            for key in keys:
                self._drop(key)
        return keys


_cache = ReaderCache()


@contextmanager
def reading(path: str | Path, opener: Callable[[Path], Any]) -> Iterator[Any]:
    """A read-only handle for the length of the block, from the process-wide cache."""
    with _cache.reading(path, opener) as store:
        yield store


@contextmanager
def writing(path: str | Path, opener: Callable[[Path], Any], *,
            timeout_s: float = WRITE_LEASE_TIMEOUT_S,
            before: Callable[[Path], Any] | None = None) -> Iterator[Any]:
    """The exclusive turn, and a writable handle, for the length of the block."""
    with _cache.writing(path, opener, timeout_s=timeout_s, before=before) as store:
        yield store


def evict(path: str | Path) -> None:
    """Take this process's cached reader of a store out of use."""
    _cache.evict(path)


def release_all() -> list[str]:
    """Close every cached reader this process holds. Returns what was closed."""
    return _cache.release_all()


def publish(built: str | Path, dest: str | Path) -> Path:
    """Put a freshly built store, log and all, in ``dest``'s place under the write lock.

    ``dest``'s own log goes first, so it never replays over the store that replaces it.
    """
    built, dest = Path(built).expanduser(), Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with write_lock(dest):
        Path(str(dest) + WAL_SUFFIX).unlink(missing_ok=True)
        if Path(str(built) + WAL_SUFFIX).exists():
            promote(str(built) + WAL_SUFFIX, str(dest) + WAL_SUFFIX)
        promote(built, dest)
        evict(dest)
    return dest
