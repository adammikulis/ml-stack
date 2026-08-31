"""Copies of a store you can actually go back to.

A store is a binary file that git does not track, so a migration, a backfill or a re-embed
that goes wrong destroys state git cannot return. This is the part of the safety story that
covers it: clone, prove the clone opens and holds what the original held, and write down why
it was taken.

Three facts shape the design, each learned the hard way rather than reasoned about:

1. **A copy you never opened is not a backup.** A rebuild once dropped ten thousand embeddings
   while every node and edge count still looked right. So a snapshot is verified by reopening
   it on a *fresh handle* and comparing counts — a clone that will not open, or disagrees, is
   deleted and raises. Never "warn and continue".
2. **A store is two files whenever a writer holds it, or died holding it.** The main file and
   its write-ahead log. Data reaches the main file on close, so cloning it alone can copy a
   four-kilobyte stub and call it a backup. The log comes too and is folded in, leaving one
   self-contained file, which is what keeps a restore a single atomic rename.
3. **Copy-on-write makes this affordable.** On APFS, `clonefile` shares blocks: milliseconds
   and no bytes until the files diverge. Everywhere else there is no such trick, so it falls
   back to a real byte copy and says so loudly, because a backup that quietly costs a hundred
   megabytes a call is something an operator should hear about before the disk fills.

Locking is the caller's job. Take the write lock around a snapshot or a restore.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = "_backups"
WAL_SUFFIX = ".wal"
MANIFEST_SUFFIX = ".json"
KEEP = 10


class SnapshotError(RuntimeError):
    """A snapshot could not be taken, verified, or put back."""


@dataclass(frozen=True)
class Snapshot:
    """A verified copy, and what it was taken for."""

    path: str
    source: str
    reason: str
    created_at: float
    counts: dict[str, int]
    method: str
    size_bytes: int

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.created_at) / 86400.0)

    def describe(self) -> str:
        stamp = datetime.fromtimestamp(self.created_at, UTC).astimezone()
        held = " / ".join(f"{v} {k}" for k, v in sorted(self.counts.items()))
        return (f"{Path(self.path).name}\n"
                f"    taken   {stamp:%Y-%m-%d %H:%M:%S}  ({self.age_days:.1f}d ago, via {self.method})\n"
                f"    reason  {self.reason}\n    holds   {held}")


def _load_clonefile():
    """libc's ``clonefile(2)``, or None where there is no such thing."""
    if sys.platform != "darwin":
        return None
    lib = ctypes.util.find_library("c")
    if lib is None:
        return None
    try:
        fn = ctypes.CDLL(lib, use_errno=True).clonefile
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


_clonefile = _load_clonefile()


def clone_file(src: Path, dst: Path) -> str:
    """Copy one file, sharing blocks where the filesystem allows.

    Returns how it was done: ``clonefile`` (copy-on-write, effectively free) or ``copy``.
    """
    if _clonefile is not None:
        if _clonefile(os.fsencode(str(src)), os.fsencode(str(dst)), 0) == 0:
            return "clonefile"
        err = ctypes.get_errno()
        # not this filesystem, or a different volume. Anything else is a real failure —
        # permissions, a missing source — and a byte copy would hit the same wall
        if err not in (errno.ENOTSUP, errno.EXDEV):
            raise OSError(err, os.strerror(err), str(src))
        logger.warning("no copy-on-write for %s (%s): falling back to a full byte copy, which "
                       "costs real time and real disk", src, os.strerror(err))
    else:
        logger.warning("copy-on-write is APFS-only: falling back to a full byte copy of %s", src)
    shutil.copy2(src, dst)
    return "copy"


def clone_store(src: Path, dst: Path, *, fold: Any = None) -> str:
    """Clone a store and its write-ahead log, then fold the log in.

    ``fold`` is called with the clone's path and must open it writable and close it, which is
    what checkpoints the log away. Without it the log is copied and left, and the snapshot is
    two files rather than one.
    """
    method = clone_file(src, dst)
    wal = Path(str(src) + WAL_SUFFIX)
    if not wal.exists():
        return method
    clone_file(wal, Path(str(dst) + WAL_SUFFIX))
    if fold is None:
        return method
    try:
        fold(dst)
    except Exception as exc:  # noqa: BLE001 - the caller's opener, whatever it raises
        raise SnapshotError(
            f"could not fold {wal.name} into the snapshot of {src.name}: {exc}. The source "
            "holds writes that were never checkpointed, and is not safely copyable.") from exc
    return method


def remove_store(path: Path) -> None:
    """Delete a store and its log. Leaving the log lets it replay into whatever lands next."""
    Path(path).unlink(missing_ok=True)
    Path(str(path) + WAL_SUFFIX).unlink(missing_ok=True)


def snapshot_dir(source: str | Path) -> Path:
    return Path(source).expanduser().parent / SNAPSHOT_DIR


def _manifest(path: Path) -> Path:
    return path.with_suffix(path.suffix + MANIFEST_SUFFIX)


def read_manifest(path: str | Path) -> Snapshot | None:
    """What a snapshot says about itself, or None when it says nothing readable."""
    try:
        raw = json.loads(_manifest(Path(path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return Snapshot(**{**raw, "counts": dict(raw.get("counts") or {})})
    except TypeError:
        return None


def _clear(path: Path) -> None:
    path.unlink(missing_ok=True)
    Path(str(path) + WAL_SUFFIX).unlink(missing_ok=True)


def take(source: str | Path, *, reason: str, count: Any, fold: Any = None,
         keep: int = KEEP) -> Snapshot:
    """Clone a store, prove the clone holds what the original holds, and record why.

    ``count`` is called with a path and returns a mapping of what is in the store there. It
    must open the store on a *fresh handle*: only a fresh open sees what reached the disk.
    A clone that will not open, or disagrees with the source, is deleted and this raises.
    """
    src = Path(source).expanduser()
    if not src.exists():
        raise SnapshotError(f"cannot snapshot {src}: there is no store there")
    before = dict(count(src))

    target = snapshot_dir(src)
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    dst = target / f"{src.stem}.{stamp}{src.suffix}"
    # two snapshots inside one second must not quietly overwrite each other
    serial = 0
    while dst.exists():
        serial += 1
        dst = target / f"{src.stem}.{stamp}-{serial}{src.suffix}"

    started = time.perf_counter()
    try:
        method = clone_store(src, dst, fold=fold)
    except (SnapshotError, OSError):
        _clear(dst)
        raise
    spent = time.perf_counter() - started

    try:
        after = dict(count(dst))
    except Exception as exc:  # noqa: BLE001 - the caller's opener
        _clear(dst)
        raise SnapshotError(
            f"the snapshot of {src.name} would not open or read back: {exc}. Discarded — "
            "whatever was about to change must not proceed.") from exc
    if after != before:
        _clear(dst)
        raise SnapshotError(
            f"the snapshot of {src.name} does not match the source. source: {before}; "
            "clone: " f"{after}. Discarded — whatever was about to change must not proceed.")

    record = Snapshot(path=str(dst), source=str(src), reason=reason, created_at=time.time(),
                      counts=after, method=method, size_bytes=dst.stat().st_size)
    _manifest(dst).write_text(json.dumps(asdict(record), indent=1), encoding="utf-8")
    logger.info("snapshot %s -> %s (%s in %.0f ms, verified %s) reason: %s",
                src.name, dst.name, method, spent * 1000, after, reason)
    prune(src, keep=keep)
    return record


def snapshots(source: str | Path) -> list[Snapshot]:
    """Verified snapshots of a store, newest first."""
    target = snapshot_dir(source)
    if not target.is_dir():
        return []
    out = [r for p in target.iterdir()
           if p.suffix != MANIFEST_SUFFIX and (r := read_manifest(p)) is not None]
    return sorted(out, key=lambda r: r.created_at, reverse=True)


def unmanaged(source: str | Path) -> list[Path]:
    """Files in the snapshot directory that no manifest accounts for."""
    target = snapshot_dir(source)
    if not target.is_dir():
        return []
    return sorted(p for p in target.iterdir()
                  if p.suffix not in (MANIFEST_SUFFIX, WAL_SUFFIX) and read_manifest(p) is None)


def prune(source: str | Path, *, keep: int = KEEP) -> list[Path]:
    """Drop all but the newest ``keep`` snapshots. Returns what went."""
    gone: list[Path] = []
    for record in snapshots(source)[keep:]:
        path = Path(record.path)
        _clear(path)
        _manifest(path).unlink(missing_ok=True)
        gone.append(path)
    if gone:
        logger.info("pruned %d old snapshot(s) of %s", len(gone), Path(source).name)
    return gone


def restore(snapshot_path: str | Path, *, count: Any, fold: Any = None) -> Snapshot:
    """Put a snapshot back over its source, atomically, and reversibly.

    The current state is snapshotted first, because restoring the wrong one must not be the
    second unrecoverable act of the day. The swap is a rename of a clone, so the store is
    never half-written, and the snapshot is left intact for a second try.
    """
    snap = Path(snapshot_path).expanduser()
    record = read_manifest(snap)
    if record is None:
        raise SnapshotError(f"{snap} has no readable manifest: refusing to restore a file this "
                            "cannot identify or verify.")
    if not snap.exists():
        raise SnapshotError(f"snapshot {snap} is missing")

    # it was verified when taken; bit rot, a full disk, a stray editor all end the same way,
    # and a restore is the wrong moment to find out
    now = dict(count(snap))
    if now != record.counts:
        raise SnapshotError(
            f"{snap.name} no longer matches its manifest. manifest: {record.counts}; on disk: "
            f"{now}. Refusing to restore a snapshot that has changed since it was taken.")

    src = Path(record.source)
    if src.exists():
        before = take(src, reason=f"before restoring {snap.name}", count=count, fold=fold)
        logger.info("state before the restore saved as %s", Path(before.path).name)

    # staged beside the destination so the swap is a same-filesystem rename
    staging = src.with_suffix(src.suffix + f".restoring-{os.getpid()}")
    _clear(staging)
    try:
        clone_store(snap, staging, fold=fold)
        # the source's own log must go, and go first: it holds writes against the file being
        # replaced, and would otherwise replay on top of what was just recovered
        Path(str(src) + WAL_SUFFIX).unlink(missing_ok=True)
        os.replace(staging, src)
    finally:
        _clear(staging)
    logger.info("restored %s from %s", src.name, snap.name)
    return record
