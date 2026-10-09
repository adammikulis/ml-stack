"""The reuse store: entries the runner wrote, a hash chain over them, in-flight claims, incidents."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import psutil
import testreuse_key as keys

from poolhouse.activity import reuse
from poolhouse.activity.reuse import SCHEMA, entry_hash, row_hash

PENDING_S = 10.0


@dataclass(frozen=True)
class Hit:
    """A verified passing entry."""

    id: str
    entry: dict


def process_start(pid: int) -> float:
    """The start time of ``pid``, or 0.0 when it does not exist or has exited unreaped."""
    try:
        process = psutil.Process(pid)
        return 0.0 if process.status() == psutil.STATUS_ZOMBIE else process.create_time()
    except (psutil.Error, OSError):
        return 0.0


def alive(pid: int, started: float) -> bool:
    """Whether ``pid`` is the process that started at ``started``."""
    return pid > 0 and started > 0 and abs(process_start(pid) - started) < 1.0


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a sibling file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class Store:
    """One project's reuse store under ``folder``."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.chain = folder / "chain.jsonl"

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the store's exclusive lock."""
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / "lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def append(self, kind: str, entry: dict) -> dict:
        """Append a chain row for ``entry`` and write the entry; the caller holds the lock."""
        text = self.chain.read_text(encoding="utf-8") if self.chain.is_file() else ""
        if text and not text.endswith("\n"):
            self.chain.write_text(text[:text.rfind("\n") + 1], encoding="utf-8")
        rows = reuse.rows(self.folder)
        row = {"seq": len(rows), "kind": kind, "lookup": entry["lookup"], "id": entry["entry_sha256"][:20],
               "entry_sha256": entry["entry_sha256"], "prev": rows[-1]["row_sha256"] if rows else "",
               "at": entry["created"]}
        row["row_sha256"] = row_hash(row)
        write_atomic(self.folder / "entries" / f"{row['id']}.json", json.dumps(entry, sort_keys=True))
        with self.chain.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def put(self, fields: dict, kind: str = "pass") -> str:
        """Write an entry from the runner's facts and return its id; a pass becomes the lookup's latest."""
        with self.locked():
            rows = reuse.rows(self.folder)
            entry = {**fields, "schema": SCHEMA, "created": time.time(),
                     "prev": rows[-1]["row_sha256"] if rows else ""}
            entry["entry_sha256"] = entry_hash(entry)
            row = self.append(kind, entry)
            if kind == "pass":
                write_atomic(self.folder / "latest" / entry["lookup"], row["id"])
                marker = self.folder / "disabled" / f"{entry['lookup']}.json"
                if marker.is_file() and json.loads(marker.read_text())["manifest_digest"] != entry["manifest_digest"]:
                    marker.unlink(missing_ok=True)
            return row["id"]

    def hit(self, lookup: str, root: Path, closures: keys.Closures) -> tuple[Hit | None, str]:
        """The verified passing entry for ``lookup`` against the current tree, else why not."""
        try:
            entry_id = (self.folder / "latest" / lookup).read_text(encoding="utf-8").strip()
        except OSError:
            return None, "no earlier run"
        entry = reuse.entry(self.folder, entry_id)
        if entry is None or entry["lookup"] != lookup or entry["outcome"] != "pass":
            return None, "stored entry failed verification"
        marker = self.folder / "disabled" / f"{lookup}.json"
        if marker.is_file():
            try:
                if json.loads(marker.read_text())["manifest_digest"] == entry["manifest_digest"]:
                    return None, "reuse disabled after a canary mismatch"
            except (OSError, ValueError, KeyError):
                return None, "reuse disabled"
        if keys.manifest_digest(entry["manifest"]) != entry["manifest_digest"]:
            return None, "stored entry failed verification"
        changed = keys.manifest_holds(root, entry["manifest"], closures)
        return (None, changed) if changed else (Hit(entry_id, entry), "")

    def disable(self, lookup: str, entry: dict, detail: str) -> None:
        """Turn reuse off for ``lookup`` while its manifest stays what ``entry`` recorded."""
        write_atomic(self.folder / "disabled" / f"{lookup}.json", json.dumps({
            "manifest_digest": entry["manifest_digest"], "entry": entry["entry_sha256"][:20],
            "detail": detail, "at": time.time()}))
        with self.locked():
            rows = reuse.rows(self.folder)
            incident = {**entry, "outcome": "canary-mismatch", "created": time.time(),
                        "prev": rows[-1]["row_sha256"] if rows else "", "counts": {"detail": detail}}
            incident["entry_sha256"] = entry_hash(incident)
            self.append("incident", incident)

    def claim(self, lookup: str, owner: dict) -> dict | None:
        """Claim ``lookup`` for a run: None when claimed, else the live owner's record."""
        path = self.folder / "inflight" / f"{lookup}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"pid": os.getpid(), "started": process_start(os.getpid()), "at": time.time(), **owner}
        while True:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                current = self.claimed(lookup)
                if current is not None:
                    return current
                continue
            with os.fdopen(descriptor, "w") as handle:
                handle.write(json.dumps(record))
            return None

    def claimed(self, lookup: str) -> dict | None:
        """The live claim on ``lookup``; a claim whose owner is gone is removed and reads as None."""
        path = self.folder / "inflight" / f"{lookup}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            record = {}
            if time.time() - path.stat().st_mtime < PENDING_S:
                return {"pid": 0, "agent": {}, "pending": True}
        if record and alive(int(record.get("pid", 0)), float(record.get("started", 0))):
            return record
        path.unlink(missing_ok=True)
        return None

    def release(self, lookup: str) -> None:
        """Drop this process's claim on ``lookup``."""
        path = self.folder / "inflight" / f"{lookup}.json"
        with contextlib.suppress(OSError, ValueError):
            if json.loads(path.read_text(encoding="utf-8")).get("pid") == os.getpid():
                path.unlink(missing_ok=True)

    def note_thread(self, lookup: str, seq: int) -> None:
        """Remember the board thread of the run for ``lookup`` and show it on this process's claim."""
        write_atomic(self.folder / "threads" / lookup, str(seq))
        path = self.folder / "inflight" / f"{lookup}.json"
        with contextlib.suppress(OSError, ValueError):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("pid") == os.getpid():
                write_atomic(path, json.dumps({**record, "thread": seq}))

    def thread_of(self, lookup: str) -> int:
        """The board thread most recently opened for ``lookup``, or 0."""
        try:
            return int((self.folder / "threads" / lookup).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
