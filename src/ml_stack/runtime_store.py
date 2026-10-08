"""Records kept beside installed runtimes: verification marks, rejections, deploy state and collection."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path

from ml_stack import runtime
from ml_stack.files import read_json, write_json
from ml_stack.serve.process import running_within

KEEP = 3
UNVERIFIED_GRACE_S = 3600.0
STATE = "deploy.json"
CREATED = "created.json"
TOOL = "ml-stack-runtime"
MARK = "verified.json"
REJECTED = "rejected"


def trees(root: Path | None = None) -> list[Path]:
    """Every runtime prefix under the runtimes directory, whatever its state."""
    root = root or runtime.directory()
    return sorted(prefix for family in root.glob("[0-9a-f]" * 40) if family.is_dir() and not family.is_symlink()
                  for prefix in family.iterdir() if prefix.is_dir() and not prefix.is_symlink())


def creator_record(agent: str, command: str) -> dict:
    """The record written into a tree this tool creates."""
    return {"tool": TOOL, "command": command, "agent": agent, "pid": os.getpid(), "at": time.time()}


def mark_created(prefix: Path, agent: str, command: str) -> None:
    """Record that this tool created the tree, unless a creation record is already there."""
    if not (prefix / CREATED).exists():
        write_json(prefix / CREATED, creator_record(agent, command))


def ours(prefix: Path) -> bool:
    """Whether this tool created the tree; any other tree is never changed, selected as a fallback or removed."""
    row = read_json(prefix / CREATED, {})
    return isinstance(row, dict) and row.get("tool") == TOOL


def unmanaged(root: Path | None = None) -> list[dict]:
    """Trees in the runtimes directory this tool did not create: path, size in bytes and whether a process is inside."""
    return [{"path": str(tree), "bytes": sum(f.stat().st_size for f in tree.rglob("*") if f.is_file() and not f.is_symlink()),
             "in_use": bool(running_within(tree))} for tree in trees(root) if not ours(tree)]


def mark_verified(chosen: runtime.Runtime, epoch: int = 0) -> None:
    """Record that a runtime passed its smoke, with the runtime epoch its source declares."""
    write_json(chosen.prefix / MARK, {"commit": chosen.commit, "version": chosen.version, "epoch": epoch,
                                      "identity": chosen.identity, "verified_at": time.time()})
    (chosen.prefix / MARK).chmod(0o600)


def verified_at(prefix: Path) -> float:
    """When a prefix passed its smoke, or 0.0 when it never did."""
    row = read_json(prefix / MARK, {})
    value = row.get("verified_at") if isinstance(row, dict) else None
    return float(value) if isinstance(value, (int, float)) else 0.0


def epoch_of(prefix: Path) -> int:
    """The runtime epoch a verified runtime was built with; 0 when none was recorded."""
    row = read_json(prefix / MARK, {})
    value = row.get("epoch") if isinstance(row, dict) else None
    return value if type(value) is int else 0


def discard(prefix: Path) -> None:
    """Make one of this tool's runtimes unlaunchable, then delete it; a deletion that fails part-way leaves it rejected."""
    if not ours(prefix):
        return
    with contextlib.suppress(OSError):
        reject(prefix, "discarded")
    (prefix / MARK).unlink(missing_ok=True)
    shutil.rmtree(prefix, ignore_errors=True)


def reject(prefix: Path, reason: str) -> None:
    """Mark one of this tool's runtimes as unusable for selection and for launcher fallback."""
    if not ours(prefix):
        return
    (prefix / REJECTED).write_text(reason[:500], encoding="utf-8")


def intact(prefix: Path) -> bool:
    """Whether a tree has an interpreter, an importable ml_stack, a verification mark and no rejection."""
    python = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    package = any(prefix.glob(f"{where}/ml_stack/__init__.py") for where in ("lib/python*/site-packages", "Lib/site-packages"))
    return python.is_file() and package and bool(verified_at(prefix)) and not rejected(prefix)


def rejected(prefix: Path) -> bool:
    """Whether a runtime was marked unusable."""
    return (prefix / REJECTED).exists()


def candidates(root: Path | None = None) -> list[runtime.Runtime]:
    """Verified, unrejected runtimes, newest verification first."""
    found = []
    for prefix in trees(root):
        row = read_json(prefix / MARK, {})
        if not isinstance(row, dict) or rejected(prefix) or not ours(prefix):
            continue
        try:
            found.append((verified_at(prefix), runtime.Runtime(prefix, row["commit"], row["version"], row["identity"])))
        except KeyError:
            continue
    return [chosen for stamp, chosen in sorted(found, key=lambda pair: pair[0], reverse=True) if stamp]


def read_state(root: Path | None = None) -> dict:
    """The deploy record: source checkout, launcher directory, hold and last outcome."""
    row = read_json((root or runtime.directory()) / STATE, {})
    return row if isinstance(row, dict) else {}


def write_state(update: dict, root: Path | None = None) -> dict:
    """Merge fields into the deploy record; a None value removes its key."""
    root = root or runtime.directory()
    row = {key: value for key, value in {**read_state(root), **update}.items() if value is not None}
    write_json(root / STATE, row)
    return row


def selection(root: Path | None = None) -> dict:
    """The selection file's fields, read without running the interpreter."""
    try:
        row = json.loads(((root or runtime.directory()) / "selected.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return row if isinstance(row, dict) else {}


def collect(protect: set[Path], *, keep: int = KEEP, root: Path | None = None) -> list[Path]:
    """Delete verified runtimes beyond the newest `keep` and stale unverified ones, never one in use."""
    root = root or runtime.directory()
    kept = {chosen.prefix for chosen in candidates(root)[:keep]} | protect
    gone = []
    for prefix in (tree for tree in trees(root) if ours(tree)):
        stale = not verified_at(prefix) and time.time() - prefix.stat().st_mtime > UNVERIFIED_GRACE_S
        if prefix in kept or not (stale or verified_at(prefix) or rejected(prefix)) or running_within(prefix):
            continue
        discard(prefix)
        gone.append(prefix)
    for family in root.glob("[0-9a-f]" * 40):
        if family.is_dir() and not family.is_symlink() and not any(family.iterdir()):
            family.rmdir()
    return gone
