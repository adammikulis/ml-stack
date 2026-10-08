"""Records kept beside installed runtimes: verification marks, rejections, deploy state and collection."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from ml_stack import runtime
from ml_stack.files import read_json, write_json
from ml_stack.serve.process import running_within

KEEP = 3
"""Verified runtimes retained besides the selected one."""

UNVERIFIED_GRACE_S = 3600.0
STATE = "deploy.json"
MARK = "verified.json"
REJECTED = "rejected"


def trees(root: Path | None = None) -> list[Path]:
    """Every runtime prefix under the runtimes directory, whatever its state."""
    root = root or runtime.directory()
    return sorted(prefix for family in root.glob("[0-9a-f]" * 40) if family.is_dir()
                  for prefix in family.iterdir() if prefix.is_dir() and not prefix.is_symlink())


def mark_verified(chosen: runtime.Runtime) -> None:
    """Record that a runtime passed its smoke."""
    write_json(chosen.prefix / MARK, {"commit": chosen.commit, "version": chosen.version,
                                      "identity": chosen.identity, "verified_at": time.time()})
    (chosen.prefix / MARK).chmod(0o600)


def verified_at(prefix: Path) -> float:
    """When a prefix passed its smoke, or 0.0 when it never did."""
    row = read_json(prefix / MARK, {})
    value = row.get("verified_at") if isinstance(row, dict) else None
    return float(value) if isinstance(value, (int, float)) else 0.0


def reject(prefix: Path, reason: str) -> None:
    """Mark a runtime as unusable for selection and for launcher fallback."""
    (prefix / REJECTED).write_text(reason[:500], encoding="utf-8")


def rejected(prefix: Path) -> bool:
    """Whether a runtime was marked unusable."""
    return (prefix / REJECTED).exists()


def candidates(root: Path | None = None) -> list[runtime.Runtime]:
    """Verified, unrejected runtimes, newest verification first."""
    found = []
    for prefix in trees(root):
        row = read_json(prefix / MARK, {})
        if not isinstance(row, dict) or rejected(prefix):
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
    for prefix in trees(root):
        stale = not verified_at(prefix) and time.time() - prefix.stat().st_mtime > UNVERIFIED_GRACE_S
        if prefix in kept or not (stale or verified_at(prefix) or rejected(prefix)) or running_within(prefix):
            continue
        shutil.rmtree(prefix, ignore_errors=True)
        gone.append(prefix)
    for family in root.glob("[0-9a-f]" * 40):
        if family.is_dir() and not any(family.iterdir()):
            family.rmdir()
    return gone
