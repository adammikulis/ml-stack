"""Moving a suspect file aside and putting it back, byte for byte."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.files import CrossDevice, promote, sha256_file

__all__ = ["HELD_DIR", "MoveRefused", "hold", "managed_roots", "purge", "restore"]

HELD_DIR = ".ml-stack-quarantine"


class MoveRefused(OSError):
    """A file could not be moved aside or back, and why."""


def managed_roots() -> list[Path]:
    """The directories ml-stack owns: its state root, its cache root and the Hugging Face hub
    cache. Nothing outside them is ever moved."""
    hf = os.environ.get("HF_HUB_CACHE")
    hub = Path(hf).expanduser() if hf else Path(os.environ.get("HF_HOME")
          or home.user_home() / ".cache" / "huggingface").expanduser() / "hub"
    seen: list[Path] = []
    for root in (home.home(), home.cache(), hub):
        resolved = root.resolve()
        if resolved not in seen:
            seen.append(resolved)
    return seen


def _root_of(path: Path, roots: Iterable[Path]) -> Path:
    here = path.parent.resolve() / path.name
    for root in roots:
        if here.is_relative_to(root.resolve()):
            return root.resolve()
    raise MoveRefused(f"{path} is outside the directories ml-stack manages")


def _digest(path: Path) -> str:
    if path.is_symlink():
        return hashlib.sha256(str(path.readlink()).encode()).hexdigest()
    return sha256_file(path)


def _carry(source: Path, target: Path) -> None:
    try:
        promote(source, target)
    except CrossDevice:
        shutil.copy2(source, target, follow_symlinks=False)
        if _digest(target) != _digest(source):
            target.unlink()
            raise MoveRefused(f"copy of {source} did not match") from None
        source.unlink()


def hold(path: Path | str, ident: str, *, move: str = "file",
         roots: Iterable[Path] | None = None) -> dict[str, Any]:
    """Move ``path`` into the quarantine directory of the root it sits under and return the
    record needed to put it back. ``move="link"`` moves a symlink itself; ``"file"`` moves
    what a symlink points at."""
    given = Path(path)
    if not os.path.lexists(given):
        raise MoveRefused(f"{given} does not exist")
    subject = given if move == "link" or not given.is_symlink() else given.resolve()
    root = _root_of(subject, roots if roots is not None else managed_roots())
    if subject.is_dir() and not subject.is_symlink():
        raise MoveRefused(f"{subject} is a directory; quarantine its files")
    folder = root / HELD_DIR / ident
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    held = folder / subject.name
    digest, size = _digest(subject), subject.lstat().st_size
    link = str(subject.readlink()) if subject.is_symlink() else ""
    _carry(subject, held)
    return {"type": "file", "original": str(subject), "held": str(held), "sha256": digest,
            "bytes": size, "link": link}


def restore(action: dict[str, Any]) -> Path:
    """Put a held file back where it was. Refused when its bytes changed while held or when
    something now sits at the original path."""
    held, original = Path(action["held"]), Path(action["original"])
    if not os.path.lexists(held):
        raise MoveRefused(f"{held} is gone")
    if _digest(held) != action["sha256"]:
        raise MoveRefused(f"{held} changed while it was held")
    if os.path.lexists(original):
        raise MoveRefused(f"{original} exists; nothing is overwritten")
    original.parent.mkdir(parents=True, exist_ok=True)
    _carry(held, original)
    held.parent.rmdir()
    return original


def purge(action: dict[str, Any]) -> None:
    """Delete a held file. Only a person's confirmed purge reaches this."""
    held = Path(action["held"])
    held.unlink(missing_ok=True)
    try:
        held.parent.rmdir()
    except OSError:
        return
