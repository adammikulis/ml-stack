"""Per-agent scratch folders, namespaced by agent id and confined to their own directory."""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json
from ml_stack.windows_private import restrict
from ml_stack.workspace.identity import AGENT, Denied, Identity

__all__ = ["Scratch", "inside"]

NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
META = ".scratch.json"
VERSION = 1


def _linked(path: Path) -> bool:
    return path.is_symlink() or (os.name == "nt" and path.is_junction())


def _private(path: Path) -> None:
    if os.name == "nt":
        restrict(path)
    else:
        path.chmod(0o700)


def inside(root: Path, candidate: Path) -> bool:
    """Whether ``candidate``, with every symlink resolved, is ``root`` or below it."""
    top = os.path.realpath(root)
    path = os.path.realpath(candidate)
    return path == top or path.startswith(top + os.sep)


def _size(folder: Path) -> int:
    total = 0
    for here, directories, files in os.walk(folder):
        directories[:] = [name for name in directories if not _linked(Path(here) / name)]
        for name in files:
            try:
                total += (Path(here) / name).lstat().st_size
            except OSError:
                continue
    return total


class Scratch:
    """Folders under ``scratch/<agent id>/<name>``, owner-only, with a size limit and expiry."""

    def __init__(self, base: Path, max_bytes: int, ttl_s: float, most: int,
                 clock: Callable[[], float] = time.time) -> None:
        self.root = base / "scratch"
        self.max_bytes, self.ttl_s, self.most, self.clock = max_bytes, ttl_s, most, clock

    def _agent_dir(self, owner: str, create: bool = False) -> Path:
        path = self.root / owner
        if _linked(self.root) or _linked(path) or (path.exists() and not inside(self.root, path)):
            raise Denied(f"{owner}'s scratch directory is not a plain directory")
        if create:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            _private(path)
        return path

    def _may_touch(self, who: Identity, owner: str) -> None:
        if owner != who.id and who.role == AGENT:
            raise Denied(f"{who.id} cannot reach {owner}'s scratch folders")

    def new(self, who: Identity, name: str, ttl_s: float = 0.0) -> Path:
        """Create the folder ``name`` for ``who`` and return its absolute path."""
        if not NAME.match(name) or ".." in name:
            raise ValueError(f"{name!r} is not a usable folder name")
        mine = self._agent_dir(who.id, create=True)
        if len(self.listing(who)) >= self.most:
            raise ValueError(f"{who.id} already has {self.most} scratch folders")
        used = sum(item["bytes"] for item in self.listing(who))
        if used > self.max_bytes:
            raise ValueError(f"{who.id}'s scratch folders hold {used} bytes, over the limit")
        folder = mine / name
        if _linked(folder):
            raise Denied(f"{name} is a link, not a folder")
        folder.mkdir(mode=0o700, exist_ok=True)
        _private(folder)
        now = self.clock()
        write_json(folder / META, {"version": VERSION, "created": now,
                                   "expires": now + (ttl_s or self.ttl_s),
                                   "max_bytes": self.max_bytes})
        return folder

    def resolve(self, who: Identity, name: str, relative: str = "", owner: str = "") -> Path:
        """An absolute path inside one scratch folder; refuses anything that escapes it."""
        owner = owner or who.id
        self._may_touch(who, owner)
        if not NAME.match(name) or Path(relative).is_absolute():
            raise Denied("the path is outside the scratch folder")
        folder = self._agent_dir(owner) / name
        target = folder / relative
        if _linked(folder) or not inside(folder, target):
            raise Denied("the path is outside the scratch folder")
        return target

    def listing(self, who: Identity, owner: str = "") -> list[dict[str, Any]]:
        """``owner``'s folders (default: ``who``'s) with size, limit and expiry."""
        owner = owner or who.id
        self._may_touch(who, owner)
        mine = self._agent_dir(owner)
        if not mine.is_dir():
            return []
        out = []
        for folder in sorted(p for p in mine.iterdir() if p.is_dir() and not _linked(p)):
            meta = read_json(folder / META, {})
            used = _size(folder)
            out.append({"owner": owner, "name": folder.name, "path": str(folder), "bytes": used,
                        "limit": int(meta.get("max_bytes", self.max_bytes)),
                        "over_limit": used > int(meta.get("max_bytes", self.max_bytes)),
                        "expires": float(meta.get("expires", 0.0)),
                        "expired": bool(meta.get("expires")) and self.clock() > meta["expires"]})
        return out

    def remove(self, who: Identity, name: str, owner: str = "") -> bool:
        """Delete one folder; returns whether it existed. Never follows a link out."""
        folder = self.resolve(who, name, "", owner)
        if not folder.is_dir() or _linked(folder):
            return False
        shutil.rmtree(folder)
        return True

    def collect(self) -> list[str]:
        """Remove every expired folder of every agent; returns their paths."""
        gone = []
        for owner in (sorted(p.name for p in self.root.iterdir()) if self.root.is_dir() else []):
            if _linked(self.root / owner):
                continue
            for item in self.listing(Identity(owner, AGENT)):
                if item["expired"]:
                    shutil.rmtree(item["path"])
                    gone.append(item["path"])
        return gone
