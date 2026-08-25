"""Taking ml-stack off a machine, a piece at a time."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Item", "plan", "remove"]


@dataclass(frozen=True, slots=True)
class Item:
    """One thing an uninstall can take away."""

    key: str
    name: str
    paths: tuple[Path, ...]
    bytes: int
    default: bool
    why: str

    def public(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "bytes": self.bytes,
                "default": self.default, "why": self.why,
                "paths": [str(p) for p in self.paths]}


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def plan(root: Path | str, *, key_path: Path | str | None = None) -> list[Item]:
    """What is on this machine, and which parts are worth keeping.

    Anything the person made themselves -- their models and their datasets -- is
    offered unticked. Everything ml-stack made for itself is ticked.
    """
    root = Path(root).expanduser()
    home = Path(key_path).expanduser().parent if key_path else root.parent

    rows = [
        ("settings", "Settings and this machine's identity",
         (root / "settings.json", root / "availability.json", root / "token",
          root / "serving.json", home / "cluster.key", home / "cluster.group",
          home / "traind.log"),
         True, "The passphrase can make the key again."),
        ("conversations", "Chats",
         (root / "chats",), True, "Everything said to a model on this machine."),
        ("environment", "Training environment",
         (root / "env",), True, "Rebuilt from scratch when training next runs."),
        ("llama", "Model server",
         (root / "llama",), True, "Downloaded again when a model is next run."),
        ("datasets", "Your files",
         (root / "files",), False, "Datasets you pushed here. Nothing else has them."),
        ("models", "Models",
         (root / "models",), False,
         "Large, and downloading them again takes as long as it did the first time."),
    ]
    out = []
    for key, name, paths, default, why in rows:
        present = tuple(p for p in paths if p.exists())
        if not present:
            continue
        out.append(Item(key=key, name=name, paths=present,
                        bytes=sum(_size(p) for p in present),
                        default=default, why=why))
    return out


def remove(root: Path | str, keys: list[str], *,
           key_path: Path | str | None = None) -> dict[str, Any]:
    """Take away the chosen parts, and the startup entry. Returns what went."""
    from . import autostart

    root = Path(root).expanduser()
    chosen = {k for k in keys}
    gone: list[str] = []
    failed: dict[str, str] = {}
    freed = 0

    for item in plan(root, key_path=key_path):
        if item.key not in chosen:
            continue
        freed += item.bytes
        for path in item.paths:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except OSError as exc:
                failed[str(path)] = str(exc)
        if not failed:
            gone.append(item.key)

    startup = [str(p) for p in autostart.uninstall()]

    # An empty traind directory left behind is just litter.
    for spare in (root, root.parent):
        try:
            if spare.is_dir() and not any(spare.iterdir()):
                spare.rmdir()
        except OSError:
            pass

    return {"removed": gone, "startup": startup, "freed": freed, "failed": failed}
