"""Which entries of a state directory ``migrate_paths`` looks at, and how long it may look.

The state directory holds models, llama.cpp sources and builds, bench logs, bundles and chat history: millions
of files, none of which stores an absolute path. The default walk looks only where one can live (symlinks to a
shallow depth, a venv's ``bin`` and config, a short list of settings files) and stops at a wall-clock budget,
naming what it left. ``deep`` walks everything, never leaves a progress line more than five seconds away, and
has no budget.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from poolhouse.files import read_json

__all__ = ["BUDGET_SECONDS", "Budget", "candidates"]

BUDGET_SECONDS = 60.0
PROGRESS_SECONDS = 5.0
MAX_DEPTH = 4
"""How far below a root a symlink is looked for."""
VENV_DEPTH = 3
BACKUPS = "migrate-backup"
PRUNED = frozenset({".git", BACKUPS, "models", "bench", "bundles", "chat", "worktrees", "cache", "caches", ".cache",
                    "logs", "node_modules", "site-packages", "__pycache__", "datasets", "downloads"})
"""Directories never entered: nothing in them stores a path, and they hold most of the files."""
CONFIG_DIRS = ("workspace", "config", "services", "launchd", "daemons", "serve", "node")
"""Directories whose own files (never their subdirectories) are read for an old path."""
PROJECT_FILES = (".poolhouse-project.json", ".ml-stack-project.json")
SKIP_FILES = frozenset({"migrate.log"})
Entry = tuple[str, Path]


class Budget:
    """The wall-clock allowance, the progress line and the list of directories left unread."""

    def __init__(self, seconds: float | None, say: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds, self.say, self.clock = seconds, say, clock
        self.started = self.clock()
        self.next_line = self.started + PROGRESS_SECONDS
        self.seen = 0
        self.skipped: list[Path] = []

    @classmethod
    def of(cls, deep: bool, say: Callable[[str], None] | None = None) -> Budget:
        """The default allowance, or none at all when ``deep``."""
        return cls(None if deep else BUDGET_SECONDS, say)

    @property
    def deep(self) -> bool:
        return self.seconds is None

    def elapsed(self) -> float:
        return self.clock() - self.started

    def expired(self) -> bool:
        return self.seconds is not None and self.elapsed() >= self.seconds

    def visit(self, here: Path, count: int) -> None:
        """Note ``count`` more entries seen in ``here``; say where the scan is when five seconds have passed."""
        self.seen += count
        if self.say and self.clock() >= self.next_line:
            self.next_line = self.clock() + PROGRESS_SECONDS
            self.say(f"paths: still scanning: {self.seen} entries seen, now in {here}")

    def message(self) -> str:
        """What a tripped budget left unread, empty when nothing was."""
        if not self.skipped:
            return ""
        first = ", ".join(str(p) for p in self.skipped[:3])
        more = f" and {len(self.skipped) - 3} more" if len(self.skipped) > 3 else ""
        return (f"paths: stopped after {self.seconds:g}s with {len(self.skipped)} directories not read ({first}{more}); "
                f"run with --deep to read everything")


def _entries(here: Path) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(here) as scan:
            return sorted(scan, key=lambda e: e.name)
    except OSError:
        return []


def _small_files(here: Path, kind: str) -> Iterator[Entry]:
    for entry in _entries(here):
        if entry.is_file(follow_symlinks=False) and entry.name not in SKIP_FILES:
            yield kind, Path(entry.path)
        elif entry.is_symlink():
            yield "link", Path(entry.path)


def _venv(root: Path) -> Iterator[Entry]:
    """A venv's ``pyvenv.cfg``, everything in ``bin`` and the ``.pth`` files at the top of its site-packages."""
    yield "venv", root / "pyvenv.cfg"
    yield from _small_files(root / "bin", "venv")
    for site in (root / "lib").glob("python*/site-packages") if (root / "lib").is_dir() else ():
        yield from (item for item in _small_files(site, "venv") if item[0] == "link" or item[1].suffix == ".pth")


def _checkout_files(root: Path) -> Iterator[Entry]:
    record = read_json(root / "workspace-connections.json", {})
    for checkout in record.keys() if isinstance(record, dict) else ():
        for name in PROJECT_FILES:
            if (Path(checkout) / name).is_file():
                yield "other", Path(checkout) / name


def _known_files(root: Path) -> Iterator[Entry]:
    """The state root's own files, the files of each config directory, and a connected checkout's project file."""
    yield from (item for item in _small_files(root, "other") if item[0] == "other")
    for name in CONFIG_DIRS:
        yield from (item for item in _small_files(root / name, "other") if item[0] == "other")
    yield from _checkout_files(root)


def _descend(parent: Path, entry: os.DirEntry[str]) -> bool:
    if entry.name in PRUNED or entry.name.endswith(".gguf") or not entry.is_dir(follow_symlinks=False):
        return False
    return parent.name != "llama.cpp" or entry.name == "builds"


def _shallow(root: Path, budget: Budget) -> Iterator[Entry]:
    yield from _known_files(root)
    stack = [(root, 0)]
    while stack:
        if budget.expired():
            budget.skipped += [here for here, _depth in stack]
            return
        here, depth = stack.pop()
        entries = _entries(here)
        budget.visit(here, len(entries))
        if depth <= VENV_DEPTH and any(e.name == "pyvenv.cfg" for e in entries):
            yield from _venv(here)
            continue
        for entry in entries:
            if entry.is_symlink():
                yield "link", Path(entry.path)
            elif depth < MAX_DEPTH and _descend(here, entry):
                stack.append((Path(entry.path), depth + 1))


def _deep(root: Path, budget: Budget) -> Iterator[Entry]:
    for here, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in (".git", BACKUPS))
        budget.visit(Path(here), len(dirs) + len(files))
        for name in sorted(dirs + files):
            path = Path(here) / name
            if path.is_symlink():
                yield "link", path
            elif name in files and name not in SKIP_FILES:
                yield "file", path


def candidates(root: Path, budget: Budget) -> Iterator[Entry]:
    """``(kind, path)`` under ``root``: ``link``, ``venv`` (a file a venv rewrites), ``other`` (a settings file)
    or, only when ``budget.deep``, ``file`` (any regular file, to be classified by the caller)."""
    yield from (_deep if budget.deep else _shallow)(root, budget)
