"""Absolute paths that still name the old directories after ``poolhouse migrate`` moved them.

A rename keeps every relative link, but a symlink with an absolute target, a venv's shebangs and a config
file that stores a full path still point into ``~/.ml-stack``. This module finds them under the NEW state and
cache directories only (never follows a link, never leaves them) and:

- repoints a symlink whose target starts with an old directory, when the new target exists;
- rewrites the old path in a venv's scripts, ``pyvenv.cfg``, ``activate*`` files and ``*.pth`` files (text
  only, size-capped, written atomically, the original kept under ``migrate-backup``);
- only reports any other file that holds an old path, and any symlink it cannot repoint.

Running it again finds nothing and changes nothing.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from poolhouse.files import promote, writing
from poolhouse.migrate_scan import BACKUPS, Budget, candidates

__all__ = ["BACKUPS", "Findings", "apply", "scan", "verify"]

MAX_TEXT = 1 << 20
"""A file larger than this is never read: it is a model, a database or a log, not a script or a setting."""
SNIFF = 8192
SKIP_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".npy", ".npz", ".so", ".dylib", ".pyc",
                           ".db", ".sqlite", ".lbug", ".wal", ".zip", ".gz", ".png", ".jpg"})
SCRIPT_DIRS = frozenset({"bin", "Scripts"})
PACKAGE_FILES = frozenset({"direct_url.json", "RECORD"})
Pairs = list[tuple[Path, Path]]


@dataclass
class Findings:
    """What was found under the new directories, as paths."""

    repoint: list[tuple[Path, str]] = field(default_factory=list)
    """Symlinks to repoint, with their new target."""
    rewrite: list[Path] = field(default_factory=list)
    """Venv files whose old path is rewritten."""
    report: list[Path] = field(default_factory=list)
    """Other files that name an old path; never changed."""
    stuck: list[Path] = field(default_factory=list)
    """Symlinks into an old directory with no new target to point at."""
    dangling: list[Path] = field(default_factory=list)
    """Symlinks whose target does not exist."""
    skipped: list[Path] = field(default_factory=list)
    """Directories the time budget left unread."""
    elapsed: float = 0.0
    stopped: str = ""
    """The message about what a tripped budget left unread; empty when nothing was left."""

    def counts(self) -> str:
        return (f"{len(self.repoint)} symlinks to repoint, {len(self.rewrite)} venv files to rewrite, "
                f"{len(self.report)} other files name an old path (reported, not changed), "
                f"{len(self.stuck)} symlinks into an old directory with no new target, "
                f"{len(self.dangling)} dangling symlinks")


def _pattern(old: Path) -> re.Pattern[str]:
    return re.compile(re.escape(str(old)) + r"(?![\w.-])")


def _swap(text: str, pairs: Pairs) -> str:
    for old, new in pairs:
        text = _pattern(old).sub(lambda _m, new=new: str(new), text)
    return text


def _in_venv(path: Path, root: Path) -> bool:
    return any((parent / "pyvenv.cfg").is_file() for parent in path.parents if parent.is_relative_to(root))


def _rewritable(path: Path, root: Path) -> bool:
    if path.name == "pyvenv.cfg" or path.name.startswith("activate") or path.suffix == ".pth":
        return _in_venv(path, root)
    return path.parent.name in SCRIPT_DIRS and _in_venv(path, root)


def _names_old(path: Path, olds: list[Path]) -> bool:
    """Whether a small text file names an old path (as a whole path, not as the start of another name)."""
    try:
        if path.suffix in SKIP_SUFFIXES or path.stat().st_size > MAX_TEXT:
            return False
        data = path.read_bytes()
        text = data.decode("utf-8") if b"\0" not in data[:SNIFF] else ""
    except (OSError, UnicodeDecodeError):
        return False
    return any(_pattern(old).search(text) for old in olds)


def _skipped_package_file(path: Path) -> bool:
    """A file inside site-packages is read only when it is a record that stores a path."""
    return ("site-packages" in path.parts and path.suffix != ".pth" and path.name not in PACKAGE_FILES
            and not path.name.startswith("__editable__"))


def _link(path: Path, pairs: Pairs, found: Findings, planning: bool) -> None:
    target = str(path.readlink())
    for old, new in pairs:
        if re.match(re.escape(str(old)) + r"(?=/|$)", target):
            fresh = str(new) + target[len(str(old)):]
            if Path(target if planning else fresh).exists():
                found.repoint.append((path, fresh))
            else:
                found.stuck.append(path)
            return
    if not path.exists():
        found.dangling.append(path)


def _classify(kind: str, path: Path, root: Path, olds: list[Path], found: Findings) -> None:
    """File one candidate: a venv file or a settings file that names an old path is rewritten or reported."""
    if kind == "file" and _skipped_package_file(path):
        return
    if kind != "link" and _names_old(path, olds):
        rewrite = kind == "venv" or (kind == "file" and _rewritable(path, root))
        (found.rewrite if rewrite else found.report).append(path)


def scan(roots: list[Path], pairs: Pairs, *, planning: bool = False, budget: Budget | None = None) -> Findings:
    """What under ``roots`` names the old side of a ``(old, new)`` pair, read only where a path can live (see
    ``migrate_scan``) and for the time ``budget`` allows (60 s by default); a ``deep`` budget reads every file,
    without a time limit, and says where it is every five seconds. ``planning`` is for roots that are still the old directories: a new target
    is taken to exist when the old one does."""
    found = Findings()
    olds = [old for old, _new in pairs]
    clock = budget or Budget.of(False)
    for root in roots:
        for kind, path in candidates(root, clock):
            if kind == "link":
                _link(path, pairs, found, planning)
            else:
                _classify(kind, path, root, olds, found)
    found.skipped, found.elapsed, found.stopped = clock.skipped, clock.elapsed(), clock.message()
    return found


def _backup(path: Path, vault: Path) -> Path:
    vault.mkdir(parents=True, exist_ok=True)
    kept = vault / (hashlib.sha256(str(path).encode()).hexdigest()[:16] + ".orig")
    shutil.copy2(path, kept)
    return kept


def _rewrite(path: Path, text: str) -> None:
    with writing(path) as tmp:
        tmp.write_bytes(text.encode("utf-8"))
        shutil.copymode(path, tmp)


def _repoint(path: Path, target: str) -> None:
    tmp = path.with_name(f".{path.name}.migrating")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    promote(tmp, path)


def apply(found: Findings, pairs: Pairs, vault: Path) -> list[str]:
    """Do the repoints and rewrites; returns the lines for migrate.log, with every backup named."""
    log: list[str] = []
    for path, target in found.repoint:
        was = path.readlink()
        _repoint(path, target)
        log.append(f"paths: link {path}: {was} -> {target}")
    for path in found.rewrite:
        text = path.read_bytes().decode("utf-8")
        kept = _backup(path, vault)
        _rewrite(path, _swap(text, pairs))
        log.append(f"paths: rewrote {path} (backup {kept})")
    log += [f"paths: names an old path, left as it is: {path}" for path in found.report]
    log += [f"paths: link {path} points into an old directory and its new target is missing" for path in found.stuck]
    return log


def verify(roots: list[Path], pairs: Pairs, *, deep: bool = False, say: Callable[[str], None] | None = None,
           ) -> list[str]:
    """What is still wrong under the new directories: each line names the file and the problem."""
    found = scan(roots, pairs, budget=Budget.of(deep, say))
    bad = [f"dangling symlink {p}" for p in found.dangling + found.stuck]
    bad += [f"symlink {p} still points into an old directory" for p, _t in found.repoint]
    bad += [f"old path in {p}" for p in found.rewrite]
    return bad
