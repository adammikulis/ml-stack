"""Which project a session is in: its root, a stable id and the name shown to the person."""

from __future__ import annotations

import configparser
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ml_stack.home import user_home

__all__ = ["Project", "at", "common_git_dir", "detect"]

SCOPES = ("user", "project")


@dataclass(frozen=True, slots=True)
class Project:
    """A project: ``ident`` is ``git:<origin url>`` for a repository with an origin remote and
    ``path:<root>`` otherwise; ``key`` (a hash of it) names the project's store directory."""

    root: Path
    ident: str
    name: str
    key: str


def _git_dir(root: Path) -> Path | None:
    mark = root / ".git"
    if mark.is_dir():
        return mark
    if mark.is_file():
        line = mark.read_text(errors="replace").strip()
        if line.startswith("gitdir:"):
            return (root / line[len("gitdir:"):].strip()).resolve()
    return None


def common_git_dir(root: Path) -> Path | None:
    """The git directory every worktree of ``root``'s repository shares, or None outside a repository."""
    git = _git_dir(Path(root).resolve())
    if git is None:
        return None
    common = git / "commondir"
    return (git / common.read_text(errors="replace").strip()).resolve() if common.is_file() else git


def _origin(git: Path) -> str:
    common = git / "commondir"
    if common.is_file():
        git = (git / common.read_text(errors="replace").strip()).resolve()
    parser = configparser.RawConfigParser(strict=False)
    try:
        parser.read(git / "config", encoding="utf-8")
        return " ".join(parser.get('remote "origin"', "url").split())
    except (configparser.Error, OSError, UnicodeDecodeError):
        return ""


def at(root: Path) -> Project:
    """The project rooted at ``root`` (which need not exist, to name a project that has moved)."""
    root = Path(root).expanduser().resolve()
    git = _git_dir(root)
    origin = _origin(git) if git is not None else ""
    ident = f"git:{origin}" if origin else f"path:{root}"
    return Project(root, ident, root.name or str(root), hashlib.sha256(ident.encode()).hexdigest()[:16])


def detect(start: Path | None = None, *, explicit: Path | None = None) -> Project | None:
    """The project for ``explicit``, or else the git toplevel above ``start`` (default the
    working directory), or else ``start``; none when that is the home directory or the
    filesystem root. ``ValueError`` when ``explicit`` is not a directory."""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"{explicit} is not a directory")
        return at(root)
    here = Path(start or Path.cwd()).expanduser().resolve()
    root = next((p for p in (here, *here.parents) if (p / ".git").exists()), here)
    if root in (user_home().resolve(), Path(root.anchor)):
        return None
    return at(root)
