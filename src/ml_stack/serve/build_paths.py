"""Where a managed build lives on disk, and the patches carried on top of upstream."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ml_stack import home
from ml_stack.serve.binary import managed_current, managed_named, managed_root

__all__ = ["BuildFailed", "builds_dir", "current_link", "named_dest", "named_dir",
           "named_src_dir", "patch_files", "patch_stamp", "patches_dir", "root",
           "slug", "src_dir"]


class BuildFailed(RuntimeError):
    """A step of the build did not do what the next step needs.

    Raised instead of letting the next step fail on a symptom -- a missing binary, a build
    directory with no ``bin/`` -- so the error names what actually went wrong.
    """


def root() -> Path:
    """Where a managed build lives."""
    return managed_root()


def src_dir() -> Path:
    """The llama.cpp checkout `ml-stack-serve build` builds from."""
    return root() / "src"


def builds_dir() -> Path:
    """The directory holding one directory per build made."""
    return root() / "builds"


def current_link() -> Path:
    """The link naming the build `find_binary` prefers."""
    return managed_current()


def named_dir() -> Path:
    """The directory of links to builds kept beside ``current``."""
    return managed_named()


def named_src_dir() -> Path:
    """The directory holding one checkout per named build."""
    return root() / "named-src"


def named_dest(name: str, commit: str) -> Path:
    """Where a fork's named build, kept beside `current`, is installed."""
    return builds_dir() / f"{name}-{commit}"


def slug(text: str) -> str:
    """``text`` cut down to a directory-name-safe slug."""
    cleaned = "".join(c if c.isalnum() else "-" for c in text.strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:40]


def patches_dir() -> Path:
    """The directory the llama.cpp patches are read from."""
    override = os.environ.get("MLSTACK_LLAMA_PATCHES", "")
    if override:
        return home.expand(override)
    packaged = Path(__file__).resolve().parent / "_patches" / "llama.cpp"
    if packaged.is_dir():
        return packaged
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "patches" / "llama.cpp"
        if candidate.is_dir():
            return candidate
    return packaged


def patch_files() -> list[Path]:
    """Every patch a source build applies, in name order."""
    where = patches_dir()
    return sorted(where.glob("*.patch")) if where.is_dir() else []


def patch_stamp(files: list[Path] | None = None) -> str:
    """A short digest of the patch set, empty when there are none."""
    files = patch_files() if files is None else files
    if not files:
        return ""
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode())
        digest.update(item.read_bytes())
    return "p" + digest.hexdigest()[:7]
