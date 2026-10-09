"""The wheel in dist/, rebuilt whenever what it is built from has changed.

Tests used to take whatever wheel sat in dist/ (or compared modification times of src/*.py with
the wheel's, which a checkout of an older branch, a changed pyproject.toml or a changed
packaging/ script all defeat). The wheel is stamped with a digest of its inputs when it is built
here, and reused only while the digest still matches.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

INPUTS = ("src", "packaging", "pyproject.toml", "version.txt", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "README.md")
SKIP = frozenset({"__pycache__", ".DS_Store"})
STAMP = ".built-from"


def digest(repo: Path) -> str:
    """A digest of every file the wheel is built from, by path and content."""
    h = hashlib.sha256()
    for name in INPUTS:
        base = repo / name
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
        for path in files:
            if not SKIP.intersection(path.parts):
                h.update(str(path.relative_to(repo)).encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def build_with_packaging(repo: Path) -> None:
    subprocess.run([sys.executable, str(repo / "packaging" / "build.py")], check=True, stdout=subprocess.DEVNULL, timeout=900)


def fresh_wheel(repo: Path, build: Callable[[Path], None] = build_with_packaging) -> Path:
    """The newest wheel in ``repo/dist``, built first unless it was built from exactly the current inputs."""
    dist, wanted = repo / "dist", digest(repo)
    stamp = dist / STAMP
    held = stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else ""
    wheels = sorted(dist.glob("poolhouse-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels or held != wanted:
        build(repo)
        wheels = sorted(dist.glob("poolhouse-*.whl"), key=lambda p: p.stat().st_mtime)
        if not wheels:
            raise FileNotFoundError(f"{dist} holds no poolhouse wheel after the build")
        dist.mkdir(exist_ok=True)
        stamp.write_text(wanted + "\n", encoding="utf-8")
    return wheels[-1]
