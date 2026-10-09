"""Remembers that scripts/redteam_coverage.py --check passed, and for which inputs."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

from . import _store

MAP = "docs/redteam/coverage-map.toml"
FILES = ("pyproject.toml", MAP, "docs/redteam/coverage.json",
         "app/src-tauri/capabilities/main.json", "scripts/test-on-linux",
         "scripts/redteam_coverage.py", "scripts/gates/_util.py", "scripts/gates/_store.py",
         "scripts/gates/_redteam_memo.py")
"""Files outside src/poolhouse that the check reads or runs."""


def mapped_tests(root: Path) -> list[str]:
    """The test files the coverage map names, as repo-relative paths."""
    try:
        table = tomllib.loads((root / MAP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted({ref.partition("::")[0] for group in table.get("group", [])
                   for ref in group.get("tests", []) if isinstance(ref, str)})


def environment(root: Path) -> str:
    """The interpreter, the POOLHOUSE variables, and what is installed beside the interpreter."""
    seen = [sys.version, sys.executable]
    seen += sorted(f"{k}={v}" for k, v in os.environ.items() if k.startswith("POOLHOUSE"))
    for entry in (e for e in sys.path if e and root.resolve() not in Path(e).resolve().parents
                  and Path(e).resolve() != root.resolve()):
        try:
            seen.append(entry + ":" + ",".join(sorted(p.name for p in Path(entry).iterdir())))
        except OSError:
            seen.append(entry)
    return _store.digest("\0".join(seen).encode())


def inputs(root: Path) -> str:
    """A hash of every file the check reads, of the interpreter and of the environment."""
    paths = [*sorted(p for p in (root / "src" / "poolhouse").rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts),
             *(root / name for name in (*FILES, *mapped_tests(root)))]
    lines = [environment(root)]
    for path in paths:
        lines.append(f"{path.relative_to(root).as_posix()}:{_store.file_digest(path)}")
    return _store.digest("\n".join(lines).encode())


def name(root: Path) -> str:
    """The cache entry for the tree at root."""
    return "redteam-check-" + _store.digest(str(root.resolve()).encode())[:16]


def passed(root: Path, fingerprint: str) -> str | None:
    """The summary printed by the last passing check with these inputs, None if there is none."""
    if _store.forced():
        return None
    kept = _store.load(name(root))
    summary = kept.get("summary")
    return summary if kept.get("inputs") == fingerprint and isinstance(summary, str) else None


def record(root: Path, fingerprint: str, summary: str) -> None:
    """Store that a check with these inputs passed and printed this summary."""
    _store.save(name(root), {"inputs": fingerprint, "summary": summary})
