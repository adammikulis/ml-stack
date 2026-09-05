"""Shared parsing for the checkers: which files to read, and how to name a call."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}


@lru_cache(maxsize=None)
def python_files(root: Path, roots: tuple[str, ...]) -> tuple[Path, ...]:
    """Every .py file under the given repo-relative directories."""
    out: list[Path] = []
    for name in roots:
        base = root / name
        if not base.exists():
            continue
        if base.is_file():
            out.append(base)
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
                continue
            out.append(path)
    return tuple(sorted(out))


def rel(path: Path, root: Path) -> str:
    """The path as written relative to the repo root, with forward slashes."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


@lru_cache(maxsize=None)
def read(path: Path) -> str:
    """The file's text, empty if it cannot be decoded."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


@lru_cache(maxsize=None)
def parse(path: Path) -> ast.Module | None:
    """The file's syntax tree, or None if it does not parse."""
    source = read(path)
    if not source:
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None


def exempt(relpath: str, prefixes: tuple[str, ...]) -> bool:
    """True when the path is one of the owners, or inside one."""
    return any(relpath == p or relpath.startswith(p.rstrip("/") + "/") for p in prefixes)


def dotted(node: ast.expr) -> str:
    """The dotted source text of a Name/Attribute expression, empty otherwise."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def import_map(tree: ast.Module) -> dict[str, str]:
    """Local name -> the module path it was imported from."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    head = alias.name.split(".")[0]
                    bound[head] = head
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


def qualify(name: str, bound: dict[str, str]) -> str:
    """The dotted name with its first segment expanded through the imports."""
    if not name:
        return ""
    head, _, rest = name.partition(".")
    target = bound.get(head)
    if target is None:
        return name
    return f"{target}.{rest}" if rest else target


@lru_cache(maxsize=None)
def calls(tree: ast.Module) -> tuple[tuple[ast.Call, str], ...]:
    """Every call in the tree with its qualified dotted name."""
    bound = import_map(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            out.append((node, qualify(dotted(node.func), bound)))
    return tuple(out)
