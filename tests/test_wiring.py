"""Every package and top-level module under src/ml_stack has a caller, a command, or a reason."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "ml_stack"

STANDALONE = {
    "testing": "fakes and markers the suite imports; not library code",
    "geo": "reached only through its own tests; no library caller yet",
    "speech": "reached only through its own tests; no library caller yet",
    "web": "loaded by name as python:ml_stack.web:tools, never imported",
}


def _pieces() -> list[str]:
    names = []
    for path in sorted(SRC.iterdir()):
        if path.name.startswith((".", "_")):
            continue
        if path.is_dir() and any(path.glob("*.py")):
            names.append(path.name)
        elif path.suffix == ".py":
            names.append(path.stem)
    return names


def _segment(path: Path) -> str:
    parts = path.relative_to(SRC).parts
    return parts[0] if len(parts) > 1 else path.stem


def _imported_by(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    package = ["ml_stack", *path.relative_to(SRC).parts[:-1]]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "ml_stack" and len(parts) > 1:
                    out.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                parts = base + (node.module.split(".") if node.module else [])
            else:
                parts = (node.module or "").split(".")
            if parts[:1] != ["ml_stack"]:
                continue
            if len(parts) > 1:
                out.add(parts[1])
            else:
                out.update(alias.name for alias in node.names)
    return out


def _callers() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        mine = _segment(path)
        for name in _imported_by(path):
            if name != mine:
                found.setdefault(name, set()).add(path.relative_to(REPO).as_posix())
    return found


def _commands() -> set[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    out = set()
    for target in data.get("project", {}).get("scripts", {}).values():
        parts = target.split(":")[0].split(".")
        if parts[:1] == ["ml_stack"] and len(parts) > 1:
            out.add(parts[1])
    return out


CALLERS = _callers()
COMMANDS = _commands()


@pytest.mark.parametrize("name", _pieces())
def test_piece_is_wired_in(name: str) -> None:
    if name in CALLERS or name in COMMANDS or name in STANDALONE:
        return
    pytest.fail(
        f"ml_stack.{name} is imported by nothing in src/ml_stack, backs no console script, "
        f"and is not in STANDALONE. Wire it in, give it a command, or name it in STANDALONE "
        f"in tests/test_wiring.py with a one-line reason."
    )


def test_standalone_names_only_real_pieces() -> None:
    stale = sorted(set(STANDALONE) - set(_pieces()))
    assert not stale, f"STANDALONE names what is not there: {', '.join(stale)}"
