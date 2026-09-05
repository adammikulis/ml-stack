"""JSON records written from a dict literal with no version key."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import calls, parse, python_files, rel

NAME = "unversioned-records"
OWNER = "ml_stack.files.write_json"
ROOTS = ("src/ml_stack",)
SINKS = ("write_text", "write_json", "write_bytes")


def describe() -> str:
    return "A record written with no version key; a reader cannot tell which shape it has."


def _payload(node: ast.Call) -> ast.Dict | None:
    if not node.args:
        return None
    first = node.args[0]
    return first if isinstance(first, ast.Dict) else None


def _versioned(payload: ast.Dict) -> bool:
    return any(
        isinstance(key, ast.Constant) and key.value == "version" for key in payload.keys
    )


def _to_a_path(tree: ast.Module) -> set[int]:
    sunk: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name not in SINKS:
            continue
        for argument in node.args:
            for inner in ast.walk(argument):
                sunk.add(id(inner))
    return sunk


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        sunk = _to_a_path(tree)
        for node, name in calls(tree):
            if name not in ("json.dump", "json.dumps"):
                continue
            if name == "json.dumps" and id(node) not in sunk:
                continue
            payload = _payload(node)
            if payload is not None and not _versioned(payload):
                out.append(Finding(where, node.lineno, name))
    return out
