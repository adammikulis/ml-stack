"""HTTP request handlers outside the served graph page."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import dotted, exempt, parse

NAME = "http-servers"
OWNER = "ml_stack.graph.serve"
INCREMENTAL = True
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/graph/serve.py", "src/ml_stack/testing/fakes.py",
        "src/ml_stack/testing/fakehub.py")


def describe() -> str:
    return ("A second HTTP handler; ml_stack.graph.serve takes a route mixin and\n"
            "    ml_stack.testing.fakes the stand-in servers.")


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    if exempt(where, OWNS):
        return out
    tree = parse(path)
    if tree is None:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if dotted(base).endswith("BaseHTTPRequestHandler"):
                out.append(Finding(where, node.lineno, node.name))
    return out


find = finder(ROOTS, scan)
