"""HTTP request handlers outside the served graph page."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import dotted, exempt, parse, python_files, rel

NAME = "http-servers"
OWNER = "ml_stack.graph.serve"
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/graph/serve.py", "src/ml_stack/testing/fakes.py")


def describe() -> str:
    return ("A second HTTP handler; ml_stack.graph.serve takes a route mixin and\n"
            "    ml_stack.testing.fakes the stand-in servers.")


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        if exempt(where, OWNS):
            continue
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                if dotted(base).endswith("BaseHTTPRequestHandler"):
                    out.append(Finding(where, node.lineno, node.name))
    return out
