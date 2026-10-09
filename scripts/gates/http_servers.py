"""HTTP request handlers outside the served graph page."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import dotted, exempt, parse

NAME = "http-servers"
OWNER = "poolhouse.graph.serve"
INCREMENTAL = True
ROOTS = ("src/poolhouse",)
OWNS = ("src/poolhouse/graph/serve.py", "src/poolhouse/testing/fakes.py",
        "src/poolhouse/testing/fakehub.py")


def describe() -> str:
    return ("A second HTTP handler; poolhouse.graph.serve takes a route mixin and\n"
            "    poolhouse.testing.fakes the stand-in servers.")


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
