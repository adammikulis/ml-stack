"""Hand-rolled urllib requests outside the HTTP client."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import calls, exempt, parse, python_files, rel

NAME = "urlopen-calls"
OWNER = "ml_stack.client.http.request_json"
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/client/http.py",)
TARGETS = {"urllib.request.urlopen", "urllib.request.Request"}


def describe() -> str:
    return "A urllib request built by hand; ml_stack.client.http.request_json already does it."


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        if exempt(where, OWNS):
            continue
        tree = parse(path)
        if tree is None:
            continue
        for node, name in calls(tree):
            if name in TARGETS:
                out.append(Finding(where, node.lineno, name))
    return out
