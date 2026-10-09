"""Hand-rolled urllib requests outside the HTTP client."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import calls, exempt, parse

NAME = "urlopen-calls"
OWNER = "poolhouse.http.request_json"
INCREMENTAL = True
ROOTS = ("src/poolhouse",)
OWNS = ("src/poolhouse/http.py",)
TARGETS = {"urllib.request.urlopen", "urllib.request.Request"}


def describe() -> str:
    return "A urllib request built by hand; poolhouse.http.request_json already does it."


def scan(path: Path, where: str) -> list[Finding]:
    out = []
    if exempt(where, OWNS):
        return out
    tree = parse(path)
    if tree is None:
        return out
    for node, name in calls(tree):
        if name in TARGETS:
            out.append(Finding(where, node.lineno, name))
    return out


find = finder(ROOTS, scan)
