"""Hand-rolled atomic file replacement outside the file helpers."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._perfile import finder
from ._util import calls, exempt, parse

NAME = "atomic-writes"
OWNER = "poolhouse.files.write_json"
INCREMENTAL = True
ROOTS = ("src/poolhouse",)
OWNS = ("src/poolhouse/files.py",)
TARGETS = {"os.replace", "os.rename", "shutil.move"}
# ``Path.replace``/``Path.rename`` are the same syscall under another spelling. A string's
# ``replace`` takes two arguments and a string has no ``rename``, so one argument tells them
# apart without knowing the receiver's type.
METHODS = {"replace": 1, "rename": 1}


def describe() -> str:
    return "A write-then-replace done by hand; poolhouse.files.write_json is the atomic write."


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
            continue
        method = name.rsplit(".", 1)[-1]
        receiver = name.rsplit(".", 1)[0] if "." in name else ""
        if (method in METHODS and receiver not in ("", "self", "cls")
                and not node.keywords and len(node.args) == METHODS[method]):
            out.append(Finding(where, node.lineno, f"{method}() on a path"))
    return out


find = finder(ROOTS, scan)
