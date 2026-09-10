"""Hand-rolled atomic file replacement outside the file helpers."""

from __future__ import annotations

from pathlib import Path

from . import Finding
from ._util import calls, exempt, parse, python_files, rel

NAME = "atomic-writes"
OWNER = "ml_stack.files.write_json"
ROOTS = ("src/ml_stack",)
OWNS = ("src/ml_stack/files.py",)
TARGETS = {"os.replace", "os.rename", "shutil.move"}
# ``Path.replace``/``Path.rename`` are the same syscall under another spelling. A string's
# ``replace`` takes two arguments and a string has no ``rename``, so one argument tells them
# apart without knowing the receiver's type.
METHODS = {"replace": 1, "rename": 1}


def describe() -> str:
    return "A write-then-replace done by hand; ml_stack.files.write_json is the atomic write."


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
                continue
            method = name.rsplit(".", 1)[-1]
            receiver = name.rsplit(".", 1)[0] if "." in name else ""
            if (method in METHODS and receiver not in ("", "self", "cls")
                    and not node.keywords and len(node.args) == METHODS[method]):
                out.append(Finding(where, node.lineno, f"{method}() on a path"))
    return out
