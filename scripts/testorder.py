"""Pytest plugin: start the files that took longest last time first, so no worker is left holding the long pole.

xdist hands tests out in collection order. A run that reaches a 200 s test late finishes 200 s
after that, while every other worker idles. ``DEV_TEST_ORDER`` names a snapshot (written once by
``scripts/test`` before the workers start, so each worker orders alike) of the seconds each file and
each long test took in the history store; the files are reordered by it, longest first, and a long
test moves to the front of its own file. Tests inside a file keep their order otherwise, so a
module-scoped fixture is still built once per stretch of a file, and a run without the snapshot
keeps pytest's order.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import testhistory

LONG_TEST_S = 20.0


def snapshot(tests: dict[str, dict]) -> dict:
    """Per-file totals and the tests of ``LONG_TEST_S`` or more, from history entries."""
    files: dict[str, float] = {}
    nodes: dict[str, float] = {}
    for node, entry in tests.items():
        spent = testhistory.seconds(entry)
        files[node.split("::", 1)[0]] = files.get(node.split("::", 1)[0], 0.0) + spent
        if spent >= LONG_TEST_S:
            nodes[node] = spent
    return {"files": files, "nodes": nodes}


def write(tests: dict[str, dict]) -> str:
    """Write the snapshot to a temporary file and return its path."""
    descriptor, name = tempfile.mkstemp(prefix="test-order-", suffix=".json")
    with os.fdopen(descriptor, "w") as stream:
        json.dump(snapshot(tests), stream)
    return name


def ordered(ids: list[str], order: dict) -> list[str]:
    """``ids`` with the longest files first, the long tests first within their file, the rest in
    the order given. A file the snapshot has never seen counts as ``DEFAULT_FILE_S`` seconds."""
    files, nodes = order.get("files", {}), order.get("nodes", {})
    groups: dict[str, list[str]] = {}
    for node in ids:
        groups.setdefault(node.split("::", 1)[0], []).append(node)
    first_seen = {file: place for place, file in enumerate(groups)}
    out: list[str] = []
    for file in sorted(groups, key=lambda f: (-files.get(f, testhistory.DEFAULT_FILE_S), first_seen[f])):
        members = groups[file]
        long = sorted((n for n in members if n in nodes), key=lambda n: -nodes[n])
        out.extend(long + [n for n in members if n not in nodes])
    return out


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    path = os.environ.get("DEV_TEST_ORDER")
    if not path:
        return
    try:
        order = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    by_id = {item.nodeid: item for item in items}
    items[:] = [by_id[node] for node in ordered([item.nodeid for item in items], order)]
