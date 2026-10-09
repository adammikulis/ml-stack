"""A read-only look at the machine's test-slot leases."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from poolhouse import home
from poolhouse.workspace.claims import alive
from poolhouse.workspace.screen import clean_label

__all__ = ["SLOTS_ENV", "slots_dir", "testslots"]

SLOTS_ENV = "DEV_TEST_SLOTS_DIR"


def slots_dir() -> Path:
    """Where the test-slot leases live."""
    named = os.environ.get(SLOTS_ENV)
    return Path(named).expanduser() if named else home.user_home() / ".cache" / "dev-test-slots"


def _number(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _agent(data: object, known: Callable[[str], object] | None) -> dict[str, Any] | None:
    """The workspace agent a lease names, cleaned, and whether the workspace has that identity."""
    if not isinstance(data, dict) or not isinstance(data.get("id"), str):
        return None
    return {"id": clean_label(data["id"]), "parent": clean_label(data.get("parent", "")), "job": clean_label(data.get("job", "")),
            "registered": bool(known and known(data["id"]))}


def testslots(known: Callable[[str], object] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The running and waiting leases, labels cleaned; nothing is created or removed.

    A lease that names a workspace agent shows it, with ``registered`` saying whether ``known``
    recognises the identity, and the agent replaces the free-text label.
    """
    running: list[dict[str, Any]] = []
    waiting: list[dict[str, Any]] = []
    folder = slots_dir()
    for path in sorted(folder.glob("*.slot")) if folder.is_dir() else []:
        try:
            data = json.loads(path.read_text() or "{}")
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not alive(_number(data.get("pid"))):
            continue
        item = {"pid": _number(data.get("pid")), "label": clean_label(data.get("label", "")),
                "granted": _number(data.get("granted")), "want": _number(data.get("want")),
                "minimum": _number(data.get("minimum")), "agent": _agent(data.get("agent"), known)}
        if item["agent"]:
            who = item["agent"]
            item["label"] = f"{who['id']} ({item['label']})"
        (running if item["granted"] > 0 else waiting).append(item)
    return {"running": running, "waiting": waiting}
