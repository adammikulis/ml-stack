"""A read-only look at the machine's test-slot leases."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.workspace.claims import alive
from ml_stack.workspace.screen import clean_label

__all__ = ["SLOTS_ENV", "slots_dir", "testslots"]

SLOTS_ENV = "DEV_TEST_SLOTS_DIR"


def slots_dir() -> Path:
    """Where the test-slot leases live."""
    named = os.environ.get(SLOTS_ENV)
    return Path(named).expanduser() if named else home.user_home() / ".cache" / "dev-test-slots"


def _number(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def testslots() -> dict[str, list[dict[str, Any]]]:
    """The running and waiting leases, labels cleaned; nothing is created or removed."""
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
                "granted": _number(data.get("granted")), "want": _number(data.get("want"))}
        (running if item["granted"] > 0 else waiting).append(item)
    return {"running": running, "waiting": waiting}
