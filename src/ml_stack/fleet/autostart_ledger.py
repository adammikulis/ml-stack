"""What a person has installed from a manifest, which manifests are spent, and the last board report."""

from __future__ import annotations

from typing import Any

from ml_stack import home
from ml_stack.files import read_json, write_json

__all__ = ["VERSION", "read", "write"]

VERSION = 1


def _path():
    return home.state("autostart", "ledger.json")


def read() -> dict[str, Any]:
    """The ledger: ``installed`` roles, ``used`` manifest ids, ``last`` install and ``reported`` drift."""
    raw = read_json(_path(), {})
    raw = raw if isinstance(raw, dict) and raw.get("version") == VERSION else {}
    return {"version": VERSION, "installed": raw.get("installed") or {}, "used": raw.get("used") or [],
            "last": raw.get("last") or "", "reported": raw.get("reported") or {}}


def write(ledger: dict[str, Any]) -> None:
    """Replace the ledger."""
    write_json(_path(), {**ledger, "version": VERSION})
