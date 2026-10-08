"""The autostart state in `ml-stack-workspace status`, and the one board line a drifted unit earns."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from ml_stack.fleet import autostart_check, autostart_ledger
from ml_stack.workspace.identity import Denied
from ml_stack.workspace.rates import RateLimited
from ml_stack.workspace.screen import Refused

__all__ = ["REPEAT_S", "fields"]

REPEAT_S = 24 * 3600
LIMIT = 190


def _post(ws: Any, token: str, report: autostart_check.Report, now: float) -> None:
    """Announce a drift under the caller's own identity unless the same drift was announced within a day."""
    ledger = autostart_ledger.read()
    seen = hashlib.sha256("\n".join(report.reasons).encode()).hexdigest()[:16]
    last = ledger["reported"]
    if last.get("fingerprint") == seen and now - float(last.get("at", 0)) < REPEAT_S:
        return
    text = f"autostart drifted: {report.reasons[0]}"[:LIMIT].replace("\n", " ")
    try:
        ws.announce(token, "blocked", text)
    except (Denied, Refused, RateLimited):
        return
    autostart_ledger.write({**ledger, "reported": {"fingerprint": seen, "at": now}})


def fields(ws: Any, token: str, *, now: float = 0.0) -> dict[str, Any]:
    """``{"autostart": {...}}`` for a status listing; reports a drift to the board once a day."""
    try:
        report = autostart_check.check()
    except (OSError, ValueError, RuntimeError, KeyError):
        return {"autostart": {"state": "unknown", "reasons": [], "manifest": ""}}
    if report.state == "drifted":
        _post(ws, token, report, now or time.time())
    return {"autostart": {"state": report.state, "reasons": list(report.reasons[:3]), "manifest": report.manifest}}
