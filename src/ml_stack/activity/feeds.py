"""What flows into the log without the source calling `record` itself: sentinel's events and the
workspace's audit trail."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from ml_stack import home, sentinel
from ml_stack.activity import writer
from ml_stack.activity.schema import valid_kind
from ml_stack.files import read_json, write_json
from ml_stack.lock import only_one
from ml_stack.sentinel.events import Event

__all__ = ["attach", "mirror", "workspace_sync"]

_ATTACHED: set[int] = set()
_LOCK = threading.Lock()
_CLAIMS = {"claim": "claimed", "release": "released", "claim.conflict": "conflict",
           "claim.expired": "expired"}


def mirror(event: Event) -> None:
    """Write a sentinel event as a ``security.*`` record; a download has its own record."""
    if event.kind.startswith("net.download"):
        return
    kind = "security." + event.kind
    meta: dict[str, Any] = {"severity": event.severity.name.lower(), "source": event.source}
    if not valid_kind(kind):
        kind, meta["event"] = "security.event", event.kind
    outcome = str(event.evidence.get("outcome", "")) if isinstance(event.evidence, dict) else ""
    meta.update({k: v for k, v in dict(event.evidence).items() if k != "outcome"})
    writer.record(kind, actor="system", subject=event.subject, outcome=outcome, meta=meta, ts=event.ts)


def attach() -> None:
    """Mirror this process's sentinel events into the log, and catch the log up with the
    workspace. Idempotent."""
    node = sentinel.default()
    with _LOCK:
        if id(node.bus) in _ATTACHED:
            return
        _ATTACHED.add(id(node.bus))
    node.bus.subscribe(mirror)
    workspace_sync()


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for text in lines:
        try:
            row = json.loads(text)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("seq"), int):
            rows.append(row)
    return rows


def workspace_sync(base: Path | None = None) -> int:
    """Record the workspace audit rows written since the last sync (who, what, to whom; the
    size and thread of a message from the bus; never a body). Returns how many were written."""
    base = base or _workspace_root()
    cursor_file = writer.directory() / "workspace.cursor"
    try:
        with only_one(cursor_file.with_suffix(".lock"), timeout=5, announce=lambda _m: None):
            seen = int(read_json(cursor_file, {}).get(str(base), 0))
            fresh = [r for r in _rows(base / "audit.jsonl") if r["seq"] > seen]
            if not fresh:
                return 0
            bus = {r["seq"]: r for r in _rows(base / "bus.jsonl")}
            done = 0
            for row in fresh:
                if not _one(row, bus):
                    break
                seen, done = row["seq"], done + 1
            write_json(cursor_file, {**read_json(cursor_file, {}), str(base): seen})
            return done
    except (OSError, ValueError, RuntimeError):
        return 0


def _workspace_root() -> Path:
    named = os.environ.get("ML_STACK_WORKSPACE_HOME")
    return Path(named).expanduser() if named else home.state("workspace")


def _one(row: dict[str, Any], bus: dict[int, dict[str, Any]]) -> bool:
    event, who = str(row.get("event", "")), str(row.get("who", ""))
    skip = {"v", "seq", "prev", "hash", "ts", "event", "who"}
    detail = {k: v for k, v in row.items() if k not in skip}
    if event == "message":
        sent = bus.get(int(row.get("msg", 0)), {})
        return writer.record(
            "workspace.message", actor=who, subject=f"msg:{row.get('msg')}", outcome="held" if row.get("held") else "sent",
            refs={"from": who, "to": row.get("to", ""), "thread": sent.get("thread", "")},
            meta={"type": row.get("type", ""), "size": len(str(sent.get("body", "")))}, ts=row.get("ts"))
    if event in _CLAIMS:
        return writer.record("workspace.claim", actor=who, subject=f"{detail.get('kind')}:{detail.get('key')}",
                             outcome=_CLAIMS[event], refs={"claim": f"{detail.get('kind')}:{detail.get('key')}"},
                             ts=row.get("ts"))
    return writer.record("workspace.event", actor=who or "system", subject=event, meta=detail, ts=row.get("ts"))
