"""A read-only view of the person record for the board: every row is a person-attestation made by a harness hook, never a message from a person."""

from __future__ import annotations

from typing import Any

from ml_stack.workspace import person_store

__all__ = ["listing"]

DETAIL = ("excerpt", "kind", "target", "how", "state", "label", "by", "reason", "uses")


def listing(limit: int = 50, session: str = "") -> list[dict[str, Any]]:
    """The newest ``limit`` records, oldest first, each shown as attested by the harness hook for its session."""
    try:
        rows = person_store.records()
    except (person_store.Unreadable, OSError, ValueError) as error:
        return [{"kind": person_store.ATTESTATION, "ok": False,
                 "text": f"the person record cannot be trusted: {error}"}]
    shown = [r for r in rows if not session or r.get("session_id") == session]
    return [_row(r) for r in shown[-max(1, limit):]]


def _row(row: dict[str, Any]) -> dict[str, Any]:
    session = row.get("session_id") or "(session of its authorization)"
    return {"kind": person_store.ATTESTATION, "ok": True, "record": row.get("type"), "seq": row.get("seq"),
            "hash": str(row.get("hash", ""))[:12], "ts": row.get("ts"),
            "text": f"attested by the harness hook for session {session}",
            "detail": {k: row[k] for k in DETAIL if k in row}}
