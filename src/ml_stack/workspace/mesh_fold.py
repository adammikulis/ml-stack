"""Folds merged journal rows into the message bus and the notes log."""

from __future__ import annotations

from typing import Any

from ml_stack.workspace import journal_merge as rules
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import AGENT, HUMAN

__all__ = ["apply", "message_payload", "note_payload"]

MESSAGE_FIELDS = ("type", "from", "role", "to", "subject", "body", "label", "mentions", "flags",
                  "model", "model_state", "expires", "file")
NOTE_FIELDS = ("nkind", "title", "body", "source", "tags", "verify_cmd", "ttl_s", "flags")


def _qualified(actor: str, origin: str) -> str:
    return f"{actor}@{origin[:6]}"


def message_payload(ws, row: dict[str, Any]) -> dict[str, Any]:
    """The journal body of a message row about to be posted."""
    payload = {key: row[key] for key in MESSAGE_FIELDS if key in row}
    for field, link in (("reply_to", "reply_jid"), ("thread", "thread_jid")):
        parent = ws.bus.get(int(row.get(field) or 0)) if row.get(field) else None
        payload[link] = parent.get("jid", "") if parent else ""
    return payload


def note_payload(ws, who, fields: dict[str, Any]) -> dict[str, Any]:
    """The journal body of a note about to be added."""
    notes = {r["seq"]: r.get("jid", "") for r in ws.notes.log.rows() if r.get("op") == "add"}
    payload = {key: fields[key] for key in NOTE_FIELDS if key in fields}
    return {**payload, "author": who.id, "role": who.role,
            "supersedes_jids": [notes.get(int(n), "") for n in fields.get("supersedes", [])]}


class _Views:
    """The journal ids already in the bus and the notes log, with the local sequence of each."""

    def __init__(self, ws) -> None:
        self.messages = {r["jid"]: r["seq"] for r in ws.bus.log.rows() if r.get("jid")}
        self.notes = {r["jid"]: r["seq"] for r in ws.notes.log.rows() if r.get("jid")}


def apply(ws) -> int:
    """Fold every journal row not yet applied into the views, in merged order; returns how
    many rows changed a view."""
    mesh = ws.mesh
    now_ms = int(ws.clock() * 1000)
    with held(ws.base / "mesh-fold.lock"):
        batches: dict[str, list[dict[str, Any]]] = {}
        for origin in mesh.journals.origins():
            rows = [r for r in mesh.journals.rows(origin) if r["seq"] > int(mesh.applied.get(origin, 0))]
            if origin != mesh.origin:
                rows = rows[:next((i for i, r in enumerate(rows) if rules.held_back(r, now_ms)), len(rows))]
            if rows:
                batches[origin] = rows
        if not batches:
            return 0
        views, made = _Views(ws), 0
        for row in rules.merge(batches):
            made += _fold(ws, row, views)
        for origin, rows in batches.items():
            mesh.applied.put(origin, rows[-1]["seq"])
        return made


def _fold(ws, row: dict[str, Any], views: _Views) -> int:
    jid = rules.row_id(row)
    if row["kind"] in ("message", "announce") and jid not in views.messages:
        views.messages[jid] = _message(ws, row, jid, views)
        return 1
    if row["kind"] == "note" and jid not in views.notes:
        views.notes[jid] = _note(ws, row, jid, views)
        return 1
    return 0


def _message(ws, row: dict[str, Any], jid: str, views: _Views) -> int:
    body = dict(row["body"])
    foreign = row["origin"] != ws.mesh.origin
    parent = int(views.messages.get(body.pop("reply_jid", ""), 0))
    root = int(views.messages.get(body.pop("thread_jid", ""), 0)) or parent
    to = str(body["to"])
    if to.endswith("@" + ws.mesh.origin[:6]):
        to = to.rpartition("@")[0]
    made = ws.bus.append({
        **body, "to": to, "reply_to": parent, "thread": root, "held": "", "jid": jid, "idem": row["idem"],
        "origin_actor": row["actor"], "flags": list(body.get("flags", [])), "mentions": [],
        "from": _qualified(body["from"], row["origin"]) if foreign else body["from"],
        "role": (AGENT if body["role"] == HUMAN else body["role"]) if foreign else body["role"],
        "expires": float(body.get("expires", 0.0)), "label": str(body.get("label", "")),
        "subject": str(body.get("subject", "")), "model": str(body.get("model", "")),
        "model_state": str(body.get("model_state", ""))})
    return int(made["seq"])


def _note(ws, row: dict[str, Any], jid: str, views: _Views) -> int:
    body = dict(row["body"])
    foreign = row["origin"] != ws.mesh.origin
    old = [int(views.notes[j]) for j in body.pop("supersedes_jids", []) if j in views.notes]
    author, role = str(body.pop("author")), str(body.pop("role"))
    made = ws.notes.log.append({
        "op": "add", **body, "jid": jid, "supersedes": old, "held": "", "flags": list(body.get("flags", [])),
        "author": _qualified(author, row["origin"]) if foreign else author,
        "role": (AGENT if role == HUMAN else role) if foreign else role})
    return int(made["seq"])

