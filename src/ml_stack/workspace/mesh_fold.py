"""Folds merged journal rows into the message bus and the notes log.

Only announcements and `#general` posts, and notes, are journaled for other devices; a direct
message or a post to any other board stays on the device that wrote it. A row from another
device is checked against the same size, screen, quarantine and quota rules as a local post
under an identity named `name@dN`, always an agent."""

from __future__ import annotations

import math
from typing import Any

from ml_stack.workspace import journal_merge as rules, plain, wake
from ml_stack.workspace.boards import ANNOUNCE, ANNOUNCE_KINDS, GENERAL
from ml_stack.workspace.bus import TYPES
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import AGENT, Denied, Identity, valid_name
from ml_stack.workspace.notes import KINDS
from ml_stack.workspace.quarantine import PLACEHOLDER
from ml_stack.workspace.screen import Refused

__all__ = ["apply", "message_payload", "note_payload", "replicated"]

MESSAGE_KEYS = frozenset({"type", "from", "role", "to", "subject", "body", "label", "model",
                          "model_state", "reply_jid", "thread_jid"})
NOTE_KEYS = frozenset({"nkind", "title", "body", "source", "tags", "verify_cmd", "ttl_s", "author",
                       "role", "supersedes_jids"})
MAX_FOLD = 500
MAX_TTL_S = 10 * 365 * 86400.0
REJECTIONS = (ValueError, KeyError, TypeError, OverflowError, Refused, Denied)


class Rejected(ValueError):
    """A foreign row that is not folded."""


def replicated(row: dict[str, Any]) -> bool:
    """Whether a message row is journaled for other devices."""
    return row["to"] in (ANNOUNCE, GENERAL) and not row.get("held")


def message_payload(ws, row: dict[str, Any]) -> dict[str, Any] | None:
    """The journal body of a message row about to be posted; None when it stays on this device."""
    if not replicated(row):
        return None
    payload = {key: row[key] for key in MESSAGE_KEYS if key in row}
    for field, link in (("reply_to", "reply_jid"), ("thread", "thread_jid")):
        parent = ws.bus.get(int(row.get(field) or 0)) if row.get(field) else None
        payload[link] = parent.get("jid", "") if parent else ""
    return payload


def note_payload(ws, who, fields: dict[str, Any]) -> dict[str, Any]:
    """The journal body of a note about to be added."""
    notes = {r["seq"]: r.get("jid", "") for r in ws.notes.log.rows() if r.get("op") == "add"}
    payload = {key: fields[key] for key in NOTE_KEYS if key in fields}
    return {**payload, "author": who.id, "role": who.role,
            "supersedes_jids": [notes.get(int(n), "") for n in fields.get("supersedes", [])]}


class _Views:
    """The journal ids already in the bus and the notes log, with the local sequence of each."""

    def __init__(self, ws) -> None:
        self.messages = {r["jid"]: r["seq"] for r in ws.bus.log.rows() if r.get("jid")}
        self.notes = {r["jid"]: r["seq"] for r in ws.notes.log.rows() if r.get("jid")}


def apply(ws) -> int:
    """Fold every journal row not yet applied into the views, in merged order; returns how
    many rows changed a view. A row that cannot be folded is recorded and skipped."""
    mesh = ws.mesh
    now_ms = int(ws.clock() * 1000)
    with held(ws.base / "mesh-fold.lock"):
        batches: dict[str, list[dict[str, Any]]] = {}
        for origin in mesh.journals.origins():
            rows = [r for r in mesh.journals.rows(origin) if r["seq"] > int(mesh.applied.get(origin, 0))]
            if origin != mesh.origin:
                rows = rows[:next((i for i, r in enumerate(rows) if rules.held_back(r, now_ms)), len(rows))][:MAX_FOLD]
            if rows:
                batches[origin] = rows
        if not batches:
            return 0
        views, made = _Views(ws), 0
        for row in rules.merge(batches):
            try:
                made += _fold(ws, row, views)
            except REJECTIONS as why:
                mesh.reject(rules.row_id(row), str(why))
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


def _fields(body: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise Rejected(f"unknown field {plain.line(unknown[0], 40)}")
    return body


def _text(value: object, name: str, most: int) -> str:
    if type(value) is not str or len(value.encode()) > most:
        raise Rejected(f"{name} is not text within {most} bytes")
    return value


def _line(value: object, name: str, most: int) -> str:
    text = _text(value, name, most)
    if plain.line(text, most) != text:
        raise Rejected(f"{name} is not one clean line")
    return text


def _sender(ws, row: dict[str, Any], declared: object) -> Identity:
    """The agent identity a row from another device speaks as."""
    actor = row["actor"]
    if declared != actor or not valid_name(actor) or ws.registry.role_of(actor):
        raise Rejected("the sender is not a usable name on this device")
    return Identity(f"{actor}@{ws.mesh.label(row['origin'])}", AGENT)


def _message(ws, row: dict[str, Any], jid: str, views: _Views) -> int:
    body = _fields(row["body"], MESSAGE_KEYS)
    announce = row["kind"] == "announce"
    to, kind = body.get("to"), body.get("type")
    if to != (ANNOUNCE if announce else GENERAL) or kind not in (ANNOUNCE_KINDS if announce else TYPES):
        raise Rejected("the post is not an announcement or a #general message")
    limits = ws.limits
    subject = _line(body.get("subject", ""), "the subject", limits.subject_chars)
    text = _text(body.get("body"), "the body", limits.body_bytes)
    foreign = row["origin"] != ws.mesh.origin
    who = _sender(ws, row, body.get("from")) if foreign else Identity(str(body["from"]), str(body["role"]))
    qid, flags, model, state, label = "", [], "", "", ""
    if foreign:
        ws._screen(who, "the message", limits.body_bytes, subject, text)
        qid, flags = ws._hold(who, "message", f"{who.id}->{to}", subject, text)
        subject, text = ("", PLACEHOLDER.format(qid=qid, why=", ".join(flags))) if qid else (subject, text)
    else:
        model, state = _line(body.get("model", ""), "the model", 200), _line(body.get("model_state", ""), "the state", 40)
        label = _line(body.get("label", ""), "the label", 40)
    parent_seq = int(views.messages.get(body.get("reply_jid", ""), 0))
    parent = ws.bus.get(parent_seq) if parent_seq else None
    parent = parent if parent and parent["to"] == to else None
    made = ws.bus.append({
        "type": kind, "from": who.id, "role": who.role, "to": to, "subject": subject, "body": text,
        "reply_to": parent["seq"] if parent else 0, "thread": (parent.get("thread") or parent["seq"]) if parent else 0,
        "label": label, "mentions": [], "held": qid, "flags": flags, "expires": 0.0, "model": model,
        "model_state": state, "jid": jid, "idem": row["idem"]})
    wake.signal(ws.base / "wake", ws.board.wake_names(made))
    return int(made["seq"])


def _note(ws, row: dict[str, Any], jid: str, views: _Views) -> int:
    body = _fields(row["body"], NOTE_KEYS)
    limits = ws.limits
    foreign = row["origin"] != ws.mesh.origin
    kind = body.get("nkind")
    if kind not in KINDS:
        raise Rejected("the note kind is unknown")
    title = _line(body.get("title", ""), "the title", limits.subject_chars)
    text = _text(body.get("body"), "the body", limits.note_body_bytes)
    source = _line(body.get("source", ""), "the source", 300)
    tags = body.get("tags", [])
    if type(tags) is not list or len(tags) > 10:
        raise Rejected("tags are at most ten words")
    tags = [_line(t, "a tag", 40) for t in tags]
    ttl = body.get("ttl_s", 0.0)
    if type(ttl) not in (int, float) or not math.isfinite(ttl) or not 0 <= ttl <= MAX_TTL_S:
        raise Rejected("the lifetime is not a bounded number")
    old = [int(views.notes[j]) for j in body.get("supersedes_jids", []) if type(j) is str and j in views.notes]
    author, role, cmd, qid, flags = str(body.get("author")), str(body.get("role")), "", "", []
    if foreign:
        who = _sender(ws, row, author)
        if ws.notes.count_by(who.id) >= limits.notes_per_agent:
            raise Rejected("the sender has written as many notes as it may")
        ws._screen(who, "the note", limits.note_body_bytes, title, text, source, " ".join(tags))
        qid, flags = ws._hold(who, "note", f"{who.id}:{title[:40]}", title, text)
        title, text = ("", PLACEHOLDER.format(qid=qid, why=", ".join(flags))) if qid else (title, text)
        author, role = who.id, AGENT
        try:
            ws.notes.check_supersedes(who, old)
        except (ValueError, Denied):
            old = []
    else:
        cmd = _line(body.get("verify_cmd", ""), "the command", 500)
    made = ws.notes.log.append({
        "op": "add", "author": author, "role": role, "nkind": kind, "title": title, "body": text,
        "source": source, "tags": tags, "supersedes": old, "verify_cmd": cmd, "ttl_s": float(ttl),
        "held": qid, "flags": flags, "jid": jid})
    return int(made["seq"])
