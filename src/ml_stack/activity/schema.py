"""The activity record: its fields, its bounds, and the one function that makes one safe."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict, Unpack

from ml_stack.sentinel.explain import show
from ml_stack.sentinel.redaction import redact_value

__all__ = ["BODY_KEYS", "KINDS", "MAX_BYTES", "VERSION", "Entry", "Said", "build", "valid_kind"]

VERSION = 1
MAX_TEXT = 200
MAX_KEYS = 16
MAX_REFS = 12
MAX_BYTES = 4096
_KIND = re.compile(r"[a-z][a-z0-9_]{0,23}(?:\.[a-z0-9_]{1,24}){0,3}")
_KEY = re.compile(r"[a-z][a-z0-9_]{0,31}")

BODY_KEYS = frozenset({"body", "text", "content", "message", "prompt", "output", "result",
                       "arguments", "args", "stdout", "stderr", "response", "answer", "transcript"})
"""Metadata names that would carry a body; they are dropped, never stored."""

KINDS: dict[str, str] = {
    "agent.tool_call": "a model's tool call and the outcome of it (blocked, error, ok)",
    "approval.asked": "a question put to the person before a tool call runs",
    "approval.answered": "the person's answer: allow_once, always, never or no",
    "approval.rule_fired": "a saved allow/never rule decided a call without asking",
    "request.raised": "a component put a question to the person (kind, agent, project; never the call's text)",
    "request.answered": "the person answered a request: the choice and the way (terminal, ui, dialog)",
    "request.cancelled": "the raiser withdrew a request before it was answered",
    "role.changed": "the person moved a session to another role",
    "rule.added": "the person saved an allow/never rule",
    "rule.removed": "the person removed a rule",
    "rule.flipped": "the person turned an always rule into a never rule or back",
    "rule.tainted_setting": "the person changed whether a rule applies after outside text was read",
    "workspace.message": "a message between agents (from, to, type, thread, size; never the body)",
    "workspace.claim": "an agent took, released or lost a claim (claimed, released, conflict, expired)",
    "workspace.event": "any other workspace audit row (minted, revoked, refused, held, released)",
    "security.*": "every sentinel event: keystore operations, quarantines, releases, dialogs",
    "reputation.*": "a source's standing changed",
    "model.lease": "a model server was started for someone (model, quant, flags, GPU share)",
    "net.download": "a download or pull (host, size, hash, outcome)",
    "bench.run": "a bench, jevbench or decide run (command, model, build, result summary, path)",
    "test.result": "a test run on a tree: pass and fail counts, duration",
    "activity.off": "the person turned the log off for a process",
    "activity.off_refused": "an agent asked for the log to be turned off and was refused",
    "activity.gap": "records were dropped before this one (count and cause)",
    "activity.export": "the person wrote the log out to a file",
}
"""The kinds the stack writes. A well-formed kind not listed here is accepted."""


def valid_kind(kind: str) -> bool:
    """Whether ``kind`` is dotted lower-case words of bounded length."""
    return bool(_KIND.fullmatch(kind))


@dataclass(frozen=True, slots=True)
class Entry:
    """One record as read back: where it sits in the chain and what it says."""

    seq: int
    hash: str
    ts: float
    actor: str
    session: str
    kind: str
    subject: str
    outcome: str
    refs: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """The short name a record is shown and fetched by."""
        return self.hash[:12]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], seq: int, digest: str) -> Entry:
        return cls(seq=seq, hash=digest, ts=float(payload.get("ts", 0.0)),
                   actor=str(payload.get("actor", "")), session=str(payload.get("session", "")),
                   kind=str(payload.get("kind", "")), subject=str(payload.get("subject", "")),
                   outcome=str(payload.get("outcome", "")),
                   refs=dict(payload.get("refs") or {}), meta=dict(payload.get("meta") or {}))


def _scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, str):
        return show(value, MAX_TEXT)
    try:
        return show(json.dumps(value, default=str, sort_keys=True), MAX_TEXT)
    except (TypeError, ValueError):
        return show(repr(value), MAX_TEXT)


def _mapping(raw: Mapping[str, Any] | None, limit: int, *, drop_bodies: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in redact_value(dict(raw or {})).items():
        key = str(name).lower()
        if not _KEY.fullmatch(key) or (drop_bodies and key in BODY_KEYS) or len(out) >= limit:
            continue
        out[key] = _scalar(value)
    return out


class Said(TypedDict, total=False):
    """What a record says beyond who and when."""

    subject: str
    outcome: str
    refs: Mapping[str, Any] | None
    meta: Mapping[str, Any] | None


def build(kind: str, *, ts: float, actor: str, session: str, **said: Unpack[Said]) -> dict[str, Any]:
    """The payload to store: text redacted and escaped, sizes bounded, bodies dropped. A
    record over `MAX_BYTES` loses its metadata first, then its references."""
    subject, outcome = said.get("subject", ""), said.get("outcome", "")
    refs, meta = said.get("refs"), said.get("meta")
    payload: dict[str, Any] = {
        "schema_version": VERSION, "ts": round(ts, 3), "actor": show(actor, 64),
        "session": show(session, 32), "kind": kind if valid_kind(kind) else "invalid",
        "subject": show(subject, MAX_TEXT), "outcome": show(outcome, 48),
        "refs": {k: str(v) for k, v in _mapping(refs, MAX_REFS, drop_bodies=False).items()},
        "meta": _mapping(meta, MAX_KEYS, drop_bodies=True)}
    if kind != payload["kind"]:
        payload["meta"] = {"asked_kind": show(kind, 48)}
    for squeezed in ("meta", "refs"):
        if len(json.dumps(payload, ensure_ascii=True)) <= MAX_BYTES:
            break
        payload[squeezed], payload["truncated"] = {}, True
    return payload
