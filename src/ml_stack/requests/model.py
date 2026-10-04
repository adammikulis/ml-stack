"""What a request is: its kinds, its choices with what each does, its states and its fingerprint."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ml_stack.safetext import escape

__all__ = ["CHOICES", "KINDS", "PENDING", "RESOLVED", "STATES", "VIAS", "Choice", "Kind", "Origin",
           "Request", "fingerprint", "make_choices", "shown"]

PENDING = "pending"
RESOLVED = ("approved", "denied", "expired", "cancelled", "superseded")
STATES = (PENDING, *RESOLVED)
VIAS = ("terminal", "ui", "dialog")
"""The three ways a person answers: a terminal at a tty, the UI's browser session, the desktop dialog."""

MOST = {"subject": 300, "reason": 400, "name": 64, "label": 40, "effect": 200}
APPROVING = frozenset({"allow-once", "allow-always", "approve", "approve-other", "release"})
"""Choices that let the thing asked about go ahead; every other choice, and expiry, does not."""

CHOICES: dict[str, tuple[str, str]] = {
    "allow-once": ("allow this time", "The call runs once; nothing is saved."),
    "allow-always": ("always allow", "The call runs and a rule is saved so a call like it runs without asking."),
    "never": ("never allow", "The call is refused and a rule is saved so a call like it is refused without asking."),
    "deny": ("no", "The call is refused this time; nothing is saved."),
    "cancel-hold": ("cancel", "The held call is dropped and does not run."),
    "release": ("release", "What is held is put back in use."),
    "keep-held": ("keep held", "What is held stays blocked and is not asked about again."),
    "later": ("later", "Nothing changes now; the question comes back in a few hours."),
    "approve": ("approve", "The request is granted as described."),
    "approve-other": ("approve elsewhere", "The request is granted, in the other scope it names instead of the one shown."),
}
"""Every choice a request may offer: its id, the label shown, and what it does in one fixed sentence."""


@dataclass(frozen=True, slots=True)
class Kind:
    """What a kind of request is: ``human_only`` kinds (a release, a host, a grant) are
    re-checked by the action itself and never approved in bulk; ``destructive`` kinds never
    offer an always choice or bulk approval."""

    summary: str
    human_only: bool = False
    destructive: bool = False


KINDS: dict[str, Kind] = {
    "tool_call": Kind("a tool call that waits for a yes"),
    "tool_call_destructive": Kind("a destructive or unsure tool call", destructive=True),
    "memory_remember": Kind("a fact the agent wants remembered"),
    "hold": Kind("a held call counting down"),
    "quarantine_release": Kind("something sentinel holds that a person may release", human_only=True),
    "host_approval": Kind("a network host an agent asked to reach", human_only=True),
    "pairing": Kind("a machine or agent asking to join", human_only=True),
    "grant": Kind("a grant of a human-only action", human_only=True),
    "keystore_unlock": Kind("the keystore asking to be unlocked", human_only=True),
    "rule_suggestion": Kind("a rule the agent suggests"),
}


@dataclass(frozen=True, slots=True)
class Choice:
    """One thing a person may answer: ``id`` is from `CHOICES`, ``label`` and ``effect`` are its fixed words."""

    id: str
    label: str
    effect: str

    @property
    def approving(self) -> bool:
        return self.id in APPROVING


@dataclass(frozen=True, slots=True)
class Origin:
    """Who raised a request: the agent, the project and the session, as text the raiser named."""

    agent: str = ""
    project: str = ""
    session: str = ""


@dataclass(frozen=True, slots=True)
class Request:
    """One question for a person. Everything but the state and the answer is fixed when it is raised."""

    id: str
    kind: str
    raised_by: Origin
    subject: str
    reason: str
    choices: tuple[Choice, ...]
    created: float
    expires: float
    state: str = PENDING
    answered_by: str = ""
    answered_at: float = 0.0
    answer: str = ""
    key: str = ""
    extra: Mapping[str, str] = field(default_factory=dict)

    @property
    def human_only(self) -> bool:
        return KINDS[self.kind].human_only

    @property
    def destructive(self) -> bool:
        return KINDS[self.kind].destructive

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)

    def choice(self, ident: str) -> Choice | None:
        return next((c for c in self.choices if c.id == ident), None)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "raised_by": _origin(self.raised_by),
                "subject": self.subject, "reason": self.reason,
                "choices": [[c.id, c.label, c.effect] for c in self.choices],
                "created": self.created, "expires": self.expires, "state": self.state,
                "answered_by": self.answered_by, "answered_at": self.answered_at,
                "answer": self.answer, "key": self.key, "extra": dict(self.extra)}

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Request:
        origin = row.get("raised_by") or {}
        return cls(
            id=str(row["id"]), kind=str(row["kind"]),
            raised_by=Origin(str(origin.get("agent", "")), str(origin.get("project", "")),
                             str(origin.get("session", ""))),
            subject=str(row["subject"]), reason=str(row["reason"]),
            choices=tuple(Choice(str(a), str(b), str(c)) for a, b, c in row["choices"]),
            created=float(row["created"]), expires=float(row["expires"]), state=str(row["state"]),
            answered_by=str(row.get("answered_by", "")), answered_at=float(row.get("answered_at", 0.0)),
            answer=str(row.get("answer", "")), key=str(row.get("key", "")),
            extra={str(k): str(v) for k, v in dict(row.get("extra") or {}).items()})


def _origin(origin: Origin) -> dict[str, str]:
    return {"agent": origin.agent, "project": origin.project, "session": origin.session}


def fingerprint(request: Request) -> str:
    """A hash of every word and choice a person is shown: a change to any of them changes it."""
    fixed = {"id": request.id, "kind": request.kind, "raised_by": _origin(request.raised_by),
             "subject": request.subject, "reason": request.reason,
             "choices": [[c.id, c.label, c.effect] for c in request.choices],
             "expires": request.expires, "extra": dict(request.extra)}
    return hashlib.sha256(json.dumps(fixed, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def shown(value: object, kind: str) -> str:
    """``value`` as one safe line cut to the bound for ``kind`` (a key of `MOST`)."""
    return escape(str(value), MOST[kind])


def make_choices(ids: Sequence[str], *, destructive: bool = False) -> tuple[Choice, ...]:
    """The choices for ``ids`` with their fixed words; ``ValueError`` for an unknown id, or
    for an always choice on a destructive request."""
    out = []
    for ident in dict.fromkeys(ids):
        if ident not in CHOICES:
            raise ValueError(f"no choice {ident!r}: the choices are {', '.join(CHOICES)}")
        if destructive and ident == "allow-always":
            raise ValueError("a destructive request never offers always allow")
        out.append(Choice(ident, *CHOICES[ident]))
    if not out:
        raise ValueError("a request offers at least one choice")
    return tuple(out)
