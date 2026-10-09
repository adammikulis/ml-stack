"""What a remembered fact is, and the checks every fact passes before it is written."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from ml_stack.guard.secrets import secrets_in
from ml_stack.guard.untrusted import fold, injection_markers
from ml_stack.sentinel.human import agent_may
from ml_stack.sentinel.redaction import redact

__all__ = ["BUILD_BOUND", "KINDS", "MAX_FACTS", "MAX_FACT_CHARS", "SOURCES", "STALE_DAYS", "STATES", "Fact", "Refused",
           "check", "clean"]

KINDS = ("preference", "machine", "result", "note")
SOURCES = ("user-said", "agent-observed", "tool-result")
MAX_FACT_CHARS = 400
MAX_FACTS = 300
STATES = ("current", "superseded")
BUILD_BOUND = frozenset({"machine", "result"})
"""Kinds that describe a particular llama.cpp build and go stale when it changes."""
STALE_DAYS = {"machine": 90.0, "result": 30.0}
"""Days after the last confirmation at which a kind of fact is marked for re-checking."""


class Refused(ValueError):
    """A fact that may not be stored, with the reason."""


@dataclass(slots=True)
class Fact:
    """One remembered thing and where it came from."""

    id: str
    text: str
    kind: str
    source: str
    created: float
    last_confirmed: float
    confirm_count: int = 1
    scope: dict[str, str] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    state: str = "current"
    links: list[str] = field(default_factory=list)
    realm: str = ""
    """``user`` or ``project`` in a merged view; empty in a store's own facts."""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> Fact:
        """The fact a stored row describes; raises ``Refused`` when the row is malformed."""
        try:
            scope = {str(k): str(v) for k, v in dict(row.get("scope") or {}).items()}
            fact = cls(str(row["id"]), str(row["text"]), str(row["kind"]), str(row["source"]),
                       float(row["created"]), float(row["last_confirmed"]),
                       int(row.get("confirm_count", 1)), scope,
                       [str(e) for e in row.get("entities") or []], str(row.get("state", "current")),
                       [str(x) for x in row.get("links") or []])
        except (KeyError, TypeError, ValueError) as exc:
            raise Refused(f"a stored fact is malformed ({exc})") from exc
        if fact.kind not in KINDS or fact.source not in SOURCES or fact.state not in STATES:
            raise Refused("a stored fact has an unknown kind, source or state")
        return fact


_AUTHORITY = tuple(re.compile(p, re.I) for p in (
    r"\bapprov\w*\b.{0,30}\b(?:hosts?|domains?|downloads?|grants?|requests?)\b",
    r"\b(?:grants?|permissions?|privileges?|roles?|authori[sz]\w*|entitle\w*)\b",
    r"\b(?:without|no need to|don'?t|do not|never|stop|skip\w*)\s+(?:to\s+)?"
    r"(?:ask\w*|confirm\w*|check\w*|prompt\w*)",
    r"\bauto[- ]?(?:approve|accept|confirm|allow|yes)\w*\b",
    r"\b(?:disabl\w*|turn\w*\s+off|bypass\w*|weaken\w*|relax\w*|override\w*)\b.{0,30}"
    r"\b(?:guard|rails?|sentinel|polic\w*|confirm\w*|fence|checks?)\b",
    r"\b(?:you\s+(?:may|can|are\s+allowed\s+to|should|must|will)|always|from\s+now\s+on)\b.{0,40}"
    r"\b(?:call|run|execute|invoke|start|download|fetch|approve|release|purge|mint)\b",
    r"\bquarantin\w*\b",
))
_PERSONAL = (
    ("an email address", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("a phone number", re.compile(r"(?<![\w.])\+?\d[\d ()./-]{8,}\d(?![\w.])")),
    ("a card or id number", re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b(?:\d[ -]?){13,19}\b")),
)


def clean(text: str) -> str:
    """``text`` on one line with look-alike characters folded and invisible, control and
    unassigned characters removed."""
    kept = []
    for char in fold(str(text)):
        if char.isspace() or unicodedata.category(char) == "Cc":
            kept.append(" ")
        elif unicodedata.category(char) not in ("Cs", "Co", "Cn"):
            kept.append(char)
    return " ".join("".join(kept).split())


def check(text: str, *, person: bool = False) -> str:
    """The cleaned ``text``, or ``Refused`` saying why it may not be stored: empty, over the
    length limit, holding a credential, reading as an instruction, claiming authority or
    naming sentinel's state, or (unless ``person`` typed it) holding personal details."""
    text = clean(text)
    if not text:
        raise Refused("the fact is empty")
    if len(text) > MAX_FACT_CHARS:
        raise Refused(f"the fact is {len(text)} characters; the limit is {MAX_FACT_CHARS}")
    kinds = secrets_in(text)
    if kinds or redact(text) != text:
        raise Refused(f"the fact holds a credential ({', '.join(kinds) or 'secret-like text'})")
    marks = injection_markers(text)
    if marks:
        raise Refused(f"the fact reads like an instruction to a model ({', '.join(marks)})")
    if any(p.search(text) for p in _AUTHORITY):
        raise Refused("a fact cannot grant, change or waive a permission, a role or a check")
    why = agent_may("memory", {"fact": text})
    if why:
        raise Refused(f"a fact cannot touch what only a person controls ({why})")
    if not person:
        for what, pattern in _PERSONAL:
            if pattern.search(text):
                raise Refused(f"the fact holds {what}; the person can add it with ml-stack-memory add")
    return text
