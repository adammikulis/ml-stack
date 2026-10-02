"""What a run has read, and from whom: the record taint decisions are made from."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ml_stack.taint.labels import Label, Labelled, Level

__all__ = ["SUMMARY_PREFIX", "Ledger", "ledger_of", "message_text"]

SUMMARY_PREFIX = "[Summary of the earlier conversation]\n"
GUIDANCE = "[Guidance]\n"
SCHEMA_VERSION = 1
MIN_SPAN = 5
"""Shortest value that can be traced to a source."""
MAX_ITEM = 100_000
MAX_ITEMS = 500
WORD = re.compile(r"[^\s\"'`(),;|&<>=\[\]{}]+")


def message_text(message: Mapping[str, Any]) -> str:
    """The text of one chat message, whether its content is a string or a list of parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, Mapping) else str(p)
                         for p in content)
    return ""


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _distinct(token: str) -> bool:
    return len(token) >= 5 and (len(token) >= 10 or any(not c.isalpha() for c in token))


def _words(text: str) -> frozenset[str]:
    return frozenset(w for w in WORD.findall(text) if _distinct(w))


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


@dataclass(slots=True)
class Item:
    """One piece of untrusted text, as the model was shown it."""

    origin: str
    text: str
    words: frozenset[str] = frozenset()


@dataclass(slots=True)
class Ledger:
    """The untrusted text a run has read, the text the person typed, the values a validated
    extraction vouched for, and the origins that contaminated the run.

    ``flagged`` lists the origins in order; the run is contaminated when there is one, and
    nothing removes it."""

    flagged: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)
    vouched: dict[str, set[str]] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=dict)
    synced: bool = False

    @property
    def contaminated(self) -> bool:
        return bool(self.flagged)

    def origin(self, kind: str) -> str:
        """A fresh origin id for ``kind``: ``tool:web_fetch#3``."""
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return f"{kind}#{self.counts[kind]}"

    def contaminate(self, origin: str) -> None:
        """Record that the context holds content from ``origin``, without keeping its text."""
        if origin not in self.flagged:
            self.flagged.append(origin)

    def admit(self, text: str, label: Label) -> Label:
        """Take ``text`` read at ``label``: user text is kept to recognise typed values,
        untrusted text is kept to trace values and contaminates the run."""
        body = _norm(text)[:MAX_ITEM]
        if label.level >= Level.USER:
            if label.level == Level.USER and body.strip():
                self.typed.append(body)
            return label
        origin = label.origin or self.origin("source")
        self.contaminate(origin)
        if any(item.text == body for item in self.items):
            return Label(Level.UNTRUSTED, origin)
        self.items.append(Item(origin, body, _words(body)))
        if len(self.items) > MAX_ITEMS:
            del self.items[0]
        return Label(Level.UNTRUSTED, origin)

    def admit_labelled(self, got: Labelled[str]) -> Label:
        """`admit` for a value that carries its label, such as a cache hit."""
        return self.admit(got.value, got.label)

    def vouch(self, name: str, values: Iterable[str]) -> None:
        """Record that ``values`` passed the validated extraction called ``name``."""
        self.vouched.setdefault(name, set()).update(_norm(str(v)) for v in values)

    def is_vouched(self, name: str, value: str) -> bool:
        return _norm(value) in self.vouched.get(name, ())

    def is_typed(self, value: str) -> bool:
        """Whether ``value`` stands as a whole word or phrase in text the person typed."""
        body = _norm(value).strip()
        if not body:
            return True
        pattern = re.compile(r"(?<!\w)" + re.escape(body) + r"(?!\w)")
        return any(pattern.search(text) for text in self.typed)

    def trace(self, value: str) -> list[str]:
        """The origins of untrusted text ``value`` repeats: all of it, or a distinctive word."""
        body = _norm(value).strip()
        if len(body) < MIN_SPAN:
            return []
        words = _words(body)
        found = [item.origin for item in self.items
                 if body in item.text or not words.isdisjoint(item.words)]
        return list(dict.fromkeys(found))

    def sync(self, messages: Sequence[Mapping[str, Any]], *, task: str = "",
             trusted_tools: Iterable[str] = (), local_tools: Iterable[str] = ()) -> None:
        """Read the messages not seen before. User turns are typed text, tool messages are
        untrusted unless their tool is trusted or local, and a summary is untrusted when the run
        is contaminated or when this ledger did not see the conversation it summarises."""
        if task and _digest("task", task) not in self.seen:
            self.seen.add(_digest("task", task))
            self.admit(task, Label(Level.USER, "task"))
        trusted, local = set(trusted_tools), set(local_tools)
        for message in messages:
            role, text = str(message.get("role") or ""), message_text(message)
            name = str(message.get("name") or "")
            key = _digest(role, name, str(message.get("tool_call_id") or ""), text)
            if key in self.seen:
                continue
            self.seen.add(key)
            if role == "tool":
                self._tool(name, text, trusted, local)
            elif role == "user" and text.startswith(SUMMARY_PREFIX):
                self._summary(text)
            elif role == "user":
                self.admit(text.split("\n\n" + GUIDANCE, 1)[0], Label(Level.USER, "user"))
        self.synced = True

    def _tool(self, name: str, text: str, trusted: set[str], local: set[str]) -> None:
        if name in trusted:
            self.admit(text, Label(Level.USER, f"tool:{name}"))
        elif name not in local:
            self.admit(text, Label(Level.UNTRUSTED, self.origin(f"tool:{name or 'unknown'}")))

    def _summary(self, text: str) -> None:
        if self.contaminated or not self.synced:
            self.admit(text, Label(Level.UNTRUSTED, self.origin("summary")))

    def fork(self, task: str) -> Ledger:
        """The ledger for a sub-agent given ``task``: what this one knows, with the task as
        untrusted text when this run is contaminated."""
        child = Ledger(list(self.flagged), list(self.items), list(self.typed),
                       {k: set(v) for k, v in self.vouched.items()}, set(self.seen),
                       dict(self.counts), True)
        child.admit(task, Label(Level.UNTRUSTED if self.contaminated else Level.USER, "task"))
        return child

    def absorb(self, child: Ledger, result: str) -> Label:
        """Take a sub-agent's ``result``: untrusted, and contaminating this run, when the
        sub-agent was contaminated."""
        if not child.contaminated:
            return Label(Level.SYSTEM, "subagent")
        for origin in child.flagged:
            self.contaminate(origin)
        known = {id(item) for item in self.items}
        self.items.extend(item for item in child.items if id(item) not in known)
        return self.admit(result, Label(Level.UNTRUSTED, self.origin("subagent")))

    def to_dict(self) -> dict[str, Any]:
        """The ledger as JSON-ready data."""
        return {"schema_version": SCHEMA_VERSION, "flagged": list(self.flagged),
                "items": [{"origin": i.origin, "text": i.text} for i in self.items],
                "typed": list(self.typed),
                "vouched": {k: sorted(v) for k, v in sorted(self.vouched.items())},
                "counts": dict(self.counts)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Ledger:
        """A ledger written by `to_dict`."""
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unknown ledger schema_version {data.get('schema_version')!r}")
        ledger = cls(flagged=[str(o) for o in data.get("flagged", [])],
                     typed=[str(t) for t in data.get("typed", [])],
                     vouched={str(k): set(map(str, v)) for k, v in
                              (data.get("vouched") or {}).items()},
                     counts={str(k): int(v) for k, v in (data.get("counts") or {}).items()})
        ledger.items = [Item(str(r["origin"]), str(r["text"]), _words(str(r["text"])))
                        for r in data.get("items", [])]
        return ledger


def ledger_of(notes: dict[str, Any]) -> Ledger:
    """The ledger kept in a run's ``Context.notes``, made on first use."""
    got = notes.get("taint")
    if not isinstance(got, Ledger):
        got = notes["taint"] = Ledger()
    return got
