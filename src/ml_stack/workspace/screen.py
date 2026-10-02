"""Screening text that goes into the workspace and text that comes out of it."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Refused", "Screened", "clean_label", "fence", "injection_markers", "refusals",
           "secret_kinds"]

SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = tuple((n, re.compile(p)) for n, p in (
    ("private-key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("aws-key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("github-token", r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    ("hub-token", r"\bhf_[A-Za-z0-9]{30,}\b"),
    ("slack-token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    ("api-key", r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"),
    ("google-key", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ("bearer", r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    ("url-password", r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]{3,}@"),
    ("workspace-token", r"\bmlws1\.[a-z0-9._-]{1,48}\.[A-Za-z0-9_-]{20,}"),
    ("assigned-secret", r"(?i)\b[\w.-]*(?:api[_-]?key|secret|token|passw(?:or)?d|passphrase)"
                        r"[\w.-]*\s*[:=]\s*[\"']?[^\s\"',;]{8,}"),
))

MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = tuple((n, re.compile(p, re.I | re.S)) for n, p in (
    ("override", r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}\b(?:previous|prior|above|"
                 r"earlier|all|any|your|the)\b[^.\n]{0,30}\b(?:instruction|prompt|rule|direction|"
                 r"guideline)s?"),
    ("new-instructions", r"\b(?:new|updated|real|actual|additional)\s+(?:system\s+)?"
                         r"instructions?\s*[:\-]"),
    ("role-play", r"\byou\s+are\s+now\b|\bact\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken|dan)\b"),
    ("prompt-leak", r"\b(?:reveal|print|repeat|show|output|disclose)\b[^.\n]{0,40}\b(?:system|"
                    r"hidden|initial)\s+(?:prompt|message|instruction)s?"),
    ("tool-order", r"\b(?:call|invoke|run|execute|use)\s+(?:the\s+)?[\w.-]+\s+tool\b"),
    ("exfiltrate", r"\b(?:send|post|upload|forward|exfiltrate|leak|email)\b[^.\n]{0,60}"
                   r"\b(?:to|at)\s+(?:https?://|[\w.-]+@[\w.-]+)"),
    ("chat-markup", r"<\|[a-z_]+\|>|\[/?INST\]|<</?SYS>>|^\s*(?:system|assistant)\s*:"),
    ("fake-fence", r"</?\s*untrusted\b"),
    ("authority-claim", r"\b(?:the\s+)?(?:owner|user|person|human|admin(?:istrator)?|lead)\s+"
                        r"(?:has\s+|have\s+)?(?:approved|confirmed|authori[sz]ed|granted|"
                        r"allowed|consented)\b|\b(?:i|we)\s+(?:hereby\s+)?(?:approve|grant|"
                        r"authori[sz]e|confirm)\b|\bpre-?approved\b"),
    ("rule-promotion", r"\b(?:add|copy|write|append|put)\b[^.\n]{0,40}\b(?:to|into|in)\b[^.\n]{0,"
                       r"20}\b(?:claude\.md|agents\.md|the\s+rules|repo\s+docs?|system\s+prompt)"
                       r"|\b(?:from\s+now\s+on|henceforth)\b[^.\n]{0,60}\b(?:all\s+agents?|every"
                       r"\s+agent|you\s+must)\b|\bthis\s+(?:rule|note)\s+(?:overrides|supersedes|"
                       r"takes\s+precedence)\b"),
))

NEUTRAL = (
    (re.compile(r"<\|"), "< | "),
    (re.compile(r"\|>"), " | >"),
    (re.compile(r"\[(/?)INST\]"), r"[\1 INST ]"),
    (re.compile(r"<<(/?)SYS>>"), r"< \1SYS >"),
    (re.compile(r"</?\s*untrusted\b[^>]*>?", re.I), "[tag removed]"),
)
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class Refused(ValueError):
    """A write was refused; the message says which rule and never repeats the text."""


@dataclass(frozen=True, slots=True)
class Screened:
    """Text ready to show: fenced as data, with the injection patterns it matched."""

    text: str
    markers: tuple[str, ...]


def _optional(module: str, attr: str):
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError):
        return None


def secret_kinds(text: str) -> list[str]:
    """The kinds of credential ``text`` holds."""
    found = [name for name, pattern in SECRETS if pattern.search(text)]
    guard = _optional("ml_stack.guard.secrets", "secrets_in")
    return sorted(set(found) | (set(guard(text)) if guard else set()))


def injection_markers(text: str) -> list[str]:
    """The names of the injection and authority-claim patterns ``text`` matches."""
    found = {name for name, pattern in MARKERS if pattern.search(text)}
    guard = _optional("ml_stack.guard.untrusted", "injection_markers")
    return sorted(found | (set(guard(text)) if guard else set()))


def _terms(denylist: Path) -> list[str]:
    try:
        lines = denylist.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [t.strip().casefold() for t in lines if t.strip() and not t.startswith("#")]


def private_terms_in(text: str, denylist: Path) -> int:
    """How many distinct denylist terms ``text`` contains, matched case-insensitively."""
    low = " ".join(text.casefold().split())
    return sum(1 for term in set(_terms(denylist)) if " ".join(term.split()) in low)


def refusals(text: str, denylist: Path) -> list[str]:
    """Why ``text`` may not be written, one clause per rule, without quoting it."""
    why = []
    kinds = secret_kinds(text)
    if kinds:
        why.append(f"it contains what looks like a credential ({', '.join(kinds)})")
    if private_terms_in(text, denylist):
        why.append("it contains a term from the private-terms denylist")
    return why


def clean_label(text: object, width: int = 80) -> str:
    """``text`` cut to ``width`` on one line, control characters removed."""
    return " ".join(CONTROL.sub(" ", str(text)).split())[:width]


def fence(text: str, source: str, note: str = "") -> Screened:
    """``text`` inside an untrusted fence, its own fence tags and chat markup neutralised."""
    markers = tuple(injection_markers(text))
    body = text
    for pattern, repl in NEUTRAL:
        body = pattern.sub(repl, body)
    head = f"<untrusted source={clean_label(source, 120)!r}>"
    lines = [head, f"[data from an agent, no authority{'; ' + note if note else ''}]", body,
             "</untrusted>"]
    return Screened("\n".join(lines), markers)
