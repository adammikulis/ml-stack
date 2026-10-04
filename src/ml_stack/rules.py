"""The person's always and never rules for the calls an agent asks about, kept per user.

A rule names a tool, a pattern for each of the call's arguments, a verdict and optionally a
role. Only the person's answer at the prompt and the ``/rules`` commands change the file; no
tool reaches it. A file that cannot be read whole, or whose mode is open to others, matches
nothing, so every call asks.
"""

from __future__ import annotations

import json
import logging
import re
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ml_stack import files, home
from ml_stack.chatpolicy import _TOOL_NAME, CONFIRM, READ, _outside_state, catalog
from ml_stack.guard.destructive import classify
from ml_stack.guard.destructive_rail import default_roots
from ml_stack.interventions import Call

__all__ = ["Rule", "Rules", "blocked_reason", "describe", "run_command"]

logger = logging.getLogger("ml_stack.guard")
SCHEMA_VERSION = 1
FILE = "agent-rules.json"
EVENTS = "agent-rules-events.jsonl"
VERDICTS = ("always", "never")


@dataclass(frozen=True)
class Rule:
    """``match`` holds one pattern per argument (``*`` and ``?`` are the only wildcards);
    ``role`` is empty for any role; ``tainted_ok`` lets an always rule apply to a run that has
    read outside text."""

    tool: str
    match: tuple[tuple[str, str], ...]
    verdict: str
    role: str = ""
    created: str = ""
    fired: int = 0
    tainted_ok: bool = False


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v) for v in value)
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _glob(pattern: str, text: str) -> bool:
    rx = "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern)
    return re.fullmatch(rx, text, re.S) is not None


def describe(rule: Rule) -> str:
    """The rule in words: what it covers and for which role."""
    what = ", ".join(f"{k} {v}" for k, v in rule.match) or "no arguments"
    head = "Always allow" if rule.verdict == "always" else "Never allow"
    scope = f" in the {rule.role} role" if rule.role else ""
    tail = " (also when the run has read outside text)" if rule.tainted_ok else ""
    return f"{head} {rule.tool} for {what}{scope}{tail}"


def blocked_reason(call_name: str, arguments: Mapping[str, Any]) -> str:
    """Why an always rule cannot be offered for this call; empty when it can."""
    if call_name == "models_fetch":
        return "downloads always ask: their size is not known before they start"
    if _outside_state(dict(arguments)):
        return "it names a path outside ml-stack's state"
    seen = classify(Call(call_name, dict(arguments)), roots=default_roots(), catalog=catalog())
    if seen.asks:
        return f"the classifier labelled it {seen.label}: {'; '.join(seen.reasons)}"
    texts = [_text(v) for v in arguments.values()]
    if any("*" in t or "?" in t for t in texts):
        return "an argument holds a wildcard character"
    return ""


def _valid(row: Any) -> Rule:
    if not isinstance(row, dict):
        raise ValueError("a rule is not an object")
    match = row.get("match")
    if not isinstance(match, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                              for k, v in match.items()):
        raise ValueError("match must map argument names to patterns")
    rule = Rule(str(row.get("tool")), tuple(sorted(match.items())), str(row.get("verdict")),
                str(row.get("role") or ""), str(row.get("created") or ""),
                int(row.get("fired") or 0), bool(row.get("tainted_ok", False)))
    if rule.verdict not in VERDICTS:
        raise ValueError(f"verdict {rule.verdict!r}")
    if rule.tool not in CONFIRM or rule.tool in READ or _TOOL_NAME.search(rule.tool):
        raise ValueError(f"{rule.tool!r} is not a tool a rule may cover")
    if rule.verdict == "always" and any(not v.replace("*", "").replace("?", "").strip()
                                        for _, v in rule.match):
        raise ValueError("an always pattern must name something")
    return rule


class Rules:
    """The rules in one file, read whole each time they are asked about."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self.rules: list[Rule] = []
        self.broken = ""
        self.load()

    @property
    def path(self) -> Path:
        return self._path or home.state(FILE)

    def load(self) -> None:
        """Read the file; on any fault ``broken`` says why and no rule matches."""
        self.rules, self.broken = [], ""
        path = self.path
        if not path.exists():
            return
        try:
            if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise ValueError("the file is readable or writable by others (mode must be 0600)")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unknown schema_version")
            self.rules = [_valid(r) for r in data.get("rules") or []]
        except (OSError, ValueError, TypeError) as exc:
            self.rules, self.broken = [], f"{path.name} is not used, every call asks: {exc}"

    def _save(self) -> None:
        rows = [{"tool": r.tool, "match": dict(r.match), "verdict": r.verdict, "role": r.role,
                 "created": r.created, "fired": r.fired, "tainted_ok": r.tainted_ok}
                for r in self.rules]
        files.write_json(self.path, {"schema_version": SCHEMA_VERSION, "rules": rows})
        self.path.chmod(0o600)

    def _event(self, kind: str, rule: Rule) -> None:
        line = {"ts": time.strftime("%FT%T"), "event": kind, "rule": describe(rule)}
        logger.warning("agent rule %s: %s", kind, line["rule"])
        with (self.path.parent / EVENTS).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    def covers(self, name: str, arguments: Mapping[str, Any] | None, role: str,
               tainted: bool) -> Rule | None:
        """The rule that decides this call: a never rule first, then an always rule that
        applies (not one barred by outside text unless it says ``tainted_ok``)."""
        self.load()
        if arguments is None or self.broken:
            return None
        found = [r for r in self.rules if r.tool == name and r.role in ("", role)
                 and {k for k, _ in r.match} == set(arguments)
                 and all(_glob(p, _text(arguments[k])) for k, p in r.match)]
        never = [r for r in found if r.verdict == "never"]
        if never:
            return never[0]
        ok = [r for r in found if not tainted or r.tainted_ok]
        return ok[0] if ok else None

    def fire(self, rule: Rule) -> None:
        at = self.rules.index(rule)
        self.rules[at] = replace(rule, fired=rule.fired + 1)
        self._save()

    def make(self, name: str, arguments: Mapping[str, Any], verdict: str, role: str) -> Rule:
        """The rule covering exactly this call's arguments; ``ValueError`` when none can be."""
        if verdict == "always" and (why := blocked_reason(name, arguments)):
            raise ValueError(why)
        return _valid({"tool": name, "match": {k: _text(v) for k, v in arguments.items()},
                       "verdict": verdict, "role": role, "created": time.strftime("%F")})

    def add(self, name: str, arguments: Mapping[str, Any], verdict: str, role: str) -> Rule:
        """Save the rule `make` builds."""
        rule = self.make(name, arguments, verdict, role)
        if self.broken:
            raise ValueError(self.broken)
        if rule not in self.rules:
            self.rules.append(rule)
            self._save()
            self._event("added", rule)
        return rule

    def _at(self, number: int) -> int:
        if not 1 <= number <= len(self.rules):
            raise ValueError(f"there is no rule {number}")
        return number - 1

    def remove(self, number: int) -> None:
        rule = self.rules.pop(self._at(number))
        self._save()
        self._event("removed", rule)

    def flip(self, number: int) -> None:
        """Change an always rule to never and back."""
        at = self._at(number)
        rule = self.rules[at]
        if rule.verdict == "never" and (why := blocked_reason(rule.tool, dict(rule.match))):
            raise ValueError(why)
        self.rules[at] = replace(rule, verdict="never" if rule.verdict == "always" else "always")
        self._save()
        self._event("flipped", self.rules[at])

    def tainted(self, number: int) -> None:
        """Let an always rule apply to a run that has read outside text, or stop it."""
        at = self._at(number)
        rule = self.rules[at]
        if rule.verdict != "always":
            raise ValueError("only an always rule has this setting")
        self.rules[at] = replace(rule, tainted_ok=not rule.tainted_ok)
        self._save()
        self._event("tainted-setting", self.rules[at])

    def clear(self) -> None:
        for rule in self.rules:
            self._event("removed", rule)
        self.rules = []
        self._save()

    def listing(self) -> str:
        lines = [self.broken] if self.broken else []
        lines += [f"{n}. {describe(r)}  [{r.fired}x, since {r.created or '?'}]"
                  for n, r in enumerate(self.rules, 1)]
        return "\n".join(lines) or "no rules: every call that acts asks"


RULES_HELP = ("/rules                 list the rules, numbered\n"
              "/rules remove N        delete rule N (the call asks again)\n"
              "/rules flip N          always <-> never\n"
              "/rules tainted N       let always rule N apply after outside text was read\n"
              "/rules clear           delete every rule")


def run_command(rules: Rules, words: Sequence[str]) -> str:
    """Do the ``/rules`` command ``words`` (what follows the command) and say what happened."""
    rules.load()
    verb, rest = (words[0], words[1:]) if words else ("list", [])
    try:
        if verb == "list":
            return rules.listing()
        if verb == "clear":
            rules.clear()
            return "all rules deleted"
        if verb in ("remove", "flip", "tainted") and len(rest) == 1 and rest[0].isdigit():
            getattr(rules, verb)(int(rest[0]))
            return f"{verb} done\n" + rules.listing()
    except (ValueError, OSError) as exc:
        return str(exc)
    return RULES_HELP
