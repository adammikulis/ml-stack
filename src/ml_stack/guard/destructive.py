"""Labels a tool call ``safe``, ``reversible``, ``destructive`` or ``unsure`` without running it.

`classify` reads the tool's metadata (a catalog, MCP annotations, the verbs in its name) and its
arguments (shell commands, SQL, paths, flags, sizes) and returns a `Verdict` from fixed rules.
The text of a call is data: it is never executed, never followed, and a longer call is not
read past `MAX_INPUT` characters but labelled ``unsure``. `combine` merges a model's verdict into
a deterministic one and can only raise it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.guard import argscan
from ml_stack.guard.harm import ASKS, LABELS, RANK, Finding, Verdict, worst
from ml_stack.guard.verbs import verb_finding, words_of
from ml_stack.interventions import Call

__all__ = ["ASKS", "LABELS", "MAX_INPUT", "Verdict", "classify", "combine", "reason_text"]

MAX_INPUT = 20_000
MAX_REASONS = 4


def _hint(annotations: Mapping[str, Any] | None) -> Finding | None:
    if not annotations:
        return None
    if annotations.get("destructiveHint") is True:
        return Finding("destructive", "the tool declares itself destructive")
    if annotations.get("readOnlyHint") is True:
        return Finding("safe", "the tool declares itself read-only")
    return None


def _named(name: str, catalog: Mapping[str, str], annotations: Mapping[str, Any] | None
           ) -> tuple[Finding | None, bool]:
    """The label the tool's metadata gives, and whether the catalog (which also vouches for how
    its arguments are read) gave it."""
    if name in catalog:
        return Finding(catalog[name], f"{name} is {catalog[name]} in the tool catalog"), True
    hint = _hint(annotations)
    verb = verb_finding(words_of(name), f"{name}'s")
    if hint is not None and hint.label == "destructive":
        return hint, False
    if verb is not None:
        return verb, False
    return hint, False


def reason_text(verdict: Verdict) -> str:
    """The verdict's reasons as one sentence for the person."""
    return "; ".join(verdict.reasons) if verdict.reasons else verdict.label


def classify(call: Call, *, roots: Sequence[str] = (), catalog: Mapping[str, str] | None = None,
             annotations: Mapping[str, Mapping[str, Any]] | None = None) -> Verdict:
    """The verdict for ``call``. ``roots`` are the directories writes may go to (the first is
    where relative paths are read); ``catalog`` maps tool names to a label; ``annotations`` maps
    tool names to their MCP annotations. The same input gives the same verdict."""
    cat = catalog or {}
    args = call.arguments
    if args is None:
        return Verdict("unsure", ["the arguments are not a JSON object, so they cannot be read"],
                       confidence=0.0)
    try:
        size = len(json.dumps(args, ensure_ascii=False, default=str))
    except (TypeError, ValueError, RecursionError):
        return Verdict("unsure", ["the arguments cannot be read"], confidence=0.0)
    if size > MAX_INPUT:
        return Verdict("unsure", [f"the call is {size} characters, too long to read"], confidence=0.0)
    where = tuple(roots)
    named, vouched = _named(call.name, cat, (annotations or {}).get(call.name))
    found: list[Finding] = [named] if named else []
    read = False
    if not vouched:
        more, read = argscan.content(call.name, args, where)
        found += more
    found += argscan.generic(call.name, args, where, named.label if named else "")
    if not found:
        return Verdict("unsure", [f"{call.name} is not a tool the classifier knows and its "
                                  "arguments say nothing about what it does"], confidence=0.0)
    if not read and named is None:
        found.append(Finding("unsure", f"{call.name} is not a tool the classifier knows"))
    label = worst(found)
    reasons = list(dict.fromkeys(f.reason for f in found if f.label == label))
    if label == "safe":
        reasons = ["only reads"]
    return Verdict(label, reasons[:MAX_REASONS], confidence=0.0 if label == "unsure" else 1.0)


def combine(base: Verdict, other: Verdict | None) -> Verdict:
    """``base`` unless ``other`` is more severe: a second opinion can raise a verdict, never
    lower it."""
    if other is None or RANK[other.label] <= RANK[base.label]:
        return base
    return other
