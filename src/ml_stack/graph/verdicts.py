"""The verdicts and merges a store remembers, so a pair is judged once: reading both
documents, writing one back at a time, and matching a verdict to a node the store has
rebuilt under another id."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

from ml_stack.graph.hygiene import Report

__all__ = ["DECISIONS", "MERGES", "SECTIONS", "conflict_key", "conflict_remembered",
           "decisions_in", "keep_decision", "keep_merge", "keep_merges", "merges_in", "now",
           "pair_key", "pair_remembered", "remembered", "stale", "suspect_verdict"]

DECISIONS = "tidy:decisions"      # the store document every judged pair is written to
MERGES = "tidy:merges"            # the store document every merge is written to
SECTIONS = ("pairs", "conflicts", "definitions", "suspects")


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))



def now() -> str:
    """This moment, as a verdict records it."""
    return time.strftime("%FT%T")


def decisions_in(store: Any) -> dict[str, dict[str, Any]]:
    """Every verdict the store remembers, by section: ``pairs`` (two names), ``conflicts``
    (two verbs between the same ends), ``definitions``, ``suspects``."""
    doc = store.get_doc(DECISIONS) if hasattr(store, "get_doc") else None
    doc = doc if isinstance(doc, dict) else {}
    return {name: (dict(doc[name]) if isinstance(doc.get(name), dict) else {})
            for name in SECTIONS}



def merges_in(store: Any) -> list[dict[str, Any]]:
    """Every merge the store remembers, oldest first."""
    doc = store.get_doc(MERGES) if hasattr(store, "get_doc") else None
    doc = doc.get("merges") if isinstance(doc, dict) else doc
    return [dict(m) for m in doc if isinstance(m, Mapping)] if isinstance(doc, list) else []



def keep_merge(store: Any, merges: list[dict[str, Any]], entry: Mapping[str, Any]) -> None:
    """One merge into the store's merges document, at once."""
    keep_merges(store, merges, [entry])



def keep_merges(store: Any, merges: list[dict[str, Any]],
                 entries: Iterable[Mapping[str, Any]]) -> None:
    """``entries`` into the store's merges document in one write; a (kept, gone) pair
    already there is not written twice, and a read-only store is not written."""
    seen = {(m.get("kept"), m.get("gone")) for m in merges}
    new = []
    for entry in entries:
        pair = (entry["kept"], entry["gone"])
        if pair not in seen:
            seen.add(pair)
            new.append(dict(entry))
    if not new:
        return
    merges.extend(new)
    if not hasattr(store, "put_doc") or getattr(store, "read_only", False):
        return
    store.put_doc(MERGES, {"merges": merges, "hidden": True})



def remembered(decisions: dict[str, dict[str, Any]], section: str, key: str, *, rejudge: bool,
          report: Report) -> dict[str, Any] | None:
    """The verdict the store remembers under ``key``; None when there is none, or when it
    is to be asked again."""
    found = decisions[section].get(key)
    if found is not None and rejudge:
        report.rejudged += 1
        return None
    return found



def keep_decision(store: Any, decisions: dict[str, dict[str, Any]], section: str, key: str,
                   decision: Mapping[str, Any]) -> None:
    """One verdict into the store's decisions document, at once: a judge run killed
    halfway keeps what it paid for."""
    decisions.setdefault(section, {})[key] = dict(decision)
    if not hasattr(store, "put_doc") or getattr(store, "read_only", False):
        return
    store.put_doc(DECISIONS, {**{name: decisions.get(name) or {} for name in SECTIONS},
                              "hidden": True})



def conflict_key(ends: tuple[str, str], one: str, other: str) -> str:
    return "|".join(ends) + "::" + "|".join(sorted((one, other)))



def suspect_verdict(decisions: dict[str, dict[str, Any]], node: Mapping[str, Any], *,
                     rejudge: bool, report: Report) -> dict[str, Any] | None:
    """The verdict the store holds for one doubtful label, by id or by the label itself.

    A rebuild from the reads can give a node an id the verdict was not written against,
    so the label and kind the verdict recorded stand in for it.
    """
    decision = remembered(decisions, "suspects", str(node.get("id") or ""), rejudge=rejudge,
                 report=report)
    if decision is not None or rejudge:
        return decision
    label, kind = str(node.get("label") or ""), str(node.get("kind") or "")
    for one in (decisions.get("suspects") or {}).values():
        if str((one or {}).get("label") or "") == label \
                and str((one or {}).get("kind") or kind) == kind:
            return dict(one)
    return None



def pair_remembered(decisions: dict[str, dict[str, Any]], one: str, other: str,
                     kind: str) -> dict[str, Any] | None:
    """One name-pair verdict found by the two labels it recorded rather than by node id."""
    want = {one, other}
    for one in (decisions.get("pairs") or {}).values():
        one = one or {}
        if {str(one.get("a_label") or ""), str(one.get("b_label") or "")} == want \
                and str(one.get("kind") or kind) == kind:
            return dict(one)
    return None



def conflict_remembered(decisions: dict[str, dict[str, Any]], labels: set[str],
                         verbs: list[str]) -> dict[str, Any] | None:
    """One conflict verdict found by the two labels it recorded rather than by node id."""
    for one in (decisions.get("conflicts") or {}).values():
        if list((one or {}).get("verbs") or ()) == verbs \
                and set((one or {}).get("labels") or ()) == labels:
            return dict(one)
    return None



def stale(decisions: dict[str, dict[str, Any]], nodes: Mapping[str, Mapping[str, Any]]) -> int:
    """Verdicts naming a node the store does not hold, by id and by recorded label."""
    labels = {str(n.get("label") or "") for n in nodes.values()}

    def gone(node_id: Any, label: Any) -> bool:
        return str(node_id or "") not in nodes and str(label or "") not in labels

    out = 0
    for one in (decisions.get("pairs") or {}).values():
        one = one or {}
        out += int(gone(one.get("a"), one.get("a_label"))
                   or gone(one.get("b"), one.get("b_label")))
    for one in (decisions.get("conflicts") or {}).values():
        ends = list((one or {}).get("ends") or ("", ""))
        said = list((one or {}).get("labels") or ("", ""))
        out += int(any(gone(end, name) for end, name in zip(ends, said + ["", ""])))
    for key, one in (decisions.get("suspects") or {}).items():
        out += int(gone(key, (one or {}).get("label")))
    return out
