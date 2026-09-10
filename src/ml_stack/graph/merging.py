"""Folding one thing into another once something has said to: a node into the node it turned
out to be, an incoming node onto the node the store already held, a rejected edge into the
edge that was kept, and a doubtful label renamed, dropped or kept."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from ml_stack.graph.hygiene import SOURCE_LINKS, Report, kept_spans, union
from ml_stack.graph.names import TRAILING_PREPOSITION
from ml_stack.graph.verdicts import (
    conflict_key,
    conflict_remembered,
    keep_decision,
    keep_merge,
    now,
    pair_key,
    remembered,
)

__all__ = ["drop_edge", "fold_node", "merge_nodes", "resolve_conflict", "resolve_suspect"]

def fold_node(kept: Mapping[str, Any], gone: Mapping[str, Any]) -> dict[str, Any]:
    """An incoming node onto the node it turned out to be: mentions summed, provenance and
    aliases unioned, both definitions kept."""
    node = dict(kept)
    attrs = dict(node.get("attrs") or {})
    aliases = list(attrs.get("aliases") or [])
    for alias in [str(gone.get("label") or ""),
                  *((gone.get("attrs") or {}).get("aliases") or ())]:
        if alias and alias != node.get("label") and alias not in aliases:
            aliases.append(alias)
    attrs["aliases"] = aliases
    for key, value in (gone.get("attrs") or {}).items():
        if (key not in ("aliases", "hidden", "suspect", "passage", "definition",
                        "definitions_also") and value and not attrs.get(key)):
            attrs[key] = value
    _merge_definitions(node, gone, attrs, judge=None, decisions=None, store=None, report=None,
                       note=None)
    node["attrs"] = attrs
    node["mentions"] = int(node.get("mentions") or 0) + int(gone.get("mentions") or 0)
    node["provenance"] = union(node.get("provenance"), gone.get("provenance"))
    kept_spans(node, node, gone)
    return node


def resolve_conflict(store: Any, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]],
                      ends: tuple[str, str], group: dict[str, dict[str, Any]], rels: list[str],
                      *, judge: Any, decisions: dict[str, dict[str, Any]], report: Report,
                      note: Callable[[str], None], rejudge: bool = False,
                      dry_run: bool = False) -> list[str]:
    """The verbs left between two nodes after each pair is settled.

    A verdict the store holds is applied whether or not there is a judge; a pair with no
    verdict and no judge is left for the conflict report.
    """
    one, other = nodes[ends[0]], nodes[ends[1]]
    left = list(rels)
    at = 0
    while at < len(left) - 1:
        a_rel, b_rel = left[at], left[at + 1]
        key = conflict_key(ends, a_rel, b_rel)
        decision = remembered(decisions, "conflicts", key, rejudge=rejudge, report=report)
        if decision is None and not rejudge:
            decision = conflict_remembered(
                decisions, {str(one.get("label") or ""), str(other.get("label") or "")},
                sorted((a_rel, b_rel)))
        replayed = decision is not None
        if decision is None:
            if judge is None:
                report.unjudged += 1
                at += 1
                continue
            decision = judge.decide_conflict(one, other, group[a_rel], group[b_rel])
            decision = {**decision, "ends": list(ends), "verbs": sorted((a_rel, b_rel)),
                    "labels": [str(one.get("label") or ""), str(other.get("label") or "")],
                    "model": getattr(judge, "model", ""), "when": now()}
            if not decision.get("failed"):
                keep_decision(store, decisions, "conflicts", key, decision)
        report.conflicts_judged += 1
        report.replayed += int(replayed)
        verdict = str(decision.get("verdict") or "unsure")
        drop = b_rel if verdict == f"keep {a_rel}" else a_rel if verdict == f"keep {b_rel}" else ""
        if not drop:
            note(f"conflict judged ({verdict}): {one['label']!r} and {other['label']!r} joined "
                 f"by {a_rel} and {b_rel} -- {decision.get('why', '')}")
            at += 1
            continue
        kept_rel = a_rel if drop == b_rel else b_rel
        drop_edge(store, edges, group[kept_rel], group.pop(drop), dry_run=dry_run)
        left.remove(drop)
        report.conflict_edges_dropped += 1
        note(f"conflict judged: {one['label']!r} and {other['label']!r} keep {kept_rel}; "
             f"{drop} dropped, its weight and provenance folded in -- {decision.get('why', '')}")
    return left



def drop_edge(store: Any, edges: list[dict[str, Any]], keep: dict[str, Any],
               gone: dict[str, Any], *, dry_run: bool = False) -> None:
    """``gone`` out of the store and the working list, its weight and provenance into ``keep``."""
    keep["weight"] = int(keep.get("weight") or 0) + int(gone.get("weight") or 0)
    keep["provenance"] = union(keep.get("provenance"), gone.get("provenance"))
    kept_spans(keep, keep, gone)
    if not dry_run:
        store.remove_edge(gone["source"], gone["rel"], gone["target"])
        store.upsert_edge(keep)
    for at, edge in enumerate(edges):
        if edge is gone:
            edges.pop(at)
            break


def resolve_suspect(store: Any, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]],
                     node: dict[str, Any], why: str, *, judge: Any,
                     decisions: dict[str, dict[str, Any]], report: Report,
                     note: Callable[[str], None], merges: list[dict[str, Any]] | None = None,
                     decision: Mapping[str, Any] | None = None, dry_run: bool = False) -> None:
    """One doubtful label renamed, dropped, or kept with the flag cleared.

    ``decision`` is the verdict the store already carries for this label; without one the
    judge is asked and its answer written down.
    """
    replayed = decision is not None
    if decision is None:
        decision = judge.decide_suspect(node, why)
        decision = {**decision, "label": node.get("label"), "kind": node.get("kind"), "flagged": why,
                "model": getattr(judge, "model", ""), "when": now()}
        if not decision.get("failed"):
            keep_decision(store, decisions, "suspects", node["id"], decision)
    report.suspects_resolved += 1
    report.replayed += int(replayed)
    verdict = str(decision.get("verdict") or "keep")
    label = str(node.get("label") or "")
    name = str(decision.get("name") or "").strip()
    if verdict == "rename" and name and name != label:
        into = next((n for n in nodes.values()
                     if n["id"] != node["id"] and str(n.get("label") or "") == name
                     and n.get("kind") == node.get("kind")), None)
        if into is not None:
            moved = merge_nodes(store, nodes, edges, into["id"], node["id"], dry_run=dry_run,
                           judge=judge, decisions=decisions, report=report, note=note,
                           merges=merges)
            report.merged_nodes += 1
            report.merged_edges += moved
            note(f"suspect: {label!r} is {name!r}, which is already a node -- merged into it "
                 f"({moved} edge(s) moved)")
            return
        node["label"] = name
        if not dry_run:
            store.rename(node["id"], name)
            store.unset_attribute(node["id"], "suspect")
        (node.setdefault("attrs", {})).pop("suspect", None)
        note(f"suspect: {label!r} renamed to {name!r} -- {decision.get('why', '')}")
    elif verdict == "drop":
        gone = [e for e in edges if node["id"] in (e["source"], e["target"])]
        if not dry_run:
            store.drop([node["id"]])
        nodes.pop(node["id"], None)
        edges[:] = [e for e in edges if node["id"] not in (e["source"], e["target"])]
        report.suspects_dropped += 1
        note(f"suspect: {label!r} dropped with {len(gone)} edge(s) -- {decision.get('why', '') or why}")
    else:
        if not dry_run:
            store.unset_attribute(node["id"], "suspect")
        (node.setdefault("attrs", {})).pop("suspect", None)
        note(f"suspect: {label!r} kept, the flag cleared -- {decision.get('why', '') or why}")

_FRAGMENT = re.compile(r"^(that|which|who|and|or|but|of|in|to|for|with|by)\b", re.I)



def _is_fragment(text: str) -> bool:
    """Whether a definition reads as half a sentence rather than a definition."""
    words = text.split()
    return (len(words) < 3 or bool(_FRAGMENT.match(text))
            or bool(TRAILING_PREPOSITION.search(text.rstrip("."))))



def _better_definition(one: str, other: str) -> tuple[str, str]:
    """``(the definition, the other)``: the longer that is not a clause fragment."""
    if _is_fragment(one) != _is_fragment(other):
        return (other, one) if _is_fragment(one) else (one, other)
    return (one, other) if len(one) >= len(other) else (other, one)



def _definition_remembered(decisions: dict[str, dict[str, Any]] | None, a_said: str,
                           b_said: str) -> dict[str, Any] | None:
    """The verdict the store holds about which of two definitions is the definition."""
    for one in ((decisions or {}).get("definitions") or {}).values():
        one = one or {}
        if {str(one.get("a") or ""), str(one.get("b") or "")} == {a_said, b_said}:
            return dict(one)
    return None



def _substantially(one: str, other: str) -> bool:
    """Whether two definitions say different things -- neither one the start of the other."""
    a, b = " ".join(one.split()).casefold(), " ".join(other.split()).casefold()
    return not (a.startswith(b) or b.startswith(a))



def _merge_definitions(kept: Mapping[str, Any], gone: Mapping[str, Any],
                       attrs: dict[str, Any], *, judge: Any,
                       decisions: dict[str, dict[str, Any]] | None, store: Any,
                       report: Report | None, note: Callable[[str], None] | None) -> None:
    """The definition of the survivor, with the other kept in ``definitions_also``."""
    a_said = str((kept.get("attrs") or {}).get("definition") or "").strip()
    b_said = str((gone.get("attrs") or {}).get("definition") or "").strip()
    also = [d for d in (attrs.get("definitions_also") or []) if d]
    for one in ((gone.get("attrs") or {}).get("definitions_also") or []):
        if one and one not in also:
            also.append(one)
    if a_said and b_said:
        better, spare = _better_definition(a_said, b_said)
        differ = _substantially(a_said, b_said)
        remembered = _definition_remembered(decisions, a_said, b_said)
        if judge is None and remembered is not None:
            pick = str(remembered.get("keep") or "both")
            if pick in ("a", "b"):
                better, spare = ((a_said, b_said) if pick == "a" else (b_said, a_said))
            if report is not None:
                report.replayed += 1
        elif judge is not None and differ:
            answer = judge.decide_definition(kept, gone, a_said, b_said)
            pick = str((answer or {}).get("keep") or "both")
            if pick == "a":
                better, spare = a_said, b_said
            elif pick == "b":
                better, spare = b_said, a_said
            if report is not None:
                report.definitions_judged += 1
            if decisions is not None:
                keep_decision(store, decisions, "definitions",
                               pair_key(str(kept.get("id")), str(gone.get("id"))),
                               {**answer, "a": a_said, "b": b_said,
                                "model": getattr(judge, "model", ""), "when": now()})
            if note is not None:
                note(f"definition ({kept.get('label')!r}): kept {pick} -- "
                     f"{(answer or {}).get('why', '')}")
        attrs["definition"] = better
        if differ and spare not in also:
            also.append(spare)
    elif b_said and not a_said:
        attrs["definition"] = b_said
    if also:
        attrs["definitions_also"] = [d for d in also if d != attrs.get("definition")]


def merge_nodes(store: Any, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]],
           keep: str, remove: str, *, dry_run: bool, judge: Any = None,
           decisions: dict[str, dict[str, Any]] | None = None, report: Report | None = None,
           note: Callable[[str], None] | None = None,
           merges: list[dict[str, Any]] | None = None) -> int:
    """``remove`` into ``keep`` with everything kept: edges moved (summed where the
    survivor had the same one), mentions summed, provenance unioned, the name an alias,
    and the merge written to ``merges`` and the store's `MERGES` document."""
    kept, gone = nodes[keep], nodes[remove]
    entry = {"kept": keep, "kept_label": str(kept.get("label") or ""),
             "gone": remove, "gone_label": str(gone.get("label") or ""),
             "kind": str(kept.get("kind") or ""), "edges_moved": 0,
             "kept_from": list(kept.get("provenance") or ()),
             "gone_from": list(gone.get("provenance") or ()), "at": now()}
    moved = 0
    by_triple = {(e["source"], e["rel"], e["target"]): e for e in edges}
    for edge in [e for e in edges if remove in (e["source"], e["target"])]:
        source = keep if edge["source"] == remove else edge["source"]
        target = keep if edge["target"] == remove else edge["target"]
        if source == target and edge["rel"] not in SOURCE_LINKS:
            # a relation between the two names being joined says nothing once they are one
            by_triple.pop((edge["source"], edge["rel"], edge["target"]), None)
            if not dry_run:
                store.remove_edge(edge["source"], edge["rel"], edge["target"])
            continue
        other = by_triple.get((source, edge["rel"], target))
        merged = {"source": source, "rel": edge["rel"], "target": target,
                  "weight": int(edge.get("weight") or 0) + int((other or {}).get("weight") or 0),
                  "provenance": union((other or {}).get("provenance"), edge.get("provenance"))}
        kept_spans(merged, other, edge)
        by_triple.pop((edge["source"], edge["rel"], edge["target"]), None)
        by_triple[(source, edge["rel"], target)] = merged
        moved += 1
        if not dry_run:
            store.remove_edge(edge["source"], edge["rel"], edge["target"])
            store.upsert_edge(merged)
    attrs = dict(kept.get("attrs") or {})
    aliases = list(attrs.get("aliases") or [])
    for alias in [str(gone.get("label") or ""), *((gone.get("attrs") or {}).get("aliases") or [])]:
        if alias and alias != kept.get("label") and alias not in aliases:
            aliases.append(alias)
    attrs["aliases"] = aliases
    for key, value in (gone.get("attrs") or {}).items():
        if (key not in ("aliases", "hidden", "suspect", "definition", "definitions_also")
                and value and not attrs.get(key)):
            attrs[key] = value
    _merge_definitions(kept, gone, attrs, judge=judge, decisions=decisions, store=store,
                       report=report, note=note)
    kept["attrs"] = attrs
    kept["mentions"] = int(kept.get("mentions") or 0) + int(gone.get("mentions") or 0)
    kept["provenance"] = union(kept.get("provenance"), gone.get("provenance"))
    kept_spans(kept, kept, gone)
    if not dry_run:
        store.upsert_node(kept)
        store.drop([remove])
        if merges is not None:
            keep_merge(store, merges, {**entry, "edges_moved": moved})
    edges[:] = list(by_triple.values())
    nodes.pop(remove, None)
    return moved
