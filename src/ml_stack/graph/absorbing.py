"""An incoming graph reconciled against what a store already holds, on the way in, and
handed back with its ids rewritten.

Every writer of learned knowledge calls this before it upserts -- the ingest's fold, a
community extraction, a conversation kept in the graph -- because the same concepts come
back as more is read. The whole-store pass (`ml_stack.graph.tidy`) is for what was written
before it existed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.graph.hygiene import (
    NEVER_FOLDED,
    SOURCE_LINKS,
    Report,
    hidden,
    kept_spans,
    union,
)
from ml_stack.graph.merging import fold_node
from ml_stack.graph.names import kin, near, same_name
from ml_stack.graph.store import GraphStore
from ml_stack.graph.verdicts import (
    decisions_in,
    keep_decision,
    keep_merges,
    merges_in,
    now,
    pair_key,
)

__all__ = ["absorb"]

def absorb(store: Any, graph: Mapping[str, Any], *, judge: Any = None,
           sources: Callable[[str], str] | None = None,
           log: Callable[[str], None] | None = None) -> Report:
    """Reconcile an incoming ``{nodes, edges}`` graph against what a store already holds,
    on the way in, and return it with its ids rewritten (``report.graph``).

    Every writer of learned knowledge calls this before it upserts -- the ingest's fold, a
    community extraction, a conversation kept in the graph -- because the same concepts come
    back as more is read. The whole-store pass (`tidy`) is for what was written before it
    existed.

    An incoming node whose name is an existing node's under case, spacing and hyphens
    (`same_name`), or its plural, is rewritten onto that node and its edges with it. A close
    spelling goes to ``judge``, with the incoming node's own passage (``attrs.passage``, or
    ``sources`` over its provenance) beside the existing node's: ``same`` rewrites it onto
    the existing node, ``different`` is written to ``tidy:decisions`` and the node stays new,
    ``unsure`` stays new and is reported. Every name landed on an existing node is written
    to `MERGES`. Nothing else in the store is written; a read-only store gets neither.
    """
    if isinstance(store, (str, Path)):
        with GraphStore(store, read_only=judge is None) as opened:
            return absorb(opened, graph, judge=judge, sources=sources, log=log)
    report = Report(dry_run=True)
    say = log or (lambda _line: None)

    def note(line: str) -> None:
        report.lines.append(line)
        say(line)

    incoming = [dict(node) for node in (graph.get("nodes") or ())]
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    by_length: dict[str, dict[int, list[str]]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for node in store.nodes():
        kind = str(node.get("kind") or "")
        if hidden(node) or kind in NEVER_FOLDED:
            continue
        key = same_name(node.get("label"))
        by_id[str(node["id"])] = node
        if key not in by_key.setdefault(kind, {}):
            by_key[kind][key] = node
            by_length.setdefault(kind, {}).setdefault(len(key), []).append(key)

    decisions = decisions_in(store)
    merges = merges_in(store)
    landed: dict[str, dict[str, Any]] = {}
    carried: dict[str, str] = {}
    original = getattr(judge, "sources", None)
    if judge is not None:
        def _text(unit: str) -> str:
            if unit in carried:
                return carried[unit]
            if sources is not None:
                text = sources(unit)
                if text:
                    return text
            return original(unit) if original is not None else ""

        judge.sources = _text
    mapping: dict[str, str] = {}
    try:
        for node in incoming:
            kind = str(node.get("kind") or "")
            same_kind = by_key.get(kind)
            if hidden(node) or kind in NEVER_FOLDED or not same_kind:
                continue
            node_id = str(node.get("id") or "")
            key = same_name(node.get("label"))
            found = same_kind.get(key)
            if found is not None and str(found["id"]) == node_id:
                continue
            how = "the same name"
            if found is None:
                for other_name in kin(key):
                    found = same_kind.get(other_name)
                    if found is not None:
                        how = "a plural"
                        break
            if found is not None:
                mapping[node_id] = str(found["id"])
                landed[node_id] = dict(found)
                if how == "the same name":
                    report.mapped_same_name += 1
                else:
                    report.mapped_plural += 1
                note(f"absorb: {node.get('label')!r} is {found['label']!r}, already held "
                     f"({how})")
                continue
            for other_key in near(by_length.get(kind) or {}, key):
                other = same_kind[other_key]
                if str(other["id"]) == node_id:
                    continue
                pair = pair_key(node_id, str(other["id"]))
                verdict = "unsure"
                answer: Mapping[str, Any] = decisions["pairs"].get(pair) or {}
                if answer:
                    verdict = str(answer.get("verdict") or "unsure")
                elif judge is not None:
                    carried.clear()
                    asked = dict(node)
                    passage = (node.get("attrs") or {}).get("passage")
                    if passage:
                        carried[f"incoming:{node_id}"] = str(passage)
                        asked["provenance"] = [f"incoming:{node_id}",
                                               *(node.get("provenance") or ())]
                    answer = judge.decide(asked, other)
                    verdict = str(answer.get("verdict") or "unsure")
                    keep_decision(store, decisions, "pairs", pair,
                                   {**answer, "a": node_id, "b": str(other["id"]), "kind": kind,
                                    "a_label": str(node.get("label") or ""),
                                    "b_label": str(other.get("label") or ""),
                                    "model": getattr(judge, "model", ""), "when": now(),
                                    "absorbed": True})
                if verdict == "same":
                    mapping[node_id] = str(other["id"])
                    landed[node_id] = dict(other)
                    report.judged_same += 1
                    note(f"absorb: {node.get('label')!r} judged the same as "
                         f"{other['label']!r} -- {answer.get('why', '')}")
                    break
                if verdict == "different":
                    report.judged_different += 1
                    note(f"absorb: {node.get('label')!r} judged different from "
                         f"{other['label']!r} -- {answer.get('why', '')}")
                    continue
                report.left_possible += 1
                report.possible.append((str(node.get("label") or ""),
                                        str(other.get("label") or "")))
                note(f"absorb: {node.get('label')!r} ~ {other['label']!r} -- a spelling apart, "
                     "left as a new node")
    finally:
        if judge is not None:
            judge.sources = original

    out: dict[str, dict[str, Any]] = {}
    for node in incoming:
        node_id = str(node.get("id") or "")
        final = mapping.get(node_id, node_id)
        if final in out:
            out[final] = fold_node(out[final], node)
        elif final != node_id:
            out[final] = fold_node(by_id[final], node)
        else:
            out[final] = node
    edged: dict[tuple[str, str, str], dict[str, Any]] = {}
    rewired: dict[str, int] = {}
    for edge in (graph.get("edges") or ()):
        for end in {str(edge.get("source") or ""), str(edge.get("target") or "")} & landed.keys():
            rewired[end] = rewired.get(end, 0) + 1
        source = mapping.get(str(edge.get("source") or ""), str(edge.get("source") or ""))
        target = mapping.get(str(edge.get("target") or ""), str(edge.get("target") or ""))
        rel = str(edge.get("rel") or "")
        if source == target and rel not in SOURCE_LINKS:
            continue
        merged = {**edge, "source": source, "target": target}
        already = edged.get((source, rel, target))
        if already is not None:
            merged["weight"] = int(already.get("weight") or 0) + int(edge.get("weight") or 0)
            merged["provenance"] = union(already.get("provenance"), edge.get("provenance"))
            kept_spans(merged, already, edge)
        edged[(source, rel, target)] = merged
    report.graph = {**{k: v for k, v in graph.items() if k not in ("nodes", "edges")},
                    "nodes": list(out.values()), "edges": list(edged.values())}
    report.mapping = dict(mapping)
    by_incoming = {str(n.get("id") or ""): n for n in incoming}
    keep_merges(store, merges, [
        {"kept": str(found["id"]), "kept_label": str(found.get("label") or ""),
         "gone": node_id, "gone_label": str(by_incoming[node_id].get("label") or ""),
         "kind": str(found.get("kind") or ""), "edges_moved": rewired.get(node_id, 0),
         "kept_from": list(found.get("provenance") or ()),
         "gone_from": list(by_incoming[node_id].get("provenance") or ()), "at": now()}
        for node_id, found in landed.items()])
    note(report.absorbed())
    return report
