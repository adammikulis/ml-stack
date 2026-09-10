"""The hygiene pass over a whole graph store: duplicates merged, inverses folded, the
doubtful flagged, the rest reported.

Two passes touch a knowledge graph, and they are kept apart. The *fold*
(``ml_stack.ingest``) is an upsert -- it adds nodes and edges and grows the ones it finds,
never merges, never removes. This is the other pass, the dedupe: run when a person asks, dry
by default, and what it merges it merges with everything kept on the survivor.
`ml_stack.graph.absorbing` is the third, run on the way in.

A verdict is bought once. Every one is written to ``tidy:decisions`` and every merge to
``tidy:merges`` (`ml_stack.graph.verdicts`), and this pass applies what it finds there
whether or not it has a judge, so a store rebuilt from its reads keeps what was already
settled. A verdict is matched by node id and, where the id is gone, by the labels and kind
it recorded; one whose subject the store no longer holds is counted and left in the
document.

What it does, in order:

1. **Duplicate nodes**, within a kind and across every source. Two names are the same thing
   when they differ only by case, spacing, hyphens or underscores, or when one is the
   other's plural (`ml_stack.graph.names`); the heavier name survives. A name
   `entities.close` calls one letter off another goes to the **judge**
   (`ml_stack.graph.judging`) when one is given, and is otherwise reported as a possible
   duplicate and not merged. "Same" merges into the heavier name; "different" is written
   down so the pair is never asked again; only what the model cannot settle after reading
   stays for a person, who hands the decision back as ``written`` (``{name: the name it
   is}``), applied whatever the weights. Figures, sources and runs are never folded.
2. **Relation spellings** folded to the vocabulary the store uses most (`fold_edges`), so
   ``has_part`` and ``haspart`` are one relationship.
3. **Inverse pairs**: ``X part_of Y`` beside ``Y has_part X`` is one fact stored twice. The
   canonical direction (`ml_stack.graph.relations`) is kept and the other's weight and
   provenance fold into it.
4. **Suspect labels**: a clause rather than a name, an over-generic word, a number, a
   single letter. Without a judge they are flagged and left (``attrs.suspect`` says why);
   with one, the label and its passages go to the model, which renames it, drops it, or
   keeps it and clears the flag.
5. **Verb conflicts**: two edges between the same ends with verbs that are not each
   other's inverse. Without a judge, reported. With one, both edges, the two nodes'
   definitions and the passages both were read from go to the model in one call: ``keep
   both``, ``keep <verb>`` or ``unsure``.
6. **Orphans** reported: nodes with no edge but their source link. Left alone.
7. **Self-loops** reported. Left alone.
8. **Hierarchy cycles**: each of `HIERARCHY`'s relations read on its own as a directed
   graph, and a ring among them reported as one line naming the nodes round it. Left
   alone, ``--apply`` and all.

A hidden node (``attrs.hidden`` -- an ingest run, a unit) is never touched. Idempotent: a
second run reports nothing to do.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.entities.fold import ESTABLISHED, fold_edges
from ml_stack.entities.spelling import close
from ml_stack.graph.hygiene import (
    NEVER_FOLDED,
    SOURCE_LINKS,
    Report,
    hidden,
    kept_spans,
    label_of,
    union,
)
from ml_stack.graph.merging import merge_nodes, resolve_conflict, resolve_suspect
from ml_stack.graph.names import plurals, same_name, suspect
from ml_stack.graph.relations import HIERARCHY, canonical_direction, cycles
from ml_stack.graph.store import GraphStore
from ml_stack.graph.verdicts import (
    decisions_in,
    keep_decision,
    merges_in,
    now,
    pair_key,
    pair_remembered,
    remembered,
    stale,
    suspect_verdict,
)

__all__ = ["tidy"]

def tidy(store: Any, *, dry_run: bool = True, established: int = ESTABLISHED,
         written: Mapping[str, str] | None = None, judge: Any = None,
         hierarchy: Iterable[str] = HIERARCHY,
         log: Callable[[str], None] | None = None, rejudge: bool = False) -> Report:
    """The pass, over a `GraphStore` or the path to one. Dry by default. ``written`` is
    the map of duplicates a person settled (``{name: the name it is}``), applied as given.
    ``judge`` (a `ModelJudge`) decides the pairs a spelling apart, and with one the pass
    is automated: it applies everything it decides and writes every verdict to the store
    with its reason (Adam: "record the mergers but don't defer them to a human"). A pair
    once judged different is not asked again; a pair the judge cannot settle even after
    reading the source is the one thing left for a person, as ``written``. ``rejudge``
    asks the judge again about every verdict the store holds and writes the new ones.
    ``hierarchy`` names the relations that must form a DAG; a cycle in one is reported and
    never broken."""
    if judge is not None:
        dry_run = False
    if isinstance(store, (str, Path)):
        with GraphStore(store, read_only=dry_run) as opened:
            report = tidy(opened, dry_run=dry_run, established=established, written=written,
                          judge=judge, hierarchy=hierarchy, log=log, rejudge=rejudge)
        if not dry_run:
            _recheck(store, report, log)
        return report
    report = Report(dry_run=dry_run)
    say = log or (lambda _line: None)

    def note(line: str) -> None:
        report.lines.append(line)
        say(line)

    nodes = {n["id"]: n for n in store.nodes() if not hidden(n)}
    edges = [e for e in store.edges() if e["source"] in nodes and e["target"] in nodes]
    decisions = decisions_in(store)
    merges = merges_in(store)
    rejudge = bool(rejudge and judge is not None)
    report.stale = stale(decisions, nodes)
    if report.stale:
        note(f"{report.stale} remembered verdict(s) name a node the store no longer has, and are "
             "left in the decisions document")

    # 1. duplicate nodes, within a kind
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for node in nodes.values():
        by_kind.setdefault(str(node.get("kind") or ""), []).append(node)
    merged_into: dict[str, str] = {}
    decided = {str(k).casefold(): str(v) for k, v in (written or {}).items()}
    for kind, group in by_kind.items():
        if kind in NEVER_FOLDED:
            continue
        # the same name under two ids -- case, spacing, hyphens -- is one node; the
        # heavier survives
        by_same: dict[str, dict[str, Any]] = {}
        for node in sorted(group, key=lambda n: -int(n.get("mentions") or 0)):
            key = same_name(node.get("label"))
            if key in by_same:
                merged_into[node["id"]] = by_same[key]["id"]
            else:
                by_same[key] = node
        survivors = list(by_same.values())
        by_label = {str(n.get("label") or ""): n for n in survivors}
        weight = {label: int(n.get("mentions") or 0) for label, n in by_label.items()}
        # plurals and written decisions join whatever the weights; nothing else does
        joins = plurals(weight)
        for low, into in decided.items():
            source = next((lbl for lbl in by_label if lbl.casefold() == low), None)
            target = next((lbl for lbl in by_label if lbl.casefold() == into.casefold()), None)
            if source and target and source != target:
                joins[source.casefold()] = target
        for low, into in joins.items():
            name = next((lbl for lbl in by_label if lbl.casefold() == low), None)
            if name and into in by_label and name != into:
                merged_into[by_label[name]["id"]] = by_label[into]["id"]
        # close spellings: the judge's call when there is one, else a person's
        labels = [lbl for lbl in by_label if by_label[lbl]["id"] not in merged_into]
        for i, one in enumerate(labels):
            for other in labels[i + 1:]:
                if not close(same_name(one), same_name(other)):
                    continue
                a, b = by_label[one], by_label[other]
                key = pair_key(a["id"], b["id"])
                decision = remembered(decisions, "pairs", key, rejudge=rejudge, report=report)
                if decision is None and not rejudge:
                    decision = pair_remembered(decisions, one, other, kind)
                report.replayed += int(decision is not None)
                if decision is None and judge is not None:
                    decision = judge.decide(a, b)
                    decision = {**decision, "a": a["id"], "b": b["id"], "kind": kind,
                                "a_label": one, "b_label": other,
                                "model": getattr(judge, "model", ""),
                                "when": now()}
                    if not decision.get("failed"):
                        keep_decision(store, decisions, "pairs", key, decision)
                verdict = str((decision or {}).get("verdict") or "unsure")
                if verdict == "same":
                    heavier, lighter = ((a, b) if int(a.get("mentions") or 0)
                                        >= int(b.get("mentions") or 0) else (b, a))
                    if lighter["id"] not in merged_into:
                        merged_into[lighter["id"]] = heavier["id"]
                    report.judged_same += 1
                    note(f"judged same ({kind}): {lighter['label']!r} -> {heavier['label']!r}"
                         f" -- {decision.get('why', '')}"
                         + (f" (read {len(decision.get('read') or ())} passage(s))"
                            if decision.get("read") else ""))
                elif verdict == "different":
                    report.judged_different += 1
                    note(f"judged different ({kind}): {one!r} | {other!r}"
                         f" -- {decision.get('why', '')}")
                else:
                    report.possible.append((one, other))
                    note(f"possible ({kind}): {one!r} ~ {other!r} -- a spelling apart"
                         + ("; the judge could not settle it" if decision else "")
                         + "; hand it back as written if they are one")
    # follow chains a -> b -> c to their end
    for remove in list(merged_into):
        keep = merged_into[remove]
        seen = {remove}
        while keep in merged_into and keep not in seen:
            seen.add(keep)
            keep = merged_into[keep]
        merged_into[remove] = keep
    for remove, keep in merged_into.items():
        if remove == keep or keep not in nodes or remove not in nodes:
            continue
        gone_label, gone_kind = nodes[remove]["label"], nodes[remove]["kind"]
        moved = merge_nodes(store, nodes, edges, keep, remove, dry_run=dry_run, judge=judge,
                       decisions=decisions, report=report, note=note, merges=merges)
        report.merged_nodes += 1
        report.merged_edges += moved
        note(f"merge: {gone_label!r} -> {nodes[keep]['label']!r} "
             f"({gone_kind}, {moved} edge(s) moved)")
    if merged_into:
        edges = [e for e in edges if e["source"] in nodes and e["target"] in nodes]

    # 2. relation spellings
    keyed = {(e["source"], e["rel"], e["target"]): dict(e) for e in edges
             if e["rel"] not in SOURCE_LINKS}
    folded, records = fold_edges(keyed, label="relations", provenance="provenance",
                                 settles="the spelling the store uses more is right")
    for record in records:
        report.relations_folded += 1
        note(f"relation: {record.get('from')} -> {record.get('into')}")
    if records and not dry_run:
        for key in keyed:
            if key not in folded:
                store.remove_edge(*key)
        for key, edge in folded.items():
            store.upsert_edge({**edge, "source": key[0], "rel": key[1], "target": key[2]})
    edges = [e for e in edges if e["rel"] in SOURCE_LINKS] + list(folded.values())

    # 3. inverse pairs
    by_triple = {(e["source"], e["rel"], e["target"]): e for e in edges}
    for triple, edge in list(by_triple.items()):
        source, rel, target = triple
        keep_rel, flipped = canonical_direction(rel)
        if not flipped:
            continue
        canonical_triple = (target, keep_rel, source)
        other = by_triple.get(canonical_triple)
        merged = {"source": target, "rel": keep_rel, "target": source,
                  "weight": int(edge.get("weight") or 0) + int((other or {}).get("weight") or 0),
                  "provenance": union((other or {}).get("provenance"), edge.get("provenance"))}
        kept_spans(merged, other, edge)
        report.inverses_folded += 1
        note(f"inverse: {label_of(nodes, source)} {rel} {label_of(nodes, target)} -> "
             f"{label_of(nodes, target)} {keep_rel} {label_of(nodes, source)}")
        if not dry_run:
            store.remove_edge(source, rel, target)
            store.upsert_edge(merged)
        by_triple.pop(triple)
        by_triple[canonical_triple] = merged
    edges = list(by_triple.values())

    # 4. suspect labels
    for node in list(nodes.values()):
        why = suspect(str(node.get("label") or ""))
        if not why:
            continue
        held = suspect_verdict(decisions, node, rejudge=rejudge, report=report)
        if judge is not None or held is not None:
            resolve_suspect(store, nodes, edges, node, why, judge=judge, decisions=decisions,
                            report=report, note=note, merges=merges, decision=held,
                            dry_run=dry_run)
        elif not (node.get("attrs") or {}).get("suspect"):
            report.flagged += 1
            note(f"suspect: {node['label']!r} -- {why}")
            if not dry_run:
                store.set_attribute(node["id"], "suspect", why)

    # 5. verb conflicts, 6. orphans, 7. self-loops
    by_ends: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    touched: dict[str, int] = {}
    for edge in edges:
        if edge["rel"] in SOURCE_LINKS:
            continue
        ends = (min(edge["source"], edge["target"]), max(edge["source"], edge["target"]))
        by_ends.setdefault(ends, {})[edge["rel"]] = edge
        touched[edge["source"]] = touched.get(edge["source"], 0) + 1
        touched[edge["target"]] = touched.get(edge["target"], 0) + 1
        if edge["source"] == edge["target"]:
            report.self_loops.append((edge["source"], edge["rel"]))
            note(f"self-loop: {label_of(nodes, edge['source'])} {edge['rel']} itself")
    for (a, b), group in by_ends.items():
        rels = sorted(group)
        if len(rels) < 2:
            continue
        if a in nodes and b in nodes:
            rels = resolve_conflict(store, nodes, edges, (a, b), group, rels, judge=judge,
                                     decisions=decisions, report=report, note=note,
                                     rejudge=rejudge, dry_run=dry_run)
        if len(rels) > 1:
            report.conflicts.append((a, b, rels[0], rels[1]))
            note(f"conflict: {label_of(nodes, a)} and {label_of(nodes, b)} are joined by "
                 + " and ".join(rels))
    for node_id, node in nodes.items():
        if node_id not in touched and str(node.get("kind") or "") not in ("figure", "source"):
            report.orphans.append(node_id)
    if report.orphans:
        note(f"orphans: {len(report.orphans)} node(s) with no relation, e.g. "
             + ", ".join(label_of(nodes, n) for n in report.orphans[:5]))

    # 8. hierarchy cycles, reported and never broken
    for rel, ring in cycles(edges, relations=hierarchy):
        report.cycles.append((rel, ring))
        note(f"cycle: {rel} runs round "
             + " -> ".join(label_of(nodes, node_id) for node_id in ring)
             + f" -> {label_of(nodes, ring[0])}")
    if not dry_run:
        _recount(store)
    if not dry_run and hasattr(store, "check"):
        # 2026-09-03: a store engine blanked other nodes' strings on a delete, and the
        # pass reported success over a store that no longer read back by id. Never again:
        # the store's own check runs after the writes and the report carries what it found
        report.findings = list(store.check())
    note(report.said())
    return report


def _recheck(path: str | Path, report: Report, log: Callable[[str], None] | None) -> None:
    """The store's own check again, on a fresh handle, and the report's last line with it.

    A handle that has just dropped nodes and upserted thousands of edges reads its own edge
    count stale: over a store of two sources the writer's count said 25,612 against a scan of 25,513 and
    a reopen agreed with the scan. What a pass leaves is what the next reader finds, so
    that is what is asked.
    """
    was = list(report.findings)
    with GraphStore(path, read_only=True) as reader:
        report.findings = list(reader.check())
    if report.findings == was:
        return
    if report.lines:
        report.lines[-1] = report.said()
    if log is not None:
        log(report.said())



def _recount(store: Any) -> None:
    """The ``stats`` document's ``nodes`` and ``edges`` from what the store counts."""
    if not all(hasattr(store, name) for name in ("get_doc", "put_doc", "counts")):
        return
    stats = store.get_doc("stats", None)
    if not isinstance(stats, dict):
        return
    counted = store.counts()
    fresh = {**stats, "nodes": counted["nodes"], "edges": counted["edges"]}
    if fresh != stats:
        store.put_doc("stats", fresh)
