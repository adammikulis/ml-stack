"""The hygiene pass over a whole graph store: duplicates merged, inverses folded, the
doubtful flagged, the rest reported.

Two passes touch a knowledge graph, and they are kept apart on purpose. Adam: "hygiene
pass is separate from upsert/additive pass ... you're conflating two things." The *fold*
(``ml_stack.ingest``) is an upsert -- it adds nodes and edges and grows the ones it finds,
never merges, never removes, so a graph can be written into as reads land without ever
losing something good to something broken. This is the other pass, the dedupe: run when a
person asks, dry by default, and what it merges it merges with everything kept on the
survivor.

What it does, in order:

1. **Duplicate nodes**, within a kind and across every source. Two names are the same thing
   when they differ only by case, spacing, hyphens or underscores, or when one is the
   other's plural; the heavier name survives. A name `entities.close` calls one letter off
   another is *reported* as a possible duplicate and not merged: that rule is right for a
   community's typos and wrong for a textbook, where the dry run over a biology shelf
   would have folded "Natrium" into "atrium", "Isobutene" into "isobutane" and
   "Triacylglycerol" into "Diacylglycerol". A person settles those, and hands the
   decision back as ``written`` (``{name: the name it is}``, casefolded keys), which the
   pass applies whatever the weights. Figures, books and runs are never folded. A merge
   moves every edge to the survivor (an edge the survivor already had takes the sum of the
   weights and the union of the provenance), sums mentions, unions provenance, and keeps
   the merged name as an alias.
2. **Relation spellings** folded to the vocabulary the store uses most (`fold_edges`), so
   ``has_part`` and ``haspart`` are one relationship.
3. **Inverse pairs**: ``X part_of Y`` beside ``Y has_part X`` is one fact stored twice. The
   canonical direction (`INVERSES`) is kept, the other's weight and provenance fold into
   it, and the duplicate edge goes -- a fact that is still there the other way round.
4. **Suspect labels**, flagged not removed: a clause rather than a name, an over-generic
   word, a number, a single letter. ``attrs.suspect`` says why.
5. **Verb conflicts** reported: two edges between the same ends with verbs that are not
   each other's inverse (``X causes Y`` and ``X regulates Y``). Left alone.
6. **Orphans** reported: nodes with no edge but their source link. Left alone.
7. **Self-loops** reported. Left alone.

A hidden node (``attrs.hidden`` -- an ingest run, a unit) is never touched. Idempotent: a
second run reports nothing to do.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.entities.fold import ESTABLISHED, fold_edges

__all__ = ["INVERSES", "Report", "canonical_direction", "plurals", "same_name", "suspect", "tidy",
           "written_from"]

# The direction a fact is kept in, and the verbs that say the same thing the other way.
INVERSES: dict[str, frozenset[str]] = {
    "part_of": frozenset({"has_part", "contains", "has"}),
    "causes": frozenset({"caused_by"}),
    "precedes": frozenset({"follows", "after"}),
    "produces": frozenset({"produced_by", "made_by"}),
    "created_by": frozenset({"authored", "wrote", "created", "built", "proposed"}),
    "requires": frozenset({"required_by", "enables"}),
}
"""``{canonical verb: the verbs that state it with the ends swapped}``. ``X has_part Y`` is
``Y part_of X``; the pass keeps the left-hand form."""

# What is kept off the "no edge but its source" count: the links that say where a node came
# from rather than what it stands in relation to.
_SOURCE_LINKS = frozenset({"read_from", "read_by", "illustrates"})

_CLAUSE = re.compile(r"\b(that|which|who|aims? to|is an?|are|used to|in order to)\b", re.I)
_TRAILING_PREPOSITION = re.compile(r"\b(of|in|to|for|by|with|on|at|from|into)$", re.I)
_GENERIC = frozenset({"thing", "things", "process", "form of science", "form", "type",
                      "concept", "part", "item", "object", "structure", "substance"})


@dataclass
class Report:
    """What one pass did or would do; every step's count, and a line per decision."""

    dry_run: bool = True
    merged_nodes: int = 0
    possible: list[tuple[str, str]] = field(default_factory=list)   # close spellings, unmerged
    merged_edges: int = 0
    relations_folded: int = 0
    inverses_folded: int = 0
    flagged: int = 0
    refused: int = 0
    conflicts: list[tuple[str, str, str, str]] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    self_loops: list[tuple[str, str]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    @property
    def nothing_to_do(self) -> bool:
        return not (self.merged_nodes or self.relations_folded or self.inverses_folded
                    or self.flagged)

    def said(self) -> str:
        head = "would merge" if self.dry_run else "merged"
        return (f"{head} {self.merged_nodes} node(s) ({self.merged_edges} edge(s) moved), "
                f"folded {self.relations_folded} relation spelling(s) and "
                f"{self.inverses_folded} inverse pair(s), flagged {self.flagged} label(s); "
                f"{len(self.possible)} possible duplicate(s) by spelling left for a person, "
                f"{len(self.conflicts)} verb conflict(s), {len(self.orphans)} orphan(s), "
                f"{len(self.self_loops)} self-loop(s) reported")


def written_from(path: str | Path | None) -> dict[str, str]:
    """The duplicates a person settled, from a JSON file ``{name: the name it is}``; an
    empty map when no file is named. A file that is not that shape is an error said plainly."""
    import json

    if not path:
        return {}
    held = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(held, dict) or not all(isinstance(v, str) for v in held.values()):
        raise ValueError(f"{path}: expected a JSON object of name -> name")
    return {str(k): v for k, v in held.items()}


def plurals(names: Iterable[str]) -> dict[str, str]:
    """``{plural (casefolded): singular}`` for every name whose singular is also a name."""
    held = {str(name).casefold(): str(name) for name in names}
    out: dict[str, str] = {}
    for low, _name in held.items():
        for ending, singular in (("ies", "y"), ("es", ""), ("s", "")):
            if low.endswith(ending) and len(low) > len(ending) + 2:
                stem = low[: -len(ending)] + singular
                if stem in held and stem != low:
                    out[low] = held[stem]
                    break
    return out


def suspect(label: str) -> str:
    """Why a label is doubtful as a *name*, or ``""`` when it reads as one."""
    text = " ".join(str(label or "").split())
    if not text:
        return "empty"
    words = text.split()
    if len(words) > 6:
        return f"a clause of {len(words)} words, not a name"
    if _CLAUSE.search(text):
        return "reads as a clause"
    if _TRAILING_PREPOSITION.search(text):
        return "ends in a preposition"
    if text.casefold() in _GENERIC:
        return "too generic to be one thing"
    if re.fullmatch(r"[\d.,%-]+", text):
        return "a number"
    if len(text) == 1:
        return "a single letter"
    return ""


def canonical_direction(rel: str) -> tuple[str, bool]:
    """``(canonical verb, flipped)``: the verb a fact is kept under, and whether the ends
    must be swapped to get there."""
    if rel in INVERSES:
        return rel, False
    for keep, others in INVERSES.items():
        if rel in others:
            return keep, True
    return rel, False


_NEVER_FOLDED = frozenset({"figure", "book", "run", "unit"})


def same_name(label: str) -> str:
    """The form under which two labels are one name: casefolded, with spaces, hyphens and
    underscores collapsed -- "T-cell", "t cell" and "T_cell" are one; "atrium" and
    "Natrium" are not."""
    return re.sub(r"[\s_\-]+", "", str(label or "").casefold())


def tidy(store: Any, *, dry_run: bool = True, established: int = ESTABLISHED,
         written: Mapping[str, str] | None = None,
         log: Callable[[str], None] | None = None) -> Report:
    """The pass, over a `GraphStore` or the path to one. Dry by default. ``written`` is
    the map of duplicates a person settled (``{name: the name it is}``), applied as given."""
    from ml_stack.entities.spelling import close
    from ml_stack.graph.store import GraphStore

    if isinstance(store, (str, Path)):
        with GraphStore(store, read_only=dry_run) as opened:
            return tidy(opened, dry_run=dry_run, established=established, written=written,
                        log=log)
    report = Report(dry_run=dry_run)
    say = log or (lambda _line: None)

    def note(line: str) -> None:
        report.lines.append(line)
        say(line)

    nodes = {n["id"]: n for n in store.nodes() if not _hidden(n)}
    edges = [e for e in store.edges() if e["source"] in nodes and e["target"] in nodes]

    # 1. duplicate nodes, within a kind
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for node in nodes.values():
        by_kind.setdefault(str(node.get("kind") or ""), []).append(node)
    merged_into: dict[str, str] = {}
    decided = {str(k).casefold(): str(v) for k, v in (written or {}).items()}
    for kind, held in by_kind.items():
        if kind in _NEVER_FOLDED:
            continue
        # the same name under two ids -- case, spacing, hyphens -- is one node; the
        # heavier survives
        by_same: dict[str, dict[str, Any]] = {}
        for node in sorted(held, key=lambda n: -int(n.get("mentions") or 0)):
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
        # close spellings: a person's call, reported
        labels = [lbl for lbl in by_label if by_label[lbl]["id"] not in merged_into]
        for i, one in enumerate(labels):
            for other in labels[i + 1:]:
                if close(same_name(one), same_name(other)):
                    report.possible.append((one, other))
                    note(f"possible ({kind}): {one!r} ~ {other!r} -- a spelling apart; "
                         f"hand it back as written if they are one")
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
        moved = _merge(store, nodes, edges, keep, remove, dry_run=dry_run)
        report.merged_nodes += 1
        report.merged_edges += moved
        note(f"merge: {gone_label!r} -> {nodes[keep]['label']!r} "
             f"({gone_kind}, {moved} edge(s) moved)")
    if merged_into:
        edges = [e for e in edges if e["source"] in nodes and e["target"] in nodes]

    # 2. relation spellings
    keyed = {(e["source"], e["rel"], e["target"]): dict(e) for e in edges
             if e["rel"] not in _SOURCE_LINKS}
    folded, records = fold_edges(keyed, label="relations", provenance="provenance",
                                 settles="the spelling the store uses more is right")
    for record in records:
        report.relations_folded += 1
        note(f"relation: {record.get('from')} -> {record.get('into')}")
    if records and not dry_run:
        for key in keyed:
            if key not in folded:
                store.remove_edge(key[0], key[2], key[1])
        for key, edge in folded.items():
            store.upsert_edge({**edge, "source": key[0], "rel": key[1], "target": key[2]})
    edges = [e for e in edges if e["rel"] in _SOURCE_LINKS] + list(folded.values())

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
                  "provenance": _union((other or {}).get("provenance"), edge.get("provenance"))}
        report.inverses_folded += 1
        note(f"inverse: {_label(nodes, source)} {rel} {_label(nodes, target)} -> "
             f"{_label(nodes, target)} {keep_rel} {_label(nodes, source)}")
        if not dry_run:
            store.remove_edge(source, target, rel)
            store.upsert_edge(merged)
        by_triple.pop(triple)
        by_triple[canonical_triple] = merged
    edges = list(by_triple.values())

    # 4. suspect labels
    for node in nodes.values():
        attrs = node.get("attrs") or {}
        why = suspect(str(node.get("label") or ""))
        if why and not attrs.get("suspect"):
            report.flagged += 1
            note(f"suspect: {node['label']!r} -- {why}")
            if not dry_run:
                store.set_attribute(node["id"], "suspect", why)

    # 5. verb conflicts, 6. orphans, 7. self-loops -- reported only
    by_ends: dict[tuple[str, str], set[str]] = {}
    touched: dict[str, int] = {}
    for edge in edges:
        if edge["rel"] in _SOURCE_LINKS:
            continue
        ends = tuple(sorted((edge["source"], edge["target"])))
        by_ends.setdefault(ends, set()).add(edge["rel"])
        touched[edge["source"]] = touched.get(edge["source"], 0) + 1
        touched[edge["target"]] = touched.get(edge["target"], 0) + 1
        if edge["source"] == edge["target"]:
            report.self_loops.append((edge["source"], edge["rel"]))
            note(f"self-loop: {_label(nodes, edge['source'])} {edge['rel']} itself")
    for (a, b), rels in by_ends.items():
        if len(rels) > 1:
            names = sorted(rels)
            report.conflicts.append((a, b, names[0], names[1]))
            note(f"conflict: {_label(nodes, a)} and {_label(nodes, b)} are joined by "
                 + " and ".join(names))
    for node_id, node in nodes.items():
        if node_id not in touched and str(node.get("kind") or "") not in ("book", "figure"):
            report.orphans.append(node_id)
    if report.orphans:
        note(f"orphans: {len(report.orphans)} node(s) with no relation, e.g. "
             + ", ".join(_label(nodes, n) for n in report.orphans[:5]))
    note(report.said())
    return report


def _merge(store: Any, nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]],
           keep: str, remove: str, *, dry_run: bool) -> int:
    """``remove`` into ``keep`` with everything kept: edges moved (summed where the
    survivor had the same one), mentions summed, provenance unioned, the name an alias."""
    kept, gone = nodes[keep], nodes[remove]
    moved = 0
    by_triple = {(e["source"], e["rel"], e["target"]): e for e in edges}
    for edge in [e for e in edges if remove in (e["source"], e["target"])]:
        source = keep if edge["source"] == remove else edge["source"]
        target = keep if edge["target"] == remove else edge["target"]
        if source == target and edge["rel"] not in _SOURCE_LINKS:
            # a relation between the two names being joined says nothing once they are one
            by_triple.pop((edge["source"], edge["rel"], edge["target"]), None)
            if not dry_run:
                store.remove_edge(edge["source"], edge["target"], edge["rel"])
            continue
        other = by_triple.get((source, edge["rel"], target))
        merged = {"source": source, "rel": edge["rel"], "target": target,
                  "weight": int(edge.get("weight") or 0) + int((other or {}).get("weight") or 0),
                  "provenance": _union((other or {}).get("provenance"), edge.get("provenance"))}
        by_triple.pop((edge["source"], edge["rel"], edge["target"]), None)
        by_triple[(source, edge["rel"], target)] = merged
        moved += 1
        if not dry_run:
            store.remove_edge(edge["source"], edge["target"], edge["rel"])
            store.upsert_edge(merged)
    attrs = dict(kept.get("attrs") or {})
    aliases = list(attrs.get("aliases") or [])
    for alias in [str(gone.get("label") or ""), *((gone.get("attrs") or {}).get("aliases") or [])]:
        if alias and alias != kept.get("label") and alias not in aliases:
            aliases.append(alias)
    attrs["aliases"] = aliases
    for key, value in (gone.get("attrs") or {}).items():
        if key not in ("aliases", "hidden", "suspect") and value and not attrs.get(key):
            attrs[key] = value
    kept["attrs"] = attrs
    kept["mentions"] = int(kept.get("mentions") or 0) + int(gone.get("mentions") or 0)
    kept["provenance"] = _union(kept.get("provenance"), gone.get("provenance"))
    if not dry_run:
        store.upsert_node(kept)
        store.drop([remove])
    edges[:] = list(by_triple.values())
    nodes.pop(remove, None)
    return moved


def _union(*lists: Any) -> list[str]:
    out: list[str] = []
    for held in lists:
        for item in held or ():
            if item not in out:
                out.append(item)
    return out


def _hidden(node: Mapping[str, Any]) -> bool:
    return bool((node.get("attrs") or {}).get("hidden"))


def _label(nodes: Mapping[str, Mapping[str, Any]], node_id: str) -> str:
    return str((nodes.get(node_id) or {}).get("label") or node_id)
