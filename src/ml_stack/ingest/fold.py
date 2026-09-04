"""Extractions into a graph: one section built into nodes and edges, a source folded
out of its sections, and the fold reconciled into the store."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.ingest.extract import VERBS
from ml_stack.ingest.progress import Progress
from ml_stack.ingest.reads import _slug, unit_of, units_of

__all__ = ["CORE", "build", "fold", "fold_into", "fold_source", "plurals", "write"]


CORE = frozenset(VERBS) | {"illustrates", "read_from"}
"""The verbs this library sets itself. An edge with any other verb carries ``extension``."""


def build(extraction: Mapping[str, Any], unit: Any, *, book_title: str = ""
          ) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    """One extraction as ``(nodes by id, edges by triple)``, every one pointing at where it came from.

    Names are the ids: two sections that both name the same concept are one node whose
    provenance lists both, which is the whole reason to read a source section by section
    rather than a page at a time. Provenance is pointers and nothing else -- unit ids --
    because Adam: "provenance should always be pointers to the textbook". The unit
    document holds the source, chapter, section and pages, and points at the run that read
    it; `located()` and `origin()` follow the pointers back.
    """
    where = unit.where
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def put(name: str, kind: str, **more: Any) -> str:
        clean = " ".join(str(name or "").split())
        if not clean:
            return ""
        node_id = f"concept:{_slug(clean)}"
        held = nodes.setdefault(node_id, {
            "id": node_id, "kind": kind or "concept", "label": clean, "mentions": 0,
            "attrs": {"definition": "", "aliases": [], "key_term": False},
            "provenance": []})
        held["mentions"] += 1
        if where["unit"] not in held["provenance"]:
            held["provenance"].append(where["unit"])
        attrs = held["attrs"]
        if more.get("definition") and not attrs["definition"]:
            attrs["definition"] = str(more["definition"])[:400]
            # the unit it was defined in, by pointer: `located()` turns it into a page
            attrs["defined_in"] = where["unit"]
        for alias in more.get("aliases") or ():
            clean_alias = " ".join(str(alias).split())
            if clean_alias and clean_alias.casefold() != clean.casefold() \
                    and clean_alias not in attrs["aliases"]:
                attrs["aliases"].append(clean_alias)
        if more.get("key_term"):
            attrs["key_term"] = True
        return node_id

    for concept in extraction.get("concepts") or ():
        if isinstance(concept, Mapping):
            put(concept.get("name", ""), str(concept.get("kind") or "concept"),
                definition=concept.get("definition"), aliases=concept.get("aliases"))

    for term in extraction.get("key_terms") or ():
        if isinstance(term, Mapping):
            put(term.get("term", ""), "concept", definition=term.get("definition"),
                key_term=True)

    for relation in extraction.get("relations") or ():
        if not isinstance(relation, Mapping):
            continue
        source = put(relation.get("from", ""), "concept")
        target = put(relation.get("to", ""), "concept")
        rel = _slug(relation.get("rel", "")).replace("-", "_")
        if not (source and target and rel) or source == target:
            continue
        key = (source, rel, target)
        held = edges.setdefault(key, {"source": source, "rel": rel, "target": target,
                                      "weight": 0, "provenance": []})
        if rel not in CORE:
            held["extension"] = True
        held["weight"] += 1
        if where["unit"] not in held["provenance"]:
            held["provenance"].append(where["unit"])


    for order, figure in enumerate(extraction.get("figures") or (), start=1):
        if not isinstance(figure, Mapping):
            continue
        caption = " ".join(str(figure.get("caption") or "").split())
        label = " ".join(str(figure.get("label") or "").split())
        if not (caption or label):
            continue
        node_id = f"figure:{where['unit']}:{order}"
        nodes[node_id] = {"id": node_id, "kind": "figure", "label": label or caption[:80],
                          "mentions": 1,
                          "attrs": {"caption": caption, "shows": str(figure.get("shows") or "")},
                          "provenance": [where["unit"]]}
        for name in figure.get("concepts") or ():
            target = put(name, "concept")
            if target:
                key = (node_id, "illustrates", target)
                edges.setdefault(key, {"source": node_id, "rel": "illustrates",
                                       "target": target, "weight": 0, "provenance": []})
                edges[key]["weight"] += 1
                if where["unit"] not in edges[key]["provenance"]:
                    edges[key]["provenance"].append(where["unit"])
    return nodes, edges


def fold_source(reads: Iterable[Mapping[str, Any]], units_by_id: Mapping[str, Any], *,
                book_title: str = "", log: Callable[[str], None] | None = None
                ) -> dict[str, Any]:
    """Every section of one source, folded into one graph.

    A read that carries an `error` contributes nothing: what a failed extraction left is
    kept for reading, not for believing.

    Two folds, and they are not the same fold. The *relations* are folded by
    `entities.fold_edges`, which is what stops ``has_part`` and ``haspart`` being two
    relationships. The *names* are folded by `entities.fold_names` over how often each was
    said, which is what stops "mitochondrion" and "mitochondria" being two concepts -- and
    which refuses to fold two names a source keeps using, because at that point they are
    two things the source distinguishes and merging them would be a decision nobody made.
    """
    from ml_stack.entities.fold import fold_edges, fold_names

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for read in reads:
        unit = units_by_id.get(str(read.get("unit") or ""))
        if unit is None or read.get("error") or not read.get("extracted"):
            continue
        got_nodes, got_edges = build(read["extracted"], unit, book_title=book_title)
        for node_id, node in got_nodes.items():
            held = nodes.get(node_id)
            if held is None:
                nodes[node_id] = node
                continue
            held["mentions"] += node["mentions"]
            held["provenance"] = list(dict.fromkeys(held["provenance"] + node["provenance"]))
            for key, value in node["attrs"].items():
                if key == "aliases":
                    held["attrs"]["aliases"] = list(dict.fromkeys(
                        held["attrs"].get("aliases", []) + value))
                elif value and not held["attrs"].get(key):
                    held["attrs"][key] = value
        for key, edge in got_edges.items():
            held = edges.get(key)
            if held is None:
                edges[key] = edge
                continue
            held["weight"] += edge["weight"]
            held["provenance"] = list(dict.fromkeys(held["provenance"] + edge["provenance"]))


    edges, relation_folds = fold_edges(
        edges, log=log, label="relations", provenance="provenance",
        settles="the schema's vocabulary settles which is right")
    for edge in edges.values():
        if edge["rel"] in CORE:
            edge.pop("extension", None)
        else:
            edge["extension"] = True

    weight = {node["label"]: int(node["mentions"]) for node in nodes.values()
              if node["kind"] != "figure"}
    canonical, name_folds = fold_names(weight, plurals(weight), log=log, label="concepts",
                                       settles="both spellings stay, and the source is right")
    moved = {f"concept:{_slug(name)}": f"concept:{_slug(into)}"
             for name, into in canonical.items() if into != name}
    if moved:
        nodes, edges = _apply(nodes, edges, moved)

    return {"nodes": sorted(nodes.values(), key=lambda n: n["id"]),
            "edges": sorted(edges.values(), key=lambda e: (e["source"], e["rel"], e["target"])),
            "folds": {"relations": relation_folds, "concepts": name_folds}}


def plurals(names: Iterable[str]) -> dict[str, str]:
    """``{plural (casefolded): singular}`` for every name whose singular is also a name.

    Chapter 2 of a biology book came back with `acid` and `acids`, `hydrogen ion` and
    `hydrogen ions`, each with its own edges, because `entities.close` -- rightly -- does
    not call a plural one letter off. A book does not distinguish a thing from two of it,
    so the plural folds into the singular however often each was said: handed to
    `fold_names` as the map somebody decided, which is what it is.
    """
    held = {str(name).casefold(): str(name) for name in names}
    out: dict[str, str] = {}
    for low, name in held.items():
        for ending, singular in (("ies", "y"), ("es", ""), ("s", "")):
            if low.endswith(ending) and len(low) > len(ending) + 2:
                stem = low[: -len(ending)] + singular
                if stem in held and stem != low:
                    out[low] = held[stem]
                    break
    return out


def _apply(nodes: Mapping[str, dict[str, Any]],
           edges: Mapping[tuple[str, str, str], dict[str, Any]],
           moved: Mapping[str, str]) -> tuple[dict[str, dict[str, Any]],
                                              dict[tuple[str, str, str], dict[str, Any]]]:
    """Rewrite a fold's decisions through the nodes and the edges that name them."""
    out_nodes: dict[str, dict[str, Any]] = {}
    for node_id, node in nodes.items():
        into = moved.get(node_id, node_id)
        if into == node_id:
            out_nodes.setdefault(node_id, dict(node))
            continue
        kept = out_nodes.get(into)
        if kept is None:
            out_nodes[into] = kept = dict(nodes.get(into) or {**node, "id": into})
        kept["mentions"] = int(kept.get("mentions") or 0) + int(node["mentions"])
        kept["provenance"] = list(dict.fromkeys(list(kept.get("provenance") or [])
                                                + node["provenance"]))
        aliases = kept.setdefault("attrs", {}).setdefault("aliases", [])
        for alias in [node["label"], *node["attrs"].get("aliases", [])]:
            if alias and alias != kept.get("label") and alias not in aliases:
                aliases.append(alias)
        if not kept["attrs"].get("definition") and node["attrs"].get("definition"):
            kept["attrs"]["definition"] = node["attrs"]["definition"]
    for node_id, node in nodes.items():
        into = moved.get(node_id, node_id)
        if into != node_id and into not in out_nodes:  # pragma: no cover - defensive
            out_nodes[into] = dict(node, id=into)


    out_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (source, rel, target), edge in edges.items():
        key = (moved.get(source, source), rel, moved.get(target, target))
        if key[0] == key[2]:
            continue
        held = out_edges.get(key)
        if held is None:
            out_edges[key] = dict(edge, source=key[0], target=key[2])
            continue
        held["weight"] += edge["weight"]
        held["provenance"] = list(dict.fromkeys(held["provenance"] + edge["provenance"]))
    return out_nodes, out_edges


def write(out: str | Path, graph: Mapping[str, Any], *, source: str, title: str,
          docs: Mapping[str, Any] | None = None, replace: bool = False,
          keep_units: Iterable[str] | None = None,
          shares: Mapping[str, int] | None = None) -> dict[str, int]:
    """One source's graph and its raw extractions into the store, read back before returning.

    The source itself is a node, so a store holding many can still be asked what came out
    of which; every concept hangs off it by ``read_from``.

    An upsert, and nothing more. Adam: "if the book already exists, it should append new
    nodes/connect new edges. additive." A node the store lacks is added; one it has keeps
    what other sources gave it and takes this source's part afresh: ``shares`` is this
    source's mentions per node (the fold's own count, before `absorb` rewrote its ids),
    kept in the ``ingest:shares:<source>`` document so the next fold can replace it;
    provenance is the other sources' units and this fold's; aliases union; a definition the
    fold lacks stays. An edge takes the fold's weight and provenance; nothing is merged and
    nothing is removed -- joining names is the hygiene pass's job (`ml_stack.graph.tidy`).
    Folding twice with nothing read in between changes nothing.

    ``replace`` is `fold --rebuild`: the source's own nodes and edges out first, then the
    full fold from its reads -- the one path that removes anything, for after a fix that
    changed what a read means. ``keep_units`` names the units the source now has: any
    ``ingest:unit:`` document of this source outside it goes with the nodes.
    """
    from ml_stack.graph.store import GraphStore

    source_id = f"source:{source}"
    nodes = [{"id": source_id, "kind": "source", "label": title or source, "mentions": 1,
              "attrs": {"source": source}}, *graph.get("nodes", ())]
    edges = [*graph.get("edges", ()),
             *({"source": node["id"], "rel": "read_from", "target": source_id, "weight": 1}
               for node in graph.get("nodes", ()))]
    if shares is None:
        shares = {str(n["id"]): int(n.get("mentions") or 0) for n in graph.get("nodes", ())}
    with GraphStore(out) as store:
        if replace:
            _drop_source(store, source, keep_units=keep_units)
        previous = store.get_doc(f"ingest:shares:{source}")
        if isinstance(previous, Mapping):
            previous = dict(previous)
        else:
            # a source folded before the shares were kept, or one never folded at all
            previous = None if store.get_doc(f"ingest:folds:{source}") is not None else {}
        held = {str(n["id"]): n for n in store.nodes()}
        nodes = [_joined(node, held.get(str(node["id"])), source, shares, previous)
                 for node in nodes]
        counts = store.write({"nodes": nodes, "edges": edges})
        for key, value in (docs or {}).items():
            store.put_doc(key, value)
        store.put_doc(f"ingest:folds:{source}", dict(graph.get("folds") or {}))
        store.put_doc(f"ingest:shares:{source}", dict(shares))
        back = store.query("MATCH (n:Node {id: $id}) RETURN n.id AS id", {"id": source_id})
    if not back:
        raise RuntimeError(f"{source_id} was written to {out} and did not come back")
    return counts


def _joined(node: Mapping[str, Any], existing: Mapping[str, Any] | None, source: str,
            shares: Mapping[str, int], previous: Mapping[str, int] | None) -> dict[str, Any]:
    """``node`` as the store will hold it: what other sources gave ``existing`` kept, and
    this source's part -- mentions, units, aliases, definition -- taken from the fold.

    ``previous`` is this source's share the last time it was written; None when the source
    was folded before the shares were kept, and then its share is what the store counts."""
    node_id = str(node["id"])
    if existing is None or node.get("kind") == "source":
        return dict(node)
    share = int(shares.get(node_id, node.get("mentions") or 0))
    before = int(existing.get("mentions") or 0)
    was = int(previous.get(node_id, 0)) if previous is not None else min(before, share)
    prefix = f"{source}:"
    other = [u for u in existing.get("provenance") or () if not str(u).startswith(prefix)]
    mine = [u for u in node.get("provenance") or () if str(u).startswith(prefix)]
    attrs = dict(existing.get("attrs") or {})
    for key, value in (node.get("attrs") or {}).items():
        if key != "aliases" and (value or key not in attrs):
            attrs[key] = value
    aliases = list((existing.get("attrs") or {}).get("aliases") or [])
    for alias in (node.get("attrs") or {}).get("aliases") or ():
        if alias and alias != node.get("label") and alias not in aliases:
            aliases.append(alias)
    attrs["aliases"] = aliases
    return {**node, "mentions": max(before - was, 0) + share, "attrs": attrs,
            "provenance": list(dict.fromkeys([*other, *mine]))}


def _drop_source(store: Any, source: str, *, keep_units: Iterable[str] | None = None) -> int:
    """Everything the store holds for one source, out: nodes read only from it, and its edges.

    A node is this source's when a ``read_from`` edge joins it to ``source:<slug>``; one
    that also reads from another source stays, and only the edges this source put on it go.
    The source node itself stays and is written again.
    """
    source_id = f"source:{source}"
    if keep_units is not None:
        # every unit id starts with the source's slug, so the stale documents are a prefix
        # away and no document has to be read to find them
        held, prefix = {f"ingest:unit:{u}" for u in keep_units}, f"ingest:unit:{source}:"
        for key in store.doc_keys():
            if key.startswith(prefix) and key not in held:
                store.delete_doc(key)
    read_from = store.edges("read_from")
    mine = {e["source"] for e in read_from if e["target"] == source_id}
    if not mine:
        return 0
    shared = {e["source"] for e in read_from if e["target"] != source_id}
    gone = store.drop(sorted(mine - shared), force=True)
    prefix = f"{source}:"
    for edge in store.edges():
        if edge["rel"] == "read_from" and edge["target"] == source_id:
            store.remove_edge(edge["source"], edge["rel"], edge["target"])
        elif any(str(u).startswith(prefix) for u in (edge.get("provenance") or ())):
            # this source's edge, by its pointers; an edge two sources both stated keeps
            # the other source's pointer and goes when that source is rebuilt
            store.remove_edge(edge["source"], edge["rel"], edge["target"])
    return gone


def _unit_docs(rows: Iterable[Mapping[str, Any]], slug: str) -> dict[str, Any]:
    """The ``ingest:unit:`` document for each read: its provenance, what it said, what it cost."""
    out: dict[str, Any] = {}
    for row in rows:
        unit = unit_of(row)
        if not unit.id:
            continue
        out[f"ingest:unit:{unit.id}"] = {
            "unit": unit.id, "source": slug, "where": unit.where,
            "title": unit.section_title,
            "chapter_title": str(row.get("chapter_title") or ""),
            "run": str(row.get("run") or ""),
            "extracted": row.get("extracted") or {}, "calls": list(row.get("calls") or ()),
            "seconds": float(row.get("seconds") or 0.0), "error": str(row.get("error") or "")}
    return out


def fold_into(out: str | Path, slug: str, *, title: str = "",
              reads: Sequence[Mapping[str, Any]] | None = None,
              units_by_id: Mapping[str, Any] | None = None,
              progress: Progress | None = None, rebuild: bool = False,
              dry_run: bool = False, judge: Any = None,
              log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Fold one source's reads so far, reconcile it against the store, and upsert it.

    Before the upsert the fold goes through `graph.tidy.absorb`: a name the store already
    holds under case, spacing or plural lands on that node, and a close spelling goes to
    ``judge`` when there is one (the run's own model, with the unit text in hand). Adam:
    "the same dedupe mechanism should be used whenever the model is reading a new thing
    or learning something new and saving to an existing graph." ``absorbed`` on the
    result says how many names landed on existing nodes and how the judge ruled.

    Returns the counts: units read, nodes, edges, folds, ``seconds`` -- what the fold and
    the write took -- whether the source is partial, and ``new_nodes``/``new_edges``, what
    the store lacked before this fold. Idempotent:
    folding twice with nothing read in between adds nothing. ``dry_run`` computes all of
    that and writes nothing; ``rebuild`` drops the source's own nodes and edges first and is
    the only way anything leaves the store.
    """
    from ml_stack.ingest.sources import Sources

    view = Sources(out)
    rows = list(reads) if reads is not None else view.reads(slug)
    held = view.progress.state["sources"].get(slug) or {}
    name = title or str(held.get("title") or "") or slug
    units = {**units_of(rows), **dict(units_by_id or {})}
    began = time.time()
    graph = fold_source(rows, units, book_title=name, log=log)
    new_nodes, new_edges = _missing_from(out, graph)
    folds = len(graph["folds"].get("concepts") or ()) + len(graph["folds"].get("relations") or ())
    wanted = int(held.get("sections") or 0)
    got = {"source": slug, "title": name, "units": len(rows),
           "read": sum(1 for r in rows if not r.get("error")),
           "wanted": wanted, "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
           "new_nodes": new_nodes, "new_edges": new_edges,
           "folds": folds, "partial": wanted > len(rows),
           "seconds": round(time.time() - began, 2)}
    if dry_run:
        return got
    shares = {str(n["id"]): int(n.get("mentions") or 0) for n in graph["nodes"]}
    if not rebuild and Path(out).expanduser().exists():
        from ml_stack.graph.store import GraphStore
        from ml_stack.graph.tidy import absorb

        with GraphStore(out) as store:
            taken = absorb(store, graph, judge=judge, sources=_texts_of(units), log=log)
        graph = taken.graph
        for gone, kept in taken.mapping.items():
            shares[kept] = shares.get(kept, 0) + shares.pop(gone, 0)
        got["absorbed"] = {"same_name": taken.mapped_same_name, "plural": taken.mapped_plural,
                           "judged_same": taken.judged_same,
                           "judged_different": taken.judged_different,
                           "possible": taken.left_possible}
    docs = _unit_docs(rows, slug)
    counts = write(out, graph, source=slug, title=name, docs=docs, replace=rebuild,
                   keep_units=set(units) if rebuild else None, shares=shares)
    got["seconds"] = round(time.time() - began, 2)
    record = progress if progress is not None else view.progress
    record.source(slug, title=name)
    record.folded(slug, units=len(rows), nodes=counts["nodes"], edges=counts["edges"],
                  seconds=got["seconds"])
    return got


def _texts_of(units: Mapping[str, Any]) -> Callable[[str], str] | None:
    """The unit text the judge may read, when the units in hand carry it (a run's do; a
    fold from the reads file alone does not, and then the judge reads the document again
    through `sources_for`)."""
    held = {uid: str(getattr(u, "text", "") or "") for uid, u in units.items()}
    if not any(held.values()):
        return None
    return lambda unit_id: held.get(unit_id, "")


def _missing_from(out: str | Path, graph: Mapping[str, Any]) -> tuple[int, int]:
    """How many of a fold's nodes and edges the store does not hold yet."""
    from ml_stack.graph.store import GraphStore

    if not Path(out).expanduser().exists():
        return len(graph.get("nodes") or ()), len(graph.get("edges") or ())
    with GraphStore(out, read_only=True) as store:
        have_nodes = {n["id"] for n in store.nodes()}
        have_edges = {(e["source"], e["rel"], e["target"]) for e in store.edges()}
    nodes = sum(1 for n in graph.get("nodes") or () if n["id"] not in have_nodes)
    edges = sum(1 for e in graph.get("edges") or ()
                if (e["source"], e["rel"], e["target"]) not in have_edges)
    return nodes, edges


def fold(out: str | Path, *, source: str = "", rebuild: bool = False, dry_run: bool = False,
         say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest fold --out STORE``: every source that has reads, upserted into the store.

    A source part-read is written as far as it has been read, so a run that will take days
    is answerable today. ``--dry-run`` says what each fold would add and writes nothing;
    ``--rebuild`` drops each source's own nodes and edges first.

    Every source folded, the hygiene pass runs over the store with no model: the verdicts
    it already holds are applied again, and the duplicate half of an inverse pair goes.
    What the fold rebuilt from the reads that no verdict covers is left as it is, and the
    line says how many verdicts were replayed, how many name a node the store no longer
    has, and what is left unjudged.
    """
    from ml_stack.ingest.sources import Sources

    view = Sources(out)
    wanted = [s for s in view.sources() if s.units and (not source or s.slug == source)]
    if not wanted:
        say(f"nothing to fold into {out}"
            + (f": no reads for {source}" if source
               else f": no {Path(out).name}.*.reads.json"))
        return 1
    for held in wanted:
        got = fold_into(out, held.slug, title=held.title, progress=view.progress,
                        rebuild=rebuild, dry_run=dry_run)
        what = ("would add" if dry_run else "rebuilt with" if rebuild else "added")
        say(f"{got['title']}: {got['read']} of {got['wanted'] or '?'} units read, "
            f"{got['nodes']} nodes, {got['edges']} edges, {got['folds']} fold(s); "
            f"{what} {got['new_nodes']} node(s) and {got['new_edges']} edge(s)"
            + (f" into {out}" if not dry_run else "")
            + ("  -- partial" if got["partial"] else ""))
    if not dry_run:
        from ml_stack.graph.store import GraphStore
        from ml_stack.graph.tidy import tidy

        say("kept: " + tidy(out, dry_run=False).said())
        with GraphStore(out, read_only=True) as store:
            findings = store.check()
        if findings:
            say(f"NOT SOUND: {out} does not read back whole after the fold -- "
                f"{len(findings)} finding(s), e.g. {findings[0]!r}; rebuild it from the reads "
                f"into a fresh store and report the store engine version")
            return 1
        say(f"{out}: reads back whole")
    return 0
