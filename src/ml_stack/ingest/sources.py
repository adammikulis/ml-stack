"""A store's reads as they land: every source, its graph so far, what the sources share,
and the two commands that print it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.ingest.fold import fold_source
from ml_stack.ingest.progress import GIVE_UP, Progress
from ml_stack.ingest.reads import _read_json, reads_path, tokens_of, units_of

__all__ = ["Source", "Sources", "show", "sources"]


@dataclass
class Source:
    """One source in a store: how much of it is read, and how much of that is in the store."""

    slug: str
    title: str = ""
    path: str = ""
    units: int = 0           # rows in the reads file: every unit attempted and kept
    read: int = 0            # rows an extraction came back for -- what folds
    failed: int = 0
    given_up: int = 0
    wanted: int = 0          # units the source has
    seconds: float = 0.0
    prompt_tokens: int = 0   # what the model was shown, over every call of every unit
    completion_tokens: int = 0
    folded_at: int = 0       # units read when the store was last written
    folded_nodes: int = 0
    folded_edges: int = 0
    folded_seconds: float = 0.0

    @property
    def partial(self) -> bool:
        """Units are still to be read."""
        return self.wanted > self.units

    @property
    def per_unit(self) -> float:
        return round(self.seconds / self.units, 1) if self.units else 0.0

    @property
    def tokens(self) -> int:
        """Read and written together."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_unit(self) -> int:
        return round(self.tokens / self.units) if self.units else 0

    @property
    def left(self) -> float:
        """Seconds of reading still to do, at this source's own measured rate."""
        return max(self.wanted - self.units, 0) * self.per_unit


class Sources:
    """A store's reads as they land: every source, its graph so far, and the store itself.

    Nothing here needs the run to have finished, or the documents to still be where they
    were read from: `sources` and `reads` come from the files beside the store, `graph`
    folds those in memory, and `store` opens the store read-only beside the writer.
    """

    def __init__(self, out: str | Path) -> None:
        self.out = Path(out).expanduser()
        self.progress = Progress(Progress.beside(self.out))

    def sources(self) -> list[Source]:
        """Every source with reads or progress, by slug."""
        out = []
        for slug in sorted(set(self.progress.state["sources"]) | set(self._slugs())):
            held = self.progress.state["sources"].get(slug) or {}
            rows = self.reads(slug)
            prompt, completion = tokens_of(rows)
            done = (held.get("done") or {}).values()
            out.append(Source(
                slug=slug, title=str(held.get("title") or ""), path=str(held.get("path") or ""),
                units=len(rows), read=sum(1 for r in rows if not r.get("error")),
                failed=sum(1 for r in rows if r.get("error")),
                given_up=sum(1 for e in done if e.get("error")
                             and int(e.get("attempts") or 1) >= GIVE_UP),
                wanted=int(held.get("sections") or 0),
                seconds=round(sum(float(r.get("seconds") or 0.0) for r in rows), 1),
                prompt_tokens=prompt, completion_tokens=completion,
                folded_at=int(held.get("folded_at") or 0),
                folded_nodes=int(held.get("folded_nodes") or 0),
                folded_edges=int(held.get("folded_edges") or 0),
                folded_seconds=float(held.get("folded_seconds") or 0.0)))
        return out

    def source(self, slug: str) -> Source | None:
        """One source by slug, or None."""
        return next((s for s in self.sources() if s.slug == slug), None)

    def _slugs(self) -> list[str]:
        head, tail = self.out.name + ".", ".reads.json"
        if not self.out.parent.is_dir():
            return []
        return [p.name[len(head):-len(tail)] for p in self.out.parent.glob(f"{head}*{tail}")
                if len(p.name) > len(head) + len(tail)]


    def reads(self, slug: str) -> list[dict[str, Any]]:
        """One source's extractions so far, in the order they were read."""
        held = _read_json(reads_path(self.out, slug))
        if not isinstance(held, dict):
            return []
        return [dict(row, unit=str(row.get("unit") or key))
                for key, row in held.items() if isinstance(row, Mapping)]


    def graph(self, slug: str, *, log: Callable[[str], None] | None = None) -> dict[str, Any]:
        """The source folded from its reads so far -- nodes, edges and the folds it made.

        No store and no document: the provenance each unit needs is on the row it wrote.
        """
        rows = self.reads(slug)
        held = self.progress.state["sources"].get(slug) or {}
        return fold_source(rows, units_of(rows), book_title=str(held.get("title") or ""),
                           log=log)

    def store(self, **kw: Any) -> Any:
        """A read-only `GraphStore` on these sources, openable while the run is writing."""
        from ml_stack.graph.store import GraphStore

        return GraphStore(self.out, read_only=True, **kw)

    def shared(self, store: Any = None) -> dict[str, Any]:
        """What the sources in this store hold, and what they hold in common.

        ``sources`` is one entry per source -- units read, and the nodes and edges the
        store holds for it. ``shared`` is every concept two or more sources were read
        into, most shared first, each naming them. ``between`` is every edge whose two
        ends are known to different sets of sources -- one source's vocabulary reaching
        into another's. An edge the same sources hold both ends of is inside a vocabulary
        they share, not between them. ``merged`` is `between_sources`, and ``logged``
        whether the store holds the merges document those come from at all. ``decisions``
        counts the pairs a judge has settled in the store's `graph.tidy` document.

        A source's node is one a ``read_from`` edge joins to ``source:<slug>``; a source's
        edge is one whose provenance names a unit of that source. Without a ``store`` one
        is opened read-only.
        """
        if store is None:
            with self.store() as held:
                return self.shared(held)
        nodes = list(store.nodes())
        edges = list(store.edges())
        labels = {str(n["id"]): str(n.get("label") or "") for n in nodes}
        mentions = {str(n["id"]): int(n.get("mentions") or 0) for n in nodes}
        sources_of: dict[str, set[str]] = {}
        for edge in edges:
            target = str(edge.get("target") or "")
            if edge.get("rel") == "read_from" and target.startswith("source:"):
                sources_of.setdefault(str(edge.get("source") or ""),
                                      set()).add(target[len("source:"):])
        known = {s.slug: s for s in self.sources()}
        slugs = sorted(set(known) | {str(n["id"])[len("source:"):] for n in nodes
                                     if str(n.get("id") or "").startswith("source:")})
        node_counts = {slug: 0 for slug in slugs}
        for held_sources in sources_of.values():
            for slug in held_sources:
                node_counts[slug] = node_counts.get(slug, 0) + 1
        edge_counts = {slug: 0 for slug in slugs}
        for edge in edges:
            if edge.get("rel") == "read_from":
                continue
            for slug in {str(u).split(":", 1)[0] for u in (edge.get("provenance") or ())}:
                if slug in edge_counts:
                    edge_counts[slug] += 1
        listed = []
        for slug in slugs:
            held = known.get(slug)
            listed.append({"source": slug,
                           "title": (held.title if held else "")
                                    or labels.get(f"source:{slug}", "") or slug,
                           "units": held.units if held else 0,
                           "read": held.read if held else 0,
                           "wanted": held.wanted if held else 0,
                           "nodes": node_counts.get(slug, 0),
                           "edges": edge_counts.get(slug, 0)})
        shared = [{"id": node_id, "label": labels.get(node_id, node_id),
                   "mentions": mentions.get(node_id, 0), "sources": sorted(held_sources)}
                  for node_id, held_sources in sources_of.items() if len(held_sources) > 1]
        shared.sort(key=lambda r: (-len(r["sources"]), -r["mentions"], r["label"]))
        between = []
        for edge in edges:
            if edge.get("rel") == "read_from":
                continue
            source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
            here, there = sources_of.get(source) or set(), sources_of.get(target) or set()
            if not (here and there) or here == there:
                continue
            between.append({"source": source, "source_label": labels.get(source, source),
                            "rel": str(edge.get("rel") or ""), "target": target,
                            "target_label": labels.get(target, target),
                            "weight": int(edge.get("weight") or 0),
                            "sources": [sorted(here), sorted(there)]})
        between.sort(key=lambda r: (-r["weight"], r["source_label"], r["target_label"]))
        return {"sources": listed, "shared": shared, "between": between,
                "merged": self.between_sources(store), "logged": _logged(store),
                "decisions": _decisions_in(store)}

    def between_sources(self, store: Any = None) -> list[dict[str, Any]]:
        """Every name the hygiene pass joined across two sources, heaviest first.

        One entry per merge in the store's `graph.tidy` merges document whose two names
        were read from different sources: ``a_label`` and ``a_source`` for the name kept,
        ``b_label`` and ``b_source`` for the name folded into it, ``kind``, and ``weight``
        -- the joined node's mentions (the units both were read from, when the store holds
        no count) plus the edges the merge moved.
        """
        if store is None:
            with self.store() as held:
                return self.between_sources(held)
        from ml_stack.graph.tidy import MERGES

        held = store.get_doc(MERGES) if hasattr(store, "get_doc") else None
        held = held.get("merges") if isinstance(held, Mapping) else held
        mentions = {str(n["id"]): int(n.get("mentions") or 0) for n in store.nodes()}
        out = []
        for one in held if isinstance(held, list) else ():
            if not isinstance(one, Mapping):
                continue
            kept_from = list(one.get("kept_from") or ())
            gone_from = list(one.get("gone_from") or ())
            here = {str(u).split(":", 1)[0] for u in kept_from}
            there = {str(u).split(":", 1)[0] for u in gone_from}
            if not (here and there) or here & there:
                continue
            kept = str(one.get("kept") or "")
            out.append({"a_label": str(one.get("kept_label") or kept),
                        "a_source": ", ".join(sorted(here)),
                        "b_label": str(one.get("gone_label") or one.get("gone") or ""),
                        "b_source": ", ".join(sorted(there)),
                        "kind": str(one.get("kind") or ""),
                        "weight": (mentions.get(kept) or len(kept_from) + len(gone_from))
                                  + int(one.get("edges_moved") or 0)})
        out.sort(key=lambda r: (-r["weight"], r["a_label"], r["b_label"]))
        return out


def _logged(store: Any) -> bool:
    """Whether the store holds a merges document, however many merges are in it."""
    from ml_stack.graph.tidy import MERGES

    return (store.get_doc(MERGES) if hasattr(store, "get_doc") else None) is not None


def _decisions_in(store: Any) -> dict[str, int]:
    """How many name pairs a judge has settled in the store, and how each went."""
    from ml_stack.graph.tidy import DECISIONS

    held = store.get_doc(DECISIONS) if hasattr(store, "get_doc") else None
    pairs = (held or {}).get("pairs") if isinstance(held, Mapping) else None
    pairs = pairs if isinstance(pairs, Mapping) else {}
    out = {"pairs": len(pairs), "same": 0, "different": 0, "unsure": 0}
    for one in pairs.values():
        verdict = str((one or {}).get("verdict") or "unsure") if isinstance(one, Mapping) \
            else "unsure"
        out[verdict if verdict in out else "unsure"] += 1
    return out


def show(out: str | Path, *, source: str = "", most: int = 5,
         say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest show --out STORE``: what each source was read as, in plain text.

    A sample of concepts with their kind and definition, a sample of relations with the
    verb and the page they were read on, the folds the source made, and how many figures.
    """
    view = Sources(out)
    listed = [s for s in view.sources() if s.units and (not source or s.slug == source)]
    if not listed:
        say(f"nothing read into {out}" + (f" for {source}" if source else ""))
        return 1
    for held in listed:
        graph = view.graph(held.slug)
        concepts = [n for n in graph["nodes"] if n["kind"] != "figure"]
        figures = len(graph["nodes"]) - len(concepts)
        say(f"\n{held.title or held.slug} ({held.slug}): {held.read} of "
            f"{held.wanted or '?'} units read" + (", partial" if held.partial else "")
            + (f", {held.failed} failed" if held.failed else ""))
        say(f"  {len(concepts)} concept(s), {len(graph['edges'])} relation(s), "
            f"{figures} figure(s)")
        say("  concepts")
        for node in concepts[:most]:
            definition = node["attrs"].get("definition") \
                or "(the source does not define it here)"
            say(f"    {node['label']} [{node['kind']}] -- {definition}")
        if len(concepts) > most:
            say(f"    ... and {len(concepts) - most} more")
        relations = [e for e in graph["edges"] if e["rel"] != "illustrates"]
        units = units_of(view.reads(held.slug))
        runs = sorted({str(r.get("run") or "") for r in view.reads(held.slug)})
        say("  read by " + (", ".join(_run_said(out, r) for r in runs if r) or "an earlier run"))
        say("  relations")
        for edge in relations[:most]:
            pages = sorted({p for u in edge.get("provenance") or () if u in units
                            for p in (units[u].first_page, units[u].last_page)})
            say(f"    {_label(graph, edge['source'])} --{edge['rel']}--> "
                f"{_label(graph, edge['target'])}   "
                + (f"p.{pages[0]}" + (f"-{pages[-1]}" if pages[-1] != pages[0] else "") + " "
                   if pages else "")
                + f"({', '.join(edge.get('provenance') or ())})")
        if len(relations) > most:
            say(f"    ... and {len(relations) - most} more")
        for label, records in (("names joined", graph["folds"].get("concepts") or ()),
                               ("relations joined", graph["folds"].get("relations") or ())):
            if records:
                say(f"  {label}: " + ", ".join(f"{r['from']} -> {r['into']}" for r in records))
    return 0


def sources(out: str | Path, *, most: int = 10, say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest sources --out STORE``: every source at a glance, in plain text.

    Per source, how much of it is read and what the store holds for it; the concepts more
    than one source was read into; the names the hygiene pass joined across sources, with
    a weight; the relations joining one source's vocabulary to another's; how many name
    pairs the hygiene pass has judged; and the vocabulary the sources were read with.
    `Sources.shared` is the same as data.
    """
    where = Path(out).expanduser()
    if not where.exists():
        say(f"no store at {out}")
        return 1
    held = Sources(out)
    got = held.shared()
    if not got["sources"]:
        say(f"nothing read into {out}")
        return 1
    nodes = sum(s["nodes"] for s in got["sources"])
    edges = sum(s["edges"] for s in got["sources"])
    say(f"{out}: {len(got['sources'])} source(s), {nodes} node(s), {edges} edge(s)")
    for one in got["sources"]:
        say(f"  {one['source']:<28} {one['read']:>4} / {one['wanted'] or '?':<5} units read"
            f"   {one['nodes']:>5} nodes {one['edges']:>5} edges   {one['title']}")
    shared = got["shared"]
    say(f"  concepts in more than one source ({len(shared)})"
        + ("" if shared else ": none -- the sources name nothing in common yet"))
    for one in shared[:most]:
        say(f"    {one['label']:<32} {len(one['sources'])} sources: "
            f"{', '.join(one['sources'])}")
    if len(shared) > most:
        say(f"    ... and {len(shared) - most} more")
    merged = got["merged"]
    say(f"  between sources ({len(merged)})"
        + ("" if merged else
           f": no concept merged across sources yet; ml-stack-ingest tidy --out {out}"
           if got.get("logged") else
           f": no log of the names the sources share; ml-stack-ingest fold --out {out} "
           f"re-folds each source from its reads and writes one"))
    for one in merged[:most]:
        say(f"    {one['a_label']} ({one['a_source']}) = {one['b_label']} "
            f"({one['b_source']})  {one['kind']}  {one['weight']}")
    if len(merged) > most:
        say(f"    ... and {len(merged) - most} more")
    between = got["between"]
    say(f"  relations between sources ({len(between)})"
        + ("" if between else ": none -- no relation joins one source's names to another's"))
    for one in between[:most]:
        say(f"    {one['source_label']} --{one['rel']}--> {one['target_label']}   "
            f"({', '.join(one['sources'][0])} -> {', '.join(one['sources'][1])})")
    if len(between) > most:
        say(f"    ... and {len(between) - most} more")
    judged = got["decisions"]
    say(f"  judged: {judged['pairs']} pair(s) -- {judged['same']} the same name, "
        f"{judged['different']} a spelling apart, {judged['unsure']} unsure"
        if judged["pairs"] else "  judged: nothing -- the hygiene pass has settled no pair")
    from ml_stack.ingest.vocabulary import Vocabulary

    for line in Vocabulary.read(out).lines(most=most):
        say(f"  {line}")
    return 0


def _run_said(out: str | Path, run_id: str) -> str:
    """One run node as a person reads it: the model and when."""
    from ml_stack.graph.store import GraphStore

    try:
        with GraphStore(out, read_only=True) as store:
            attrs = next((n.get("attrs") or {} for n in store.nodes(kind="run")
                          if n["id"] == run_id), {})
    except Exception:  # noqa: BLE001 - a store a writer holds; the id still says something
        attrs = {}
    model = str(attrs.get("model") or "")
    try:
        from ml_stack.hub import pretty_name
        model = pretty_name(model) if model else model
    except Exception:  # noqa: BLE001
        pass
    return f"{model or run_id} ({attrs.get('started') or run_id[4:]})"


def _label(graph: Mapping[str, Any], node_id: str) -> str:
    return next((n["label"] for n in graph["nodes"] if n["id"] == node_id), node_id)
