"""A store's reads as they land: every book, its graph so far, what the books share,
and the two commands that print it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.ingest.fold import fold_book
from ml_stack.ingest.progress import GIVE_UP, Progress
from ml_stack.ingest.reads import _read_json, reads_path, tokens_of, units_of

__all__ = ["Book", "Shelf", "shelf", "show"]


@dataclass
class Book:
    """One book on a shelf: how much of it is read, and how much of that is in the store."""

    slug: str
    title: str = ""
    path: str = ""
    units: int = 0           # rows in the reads file: every unit attempted and kept
    read: int = 0            # rows an extraction came back for -- what folds
    failed: int = 0
    given_up: int = 0
    wanted: int = 0          # units the book has
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
        """Seconds of reading still to do, at this book's own measured rate."""
        return max(self.wanted - self.units, 0) * self.per_unit


class Shelf:
    """A store's reads as they land: every book, its graph so far, and the store itself.

    Nothing here needs the run to have finished, or the PDFs to still be where they were
    read from: `books` and `reads` come from the files beside the store, `graph` folds
    those in memory, and `store` opens the store read-only beside the writer.
    """

    def __init__(self, out: str | Path) -> None:
        self.out = Path(out).expanduser()
        self.progress = Progress(Progress.beside(self.out))

    def books(self) -> list[Book]:
        """Every book with reads or progress, by slug."""
        out = []
        for slug in sorted(set(self.progress.state["books"]) | set(self._slugs())):
            held = self.progress.state["books"].get(slug) or {}
            rows = self.reads(slug)
            prompt, completion = tokens_of(rows)
            done = (held.get("done") or {}).values()
            out.append(Book(
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

    def book(self, slug: str) -> Book | None:
        """One book by slug, or None."""
        return next((b for b in self.books() if b.slug == slug), None)

    def _slugs(self) -> list[str]:
        head, tail = self.out.name + ".", ".reads.json"
        if not self.out.parent.is_dir():
            return []
        return [p.name[len(head):-len(tail)] for p in self.out.parent.glob(f"{head}*{tail}")
                if len(p.name) > len(head) + len(tail)]


    def reads(self, slug: str) -> list[dict[str, Any]]:
        """One book's extractions so far, in the order they were read."""
        held = _read_json(reads_path(self.out, slug))
        if not isinstance(held, dict):
            return []
        return [dict(row, unit=str(row.get("unit") or key))
                for key, row in held.items() if isinstance(row, Mapping)]


    def graph(self, slug: str, *, log: Callable[[str], None] | None = None) -> dict[str, Any]:
        """The book folded from its reads so far -- nodes, edges and the folds it made.

        No store and no PDF: the provenance each unit needs is on the row it wrote.
        """
        rows = self.reads(slug)
        held = self.progress.state["books"].get(slug) or {}
        return fold_book(rows, units_of(rows), book_title=str(held.get("title") or ""), log=log)

    def store(self, **kw: Any) -> Any:
        """A read-only `GraphStore` on the shelf, openable while the run is writing to it."""
        from ml_stack.graph.store import GraphStore

        return GraphStore(self.out, read_only=True, **kw)

    def shared(self, store: Any = None) -> dict[str, Any]:
        """What the books on this shelf hold, and what they hold in common.

        ``books`` is one entry per book -- units read, and the nodes and edges the store
        holds for it. ``shared`` is every concept two or more books were read into, most
        shared first, each naming them. ``between`` is every edge whose two ends were read
        from different books -- one book's vocabulary joined to another's. ``merged`` is
        `between_books`. ``decisions`` counts the pairs a judge has settled in the store's
        `graph.tidy` document.

        A book's node is one a ``read_from`` edge joins to ``book:<slug>``; a book's edge
        is one whose provenance names a unit of that book. Without a ``store`` one is
        opened read-only on the shelf.
        """
        if store is None:
            with self.store() as held:
                return self.shared(held)
        nodes = list(store.nodes())
        edges = list(store.edges())
        labels = {str(n["id"]): str(n.get("label") or "") for n in nodes}
        mentions = {str(n["id"]): int(n.get("mentions") or 0) for n in nodes}
        books_of: dict[str, set[str]] = {}
        for edge in edges:
            target = str(edge.get("target") or "")
            if edge.get("rel") == "read_from" and target.startswith("book:"):
                books_of.setdefault(str(edge.get("source") or ""),
                                    set()).add(target[len("book:"):])
        known = {b.slug: b for b in self.books()}
        slugs = sorted(set(known) | {str(n["id"])[len("book:"):] for n in nodes
                                     if str(n.get("id") or "").startswith("book:")})
        node_counts = {slug: 0 for slug in slugs}
        for held_books in books_of.values():
            for slug in held_books:
                node_counts[slug] = node_counts.get(slug, 0) + 1
        edge_counts = {slug: 0 for slug in slugs}
        for edge in edges:
            if edge.get("rel") == "read_from":
                continue
            for slug in {str(u).split(":", 1)[0] for u in (edge.get("provenance") or ())}:
                if slug in edge_counts:
                    edge_counts[slug] += 1
        books = []
        for slug in slugs:
            held = known.get(slug)
            books.append({"book": slug,
                          "title": (held.title if held else "") or labels.get(f"book:{slug}", "")
                                   or slug,
                          "units": held.units if held else 0,
                          "read": held.read if held else 0,
                          "wanted": held.wanted if held else 0,
                          "nodes": node_counts.get(slug, 0),
                          "edges": edge_counts.get(slug, 0)})
        shared = [{"id": node_id, "label": labels.get(node_id, node_id),
                   "mentions": mentions.get(node_id, 0), "books": sorted(held_books)}
                  for node_id, held_books in books_of.items() if len(held_books) > 1]
        shared.sort(key=lambda r: (-len(r["books"]), -r["mentions"], r["label"]))
        between = []
        for edge in edges:
            if edge.get("rel") == "read_from":
                continue
            source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
            here, there = books_of.get(source) or set(), books_of.get(target) or set()
            if not (here and there) or here & there:
                continue
            between.append({"source": source, "source_label": labels.get(source, source),
                            "rel": str(edge.get("rel") or ""), "target": target,
                            "target_label": labels.get(target, target),
                            "weight": int(edge.get("weight") or 0),
                            "books": [sorted(here), sorted(there)]})
        between.sort(key=lambda r: (-r["weight"], r["source_label"], r["target_label"]))
        return {"books": books, "shared": shared, "between": between,
                "merged": self.between_books(store), "decisions": _decisions_in(store)}

    def between_books(self, store: Any = None) -> list[dict[str, Any]]:
        """Every name the hygiene pass joined across two books, heaviest first.

        One entry per merge in the store's `graph.tidy` merges document whose two names
        were read from different books: ``a_label`` and ``a_book`` for the name kept,
        ``b_label`` and ``b_book`` for the name folded into it, ``kind``, and ``weight`` --
        the joined node's mentions (the units both were read from, when the store holds no
        count) plus the edges the merge moved.
        """
        if store is None:
            with self.store() as held:
                return self.between_books(held)
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
                        "a_book": ", ".join(sorted(here)),
                        "b_label": str(one.get("gone_label") or one.get("gone") or ""),
                        "b_book": ", ".join(sorted(there)), "kind": str(one.get("kind") or ""),
                        "weight": (mentions.get(kept) or len(kept_from) + len(gone_from))
                                  + int(one.get("edges_moved") or 0)})
        out.sort(key=lambda r: (-r["weight"], r["a_label"], r["b_label"]))
        return out


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


def show(out: str | Path, *, book: str = "", most: int = 5,
         say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest show --out STORE``: what each book was read as, in plain text.

    A sample of concepts with their kind and definition, a sample of relations with the
    verb and the page they were read on, the folds the book made, and how many figures.
    """
    shelf = Shelf(out)
    books = [b for b in shelf.books() if b.units and (not book or b.slug == book)]
    if not books:
        say(f"nothing read into {out}" + (f" for {book}" if book else ""))
        return 1
    for held in books:
        graph = shelf.graph(held.slug)
        concepts = [n for n in graph["nodes"] if n["kind"] != "figure"]
        figures = len(graph["nodes"]) - len(concepts)
        say(f"\n{held.title or held.slug} ({held.slug}): {held.read} of "
            f"{held.wanted or '?'} units read" + (", partial" if held.partial else "")
            + (f", {held.failed} failed" if held.failed else ""))
        say(f"  {len(concepts)} concept(s), {len(graph['edges'])} relation(s), "
            f"{figures} figure(s)")
        say("  concepts")
        for node in concepts[:most]:
            definition = node["attrs"].get("definition") or "(the book does not define it here)"
            say(f"    {node['label']} [{node['kind']}] -- {definition}")
        if len(concepts) > most:
            say(f"    ... and {len(concepts) - most} more")
        relations = [e for e in graph["edges"] if e["rel"] != "illustrates"]
        units = units_of(shelf.reads(held.slug))
        runs = sorted({str(r.get("run") or "") for r in shelf.reads(held.slug)})
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


def shelf(out: str | Path, *, most: int = 10, say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest shelf --out STORE``: the whole shelf at a glance, in plain text.

    Per book, how much of it is read and what the store holds for it; the concepts more
    than one book was read into; the names the hygiene pass joined across books, with a
    weight; the relations joining one book's vocabulary to another's; and how many name
    pairs the hygiene pass has judged. `Shelf.shared` is the same as data.
    """
    where = Path(out).expanduser()
    if not where.exists():
        say(f"no store at {out}")
        return 1
    held = Shelf(out)
    got = held.shared()
    if not got["books"]:
        say(f"nothing on the shelf at {out}")
        return 1
    nodes = sum(b["nodes"] for b in got["books"])
    edges = sum(b["edges"] for b in got["books"])
    say(f"{out}: {len(got['books'])} book(s), {nodes} node(s), {edges} edge(s)")
    for book in got["books"]:
        say(f"  {book['book']:<28} {book['read']:>4} / {book['wanted'] or '?':<5} units read"
            f"   {book['nodes']:>5} nodes {book['edges']:>5} edges   {book['title']}")
    shared = got["shared"]
    say(f"  concepts in more than one book ({len(shared)})"
        + ("" if shared else ": none -- the books name nothing in common yet"))
    for one in shared[:most]:
        say(f"    {one['label']:<32} {len(one['books'])} books: {', '.join(one['books'])}")
    if len(shared) > most:
        say(f"    ... and {len(shared) - most} more")
    merged = got["merged"]
    say(f"  between books ({len(merged)})"
        + ("" if merged else f": no concept merged across books yet; "
                             f"ml-stack-ingest tidy --out {out}"))
    for one in merged[:most]:
        say(f"    {one['a_label']} ({one['a_book']}) = {one['b_label']} ({one['b_book']})  "
            f"{one['kind']}  {one['weight']}")
    if len(merged) > most:
        say(f"    ... and {len(merged) - most} more")
    between = got["between"]
    say(f"  relations between books ({len(between)})"
        + ("" if between else ": none -- no relation joins one book's names to another's"))
    for one in between[:most]:
        say(f"    {one['source_label']} --{one['rel']}--> {one['target_label']}   "
            f"({', '.join(one['books'][0])} -> {', '.join(one['books'][1])})")
    if len(between) > most:
        say(f"    ... and {len(between) - most} more")
    judged = got["decisions"]
    say(f"  judged: {judged['pairs']} pair(s) -- {judged['same']} the same name, "
        f"{judged['different']} a spelling apart, {judged['unsure']} unsure"
        if judged["pairs"] else "  judged: nothing -- the hygiene pass has settled no pair")
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
