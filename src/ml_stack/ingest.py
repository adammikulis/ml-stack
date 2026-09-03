"""Documents into a graph: a shelf of textbooks read section by section into one store.

`ml_stack.sources.pdf` turns a book into units an extractor can take. This is the other
half: each unit through `Client.extract` against `contracts/extraction-document.schema.json`,
the extractions folded into one graph per book (`ml_stack.entities.fold`, so a relation the
model spelled two ways is one edge), and nodes, edges and the raw extractions written into a
`GraphStore`. Every node and edge carries where it was read from -- book, chapter, section,
page -- because a claim in a knowledge graph with no page behind it is a claim nobody can
check.

Three things here are not obvious and all three were paid for elsewhere in this repo:

*A run is hours, so it is resumable and it detaches.* A progress file beside the store
records every unit that finished; `--resume` skips those, `--detach` re-runs the command in
its own session with a log under ``~/.ml-stack/ingest/logs`` (a child of a shell dies with
the shell, and a ranking sweep was killed that way thirty minutes in), and
``ml-stack-ingest status`` says how many sections of how many books are done and at what
rate.

*What it cost is on record, per section.* Each extraction keeps a `ml_stack.telemetry.Call`
and the run keeps their `Spent`, so "the ten books took nine hours" can be broken down
into which book, which section, and how much of it was prompt.

*Whether it does a good job is measured, not asserted.* ``--gold FILE`` runs a set of
passages with known triples through the same extraction and scores recall and precision,
matching subjects and objects through their aliases and `entities.close` and predicates
through theirs, and lists what was missed. ``--fail-under`` makes that a gate.

Nothing here is about any one book: it reads a PDF, it asks a model, it writes a graph.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["HOME", "INSTRUCTIONS", "PER_SECTION", "VERBS", "Progress", "Scored", "build", "detach",
           "extract_unit", "fold_book", "gold_score", "main", "read_gold", "schema",
           "status", "write"]

HOME = Path(os.environ.get("MLSTACK_INGEST_HOME") or "~/.ml-stack/ingest").expanduser()
"""Where a detached run's log and its record of itself live. Not the store: the store is
the caller's, named by ``--out``."""

PER_SECTION = 300.0
"""The most one section may take before it is recorded as timed out and the next is read."""

IMAGES_PER_SECTION = 4
"""How many of a section's figures are shown to the model at once. A section of a biology
textbook has a dozen plates; a dozen images is a prompt of images with a paragraph in it."""

# Windows has no sessions; a child that survives its parent's console is asked for by flag.
_WINDOWS_DETACHED = 0x00000200 | 0x00000008     # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS


VERBS: dict[str, str] = {
    "is_a": "a kind of (mitochondria is_a organelle)",
    "part_of": "belongs inside a larger whole (nucleus part_of cell)",
    "has_part": "the whole, naming a part (cell has_part nucleus)",
    "causes": "brings about (mutation causes variation)",
    "produces": "makes or yields (mitochondria produces ATP)",
    "consumes": "uses up (photosynthesis consumes carbon dioxide)",
    "regulates": "controls the rate or amount of (insulin regulates blood glucose)",
    "located_in": "found in a place or region (mitochondria located_in cytoplasm)",
    "measured_in": "the unit a quantity takes (force measured_in newton)",
    "defined_by": "fixed by a law, equation or definition (momentum defined_by mass times velocity)",
    "example_of": "one instance of a general thing (glucose example_of monosaccharide)",
    "contrasts_with": "set against, as the text opposes them (prokaryote contrasts_with eukaryote)",
    "precedes": "comes before, in a sequence or in time (prophase precedes metaphase)",
    "requires": "cannot happen without (respiration requires oxygen)",
    "converts": "turns one thing into another (the sun converts hydrogen, fuses it, into helium)",
    "created_by": "written, proposed, discovered or built by a person or body (the declaration created_by its author)",
    "adopted_by": "enacted, ratified or taken up by a body (the declaration adopted_by the congress)",
    "member_of": "one of a group, class or body (a delegate member_of the congress)",
}
"""The closed relation vocabulary, each verb glossed with the sense the model should take.

The schema's enum is this list; a test keeps them equal. The gloss is what moved
precision on the Slack graph -- a model told what a verb means uses it for that and
nothing else -- so every verb here has one, and a verb without a gloss is not added."""


def _verbs_line() -> str:
    return "The verb phrases, and what each means: " + "; ".join(
        f"{verb} -- {gloss}" for verb, gloss in VERBS.items()) + ".\n"


INSTRUCTIONS = (
    "You are reading one section of a textbook into a knowledge graph. List the concepts "
    "the section names, how they stand to one another, what its figures show, and the "
    "terms it defines.\n"
    "Invent nothing. Every concept, relation and definition must be stated in the text "
    "you were given; a fact you know from elsewhere does not belong here.\n"
    "A definition is the book's own words, cut to one line, and only when the section "
    "defines the thing. When it does not, the definition is an empty string -- an empty "
    "string is always better than a definition you wrote yourself.\n"
    "`aliases` are other names this same section uses for the same thing: a plural, an "
    "abbreviation, a symbol. Not synonyms you happen to know.\n"
    "A relation joins two concept names from your own `concepts` list, using one of the "
    "verb phrases the schema allows and no other. State only what the section states.\n"
    + _verbs_line() +
    "A caption is marked in the text as [Figure 2.9]. For each figure, `shows` is what "
    "the picture shows in one line, and `concepts` are only those the caption or the "
    "surrounding text says it illustrates -- never a concept guessed from the picture.\n"
    "Return only JSON matching the schema."
)

WITH_IMAGES = (
    "\nThe section's figures follow the text as pictures. Use them to say what each figure "
    "shows; still take the concepts a figure illustrates from what the caption and the text "
    "say, not from the picture alone."
)


def schema() -> dict[str, Any]:
    """The document extraction shape, read from the contracts."""
    from ml_stack.contracts import load

    return dict(load("extraction-document.schema.json"))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).casefold()).strip("-") or "untitled"


# -- one section through the model ------------------------------------------------------------


@dataclass
class Read:
    """One unit, extracted once, and everything it cost."""

    unit: str
    book: str
    chapter: str
    section: str
    title: str
    pages: list[int] = field(default_factory=list)
    seconds: float = 0.0
    concepts: int = 0
    relations: int = 0
    figures: int = 0
    images: int = 0
    timed_out: bool = False
    error: str = ""
    extracted: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)


class _Recording:
    """A client that keeps a `Call` for every reply it gets, and is otherwise the client.

    `Client.extract` calls the client's own `chat`, so binding `extract` here puts every
    call an extraction makes through the recording one -- the same trick, and for the same
    reason, as the bench's `_Extracting`.
    """

    def __init__(self, client: Any, *, host: str = "", port: int = 0) -> None:
        self._client = client
        self._host, self._port = host, port
        self.calls: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.telemetry import Call

        began = time.time()
        reply = self._client.chat(*args, **kwargs)
        self.calls.append(Call.from_reply(reply, time.time() - began, tool="extract",
                                          args={}, host=self._host, port=self._port))
        return reply

    def extract(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from ml_stack.client.chat import Client

        return Client.extract(self, *args, **kwargs)  # type: ignore[arg-type]

    def _chat_extractor(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.client.chat import Client

        return Client._chat_extractor(self, *args, **kwargs)  # type: ignore[arg-type]

    def _raw_extractor(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.client.chat import Client

        return Client._raw_extractor(self, *args, **kwargs)  # type: ignore[arg-type]


def prompt_for(unit: Any, *, images: bool = False,
               most: int = IMAGES_PER_SECTION) -> tuple[list[dict[str, Any]], int]:
    """The turns one section is extracted from, and how many pictures went with it.

    The section is named before its text -- a model reading "2.1 Atoms, Isotopes, Ions and
    Molecules" knows what the pronouns in the first paragraph refer to. With ``images`` the
    figures that rendered go in as a user message of their own after the text, which is the
    `_images` convention `graph.ask` uses for a tool that brings pictures back: llama.cpp
    cannot carry an image inside anything but a user turn.
    """
    where = " / ".join(x for x in (unit.book_title, f"Chapter {unit.chapter}"
                                   if unit.chapter else "", unit.chapter_title) if x)
    head = f"{where}\n{unit.section} {unit.section_title}".strip()
    if unit.parts > 1:
        head += f" (part {unit.part} of {unit.parts})"
    terms = ("\n\nTerms this section sets in bold: " + ", ".join(unit.key_terms)
             if unit.key_terms else "")
    turns: list[dict[str, Any]] = [
        {"role": "system", "content": INSTRUCTIONS + (WITH_IMAGES if images else "")},
        {"role": "user", "content": f"{head}\n\n{unit.text}{terms}"},
    ]
    if not images:
        return turns, 0
    pictures = [f.png for f in unit.figures if f.png][:most]
    if not pictures:
        return turns, 0
    from ml_stack.vision.payloads import build_message

    seen, report = build_message("The figures of this section, in order:", list(pictures))
    kept = sum(1 for part in seen["content"] if part.get("type") == "image_url")
    if not kept:
        # a picture that cannot be prepared is not sent and not claimed: a model told to
        # look at nothing answers about nothing, confidently
        return turns, 0
    turns.append(seen)
    return turns, kept


def extract_unit(client: Any, unit: Any, shape: Mapping[str, Any], *, images: bool = False,
                 per_section: float = PER_SECTION, cache_dir: str | Path | None = None) -> Read:
    """One unit through ``client.extract``, and what it cost.

    A failure is a result, not the end of the run: the row keeps the error and the next
    section is read. ``think=False`` -- reading a page is a reading, not a reasoning, and
    the thinking channel is where a ceiling gets spent.
    """
    row = Read(unit=unit.id, book=unit.book, chapter=unit.chapter, section=unit.section,
               title=unit.section_title, pages=[unit.first_page, unit.last_page])
    turns, shown = prompt_for(unit, images=images)
    row.images = shown
    recording = _Recording(client)
    began = time.time()
    try:
        got = recording.extract(unit.text, dict(shape), messages=turns, think=False, tries=1,
                                cache_dir=cache_dir,
                                cache_extra=f"document/{unit.id}/{int(images)}")
        row.extracted = got if isinstance(got, dict) else {}
    except Exception as exc:  # noqa: BLE001 - one bad section does not end a shelf
        row.error = f"{type(exc).__name__}: {exc}"[:200]
    row.seconds = round(time.time() - began, 2)
    if per_section and row.seconds >= per_section and row.error:
        row.timed_out = True
    row.calls = [call.public() for call in recording.calls]
    got = row.extracted
    row.concepts = len(got.get("concepts") or ())
    row.relations = len(got.get("relations") or ())
    row.figures = len(got.get("figures") or ())
    return row


# -- an extraction as a graph -------------------------------------------------------------------


def build(extraction: Mapping[str, Any], unit: Any, *, book_title: str = ""
          ) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    """One extraction as ``(nodes by id, edges by triple)``, every one saying where it came from.

    Names are the ids: two sections that both name the same concept are one node whose
    provenance lists both, which is the whole reason to read a book section by section
    rather than a page at a time.
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
            "attrs": {"definition": "", "aliases": [], "key_term": False,
                      "book": where["book"], "book_title": book_title},
            "provenance": []})
        held["mentions"] += 1
        if where["unit"] not in held["provenance"]:
            held["provenance"].append(where["unit"])
        attrs = held["attrs"]
        if more.get("definition") and not attrs["definition"]:
            attrs["definition"] = str(more["definition"])[:400]
            attrs["defined_in"] = dict(where)
        for alias in more.get("aliases") or ():
            clean_alias = " ".join(str(alias).split())
            if clean_alias and clean_alias.casefold() != clean.casefold() \
                    and clean_alias not in attrs["aliases"]:
                attrs["aliases"].append(clean_alias)
        if more.get("key_term"):
            attrs["key_term"] = True
        attrs.setdefault("chapter", where["chapter"])
        attrs.setdefault("section", where["section"])
        attrs.setdefault("page", where["page"])
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
                                      "weight": 0, "provenance": [], "where": dict(where)})
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
                          "attrs": {"caption": caption, "shows": str(figure.get("shows") or ""),
                                    "book": where["book"], "chapter": where["chapter"],
                                    "section": where["section"], "page": where["page"]},
                          "provenance": [where["unit"]]}
        for name in figure.get("concepts") or ():
            target = put(name, "concept")
            if target:
                key = (node_id, "illustrates", target)
                edges.setdefault(key, {"source": node_id, "rel": "illustrates",
                                       "target": target, "weight": 0, "provenance": [],
                                       "where": dict(where)})
                edges[key]["weight"] += 1
                if where["unit"] not in edges[key]["provenance"]:
                    edges[key]["provenance"].append(where["unit"])
    return nodes, edges


def fold_book(reads: Iterable[Mapping[str, Any]], units_by_id: Mapping[str, Any], *,
              book_title: str = "", log: Callable[[str], None] | None = None
              ) -> dict[str, Any]:
    """Every section of one book, folded into one graph.

    Two folds, and they are not the same fold. The *relations* are folded by
    `entities.fold_edges`, which is what stops ``has_part`` and ``haspart`` being two
    relationships. The *names* are folded by `entities.fold_names` over how often each was
    said, which is what stops "mitochondrion" and "mitochondria" being two concepts -- and
    which refuses to fold two names a book keeps using, because at that point they are two
    things the book distinguishes and merging them would be a decision nobody made.
    """
    from ml_stack.entities.fold import fold_edges, fold_names

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for read in reads:
        unit = units_by_id.get(str(read.get("unit") or ""))
        if unit is None or not read.get("extracted"):
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

    weight = {node["label"]: int(node["mentions"]) for node in nodes.values()
              if node["kind"] != "figure"}
    canonical, name_folds = fold_names(weight, log=log, label="concepts",
                                       settles="both spellings stay, and the book is right")
    moved = {f"concept:{_slug(name)}": f"concept:{_slug(into)}"
             for name, into in canonical.items() if into != name}
    if moved:
        nodes, edges = _apply(nodes, edges, moved)

    return {"nodes": sorted(nodes.values(), key=lambda n: n["id"]),
            "edges": sorted(edges.values(), key=lambda e: (e["source"], e["rel"], e["target"])),
            "folds": {"relations": relation_folds, "concepts": name_folds}}


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


def write(out: str | Path, graph: Mapping[str, Any], *, book: str, title: str,
          docs: Mapping[str, Any] | None = None) -> dict[str, int]:
    """One book's graph and its raw extractions into the store, and read back before returning.

    The book itself is a node, so a store holding a shelf can still be asked what came out
    of which book; every concept hangs off it by ``read_from``.
    """
    from ml_stack.graph.store import GraphStore

    book_id = f"book:{book}"
    nodes = [{"id": book_id, "kind": "book", "label": title or book, "mentions": 1,
              "attrs": {"book": book}}, *graph.get("nodes", ())]
    edges = [*graph.get("edges", ()),
             *({"source": node["id"], "rel": "read_from", "target": book_id, "weight": 1}
               for node in graph.get("nodes", ()))]
    with GraphStore(out) as store:
        counts = store.write({"nodes": nodes, "edges": edges})
        for key, value in (docs or {}).items():
            store.put_doc(key, value)
        store.put_doc(f"ingest:folds:{book}", dict(graph.get("folds") or {}))
        back = store.query("MATCH (n:Node {id: $id}) RETURN n.id AS id", {"id": book_id})
    if not back:
        raise RuntimeError(f"{book_id} was written to {out} and did not come back")
    return counts


# -- what is done, and how fast --------------------------------------------------------------


class Progress:
    """The record of a run, beside the store: which units are done, and what each cost.

    A file rather than the store itself, because it is written after every section and a
    store is opened for writing once per book. ``--resume`` reads it; ``status`` prints it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.state: dict[str, Any] = {"started": time.strftime("%FT%T"), "books": {}}
        held = _read_json(self.path)
        if isinstance(held, dict) and isinstance(held.get("books"), dict):
            self.state = held

    @staticmethod
    def beside(out: str | Path) -> Path:
        """Where a store's progress file goes."""
        return Path(str(Path(out).expanduser()) + ".ingest.json")

    def book(self, slug: str, *, title: str = "", path: str = "", sections: int = 0
             ) -> dict[str, Any]:
        held = self.state["books"].setdefault(slug, {"title": title, "path": path,
                                                     "sections": sections, "done": {}})
        for key, value in (("title", title), ("path", path), ("sections", sections)):
            if value:
                held[key] = value
        return held

    def done(self, slug: str, unit: str) -> bool:
        return unit in (self.state["books"].get(slug, {}).get("done") or {})

    def note(self, slug: str, read: Read) -> None:
        """Write one finished unit down, at once: a run killed mid-book resumes from here."""
        self.book(slug)["done"][read.unit] = {
            "seconds": read.seconds, "concepts": read.concepts, "relations": read.relations,
            "figures": read.figures, "images": read.images, "error": read.error,
            "at": time.strftime("%FT%T")}
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def totals(self) -> dict[str, Any]:
        """Books, sections done of how many, seconds spent, and sections a minute."""
        done = seconds = wanted = failed = 0
        for book in self.state["books"].values():
            entries = (book.get("done") or {}).values()
            done += len(entries)
            wanted += int(book.get("sections") or 0)
            seconds += sum(float(e.get("seconds") or 0.0) for e in entries)
            failed += sum(1 for e in entries if e.get("error"))
        return {"books": len(self.state["books"]), "sections": done, "of": wanted,
                "failed": failed, "seconds": round(seconds, 1),
                "per_section": round(seconds / done, 1) if done else 0.0,
                "started": self.state.get("started", "")}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def status(out: str | Path, *, say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest status``: books and sections done, what failed, and the rate."""
    where = Progress.beside(out)
    if not where.is_file():
        say(f"nothing ingested into {out}: no {where.name}")
        return 1
    progress = Progress(where)
    totals = progress.totals()
    say(f"{out}: {totals['sections']} of {totals['of']} sections in {totals['books']} book(s), "
        f"started {totals['started']}")
    for slug, book in sorted(progress.state["books"].items()):
        entries = book.get("done") or {}
        spent = sum(float(e.get("seconds") or 0.0) for e in entries.values())
        broke = sum(1 for e in entries.values() if e.get("error"))
        say(f"  {slug:<28} {len(entries):>4} / {book.get('sections') or '?':<5} "
            f"{spent / 60:6.1f} min" + (f"  {broke} failed" if broke else ""))
    if totals["sections"]:
        say(f"  {totals['per_section']:.1f} s/section, {3600 / max(totals['per_section'], 1e-9):.0f}"
            f" sections/hour; {totals['seconds'] / 3600:.1f} h spent"
            + (f", {totals['failed']} failed" if totals["failed"] else ""))
    return 0


# -- does it do a good job: a gold set of passages with known triples -------------------------


@dataclass
class Scored:
    """What a gold run found: the rates, and every triple that was missed."""

    passages: int = 0
    wanted: int = 0
    found: int = 0
    matched: int = 0
    seconds: float = 0.0
    misses: list[dict[str, str]] = field(default_factory=list)
    spurious: list[dict[str, str]] = field(default_factory=list)
    # Gold triples whose predicate -- and none of its aliases -- is a word the schema's
    # closed vocabulary has. A constrained decode cannot say it, so the triple can never be
    # matched and the recall it drags down is a fact about the gold set, not the model.
    # Said out loud rather than quietly subtracted: which of the two to change is a
    # decision, and it is not this function's.
    unsayable: list[dict[str, str]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return round(self.matched / self.wanted, 4) if self.wanted else 0.0

    @property
    def precision(self) -> float:
        return round(self.matched / self.found, 4) if self.found else 0.0

    @property
    def f1(self) -> float:
        r, p = self.recall, self.precision
        return round(2 * r * p / (r + p), 4) if r + p else 0.0


def read_gold(path: str | Path) -> list[dict[str, Any]]:
    """A gold set: ``{"passages": [{"passage_id", "source", "text", "triples": [...]}]}``.

    A triple is ``{"subject", "predicate", "object"}`` with optional ``*_aliases`` -- the
    other names a right answer is allowed to use. The aliases are the whole point: an
    extractor that says "mitochondrion" where the gold says "mitochondria" is right, and a
    scorer with no aliases would call it wrong and send somebody tuning a prompt for a week.
    """
    held = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    passages = held.get("passages") if isinstance(held, Mapping) else held
    if not isinstance(passages, list) or not passages:
        raise ValueError(f"{path}: no passages")
    return [dict(p) for p in passages if isinstance(p, Mapping)]


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in (value or ()) if str(v).strip()]


def _same(said: str, wanted: str, aliases: Sequence[str]) -> bool:
    """Whether ``said`` names the same thing as ``wanted`` or one of its aliases.

    Exact first (casefolded, underscores and spaces the same thing), then containment either
    way -- "the mitochondria" names mitochondria -- then `entities.close`, which is what
    forgives a plural or one letter.
    """
    from ml_stack.entities import close

    def flat(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()

    here = flat(said)
    if not here:
        return False
    for other in [wanted, *aliases]:
        there = flat(other)
        if not there:
            continue
        if here == there:
            return True
        if f" {there} " in f" {here} " or f" {here} " in f" {there} ":
            return True
        if close(here.replace(" ", ""), there.replace(" ", "")):
            return True
    return False


def gold_score(client: Any, passages: Sequence[Mapping[str, Any]], shape: Mapping[str, Any],
               *, per_section: float = PER_SECTION,
               log: Callable[[str], None] | None = None) -> Scored:
    """Every gold passage through the same extraction, scored triple by triple.

    A gold triple is found when some extracted relation matches on all three: subject and
    object through their aliases and `entities.close`, predicate through its own aliases.
    Precision is against the same matching, so a relation the passage does state but the
    gold set does not list counts against it -- which is why a gold passage is written
    short, with everything it states written down.
    """
    out = Scored(passages=len(passages))
    vocabulary = {str(v) for v in
                  ((shape.get("properties") or {}).get("relations") or {})
                  .get("items", {}).get("properties", {}).get("rel", {}).get("enum") or ()}
    for passage in passages:
        text = str(passage.get("text") or "")
        unit = _passage_unit(passage)
        began = time.time()
        row = extract_unit(client, unit, shape, per_section=per_section)
        out.seconds += time.time() - began
        said = [r for r in (row.extracted.get("relations") or ()) if isinstance(r, Mapping)]
        wanted = [t for t in (passage.get("triples") or ()) if isinstance(t, Mapping)]
        out.wanted += len(wanted)
        out.found += len(said)
        for triple in wanted:
            words = {str(triple.get("predicate") or ""), *_names(triple.get("predicate_aliases"))}
            if vocabulary and not (words & vocabulary):
                out.unsayable.append({"passage": str(passage.get("passage_id") or ""),
                                      "predicate": str(triple.get("predicate") or "")})
        taken: set[int] = set()
        for triple in wanted:
            hit = next((i for i, r in enumerate(said) if i not in taken and _matches(r, triple)),
                       None)
            if hit is None:
                out.misses.append({"passage": str(passage.get("passage_id") or ""),
                                   "triple": f"{triple.get('subject')} "
                                             f"{triple.get('predicate')} {triple.get('object')}"})
            else:
                taken.add(hit)
                out.matched += 1
        for index, relation in enumerate(said):
            if index not in taken:
                out.spurious.append({"passage": str(passage.get("passage_id") or ""),
                                     "triple": f"{relation.get('from')} {relation.get('rel')} "
                                               f"{relation.get('to')}"})
        if log:
            log(f"  {row.seconds:5.1f}s  {len(said):>2} said, {len(wanted):>2} wanted  "
                f"{str(passage.get('passage_id') or text[:40])}"
                + (f"  {row.error}" if row.error else ""))
    return out


def _matches(said: Mapping[str, Any], triple: Mapping[str, Any]) -> bool:
    return (_same(str(said.get("from") or ""), str(triple.get("subject") or ""),
                  _names(triple.get("subject_aliases")))
            and _same(str(said.get("rel") or ""), str(triple.get("predicate") or ""),
                      _names(triple.get("predicate_aliases")))
            and _same(str(said.get("to") or ""), str(triple.get("object") or ""),
                      _names(triple.get("object_aliases"))))


def _passage_unit(passage: Mapping[str, Any]) -> Any:
    """A gold passage dressed as a `Unit`, so it goes through the run's own extraction.

    Scoring a path other than the one that runs is scoring nothing: the prompt, the schema,
    the sampling and the parsing are all the ones the shelf will be read with.
    """
    from ml_stack.sources.pdf import Unit

    source = str(passage.get("source") or "gold")
    return Unit(book=_slug(source), book_title=source, chapter="", chapter_title="",
                section="", section_title=str(passage.get("passage_id") or "passage"),
                first_page=0, last_page=0, text=str(passage.get("text") or ""))


def gold_lines(scored: Scored, *, most: int = 20) -> list[str]:
    """The gold report as lines: the rates, then what was missed."""
    out = [f"gold: {scored.matched} of {scored.wanted} triples over {scored.passages} passages "
           f"-- recall {scored.recall:.0%}, precision {scored.precision:.0%}, "
           f"F1 {scored.f1:.0%} ({scored.seconds:.0f}s)"]
    if scored.misses:
        out.append(f"  missed ({len(scored.misses)}):")
        out += [f"    {m['passage']}: {m['triple']}" for m in scored.misses[:most]]
        if len(scored.misses) > most:
            out.append(f"    ... and {len(scored.misses) - most} more")
    if scored.unsayable:
        words = sorted({m["predicate"] for m in scored.unsayable})
        out.append(f"  {len(scored.unsayable)} of {scored.wanted} gold triples ask for a "
                   f"predicate the schema has no word for ({', '.join(words)}); no "
                   f"constrained answer can match them, so the recall above is a floor -- "
                   f"widen the vocabulary or give those triples an alias inside it")
    if scored.spurious:
        out.append(f"  said but not in the gold ({len(scored.spurious)}):")
        out += [f"    {m['passage']}: {m['triple']}" for m in scored.spurious[:most]]
        if len(scored.spurious) > most:
            out.append(f"    ... and {len(scored.spurious) - most} more")
    return out


# -- serving the model, once, for the whole run -------------------------------------------------


@contextmanager
def _serving(args: Any, say: Callable[[str], None] = print) -> Any:
    """A client for the run: one lease, held throughout, in the model's measured shape.

    The same branch the extract bench takes, and for the same reason: a model served bare
    when a profile measured it with a build, a head and a cache type is a different program
    from the one the measurement was about.
    """
    from ml_stack.client import Client

    sampling = {k: v for k, v in (("temperature", args.temperature), ("top_p", args.top_p),
                                  ("top_k", args.top_k), ("min_p", args.min_p))
                if v is not None}
    if not args.model:
        yield Client(args.base_url, timeout=args.per_section, n_predict=args.n_predict,
                     **sampling)
        return

    from ml_stack.serve.manager import serve

    found = _find_model(args.model)
    lease: dict[str, Any] = {"port": args.serve_port, "context": args.context,
                             "parallel": 1, "timeout": 900.0, "cache_reuse": 256,
                             "warmup": False}
    manager = None
    if getattr(args, "profile", True):
        from ml_stack.serve.profile import profile_for, said

        measured = profile_for(str(found))
        if measured is not None:
            shape = measured.shape(port=args.serve_port, seats=1)
            lease = {**lease, **{k: v for k, v in shape.lease().items()
                                 if k not in ("port", "parallel")}}
            lease["context"] = max(args.context, int(lease.get("context") or 0))
            manager = shape.manager()
            say(f"    serving in its measured shape: {said(measured)}")
    if not args.images:
        lease.pop("mmproj", None)
    began = time.time()
    with serve(found, manager=manager, **lease) as server:
        say(f"    up in {time.time() - began:.0f}s")
        yield Client(server.base_url, timeout=args.per_section, n_predict=args.n_predict,
                     **sampling)


def _find_model(named: str) -> str:
    """A model by name, path or ``hf:`` reference, the way every other command finds one."""
    try:
        from ml_stack.graph.bench.serve import find_model
    except ImportError:  # pragma: no cover - the bench's extras are not required here
        return named
    return find_model(named)


# -- detaching -----------------------------------------------------------------------------------


def detach(argv: Sequence[str]) -> Path:
    """Run ``ml-stack-ingest argv`` in the background, owned by no terminal; return its log.

    A shelf of textbooks is hours. A child of a shell -- `nohup`, `&`, a redirect into a
    scratch directory -- dies with the shell, or with the agent that opened it, so the
    command re-runs itself in a new session with its output in a log under ``HOME/logs``
    and gives the shell back at once. ``status`` reads the progress file the run writes.
    """
    rest = [a for a in argv if a != "--detach"]
    logs = HOME / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"ingest-{time.strftime('%Y%m%dT%H%M%S')}.log"
    command = [sys.executable, "-m", "ml_stack.ingest", *rest]
    extra: dict[str, Any] = ({"creationflags": _WINDOWS_DETACHED}
                             if platform.system() == "Windows" else {"start_new_session": True})
    with log.open("ab") as out:
        out.write((f"argv: {' '.join(rest)}\nstarted: {time.strftime('%FT%T')}\n")
                  .encode("utf-8"))
        out.flush()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out,
                                 stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"}, **extra)
    (HOME / "ingesting.json").write_text(json.dumps({
        "pid": child.pid, "argv": list(rest), "log": str(log),
        "started": time.strftime("%FT%T")}, indent=1), encoding="utf-8")
    return log


# -- the command ---------------------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ml-stack-ingest", allow_abbrev=False,
        description="Read documents into a knowledge graph, section by section: "
                    "`ml-stack-ingest BOOK.pdf ... --out STORE`, and "
                    "`ml-stack-ingest status --out STORE` for how far a run has got.")
    ap.add_argument("docs", nargs="*", metavar="DOC",
                    help="the PDFs to read; or the single word `status`, which reports how "
                         "far the run into --out has got and stops")
    ap.add_argument("--out", default="", metavar="STORE",
                    help="the GraphStore to write into; one store holds a whole shelf. "
                         "Required to read anything; --gold writes nothing and needs none")
    ap.add_argument("--model", default="", metavar="M",
                    help="a model to put up, read with and take down: a name, a path or an "
                         "hf: reference")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080",
                    help="the model reading, when nothing is served (default: %(default)s)")
    ap.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True,
                    help="serve --model in its measured shape from ml-stack's profiles "
                         "(build, head, cache type, thinking budget, raw flags); "
                         "--no-profile serves it bare")
    ap.add_argument("--images", action="store_true",
                    help="show the model each section's figures as pictures, not only their "
                         "captions; needs a served projector, and without one the captions "
                         "are all it gets")
    ap.add_argument("--sample", type=int, default=0, metavar="N",
                    help="read only the first N sections of each book -- a smoke of the "
                         "whole path before a shelf is spent on it")
    ap.add_argument("--chapter", default="", metavar="N",
                    help="read only this chapter of each book")
    ap.add_argument("--resume", action="store_true",
                    help="skip the sections the progress file beside --out already records "
                         "as done")
    ap.add_argument("--detach", action="store_true",
                    help=f"run this in the background, owned by nobody's terminal, with its "
                         f"output in a log under {HOME / 'logs'}")
    ap.add_argument("--gold", default="", metavar="FILE",
                    help="score the extraction against a gold set of passages with known "
                         "triples -- recall, precision and the misses -- instead of reading "
                         "any book")
    ap.add_argument("--fail-under", type=float, default=None, metavar="F1",
                    help="exit 1 when --gold scores below this F1 (0-1)")
    ap.add_argument("--per-section", type=float, default=PER_SECTION, metavar="SECONDS",
                    help="the most one section may take (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=0, metavar="N",
                    help="where a long section is split, in tokens (default: the reader's "
                         "own 2500)")
    ap.add_argument("--n-predict", type=int, default=16384, metavar="N",
                    help="the answer's ceiling; a ceiling is not a budget, and a low one "
                         "truncates the extraction (default: %(default)s)")
    ap.add_argument("--context", type=int, default=32768, metavar="N",
                    help="context for a --model that is served (default: %(default)s)")
    ap.add_argument("--serve-port", type=int, default=8099)
    ap.add_argument("--cache", default="", metavar="DIR",
                    help="keep each extraction under this directory and do not ask twice "
                         "for the same section and schema")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the sampling temperature")
    ap.add_argument("--top-p", type=float, default=None, help="override top_p")
    ap.add_argument("--top-k", type=int, default=None, help="override top_k")
    ap.add_argument("--min-p", type=float, default=None, help="override min_p")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """``ml-stack-ingest``: a shelf of documents into one graph, or a gold set scored."""
    rest = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(rest)

    if args.docs[:1] == ["status"]:
        if not args.out:
            print("error: status needs --out STORE", file=sys.stderr)
            return 2
        return status(args.out)
    if not args.docs and not args.gold:
        print("error: name at least one document, or --gold FILE", file=sys.stderr)
        return 2
    if args.docs and not args.out:
        print("error: reading a document needs --out STORE to write it into", file=sys.stderr)
        return 2
    if args.detach:
        log = detach(rest)
        print(f"detached; the log is {log}")
        print(f"  ml-stack-ingest status --out {args.out}")
        return 0
    if args.gold:
        return _gold_run(args)
    return _read_run(args)


def _gold_run(args: Any) -> int:
    passages = read_gold(args.gold)
    print(f"gold: {len(passages)} passages from {args.gold}")
    with _serving(args) as client:
        scored = gold_score(client, passages, schema(), per_section=args.per_section, log=print)
    for line in gold_lines(scored):
        print(line)
    if args.fail_under is not None and scored.f1 < args.fail_under:
        print(f"error: F1 {scored.f1:.2f} is under {args.fail_under:.2f}", file=sys.stderr)
        return 1
    return 0


def _read_run(args: Any) -> int:
    from ml_stack.client.spent import Spent
    from ml_stack.sources import pdf

    progress = Progress(Progress.beside(args.out))
    spent = Spent()
    shape = schema()
    started = time.time()
    code = 0

    with _serving(args) as client:
        for path in args.docs:
            where = Path(path).expanduser()
            if not where.is_file():
                print(f"error: no such document: {where}", file=sys.stderr)
                code = 2
                continue
            began = time.time()
            document = pdf.read(where, images=args.images,
                                chapter=args.chapter or None)
            wanted = pdf.units(document, **({"max_tokens": args.max_tokens}
                                            if args.max_tokens else {}))
            if args.sample:
                wanted = wanted[:args.sample]
            slug = document.slug
            progress.book(slug, title=document.title, path=str(where), sections=len(wanted))
            print(f"{document.title}: {len(document.chapters)} chapter(s), "
                  f"{len(wanted)} unit(s) over {document.page_count} pages, headings from "
                  f"the {document.how}" + (", OpenStax" if document.openstax else "")
                  + f" -- read in {time.time() - began:.0f}s")

            units_by_id = {unit.id: unit for unit in wanted}
            reads: list[dict[str, Any]] = []
            docs: dict[str, Any] = {}
            for unit in wanted:
                if args.resume and progress.done(slug, unit.id):
                    held = _read_json(Path(str(args.out) + f".{slug}.reads.json"))
                    if isinstance(held, dict) and unit.id in held:
                        reads.append(held[unit.id])
                    continue
                row = extract_unit(client, unit, shape, images=args.images,
                                   per_section=args.per_section,
                                   cache_dir=args.cache or None)
                reads.append(asdict(row))
                docs[f"ingest:unit:{unit.id}"] = {
                    "unit": unit.id, "book": slug, "where": unit.where,
                    "title": unit.section_title, "chapter_title": unit.chapter_title,
                    "extracted": row.extracted, "calls": row.calls,
                    "seconds": row.seconds, "error": row.error}
                for call in row.calls:
                    spent.add(_call_of(call))
                progress.note(slug, row)
                print(f"  ch {unit.chapter or '-':>3}  {unit.section or unit.section_title[:12]:<8}"
                      f" {row.seconds:6.1f}s  {row.concepts:>3}c {row.relations:>3}r "
                      f"{row.figures:>2}f" + (f" {row.images}img" if row.images else "")
                      + (f"  {row.error}" if row.error else ""))

            _keep_reads(args.out, slug, reads)
            graph = fold_book(reads, units_by_id, book_title=document.title, log=None)
            counts = write(args.out, graph, book=slug, title=document.title, docs=docs)
            print(f"  {document.title}: {counts['nodes']} nodes, {counts['edges']} edges "
                  f"into {args.out}")

    totals = progress.totals()
    print(f"\n{totals['sections']} section(s) of {totals['books']} book(s) in "
          f"{(time.time() - started) / 60:.1f} min; {spent.calls} calls, "
          f"{spent.prompt_tokens} prompt and {spent.completion_tokens} completion tokens"
          + (f"; {totals['failed']} failed" if totals["failed"] else ""))
    return code


def _call_of(record: Mapping[str, Any]) -> Any:
    from ml_stack.telemetry import Call

    fields = {f for f in Call.__dataclass_fields__}
    return Call(**{k: v for k, v in record.items() if k in fields})


def _keep_reads(out: str | Path, slug: str, reads: Sequence[Mapping[str, Any]]) -> None:
    """Every unit's extraction, beside the store, keyed by unit -- what ``--resume`` folds."""
    path = Path(str(Path(out).expanduser()) + f".{slug}.reads.json")
    held = _read_json(path)
    kept = dict(held) if isinstance(held, dict) else {}
    for read in reads:
        kept[str(read.get("unit") or "")] = dict(read)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kept, indent=1), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - what `detach` re-runs
    raise SystemExit(main())
