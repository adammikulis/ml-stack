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

*A half-read book is readable.* Each unit's extraction lands in
``<store>.<slug>.reads.json`` as it finishes, and the book so far is folded into the store
as the run goes -- see `FOLD_EVERY` -- so a shelf that will take days can be asked
questions today. `Shelf` is how an application reads one::

    shelf = Shelf("./shelf.ladybug")
    for book in shelf.books():
        print(book.slug, book.units, "of", book.wanted, "partial" if book.partial else "")
    graph = shelf.graph("velthorne-open-texts")     # folded from the reads, no store needed
    with shelf.store() as store:                    # read-only, beside the running writer
        store.nodes(kind="concept")

``ml-stack-ingest fold --out STORE`` does the same fold from the shelf into the store on
demand, ``show`` prints what a book holds, and ``stop`` ends a detached run after folding
what it has read.

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

__all__ = ["FOLD_EVERY", "HOME", "INSTRUCTIONS", "PER_SECTION", "VERBS", "Book", "Progress",
           "Scored", "Shelf", "Stopped", "build", "detach", "extract_unit", "fold", "fold_book",
           "fold_into", "gold_score", "main", "read_gold", "schema", "show", "status",
           "unit_of", "units_of", "write"]

HOME = Path(os.environ.get("MLSTACK_INGEST_HOME") or "~/.ml-stack/ingest").expanduser()
"""Where a detached run's log and its record of itself live. Not the store: the store is
the caller's, named by ``--out``."""

PER_SECTION = 1200.0    # a ceiling, not a budget: a legitimate unit writes 6k tokens at ~50 tok/s
GIVE_UP = 2             # failed attempts before --resume leaves a unit alone
"""The most one section may take before it is recorded as timed out and the next is read."""

FOLD_EVERY = 25
"""Units read between folds of a book into the store.

A fold costs what `entities.fold_names` costs, which is every concept name against every
other: measured over invented units, 400 units of a 12-word vocabulary folded and wrote in
3.8 s, and 300 units of a 2,700-word one took 44 s to fold and 9 s to write. It grows with
the square of the vocabulary rather than with the units, so the interval is a real cost and
not a formality. A chapter's end folds once this many units have gone by since the last
fold; a chapter longer than twice this folds inside itself; the end of a book and a stop
always fold."""

IMAGES_PER_SECTION = 4
"""How many of a section's figures are shown to the model at once. A section of a biology
textbook has a dozen plates; a dozen images is a prompt of images with a paragraph in it."""


class Stopped(BaseException):
    """SIGTERM reached a run. Not an `Exception`: one section's extraction catches every
    `Exception` there is, and a stop is not one section's failure."""


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


INVERSES: dict[str, frozenset[str]] = {
    "created_by": frozenset({"authored", "wrote", "author_of", "created", "built", "proposed",
                             "discovered", "invented", "founded"}),
    "adopted_by": frozenset({"adopted", "enacted", "ratified"}),
    "has_part": frozenset({"part_of"}),
    "part_of": frozenset({"has_part", "contains", "has"}),
    "causes": frozenset({"caused_by"}),
    "produces": frozenset({"produced_by", "made_by"}),
    "precedes": frozenset({"follows", "after"}),
    "requires": frozenset({"required_by", "enables"}),
}
"""What each verb says when the ends are swapped: `X created_by Y` is `Y authored X`.

A gold set written by hand names facts in whichever direction the sentence did, and a
closed vocabulary says each one in one direction only. Matching the flipped triple through
this map is how the gate tells 'the vocabulary has no word for it' from 'it has the word,
pointing the other way' -- the first gold run counted 'the author authored the charter'
as unsayable when the model had said `charter created_by the author`, the same fact."""


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
    "verb phrases the schema allows and no other. State only what the section states. "
    "Both ends are concept names -- never a clause or a phrase such as 'lights the "
    "system'. When no verb says what the text says, leave the relation out: a relation "
    "with the wrong verb is worse than none.\n"
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
    raw: str = ""            # what the model wrote when it failed, whole, for reading later
    run: str = ""            # the run node that read it -- model, build, head, hashes, when
    extracted: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _Unit:
    """A unit as its reads file remembers it: the provenance `build` needs, and no PDF."""

    id: str
    book: str
    chapter: str
    section: str
    section_title: str
    first_page: int
    last_page: int

    @property
    def where(self) -> dict[str, Any]:
        return {"book": self.book, "chapter": self.chapter, "section": self.section,
                "page": self.first_page, "pages": [self.first_page, self.last_page],
                "unit": self.id}


def unit_of(read: Mapping[str, Any]) -> _Unit:
    """One row of a reads file as something `build` and `fold_book` can take.

    The unit id is the one on the row rather than one recomputed, so a section split into
    parts keeps the id its provenance already names.
    """
    pages = list(read.get("pages") or ())
    return _Unit(id=str(read.get("unit") or ""), book=str(read.get("book") or ""),
                 chapter=str(read.get("chapter") or ""), section=str(read.get("section") or ""),
                 section_title=str(read.get("title") or ""),
                 first_page=int(pages[0]) if pages else 0,
                 last_page=int(pages[-1]) if pages else 0)


def units_of(reads: Iterable[Mapping[str, Any]]) -> dict[str, _Unit]:
    """``{unit id: unit}`` for a book's reads -- what `fold_book` wants, from the shelf alone."""
    out: dict[str, _Unit] = {}
    for read in reads:
        unit = unit_of(read)
        if unit.id:
            out[unit.id] = unit
    return out


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
        # a unit that ran to the ceiling twice on the first shelf night left nothing to read
        # but 120 characters; the whole reply is kept beside the unit, so the next person
        # can see whether it looped or rambled without spending ten minutes of GPU again
        row.raw = str(getattr(exc, "body", "") or "")
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
    """One extraction as ``(nodes by id, edges by triple)``, every one pointing at where it came from.

    Names are the ids: two sections that both name the same concept are one node whose
    provenance lists both, which is the whole reason to read a book section by section
    rather than a page at a time. Provenance is pointers and nothing else -- unit ids --
    because Adam: "provenance should always be pointers to the textbook". The unit
    document holds the book, chapter, section and pages, and points at the run that read
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


def fold_book(reads: Iterable[Mapping[str, Any]], units_by_id: Mapping[str, Any], *,
              book_title: str = "", log: Callable[[str], None] | None = None
              ) -> dict[str, Any]:
    """Every section of one book, folded into one graph.

    A read that carries an `error` contributes nothing: what a failed extraction left is
    kept for reading, not for believing.

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

    weight = {node["label"]: int(node["mentions"]) for node in nodes.values()
              if node["kind"] != "figure"}
    canonical, name_folds = fold_names(weight, plurals(weight), log=log, label="concepts",
                                       settles="both spellings stay, and the book is right")
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


def write(out: str | Path, graph: Mapping[str, Any], *, book: str, title: str,
          docs: Mapping[str, Any] | None = None, replace: bool = False,
          keep_units: Iterable[str] | None = None) -> dict[str, int]:
    """One book's graph and its raw extractions into the store, and read back before returning.

    The book itself is a node, so a store holding a shelf can still be asked what came out
    of which book; every concept hangs off it by ``read_from``.

    An upsert, and nothing more. Adam: "if the book already exists, it should append new
    nodes/connect new edges. additive." A node the store lacks is added; one it has takes
    the fold's mentions, aliases, definition and provenance (the fold is over every read
    so far, so those only grow); an edge likewise; nothing is merged and nothing is
    removed -- joining names is the hygiene pass's job (`ml_stack.graph.tidy`). Folding
    twice with nothing read in between changes nothing.

    ``replace`` is `fold --rebuild`: the book's own nodes and edges out first, then the
    full fold from its reads -- the one path that removes anything, for after a fix that
    changed what a read means. ``keep_units`` names the units the book now has: any
    ``ingest:unit:`` document of this book outside it goes with the nodes.
    """
    from ml_stack.graph.store import GraphStore

    book_id = f"book:{book}"
    nodes = [{"id": book_id, "kind": "book", "label": title or book, "mentions": 1,
              "attrs": {"book": book}}, *graph.get("nodes", ())]
    edges = [*graph.get("edges", ()),
             *({"source": node["id"], "rel": "read_from", "target": book_id, "weight": 1}
               for node in graph.get("nodes", ()))]
    with GraphStore(out) as store:
        if replace:
            _drop_book(store, book, keep_units=keep_units)
        counts = store.write({"nodes": nodes, "edges": edges})
        for key, value in (docs or {}).items():
            store.put_doc(key, value)
        store.put_doc(f"ingest:folds:{book}", dict(graph.get("folds") or {}))
        back = store.query("MATCH (n:Node {id: $id}) RETURN n.id AS id", {"id": book_id})
    if not back:
        raise RuntimeError(f"{book_id} was written to {out} and did not come back")
    return counts


def _drop_book(store: Any, book: str, *, keep_units: Iterable[str] | None = None) -> int:
    """Everything the store holds for one book, out: nodes read only from it, and its edges.

    A node is this book's when a ``read_from`` edge joins it to ``book:<slug>``; one that
    also reads from another book stays, and only the edges this book put on it go. The book
    node itself stays and is written again.
    """
    book_id = f"book:{book}"
    if keep_units is not None:
        # every unit id starts with the book's slug, so the stale documents are a prefix
        # away and no document has to be read to find them
        held, prefix = {f"ingest:unit:{u}" for u in keep_units}, f"ingest:unit:{book}:"
        for key in store.doc_keys():
            if key.startswith(prefix) and key not in held:
                store.delete_doc(key)
    read_from = store.edges("read_from")
    mine = {e["source"] for e in read_from if e["target"] == book_id}
    if not mine:
        return 0
    shared = {e["source"] for e in read_from if e["target"] != book_id}
    gone = store.drop(sorted(mine - shared), force=True)
    prefix = f"{book}:"
    for edge in store.edges():
        if edge["rel"] == "read_from" and edge["target"] == book_id:
            store.remove_edge(edge["source"], edge["target"], edge["rel"])
        elif any(str(u).startswith(prefix) for u in (edge.get("provenance") or ())):
            # this book's edge, by its pointers; an edge two books both stated keeps
            # the other book's pointer and goes when that book is rebuilt
            store.remove_edge(edge["source"], edge["target"], edge["rel"])
    return gone


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
        """Finished and kept, or given up on -- a unit that failed is written down, so
        `status` can say so, and is read again by `--resume` until it has failed `GIVE_UP`
        times; after that it is left, because one APBiology unit ran to the ceiling on four
        resumes in a row at twelve minutes each."""
        entry = (self.state["books"].get(slug, {}).get("done") or {}).get(unit)
        if not isinstance(entry, dict):
            return False
        if not entry.get("error"):
            return True
        return int(entry.get("attempts") or 1) >= GIVE_UP

    def note(self, slug: str, read: Read) -> None:
        """Write one finished unit down, at once: a run killed mid-book resumes from here.

        An unreachable server is not the unit's failure and does not count as one of its
        attempts: the read is written down so `status` can say what happened, and the
        next `--resume` reads it again as if for the first time."""
        before = (self.book(slug)["done"].get(read.unit) or {})
        # an entry from before attempts were counted is one attempt; zero is a number
        attempts = int(before["attempts"]) if "attempts" in before else (1 if before else 0)
        if not read.error.startswith("ServerUnreachable"):
            attempts += 1
        self.book(slug)["done"][read.unit] = {
            "seconds": read.seconds, "concepts": read.concepts, "relations": read.relations,
            "figures": read.figures, "images": read.images, "error": read.error,
            "attempts": attempts, "at": time.strftime("%FT%T")}
        self.save()

    def folded(self, slug: str, *, units: int, nodes: int, edges: int) -> None:
        """Write down that the book is in the store as of ``units`` units read."""
        held = self.book(slug)
        held["folded_at"] = int(units)
        held["folded_nodes"] = int(nodes)
        held["folded_edges"] = int(edges)
        held["folded"] = time.strftime("%FT%T")
        self.save()

    def save(self) -> None:
        _write_json(self.path, self.state)

    def totals(self) -> dict[str, Any]:
        """Books, sections done of how many, seconds spent, and sections a minute."""
        done = seconds = wanted = failed = given_up = 0
        for book in self.state["books"].values():
            entries = (book.get("done") or {}).values()
            done += len(entries)
            wanted += int(book.get("sections") or 0)
            seconds += sum(float(e.get("seconds") or 0.0) for e in entries)
            failed += sum(1 for e in entries if e.get("error"))
            given_up += sum(1 for e in entries if e.get("error")
                            and int(e.get("attempts") or 1) >= GIVE_UP)
        return {"books": len(self.state["books"]), "sections": done, "of": wanted,
                "failed": failed, "given_up": given_up, "seconds": round(seconds, 1),
                "per_section": round(seconds / done, 1) if done else 0.0,
                "started": self.state.get("started", "")}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, value: Any) -> None:
    """JSON into ``path`` through a temporary file and a rename: a kill mid-write leaves
    the file that was there, never half of the one being written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        temp.write_text(json.dumps(value, indent=1), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def status(out: str | Path, *, say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest status``: sections done, what failed, what is folded, and how long left.

    The estimate is the units still to read at the rate this shelf has actually measured,
    so it is honest about this machine and this model rather than about any other.
    """
    where = Progress.beside(out)
    if not where.is_file():
        say(f"nothing ingested into {out}: no {where.name}")
        return 1
    progress = Progress(where)
    totals = progress.totals()
    say(f"{out}: {totals['sections']} of {totals['of']} sections in {totals['books']} book(s), "
        f"started {totals['started']}")
    left = 0.0
    for slug, book in sorted(progress.state["books"].items()):
        entries = book.get("done") or {}
        spent = sum(float(e.get("seconds") or 0.0) for e in entries.values())
        broke = sum(1 for e in entries.values() if e.get("error"))
        wanted = int(book.get("sections") or 0)
        rate = spent / len(entries) if entries else totals["per_section"]
        remaining = max(wanted - len(entries), 0) * rate
        left += remaining
        say(f"  {slug:<28} {len(entries):>4} / {book.get('sections') or '?':<5} "
            f"{spent / 60:6.1f} min" + (f"  {broke} failed" if broke else "")
            + (f"  ~{_for_long(remaining)} left" if remaining else ""))
        folded = int(book.get("folded_at") or 0)
        if folded:
            say(f"      in store: {int(book.get('folded_nodes') or 0)} nodes, "
                f"{int(book.get('folded_edges') or 0)} edges, "
                f"folded at unit {folded} of {wanted or '?'}")
        elif _book_in_store(out, slug):
            say("      in store: folded by an earlier run (units unknown)")
        else:
            say("      in store: nothing folded yet")
    if totals["sections"]:
        rate = (f", {3600 / totals['per_section']:.0f} sections/hour"
                if totals["per_section"] else "")
        say(f"  {totals['per_section']:.1f} s/section{rate}; "
            f"{totals['seconds'] / 3600:.1f} h spent"
            + (f", ~{_for_long(left)} left" if left else "")
            + (f", {totals['failed']} failed" if totals["failed"] else "")
            + (f" ({totals['given_up']} given up after {GIVE_UP} tries; the reply each wrote "
               f"is `raw` in the reads file)" if totals["given_up"] else ""))
    return 0


# -- the shelf: what has been read, folded while it is still being read -------------------


def reads_path(out: str | Path, slug: str) -> Path:
    """Where one book's extractions are kept, beside the store."""
    return Path(str(Path(out).expanduser()) + f".{slug}.reads.json")


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
    folded_at: int = 0       # units read when the store was last written
    folded_nodes: int = 0
    folded_edges: int = 0

    @property
    def partial(self) -> bool:
        """Units are still to be read."""
        return self.wanted > self.units

    @property
    def per_unit(self) -> float:
        return round(self.seconds / self.units, 1) if self.units else 0.0

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
            done = (held.get("done") or {}).values()
            out.append(Book(
                slug=slug, title=str(held.get("title") or ""), path=str(held.get("path") or ""),
                units=len(rows), read=sum(1 for r in rows if not r.get("error")),
                failed=sum(1 for r in rows if r.get("error")),
                given_up=sum(1 for e in done if e.get("error")
                             and int(e.get("attempts") or 1) >= GIVE_UP),
                wanted=int(held.get("sections") or 0),
                seconds=round(sum(float(r.get("seconds") or 0.0) for r in rows), 1),
                folded_at=int(held.get("folded_at") or 0),
                folded_nodes=int(held.get("folded_nodes") or 0),
                folded_edges=int(held.get("folded_edges") or 0)))
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


def _unit_docs(rows: Iterable[Mapping[str, Any]], slug: str) -> dict[str, Any]:
    """The ``ingest:unit:`` document for each read: its provenance, what it said, what it cost."""
    out: dict[str, Any] = {}
    for row in rows:
        unit = unit_of(row)
        if not unit.id:
            continue
        out[f"ingest:unit:{unit.id}"] = {
            "unit": unit.id, "book": slug, "where": unit.where, "title": unit.section_title,
            "chapter_title": str(row.get("chapter_title") or ""),
            "run": str(row.get("run") or ""),
            "extracted": row.get("extracted") or {}, "calls": list(row.get("calls") or ()),
            "seconds": float(row.get("seconds") or 0.0), "error": str(row.get("error") or "")}
    return out


def fold_into(out: str | Path, slug: str, *, title: str = "",
              reads: Sequence[Mapping[str, Any]] | None = None,
              units_by_id: Mapping[str, Any] | None = None,
              progress: Progress | None = None, rebuild: bool = False,
              dry_run: bool = False,
              log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Fold one book's reads so far and upsert them into the store.

    Returns the counts: units read, nodes, edges, folds, whether the book is partial, and
    ``new_nodes``/``new_edges`` -- what the store lacked before this fold. Idempotent:
    folding twice with nothing read in between adds nothing. ``dry_run`` computes all of
    that and writes nothing; ``rebuild`` drops the book's own nodes and edges first and is
    the only way anything leaves the store.
    """
    shelf = Shelf(out)
    rows = list(reads) if reads is not None else shelf.reads(slug)
    held = shelf.progress.state["books"].get(slug) or {}
    name = title or str(held.get("title") or "") or slug
    units = {**units_of(rows), **dict(units_by_id or {})}
    graph = fold_book(rows, units, book_title=name, log=log)
    new_nodes, new_edges = _missing_from(out, graph)
    folds = len(graph["folds"].get("concepts") or ()) + len(graph["folds"].get("relations") or ())
    wanted = int(held.get("sections") or 0)
    got = {"book": slug, "title": name, "units": len(rows),
           "read": sum(1 for r in rows if not r.get("error")),
           "wanted": wanted, "nodes": len(graph["nodes"]), "edges": len(graph["edges"]),
           "new_nodes": new_nodes, "new_edges": new_edges,
           "folds": folds, "partial": wanted > len(rows)}
    if dry_run:
        return got
    docs = _unit_docs(rows, slug)
    counts = write(out, graph, book=slug, title=name, docs=docs, replace=rebuild,
                   keep_units=set(units) if rebuild else None)
    record = progress if progress is not None else shelf.progress
    record.book(slug, title=name)
    record.folded(slug, units=len(rows), nodes=counts["nodes"], edges=counts["edges"])
    return got


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


def write_run(out: str | Path, record: Mapping[str, Any]) -> str:
    """The hidden ``run`` node one ingest run hangs its units on: the model that read them,
    its build, head, sampling, the schema and instructions it read with, the version, the
    host, when. Written once; every unit document points at it (``run``), and every node
    and edge points at its units (``provenance``), so "which model extracted this" is a
    walk along pointers rather than a string copied a thousand times. Adam: "a hidden
    node for each metadata with hidden edges ... probably use less mem to use pointers"."""
    from ml_stack.graph.store import GraphStore

    run_id = str(record.get("id") or f"run:{time.strftime('%Y%m%dT%H%M%S')}")
    node = {"id": run_id, "kind": "run", "label": str(record.get("label") or run_id),
            "mentions": 1, "attrs": {**{k: v for k, v in record.items() if k != "id"},
                                     "hidden": True}}
    with GraphStore(out) as store:
        store.write({"nodes": [node], "edges": []})
    return run_id


def run_record(args: Any, *, model: str = "", serving: str = "") -> dict[str, Any]:
    """What one run read with, for `write_run`: everything a person would ask later."""
    import hashlib
    import platform as _platform
    from importlib import metadata

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    try:
        version = metadata.version("ml-stack")
    except metadata.PackageNotFoundError:  # pragma: no cover - a checkout without install
        version = "unknown"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return {"id": f"run:{stamp}", "label": f"ingest {stamp}",
            "model": model or str(getattr(args, "model", "") or ""),
            "serving": serving, "images": bool(getattr(args, "images", False)),
            "n_max": getattr(args, "n_max", None),
            "sampling": {k: v for k, v in (("temperature", getattr(args, "temperature", None)),
                                           ("top_p", getattr(args, "top_p", None)),
                                           ("top_k", getattr(args, "top_k", None)),
                                           ("min_p", getattr(args, "min_p", None)))
                         if v is not None},
            "schema_sha": sha(json.dumps(schema(), sort_keys=True)),
            "instructions_sha": sha(INSTRUCTIONS + WITH_IMAGES),
            "ml_stack": version, "host": _platform.node(),
            "started": time.strftime("%FT%T"), "argv": list(sys.argv[1:])}


def located(store: Any, thing: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Where a node or edge was read: ``[{book, title, chapter, section, pages, unit}]``,
    one per unit in its provenance, resolved through the unit documents -- the pointers
    turned back into pages. A unit the store no longer holds comes back as its id alone."""
    out = []
    for unit_id in thing.get("provenance") or ():
        doc = store.get_doc(f"ingest:unit:{unit_id}") or {}
        where = doc.get("where") or {}
        out.append({"unit": unit_id, "book": str(doc.get("book") or where.get("book") or ""),
                    "title": str(doc.get("title") or ""),
                    "chapter": str(where.get("chapter") or ""),
                    "section": str(where.get("section") or ""),
                    "pages": list(where.get("pages") or ()) or ([where["page"]]
                                                                if where.get("page") else [])})
    return out


def origin(store: Any, thing: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Which run -- and so which model, build, head and instructions -- read a node or an
    edge: one record per distinct run behind its provenance, oldest first, each naming the
    units it read. A unit read before runs were recorded says so (``run: ""``)."""
    runs: dict[str, dict[str, Any]] = {}
    for unit_id in thing.get("provenance") or ():
        doc = store.get_doc(f"ingest:unit:{unit_id}") or {}
        run_id = str(doc.get("run") or "")
        held = runs.get(run_id)
        if held is None:
            attrs = {}
            if run_id:
                found = store.nodes(kind="run")
                attrs = next((n.get("attrs") or {} for n in found if n["id"] == run_id), {})
            held = runs[run_id] = {"run": run_id, "model": str(attrs.get("model") or ""),
                                   "serving": str(attrs.get("serving") or ""),
                                   "started": str(attrs.get("started") or ""),
                                   "units": []}
        held["units"].append(unit_id)
    return sorted(runs.values(), key=lambda r: r["started"])


def fold(out: str | Path, *, book: str = "", rebuild: bool = False, dry_run: bool = False,
         say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest fold --out STORE``: every book that has reads, upserted into the store.

    A book part-read is written as far as it has been read, so a shelf that will take days
    is answerable today. ``--dry-run`` says what each fold would add and writes nothing;
    ``--rebuild`` drops each book's own nodes and edges first -- the only removal there is.
    """
    shelf = Shelf(out)
    wanted = [b for b in shelf.books() if b.units and (not book or b.slug == book)]
    if not wanted:
        say(f"nothing to fold into {out}"
            + (f": no reads for {book}" if book else f": no {Path(out).name}.*.reads.json"))
        return 1
    for held in wanted:
        got = fold_into(out, held.slug, title=held.title, progress=shelf.progress,
                        rebuild=rebuild, dry_run=dry_run)
        what = ("would add" if dry_run else "rebuilt with" if rebuild else "added")
        say(f"{got['title']}: {got['read']} of {got['wanted'] or '?'} units read, "
            f"{got['nodes']} nodes, {got['edges']} edges, {got['folds']} fold(s); "
            f"{what} {got['new_nodes']} node(s) and {got['new_edges']} edge(s)"
            + (f" into {out}" if not dry_run else "")
            + ("  -- partial" if got["partial"] else ""))
    return 0


def _book_in_store(out: str | Path, slug: str) -> bool:
    """Whether the store holds ``book:<slug>`` -- written by a run before folds were
    recorded, so the progress file says nothing about it."""
    from ml_stack.graph.store import GraphStore

    if not Path(out).expanduser().exists():
        return False
    try:
        with GraphStore(out, read_only=True) as store:
            return any(n["id"] == f"book:{slug}" for n in store.nodes(kind="book"))
    except Exception:  # noqa: BLE001 - a store a writer holds, or none; say nothing
        return False


def _for_long(seconds: float) -> str:
    """A duration a person reads at a glance."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


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
            flipped = {verb for verb, other_way in INVERSES.items() if words & other_way}
            if vocabulary and not ((words | flipped) & vocabulary):
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
    """The extracted relation says the gold triple -- as written, or the other way round
    through `INVERSES` (`charter created_by orlan vesk` says `orlan vesk authored charter`)."""
    subject, subject_aliases = str(triple.get("subject") or ""), _names(triple.get("subject_aliases"))
    obj, obj_aliases = str(triple.get("object") or ""), _names(triple.get("object_aliases"))
    rel = str(said.get("rel") or "")
    if (_same(str(said.get("from") or ""), subject, subject_aliases)
            and _same(rel, str(triple.get("predicate") or ""),
                      _names(triple.get("predicate_aliases")))
            and _same(str(said.get("to") or ""), obj, obj_aliases)):
        return True
    words = {str(triple.get("predicate") or ""), *_names(triple.get("predicate_aliases"))}
    if not (words & INVERSES.get(rel, frozenset())):
        return False
    return (_same(str(said.get("from") or ""), obj, obj_aliases)
            and _same(str(said.get("to") or ""), subject, subject_aliases))


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
    # One slot, one unit at a time. Adam: "we shouldn't be handling parallel requests while
    # extracting. In fact, we should never be splitting the GPU like that" -- and the shelf
    # measured it: one worker read a unit in 86 s, two workers sharing the model averaged
    # 140 s each, slower in aggregate as well as apiece. The whole --context is the one
    # seat's: a 2,500-token unit with four figures through the projector and a reply of
    # several thousand tokens overran a 16k seat on the first night.
    seats = 1
    lease: dict[str, Any] = {"port": args.serve_port, "context": int(args.context),
                             "parallel": seats, "timeout": 900.0, "cache_reuse": 256,
                             "warmup": False}
    manager = None
    if getattr(args, "profile", True):
        from ml_stack.serve.profile import profile_for, said

        measured = profile_for(str(found))
        if measured is not None:
            shape = measured.shape(port=args.serve_port, seats=seats)
            lease = {**lease, **{k: v for k, v in shape.lease().items()
                                 if k not in ("port", "parallel")}}
            lease["context"] = max(int(args.context), int(lease.get("context") or 0))
            manager = shape.manager()
            say(f"    serving in its measured shape: {said(measured)}")
    if not args.images:
        lease.pop("mmproj", None)
    if getattr(args, "n_max", None) is not None:
        # Extraction copies definitions out of the page: the head's guesses were accepted
        # 97% of the time on a biology chapter against ~75% answering questions, so the
        # length that measured best for answering is not the length for this. Measured
        # here, per workload, with the same command that reads the shelf.
        if not lease.get("draft"):
            say("--n-max: no draft head is being served, so there is no draft to lengthen")
        else:
            lease["spec_draft_max"] = int(args.n_max)
            say(f"    draft length {args.n_max} over the profile's")
    began = time.time()
    with serve(found, manager=manager, **lease) as server:
        say(f"    up in {time.time() - began:.0f}s")
        yield Client(server.base_url, timeout=args.per_section, n_predict=args.n_predict,
                     **sampling)


def _alive(client: Any) -> bool:
    """Whether the run's server still answers at all."""
    from ml_stack.client import is_healthy

    base_url = str(getattr(client, "base_url", "") or "")
    return bool(base_url) and is_healthy(base_url, timeout=3.0)


def _serving_said(args: Any) -> str:
    """The measured shape a --model is served in, as one line, for the run record."""
    if not getattr(args, "model", ""):
        return f"base_url {getattr(args, 'base_url', '')}"
    try:
        from ml_stack.serve.profile import profile_for, said

        measured = profile_for(str(_find_model(args.model)))
        return said(measured) if measured is not None else "bare"
    except Exception:  # noqa: BLE001 - a record, never a reason not to read
        return "unknown"


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


def retry(out: str | Path, *, say: Callable[[str], None] = print) -> int:
    """``ml-stack-ingest retry --out STORE``: the units given up on are read again by the
    next ``--resume`` -- for after the fix that made them fail is in."""
    where = Progress.beside(out)
    if not where.is_file():
        say(f"nothing ingested into {out}: no {where.name}")
        return 1
    progress = Progress(where)
    freed = 0
    for book in progress.state["books"].values():
        for entry in (book.get("done") or {}).values():
            if entry.get("error") and int(entry.get("attempts") or 1) >= GIVE_UP:
                entry["attempts"] = 0
                freed += 1
    progress.save()
    say(f"{freed} unit(s) will be read again on the next --resume")
    return 0


STOP_WAIT = 900.0       # a fold over a 7,000-node book took minutes on the way out


def stop(*, say: Callable[[str], None] = print, home: Path | None = None,
         wait: float = STOP_WAIT) -> int:
    """``ml-stack-ingest stop``: end the detached run and wait for its last fold to land.

    The run folds the book it is on before it exits -- minutes, for a book of thousands of
    nodes -- so this waits up to ``wait`` seconds for the process to go, saying so every
    half minute, and then says whether the store moved. The record is kept while the run
    is still ending, so `detach` refuses to start another beside it. Whatever was read is
    kept either way, and the same command with ``--resume`` reads on.
    """
    import signal

    record = (home or HOME) / "ingesting.json"
    held = _read_json(record)
    held = held if isinstance(held, dict) else {}
    pid = int(held.get("pid") or 0)
    if not pid:
        say("no detached ingest is recorded on this machine")
        return 1
    out = _out_of(held.get("argv") or ())
    before = _folded_at(out)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        say(f"the recorded ingest (pid {pid}) had already ended")
        record.unlink(missing_ok=True)
        return 1
    waited = 0.0
    ended = _wait_for(pid, min(wait, 30.0))
    waited += min(wait, 30.0)
    while not ended and waited < wait:
        say(f"  still folding after {waited:.0f}s (pid {pid})")
        ended = _wait_for(pid, min(30.0, wait - waited))
        waited += 30.0
    if not ended:
        say(f"asked the detached ingest (pid {pid}) to stop; it had not ended after "
            f"{wait:.0f}s, so its last fold is still being written -- its record stays, and "
            f"no new run starts beside it until it has")
        return 1
    record.unlink(missing_ok=True)
    after = _folded_at(out)
    moved = [f"{slug} at unit {units}" for slug, units in sorted(after.items())
             if units != before.get(slug)]
    say(f"stopped the detached ingest (pid {pid}); "
        + (f"folded {', '.join(moved)} into {out}" if moved
           else f"nothing new was folded into {out}" if out
           else "its units so far are kept")
        + "; the same command with --resume reads on")
    return 0


def _recorded_alive(home: Path | None = None) -> int:
    """The pid of the detached run this machine records, when it is still alive; else 0."""
    held = _read_json((home or HOME) / "ingesting.json")
    pid = int(held.get("pid") or 0) if isinstance(held, dict) else 0
    if not pid:
        return 0
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return 0
    except PermissionError:  # pragma: no cover - alive, somebody else's
        return pid
    return pid


def _out_of(argv: Iterable[str]) -> str:
    """The ``--out`` a recorded run was started with."""
    argv = list(argv)
    for index, word in enumerate(argv):
        if word == "--out" and index + 1 < len(argv):
            return argv[index + 1]
        if word.startswith("--out="):
            return word[len("--out="):]
    return ""


def _folded_at(out: str | Path) -> dict[str, int]:
    """``{book: units read when it was last folded}``, from the progress file."""
    if not out:
        return {}
    held = _read_json(Progress.beside(out))
    books = held.get("books") if isinstance(held, dict) else None
    if not isinstance(books, dict):
        return {}
    return {slug: int((book or {}).get("folded_at") or 0) for slug, book in books.items()}


def _wait_for(pid: int, seconds: float) -> bool:
    """Whether ``pid`` ended within ``seconds``."""
    deadline = time.time() + max(seconds, 0.0)
    while not _ended(pid):
        if time.time() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _ended(pid: int) -> bool:
    """Whether ``pid`` is gone. A process this one started is reaped rather than waited on:
    a child that has exited answers ``kill(pid, 0)`` until somebody collects it."""
    try:
        collected, _ = os.waitpid(pid, os.WNOHANG)
    except (OSError, AttributeError, ValueError):
        collected = 0
    if collected:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


@contextmanager
def _stopping() -> Any:
    """Turn SIGTERM into `Stopped` for the length of a run, and put the old handler back."""
    import signal

    def raise_it(*_: Any) -> None:
        raise Stopped("SIGTERM")

    try:
        before = signal.signal(signal.SIGTERM, raise_it)
    except ValueError:            # not the main thread: nothing to install onto
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, before)


# -- the command ---------------------------------------------------------------------------------


_WORDS = ("status", "show", "fold", "retry", "tidy")
"""What a run does instead of reading a document, when one is named where a PDF would be."""


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ml-stack-ingest", allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Read documents into a knowledge graph, section by section: "
                    "`ml-stack-ingest BOOK.pdf ... --out STORE`.",
        epilog="Instead of documents, one of these words:\n"
               "  status   how far the run into --out has got, what failed, what is in the\n"
               "           store, and how long the rest will take\n"
               "  show     what was read: concepts, relations and the folds each book made\n"
               "  fold     fold every book that has reads -- part-read ones too -- into the\n"
               "           store, replacing what the store held for it\n"
               "  retry    let the units given up on be read again by the next --resume\n"
               "  stop     end the detached run, after it has folded what it has read\n")
    ap.add_argument("docs", nargs="*", metavar="DOC",
                    help="the PDFs to read; or one of `status`, `show`, `fold`, `retry`, "
                         "`stop` (see below), which does that and stops")
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
                         "whole path before a shelf is spent on it; with `show`, how many "
                         "concepts and relations to print per book (default 5)")
    ap.add_argument("--apply", action="store_true",
                    help="with tidy: write the merges, folds and flags; without it, say what "
                         "would be done")
    ap.add_argument("--written", default="", metavar="FILE",
                    help="with tidy: a JSON object {name: the name it is} -- the possible "
                         "duplicates a person settled")
    ap.add_argument("--rebuild", action="store_true",
                    help="with fold: drop each book's own nodes and edges first and write the "
                         "full fold from its reads -- the only way anything leaves the store, "
                         "for after a fix that changed what a read means")
    ap.add_argument("--dry-run", action="store_true",
                    help="with fold: say what each fold would add and write nothing")
    ap.add_argument("--book", default="", metavar="SLUG",
                    help="with `show` or `fold`, only this book")
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
    ap.add_argument("--n-max", type=int, default=None, metavar="N",
                    help="tokens the draft head guesses ahead each step, over the profile's "
                         "measured length -- extraction accepts far more of them than "
                         "answering does, so measure it here (default: the profile's)")
    ap.add_argument("--per-section", type=float, default=PER_SECTION, metavar="SECONDS",
                    help="the most one section may take (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=0, metavar="N",
                    help="where a long section is split, in tokens (default: the reader's "
                         "own 2500)")
    ap.add_argument("--n-predict", type=int, default=16384, metavar="N",
                    help="the answer's ceiling; a ceiling is not a budget, and a low one "
                         "truncates the extraction (default: %(default)s)")
    ap.add_argument("--context", type=int, default=32768, metavar="N",
                    help="context of the one slot a --model is served with -- extraction "
                         "reads one unit at a time and never splits the GPU (default: "
                         "%(default)s)")
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

    if args.docs[:1] == ["stop"]:
        return stop()
    word = args.docs[0] if args.docs[:1] and args.docs[0] in _WORDS else ""
    if word:
        if not args.out:
            print(f"error: {word} needs --out STORE", file=sys.stderr)
            return 2
        if word == "fold":
            return fold(args.out, book=args.book, rebuild=args.rebuild, dry_run=args.dry_run)
        if word == "show":
            return show(args.out, book=args.book, most=args.sample or 5)
        if word == "retry":
            return retry(args.out)
        if word == "tidy":
            # the hygiene pass is graph.tidy's -- a shelf, a Slack community, any store --
            # and lives beside the fold here only so the shelf's commands are in one place
            from ml_stack.graph.tidy import tidy as hygiene
            from ml_stack.graph.tidy import written_from

            hygiene(args.out, dry_run=not args.apply, written=written_from(args.written),
                    log=print)
            return 0
        return status(args.out)
    if not args.docs and not args.gold:
        print("error: name at least one document, or --gold FILE", file=sys.stderr)
        return 2
    if args.docs and not args.out:
        print("error: reading a document needs --out STORE to write it into", file=sys.stderr)
        return 2
    if args.detach:
        alive = _recorded_alive()
        if alive:
            # one run at a time, on one model, into one store: a second run beside one
            # that is still folding its way out adopted its server and lost it when the
            # first finished (2026-09-03). The record is the lease; it is cleared when
            # the run ends or `stop` sees it end
            print(f"error: a detached ingest (pid {alive}) is still running or still "
                  f"folding on its way out; `ml-stack-ingest stop` waits for it", file=sys.stderr)
            return 2
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
    stopped = False

    def keep(slug: str, title: str, rows: Sequence[Mapping[str, Any]],
             units_by_id: Mapping[str, Any]) -> dict[str, Any]:
        got = fold_into(args.out, slug, title=title, reads=rows, units_by_id=units_by_id,
                        progress=progress)
        print(f"  folded {slug} at unit {got['units']}: {got['nodes']} nodes, "
              f"{got['edges']} edges" + (" (partial)" if got["partial"] else ""))
        return got

    try:
        with _stopping(), _serving(args) as client:
            run_id = write_run(args.out, run_record(args, serving=_serving_said(args)))
            print(f"  run {run_id}: units read now point at it")
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
                progress.book(slug, title=document.title, path=str(where),
                              sections=len(wanted))
                banks = pdf.question_banks(document, **({"max_tokens": args.max_tokens}
                                                        if args.max_tokens else {}))
                print(f"{document.title}: {len(document.chapters)} chapter(s), "
                      f"{len(wanted)} unit(s) over {document.page_count} pages, headings from "
                      f"the {document.how}" + (", OpenStax" if document.openstax else "")
                      + (f", {banks} question-bank part(s) skipped" if banks else "")
                      + f" -- read in {time.time() - began:.0f}s")

                units_by_id = {unit.id: unit for unit in wanted}
                held_reads = _read_json(reads_path(args.out, slug))
                held_reads = held_reads if isinstance(held_reads, dict) else {}
                reads_by_unit: dict[str, dict[str, Any]] = {}
                to_read = []
                for unit in wanted:
                    if args.resume and progress.done(slug, unit.id) and unit.id in held_reads:
                        reads_by_unit[unit.id] = held_reads[unit.id]
                        continue
                    to_read.append(unit)

                # one at a time, on the one slot: each unit is written down the moment it
                # finishes, so a run killed mid-book loses at most the unit in flight, and
                # the book is folded into the store as it goes, so a shelf that will take
                # days can be asked questions today
                since = 0
                try:
                    for index, unit in enumerate(to_read):
                        row = extract_unit(client, unit, shape, images=args.images,
                                           per_section=args.per_section,
                                           cache_dir=args.cache or None)
                        row.run = run_id
                        reads_by_unit[unit.id] = asdict(row)
                        for call in row.calls:
                            spent.add(_call_of(call))
                        progress.note(slug, row)
                        _keep_reads(args.out, slug, [reads_by_unit[unit.id]])
                        if row.error.startswith("ServerUnreachable") and not _alive(client):
                            # the server is gone -- killed, crashed, evicted. Every unit
                            # after this one would fail in a second and be written down as
                            # a failure (2026-09-03: 209 of them, in under a minute), so the
                            # run folds what it has and ends; the unit is written down but
                            # not counted against, and --resume reads on once something
                            # serves again
                            print(f"  the model server went away at {unit.id}; folding what "
                                  f"was read and stopping -- --resume reads on")
                            raise Stopped("server gone")
                        since += 1
                        print(f"  ch {unit.chapter or '-':>3}  "
                              f"{unit.section or unit.section_title[:12]:<8}"
                              f" {row.seconds:6.1f}s  {row.concepts:>3}c {row.relations:>3}r "
                              f"{row.figures:>2}f" + (f" {row.images}img" if row.images else "")
                              + (f"  {row.error}" if row.error else ""))
                        ahead = to_read[index + 1] if index + 1 < len(to_read) else None
                        if ahead is not None and _time_to_fold(
                                since, ahead.chapter != unit.chapter):
                            keep(slug, document.title, _rows(wanted, reads_by_unit), units_by_id)
                            since = 0
                except Stopped:
                    stopped = True

                counts = keep(slug, document.title, _rows(wanted, reads_by_unit), units_by_id)
                print(f"  {document.title}: {counts['nodes']} nodes, {counts['edges']} edges "
                      f"into {args.out}")
                if stopped:
                    break
    except Stopped:
        stopped = True

    totals = progress.totals()
    print(f"\n{totals['sections']} section(s) of {totals['books']} book(s) in "
          f"{(time.time() - started) / 60:.1f} min; {spent.calls} calls, "
          f"{spent.prompt_tokens} prompt and {spent.completion_tokens} completion tokens"
          + (f"; {totals['failed']} failed" if totals["failed"] else ""))
    if stopped:
        print("stopped: what was read is folded into the store; "
              f"the same command with --resume reads on ({args.out})")
    return code


def _time_to_fold(since: int, boundary: bool) -> bool:
    """Whether the book in flight should be folded into the store now."""
    return since >= (FOLD_EVERY if boundary else 2 * FOLD_EVERY)


def _rows(wanted: Iterable[Any], reads_by_unit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A book's reads so far, in the order the book has them."""
    return [reads_by_unit[unit.id] for unit in wanted if unit.id in reads_by_unit]


def _call_of(record: Mapping[str, Any]) -> Any:
    from ml_stack.telemetry import Call

    fields = {f for f in Call.__dataclass_fields__}
    return Call(**{k: v for k, v in record.items() if k in fields})


def _keep_reads(out: str | Path, slug: str, reads: Sequence[Mapping[str, Any]]) -> None:
    """Every unit's extraction, beside the store, keyed by unit -- what ``--resume`` folds."""
    path = reads_path(out, slug)
    held = _read_json(path)
    kept = dict(held) if isinstance(held, dict) else {}
    for read in reads:
        kept[str(read.get("unit") or "")] = dict(read)
    _write_json(path, kept)


if __name__ == "__main__":  # pragma: no cover - what `detach` re-runs
    raise SystemExit(main())
