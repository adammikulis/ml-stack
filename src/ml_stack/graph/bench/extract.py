"""Whether a model reads a message right, scored against the truth that wrote the message.

`bench` measures the asking: a graph exists and a model is asked questions of it. Before
any of that, a graph has to be *read out* of what people said, one message at a time, and
which model does that best was never measured -- there was nothing to score an extraction
against, because nobody knows the truth behind a real message. An invented world does:
`ml_stack.world` generates its messages from a graph it holds, so every person, place and
organisation a message names is on record, and so is every relation between them.

So: sample the world's messages, have a model extract each into a generic shape
(`contracts/extraction.schema.json` -- people, organisations, topics, places, relations),
fold the extractions into one graph by name, and score that against the gold: the union
of what the sampled messages assert, ``attrs["asserts"]`` as the simulation wrote it --
the ids the writer put into each sentence and the relations it stated, labels read off
the truth graph by id. Nothing here infers the gold back out of the text. A template-
written message's record is exact and is scored strictly; a model-written one's is a lower
bound (the persona may have named more) and is scored separately, coverage read as
"against a lower bound".

Reported the way the knowledge-graph construction benchmarks report: precision and
coverage as separate columns per kind, never only F1; ``invented``, the count and rate of
extracted people and organisations matching nothing in the gold -- the hallucination rate,
the extraction-side twin of the answering bench's ``made``; and a topology line, the
folded graph's connected components and the share of its nodes in the largest against the
gold graph's own, so a model that scores well on triples and builds a fragmented graph is
seen. Under each run a detail block adds what the same benchmarks add: *conformance*, the
share of extracted relations named in the world's own vocabulary and of entries the schema
has a kind for; *fact survival*, the share of each message's assertions still present
after the fold, averaged, so a fold that merges two people into one is caught; and
*resolution*, how many extracted nodes stand for one gold node (``splits``, 1.0 is
perfect) and how many gold nodes one extracted node absorbed (``merges``). ``--twice``
reads the sample a second time with the model's own card and reports the Jaccard of the
two graphs -- a model that gives a different graph each run is a finding. Runs are kept
beside the answering runs, marked ``kind: "extract"``, and printed in a table of their own.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.entities.spelling import close
from ml_stack.graph.bench import (
    HOME,
    PER_QUESTION,
    Counting,
    RunNotKept,
    _idle,
    _shown,
    _which,
    find_model,
    footprint,
    runs,
    sampling_from,
    smoke_first,
    smoked,
    stamped,
    wants_smoke,
)

__all__ = ["BUCKETS", "GUESS_SECONDS", "INSTRUCTIONS", "KIND", "MessageRow", "SAMPLE",
           "SMOKE_MESSAGES", "add_arguments", "as_extraction", "estimate", "extract_one",
           "consistency", "fold", "gold", "load_world", "main", "measure", "only",
           "read_back", "same", "sample_messages", "save", "schema", "score", "table",
           "detail", "topology"]

# The record's `kind`, which is what tells an extraction run from an answering one in the
# one store both are kept in.
KIND = "extract"
SAMPLE = 40
# Three, not `bench.SMOKE`'s two: two messages can both land in one stratum, and a smoke
# run exists to prove the path -- the stratified sample included.
SMOKE_MESSAGES = 3
# What a message is guessed to cost before any run of that model has said otherwise; the
# estimate is printed before the clock starts so the wall clock is known up front.
GUESS_SECONDS = 15.0
# The buckets the generic schema has a word for, as `simulate.asserts_of` files them. A
# world asserts more -- departments, projects, events, under ``others`` -- and an
# extraction naming one of those is neither right nor wrong.
BUCKETS = ("people", "orgs", "topics", "places")
# How many working days to simulate when the world has no messages of its own.
DAYS = 5

INSTRUCTIONS = (
    "Read this message from an organised group and list the people, organisations, topics, "
    "places and relations it states; invent nothing. The sender is named before the "
    "message: include them among the people. A topic is a subject area the message names "
    "-- a field, a technology, a craft, an activity -- written as the message writes it, "
    "at most three; the message's purpose, a project, an event, a request or a feeling is "
    "not a topic, and a message about nothing in particular has none. An organisation is "
    "a named company, team, institution or group, not a role or a department in passing. "
    "A field the message does not give is an empty string. A relation joins two names "
    "from this message with a short lower-case verb phrase, underscores for spaces, from "
    "these where one fits: works_at, works_with, works_on, reports_to, part_of, based_in, "
    "advises, attended, experienced_in; only otherwise a phrase of your own. "
    "Return only JSON matching the schema."
)


@dataclass
class MessageRow:
    """One message, extracted once, and everything it cost."""

    id: str
    sender: str
    channel: str = ""
    arc: str = ""
    kind: str = ""
    seconds: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    processed_tokens: int = 0
    completion_tokens: int = 0
    timed_out: bool = False
    error: str = ""
    # whether the message's gold is exact (template-written) or a lower bound (model-written)
    exact: bool = True
    extracted: dict[str, Any] = field(default_factory=dict)


# -- the world and its messages -----------------------------------------------------------------

def schema() -> dict[str, Any]:
    """The generic extraction shape, read from the contracts."""
    from ml_stack.contracts import load

    return dict(load("extraction.schema.json"))


def load_world(where: str | Path, *, days: int = DAYS) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """The truth and the messages: ``(graph, messages, note)``.

    ``where`` is what `ml-stack-world make --out` wrote, or what `simulate` wrote beside it
    (that one has ``messages.jsonl``). A world with no messages is simulated for ``days``
    working days with the template writer -- no model -- into a temporary directory, and
    the note says so; the truth is then the graph *after* the simulation, since an arc's
    end writes a fact into it.
    """
    from ml_stack.files import read_json

    where = Path(where).expanduser()
    if not (where / "graph.json").is_file():
        raise FileNotFoundError(f"no graph.json in {where}")
    talk = where / "messages.jsonl"
    note = ""
    if not talk.is_file():
        from ml_stack.world.simulate import run

        seed = int((read_json(where / "world.json", {}) or {}).get("seed", 0) or 0)
        out = Path(tempfile.mkdtemp(prefix="ml-stack-extract-"))
        counts = run(where, out, days=days, mix=0.0, seed=seed)
        note = (f"{where} has no messages.jsonl; simulated {days} working days with the "
                f"template writer into {out}: {counts['messages']} messages in "
                f"{counts['threads']} threads")
        where, talk = out, out / "messages.jsonl"
    graph = read_json(where / "graph.json", None)
    if not isinstance(graph, Mapping):
        raise FileNotFoundError(f"no graph.json in {where}")
    messages = []
    for line in talk.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            messages.append(json.loads(line))
    return dict(graph), messages, note


def _stratum(message: Mapping[str, Any]) -> str:
    attrs = message.get("attrs") or {}
    return ("arc" if attrs.get("arc") else "chat") + ":" + str(attrs.get("kind") or "")


def sample_messages(messages: Sequence[Mapping[str, Any]], n: int, *,
                    seed: int = 0) -> list[dict[str, Any]]:
    """``n`` messages with every kind of conversation still in them, seeded.

    Stratified the way `bench.sample` stratifies questions: one from each stratum first --
    an arc's thread and routine chatter, by conversation kind -- rarest first, then in
    proportion. An arc is a handful of threads in a fortnight of chatter, and a plain
    draw of forty would miss it as often as not; an arc is also where names and
    outcomes are stated, which is what an extractor is for. The same seed gives the same
    sample, so two models are read on the same messages.
    """
    everything = [dict(m) for m in messages]
    if n <= 0 or n >= len(everything):
        return everything
    rng = random.Random(f"extract/{seed}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for m in everything:
        grouped.setdefault(_stratum(m), []).append(m)
    for held in grouped.values():
        rng.shuffle(held)
    order = sorted(grouped, key=lambda k: (len(grouped[k]), k))
    taken: list[dict[str, Any]] = []
    for key in order:
        if len(taken) < n:
            taken.append(grouped[key].pop(0))
    for key in order:
        share = max(0, round((n - len(order)) * len(grouped[key]) / len(everything)))
        for _ in range(min(share, len(grouped[key]))):
            if len(taken) < n:
                taken.append(grouped[key].pop(0))
    for key in order:
        while grouped[key] and len(taken) < n:
            taken.append(grouped[key].pop(0))
    ids = {id(m) for m in taken}
    return [m for m in everything if id(m) in ids][:n]


# -- names --------------------------------------------------------------------------------------

_NOT_A_WORD = re.compile(r"[^\w]+|_+")


def _norm(text: Any) -> str:
    """Lower-cased words with single spaces between: how two names are compared."""
    return _NOT_A_WORD.sub(" ", str(text or "").casefold()).strip()


def same(a: Any, b: Any) -> bool:
    """Whether two names are one name: equal once normalised, or the same number of words
    each pair of which `spelling.close` calls one word spelled twice."""
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return False
    if x == y:
        return True
    xs, ys = x.split(), y.split()
    return len(xs) == len(ys) and all(close(p, q) for p, q in zip(xs, ys))


def _same_rel(a: Any, b: Any) -> bool:
    """Relation names, loosely: case, underscores and spaces aside, then near-spellings."""
    return same(a, b)


def _first(label: Any) -> str:
    return (_norm(label).split() or [""])[0]


def gold(graph: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The corpus gold: the union of what ``messages`` assert, labelled from ``graph`` by id.

    ``{"nodes": {bucket: {id: node}}, "others": {id: node}, "relations": [[s, rel, t]],
    "exact": bool, "vocabulary": [rels]}`` -- ``exact`` when every message's record is,
    ``vocabulary`` every relation name the truth graph uses. A message with no
    ``attrs["asserts"]`` was written before the simulation recorded them, and is refused
    rather than guessed at: the point of the gold is that nobody inferred it.
    """
    nodes = {str(n.get("id")): n for n in (graph.get("nodes") or ())}
    out: dict[str, Any] = {"nodes": {b: {} for b in BUCKETS}, "others": {}, "relations": [],
                           "exact": True,
                           "vocabulary": sorted({str(e.get("rel") or "")
                                                 for e in (graph.get("edges") or ())} - {""})}
    seen: set[tuple[str, str, str]] = set()
    for m in messages:
        held = (m.get("attrs") or {}).get("asserts")
        if not isinstance(held, Mapping):
            raise ValueError(f"message {m.get('id')!r} carries no attrs.asserts; simulate the "
                             f"world again with a build that records what each message states")
        if not (m.get("attrs") or {}).get("asserts_exact", True):
            out["exact"] = False
        for bucket in BUCKETS:
            for one in held.get(bucket) or ():
                if str(one) in nodes:
                    out["nodes"][bucket][str(one)] = nodes[str(one)]
        for one in held.get("others") or ():
            if str(one) in nodes:
                out["others"][str(one)] = nodes[str(one)]
        for r in held.get("relations") or ():
            if len(r) == 3 and str(r[0]) in nodes and str(r[2]) in nodes:
                key = (str(r[0]), str(r[1]), str(r[2]))
                if key not in seen:
                    seen.add(key)
                    out["relations"].append(list(key))
    return out


def as_extraction(held: Mapping[str, Any]) -> dict[str, Any]:
    """The gold written back in the schema's shape: what a perfect extractor would return,
    and what the scorer must give 100% to."""
    label = {i: str(n.get("label") or i) for b in BUCKETS for i, n in held["nodes"][b].items()}
    label.update({i: str(n.get("label") or i) for i, n in held["others"].items()})
    return {"people": [{"name": label[i], "role": "", "org": "", "place": ""}
                       for i in held["nodes"]["people"]],
            "orgs": [{"name": label[i], "kind": ""} for i in held["nodes"]["orgs"]],
            "topics": [label[i] for i in held["nodes"]["topics"]],
            "places": [label[i] for i in held["nodes"]["places"]],
            "relations": [{"from": label.get(s, s), "rel": r, "to": label.get(t, t)}
                          for s, r, t in held["relations"]]}


# -- folding what was extracted into one graph -----------------------------------------------------

def fold(extractions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every extraction folded into one graph by name.

    ``{"nodes": {kind: [{"name", "names", "attrs"}]}, "relations": [{"from", "rel", "to"}]}``
    where each node is a cluster of names `same` joins -- ``Pellard Foundry``, ``Pellard
    foundry`` and ``Pelard Foundry`` are one organisation -- and a person named by first
    name alone joins the person whose first name it is when there is exactly one. A
    relation's ends are the clusters' first-seen names.
    """
    clusters: dict[str, list[dict[str, Any]]] = {k: [] for k in BUCKETS}

    def place(kind: str, name: str, attrs: Mapping[str, Any] | None = None) -> str:
        name = str(name or "").strip()
        if not name:
            return ""
        held = clusters[kind]
        hit = next((c for c in held if any(same(name, n) for n in c["names"])), None)
        if hit is None and kind == "people" and len(_norm(name).split()) == 1:
            by_first = [c for c in held if any(_first(n) == _norm(name) for n in c["names"])]
            hit = by_first[0] if len(by_first) == 1 else None
        if hit is None and kind == "people":
            # a full name arriving after its first name alone
            alone = [c for c in held if all(len(_norm(n).split()) == 1 for n in c["names"])
                     and any(_norm(n) == _first(name) for n in c["names"])]
            hit = alone[0] if len(alone) == 1 else None
        if hit is None:
            hit = {"name": name, "names": [], "attrs": {}}
            held.append(hit)
        if name not in hit["names"]:
            hit["names"].append(name)
        if len(_norm(name).split()) > len(_norm(hit["name"]).split()):
            hit["name"] = name                 # the fullest spelling names the cluster
        for key, value in (attrs or {}).items():
            if value and not hit["attrs"].get(key):
                hit["attrs"][key] = str(value)
        return hit["name"]

    relations: list[dict[str, Any]] = []
    for one in extractions:
        if not isinstance(one, Mapping):
            continue
        for p in one.get("people") or ():
            if isinstance(p, Mapping):
                place("people", p.get("name", ""),
                      {"role": p.get("role", ""), "org": p.get("org", ""),
                       "place": p.get("place", "")})
                if p.get("org"):
                    place("orgs", p["org"])
                if p.get("place"):
                    place("places", p["place"])
        for o in one.get("orgs") or ():
            if isinstance(o, Mapping):
                place("orgs", o.get("name", ""), {"kind": o.get("kind", "")})
        for t in one.get("topics") or ():
            place("topics", t)
        for p in one.get("places") or ():
            place("places", p)
    for one in extractions:
        if not isinstance(one, Mapping):
            continue
        for r in one.get("relations") or ():
            if not isinstance(r, Mapping):
                continue
            src, rel, dst = str(r.get("from") or ""), str(r.get("rel") or ""), str(r.get("to") or "")
            if not (src and rel and dst):
                continue
            relations.append({"from": _named(clusters, src), "rel": rel,
                              "to": _named(clusters, dst)})
    return {"nodes": clusters, "relations": relations}


def _named(clusters: Mapping[str, Sequence[Mapping[str, Any]]], name: str) -> str:
    """The cluster name a relation's end refers to, or the name itself when no list held it."""
    for kind in BUCKETS:
        for c in clusters[kind]:
            if any(same(name, n) for n in c["names"]):
                return str(c["name"])
    return name


# -- scoring ------------------------------------------------------------------------------------

def _resolve(name: str, truth: Mapping[str, Mapping[str, Any]], *, people: bool = False) -> str:
    """The gold node among ``truth`` that ``name`` names, or ""; a person also by first name
    alone when exactly one has it."""
    for node_id, n in truth.items():
        if _norm(n.get("label")) == _norm(name):
            return node_id
    for node_id, n in truth.items():
        if same(name, n.get("label")):
            return node_id
    if people and len(_norm(name).split()) == 1:
        by_first = [i for i, n in truth.items() if _first(n.get("label")) == _norm(name)
                    or close(_first(n.get("label")), _norm(name))]
        if len(by_first) == 1:
            return by_first[0]
    return ""


def _rates(found: int, of: int, said: int) -> dict[str, float]:
    coverage = found / of if of else 0.0
    precision = found / said if said else 0.0
    f1 = 2 * coverage * precision / (coverage + precision) if coverage + precision else 0.0
    return {"coverage": round(coverage, 4), "precision": round(precision, 4), "f1": round(f1, 4)}


def _components(nodes: Sequence[str], edges: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """How many pieces a graph is in, and the share of its nodes in the largest."""
    parent = {n: n for n in nodes}

    def root(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b in edges:
        if a in parent and b in parent:
            parent[root(a)] = root(b)
    sizes: dict[str, int] = {}
    for n in nodes:
        sizes[root(n)] = sizes.get(root(n), 0) + 1
    return {"nodes": len(nodes), "edges": len(edges), "components": len(sizes),
            "largest_share": round(max(sizes.values()) / len(nodes), 4) if nodes else 0.0}


def topology(folded: Mapping[str, Any], held: Mapping[str, Any]) -> dict[str, Any]:
    """The folded graph's shape against the gold's: nodes, edges, connected components and
    the share of nodes in the largest, so a model that names the right things and joins
    none of them is seen. A person's ``org`` and ``place`` count as edges the extraction
    stated; the gold's edges are the relations the messages asserted."""
    names = [str(c["name"]) for b in BUCKETS for c in (folded.get("nodes") or {}).get(b) or ()]
    edges = [(str(r["from"]), str(r["to"])) for r in folded.get("relations") or ()]
    for c in (folded.get("nodes") or {}).get("people") or ():
        for key in ("org", "place"):
            if c.get("attrs", {}).get(key):
                edges.append((str(c["name"]), _named(folded["nodes"], str(c["attrs"][key]))))
    ids = [i for b in BUCKETS for i in held["nodes"][b]]
    return {"extracted": _components(names, edges),
            "gold": _components(ids, [(s, t) for s, _, t in held["relations"]])}


def score(folded: Mapping[str, Any], held: Mapping[str, Any], *,
          per_message: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The folded extraction against the gold.

    Per bucket: ``of`` gold nodes, ``found`` of them (``coverage``), ``said`` extracted
    entries, ``invented`` of those -- an entry naming nothing in the gold of that bucket --
    and ``precision``. ``nodes`` is the same over every bucket together, with F1;
    ``relations`` likewise: an extracted relation matches a gold one when both ends
    resolve to its ends, in that direction, and the names are `same` (case, underscores
    and spaces aside). ``invented`` on its own is the count and rate over extracted people
    and organisations -- the hallucination rate. An entry naming something the messages
    asserted under ``others`` -- a project, a department -- is dropped, neither found nor
    invented, and so is a relation with such an end. ``attrs`` counts, of the organisations
    and places the extraction put on people the gold also joins to one, how many were the
    gold's: what it said, not what it left blank. ``topology`` is `topology`.

    ``conformance``: of the extracted relations, how many are named in the gold's
    ``vocabulary`` (loosely, as everywhere here) and of the extracted entries how many
    are under a key the schema has -- the rest is ``off_schema``. ``survival``, given
    ``per_message`` (each sampled message's ``asserts``): the share of each message's
    assertions -- its ids under the four buckets and its relations -- present in the
    folded graph, averaged over the messages that assert anything. ``resolution``:
    ``splits``, the mean number of extracted nodes standing for one gold node that any
    stood for, and ``merges``, the mean number of gold nodes one extracted node's names
    resolve to; 1.0 is perfect for both.
    """
    by_kind: dict[str, dict[str, Any]] = {}
    where: dict[str, str] = {}                       # cluster name -> gold id
    absorbed: list[int] = []                         # per mapped cluster, gold ids it names
    truth_all: dict[str, Mapping[str, Any]] = {i: n for b in BUCKETS for i, n in held["nodes"][b].items()}
    found_total = of_total = said_total = invented_total = 0
    for bucket in BUCKETS:
        truth = held["nodes"][bucket]
        found: set[str] = set()
        said = invented = 0
        for cluster in (folded.get("nodes") or {}).get(bucket) or ():
            hits = [h for n in cluster["names"]
                    if (h := _resolve(n, truth, people=bucket == "people"))]
            hit = hits[0] if hits else ""
            if hit:
                found.add(hit)
                where[cluster["name"]] = hit
                absorbed.append(len(set(hits)))
                said += 1
            elif any(_resolve(n, held["others"]) for n in cluster["names"]):
                continue
            else:
                said += 1
                invented += 1
        by_kind[bucket] = {**_rates(len(found), len(truth), said), "of": len(truth),
                           "found": len(found), "said": said, "invented": invented}
        found_total += len(found)
        of_total += len(truth)
        said_total += said
        invented_total += invented
    nodes = {**_rates(found_total, of_total, said_total), "of": of_total, "found": found_total,
             "said": said_total, "invented": invented_total}
    named = sum(by_kind[b]["said"] for b in ("people", "orgs"))
    made_up = sum(by_kind[b]["invented"] for b in ("people", "orgs"))
    invented = {"count": made_up, "of": named, "rate": round(made_up / named, 4) if named else 0.0}

    def end(name: str) -> str | None:
        """A gold id, "" for something asserted under others, None for an invented thing."""
        if name in where:
            return where[name]
        hit = _resolve(name, truth_all, people=True)
        if hit:
            return hit
        return "" if _resolve(name, held["others"]) else None

    matched: set[int] = set()
    said_rels = right_rels = 0
    # the gold relations that can be scored at all: one with an end the messages asserted
    # under ``others`` -- a project, a department -- is dropped from both sides, the way an
    # entry naming one is, since the schema had no bucket to put it in and an extraction
    # naming it is neither right nor wrong
    scored = [i for i, (s, _rel, t) in enumerate(held["relations"])
              if s in truth_all and t in truth_all]
    for r in folded.get("relations") or ():
        src, dst = end(str(r["from"])), end(str(r["to"]))
        if src == "" or dst == "":
            continue
        said_rels += 1
        if src is None or dst is None:
            continue
        for i in scored:
            s, rel, t = held["relations"][i]
            if s == src and t == dst and same(r["rel"], rel):
                matched.add(i)
                right_rels += 1
                break
    relations = {**_rates(len(matched), len(scored), said_rels),
                 "of": len(scored), "found": len(matched), "said": said_rels,
                 "invented": said_rels - right_rels}

    labels = {i: str(n.get("label") or "") for i, n in truth_all.items()}
    has: dict[str, dict[str, list[str]]] = {"org": {}, "place": {}}
    for s, _, t in held["relations"]:
        for key, bucket in (("org", "orgs"), ("place", "places")):
            if s in held["nodes"]["people"] and t in held["nodes"][bucket]:
                has[key].setdefault(s, []).append(labels[t])
    attrs: dict[str, dict[str, int]] = {}
    for key in ("org", "place"):
        stated = right = 0
        for cluster in (folded.get("nodes") or {}).get("people") or ():
            who = where.get(cluster["name"])
            claimed = cluster.get("attrs", {}).get(key)
            if not who or not claimed or who not in has[key]:
                continue
            stated += 1
            right += any(same(claimed, true) for true in has[key][who])
        attrs[key] = {"stated": stated, "right": right}
    # conformance: relations named in the world's own vocabulary, entries under a key the
    # schema has -- a grammar makes the second always so, and a path without one may not
    vocabulary = list(held.get("vocabulary") or ())
    rels_all = list(folded.get("relations") or ())
    in_vocab = sum(1 for r in rels_all if any(same(r["rel"], v) for v in vocabulary))
    on_schema = sum(len(v) for k, v in (folded.get("nodes") or {}).items() if k in BUCKETS)
    off_schema = sum(len(v) for k, v in (folded.get("nodes") or {}).items() if k not in BUCKETS)
    conformance = {"relations": {"in_vocabulary": in_vocab, "of": len(rels_all),
                                 "share": round(in_vocab / len(rels_all), 4) if rels_all else None},
                   "entities": {"in_schema": on_schema, "of": on_schema + off_schema,
                                "share": round(on_schema / (on_schema + off_schema), 4)
                                if on_schema + off_schema else None},
                   "off_schema": (len(rels_all) - in_vocab) + off_schema}
    # fact survival: per message, what of its own assertions the folded graph still holds
    present = set(where.values())
    kept_rels = {tuple(held["relations"][i]) for i in matched}
    shares: list[float] = []
    for asserted in per_message:
        facts: list[Any] = [str(i) for b in BUCKETS for i in (asserted.get(b) or ())
                            if str(i) in truth_all]
        for r in asserted.get("relations") or ():
            triple = tuple(map(str, r))
            if len(triple) == 3 and triple[0] in truth_all and triple[2] in truth_all:
                facts.append(triple)
        if facts:
            shares.append(sum(1 for f in facts if (f in present if isinstance(f, str)
                                                   else f in kept_rels)) / len(facts))
    survival = {"mean": round(sum(shares) / len(shares), 4) if shares else None,
                "messages": len(shares)}
    # resolution: extracted nodes per gold node, and gold nodes per extracted node
    per_gold: dict[str, int] = {}
    for one in where.values():
        per_gold[one] = per_gold.get(one, 0) + 1
    resolution = {"splits": round(sum(per_gold.values()) / len(per_gold), 4) if per_gold else None,
                  "merges": round(sum(absorbed) / len(absorbed), 4) if absorbed else None}
    return {"nodes": nodes, "by_kind": by_kind, "relations": relations, "invented": invented,
            "attrs": attrs, "topology": topology(folded, held), "conformance": conformance,
            "survival": survival, "resolution": resolution}


def consistency(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """How alike two folds of the same messages are: the Jaccard of their node sets (bucket
    and normalised name) and of their relation sets (normalised ends and name)."""
    def nodes(folded: Mapping[str, Any]) -> set[tuple[str, str]]:
        return {(b, _norm(c["name"])) for b in BUCKETS
                for c in (folded.get("nodes") or {}).get(b) or ()}

    def triples(folded: Mapping[str, Any]) -> set[tuple[str, str, str]]:
        return {(_norm(r["from"]), _norm(r["rel"]), _norm(r["to"]))
                for r in folded.get("relations") or ()}

    def jaccard(a: set[Any], b: set[Any]) -> float | None:
        return round(len(a & b) / len(a | b), 4) if a | b else None

    return {"nodes": jaccard(nodes(first), nodes(second)),
            "relations": jaccard(triples(first), triples(second))}


# -- extracting ---------------------------------------------------------------------------------

class _Extracting(Counting):
    """`Counting`, with `Client.extract` run through its own counted, deadlined `chat`.

    `Counting` delegates what it does not define to the client, and the client's `extract`
    calls the client's own `chat` -- uncounted, and past no deadline. Binding the extractor
    here puts every call it makes through the counting one.
    """

    def extract(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from ml_stack.client.chat import Client

        return Client.extract(self, *args, **kwargs)  # type: ignore[arg-type]

    def _chat_extractor(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.client.chat import Client

        return Client._chat_extractor(self, *args, **kwargs)  # type: ignore[arg-type]

    def _raw_extractor(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.client.chat import Client

        return Client._raw_extractor(self, *args, **kwargs)  # type: ignore[arg-type]


def prompt_for(message: Mapping[str, Any], sender: str) -> list[dict[str, str]]:
    """The two turns a message is extracted from: the instructions, and the message with
    its sender named ahead of it."""
    where = str(message.get("channel") or "")
    head = f"From {sender}" + (f" in {where}" if where else "") + ":\n"
    return [{"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": head + str(message.get("text") or "")}]


def extract_one(client: Any, message: Mapping[str, Any], sender: str, shape: Mapping[str, Any],
                *, per_message: float = PER_QUESTION) -> MessageRow:
    """One message through ``client.extract``, and what it cost.

    Past ``per_message`` seconds the row is kept as timed out -- nothing extracted, the cap
    as its wall clock -- and the next message is read. ``think=False``: an extraction is a
    reading, not a reasoning, and the thinking channel is where a ceiling is spent.
    """
    attrs = message.get("attrs") or {}
    row = MessageRow(id=str(message.get("id") or ""), sender=sender,
                     channel=str(message.get("channel") or ""),
                     arc=str(attrs.get("arc") or ""), kind=str(attrs.get("kind") or ""),
                     exact=bool(attrs.get("asserts_exact", True)))
    began = time.time()
    counting = _Extracting(client, deadline=began + per_message if per_message else None)
    try:
        got = counting.extract(str(message.get("text") or ""), dict(shape),
                               messages=prompt_for(message, sender), think=False, tries=1)
        row.extracted = got if isinstance(got, dict) else {}
    except Exception as exc:  # noqa: BLE001 - a failure is a result, not the end of the run
        row.error = f"{type(exc).__name__}: {exc}"[:200]
    row.seconds = round(time.time() - began, 2)
    if counting.timed_out:
        row.timed_out = True
        row.error = f"timed out after {per_message:.0f}s"
        row.seconds = float(per_message)
        row.extracted = {}
    row.prompt_tokens = counting.prompt_tokens
    row.cached_tokens = int(counting.cached_tokens or 0)
    row.processed_tokens = counting.processed_tokens
    row.completion_tokens = counting.completion_tokens
    return row


def measure(client: Any, messages: Sequence[Mapping[str, Any]], graph: Mapping[str, Any], *,
            per_message: float = PER_QUESTION,
            log: Callable[[str], None] | None = None) -> tuple[list[MessageRow], dict[str, Any]]:
    """Extract every message, fold, and score against the gold those messages assert.

    Returns ``(rows, scores)`` where ``scores`` is `score` over the messages whose gold is
    exact, with the ``folded`` graph it scored, plus ``"lower_bound"``: the same over the
    model-written ones, when there are any, whose coverage is against a lower bound. The
    gold is read before anything is extracted, so a corpus without one costs no model time.
    """
    exact = [m for m in messages if (m.get("attrs") or {}).get("asserts_exact", True)]
    loose = [m for m in messages if m not in exact]
    golds = {"exact": gold(graph, exact), "lower_bound": gold(graph, loose) if loose else None}
    labels = {str(n.get("id")): str(n.get("label") or n.get("id"))
              for n in (graph.get("nodes") or ())}
    shape = schema()
    rows: list[MessageRow] = []
    for m in messages:
        sender = labels.get(str(m.get("sender") or ""), str(m.get("sender") or ""))
        row = extract_one(client, m, sender, shape, per_message=per_message)
        rows.append(row)
        if log:
            got = row.extracted
            counted = (f"{len(got.get('people') or ())}p {len(got.get('orgs') or ())}o "
                       f"{len(got.get('topics') or ())}t {len(got.get('places') or ())}pl "
                       f"{len(got.get('relations') or ())}r") if got else "-"
            log(f"  {row.seconds:5.1f}s {counted:>18}  {str(m.get('text') or '')[:56]}"
                + ("  TIMED OUT" if row.timed_out else f"  {row.error}" if row.error else ""))
    got = {r.id: r.extracted for r in rows}
    asserted = lambda ms: [(m.get("attrs") or {}).get("asserts") or {} for m in ms]  # noqa: E731
    folded = fold([got[str(m.get("id") or "")] for m in exact])
    scores = score(folded, golds["exact"], per_message=asserted(exact))
    scores["folded"] = folded
    if golds["lower_bound"] is not None:
        scores["lower_bound"] = score(fold([got[str(m.get("id") or "")] for m in loose]),
                                      golds["lower_bound"], per_message=asserted(loose))
    return rows, scores


# -- keeping and showing --------------------------------------------------------------------------

def save(store: str | Path, rows: Sequence[MessageRow], *, label: str, model: str,
         world: Mapping[str, Any], scores: Mapping[str, Any], sample: Mapping[str, Any],
         held: Mapping[str, Any] | None = None) -> str:
    """Keep an extraction run beside the answering ones, and read it back before returning.

    The same discipline as `bench.save`, for the same reason: the store once took twelve
    runs and gave back nothing, and the read-back is the only proof a run exists.
    """
    from ml_stack.graph.bench import _plain
    from ml_stack.graph.store import GraphStore

    stem = f"bench:{label}:{time.strftime('%Y%m%dT%H%M%S')}"
    record = _plain({"at": time.strftime("%FT%T"), "label": label, "kind": KIND, "model": model,
                     "world": dict(world), "sample": dict(sample), "server": stamped(held),
                     "scores": dict(scores), "rows": [asdict(r) for r in rows]})
    record = json.loads(json.dumps(record))
    with GraphStore(store) as writer:
        key, n = stem, 1
        while writer.get_doc(key) is not None:
            key, n = f"{stem}-{n}", n + 1
        writer.put_doc(key, record)
    back = next((r for r in runs(store) if r.get("key") == key), None)
    if back is None:
        raise RunNotKept(f"{key} was written to {store} and did not come back")
    back = {k: v for k, v in back.items() if k != "key"}
    if back != record:
        differs = sorted(k for k in set(back) | set(record) if back.get(k) != record.get(k))
        raise RunNotKept(f"{key} came back from {store} changed: {', '.join(differs)} differ")
    return key


def only(kept: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The extraction runs among ``kept``."""
    return [dict(r) for r in kept if r.get("kind") == KIND]


def read_back(store: str | Path, keys: Sequence[str]) -> list[dict[str, Any]]:
    """The extraction runs under ``keys``, read the way `show` reads them."""
    kept = {r["key"]: r for r in only(runs(store))}
    lost = [k for k in keys if k not in kept]
    if lost:
        raise RunNotKept(f"{len(lost)} run(s) saved to {store} did not come back: "
                         + ", ".join(lost))
    return [kept[k] for k in keys]


def estimate(store: str | Path, model: str, n: int) -> tuple[float, str]:
    """``(seconds per message, where that came from)``: the mean over earlier runs of the
    same model that were not timeouts, else `GUESS_SECONDS`."""
    if Path(store).expanduser().exists():
        seen = [float(r.get("seconds") or 0) for one in only(runs(store))
                if one.get("model") == model
                for r in (one.get("rows") or ()) if not r.get("timed_out")]
        if seen:
            return sum(seen) / len(seen), f"{len(seen)} earlier messages of {model}"
    return GUESS_SECONDS, "a guess, no earlier run of this model"


def _pct(v: Any) -> str:
    return f"{100 * float(v):.0f}%" if v is not None else "-"


def _line(label: str, model: str, rows: Sequence[Mapping[str, Any]], scores: Mapping[str, Any],
          rss: Any) -> str:
    n = len(rows)
    by = scores.get("by_kind") or {}
    nodes, rels = scores.get("nodes") or {}, scores.get("relations") or {}
    made = scores.get("invented") or {}
    attrs = scores.get("attrs") or {}
    seconds = sum(float(r.get("seconds") or 0) for r in rows)
    tokens = sum(int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
                 for r in rows)
    timed = sum(1 for r in rows if r.get("timed_out"))

    def pair(bucket: str) -> str:
        held = by.get(bucket) or {}
        return f"{_pct(held.get('coverage')):>7} {_pct(held.get('precision')):>8}"

    def ratio(key: str) -> str:
        held = attrs.get(key) or {}
        return f"{held.get('right', 0)}/{held.get('stated', 0)}" if held else "-"

    invented = (f"{made.get('count', 0)} ({_pct(made.get('rate'))})"
                if made.get("of") else f"{made.get('count', 0)}")
    return (f"{_shown(label):28} {_shown(model, 22):22} {n:>4} "
            f"{(seconds / n if n else 0):>6.1f} {(tokens / n if n else 0):>7.0f} "
            f"{(str(timed) if timed else ''):>4} "
            f"{pair('people')} {pair('orgs')} {pair('topics')} {pair('places')} "
            f"{_pct(rels.get('coverage')):>7} {_pct(rels.get('precision')):>8} "
            f"{_pct(nodes.get('f1')):>5} {_pct(rels.get('f1')):>5} {invented:>10} "
            f"{ratio('org'):>5} {ratio('place'):>5} "
            f"{(f'{rss / 2**30:.2f}G' if rss else '-'):>9}")


def table(kept: Sequence[Mapping[str, Any]]) -> None:
    """Every extraction run, one per line, and its topology under it.

    Coverage and precision are separate columns per kind on purpose -- an F1 alone cannot
    say whether a model missed things or made them up, and those are fixed by opposite
    changes to the asking. `invented` is the count and rate of extracted people and
    organisations matching nothing in the gold. A run with model-written messages gets a
    second line, ``~ lower bound``, scored against the gold those messages are known to
    hold at least; its coverage reads high for that reason and its precision does not.
    """
    if not kept:
        print("no extraction runs kept yet")
        return
    head = (f"{'run':28} {'model':22} {'msgs':>4} {'s/msg':>6} {'tok/msg':>7} {'t/o':>4} "
            f"{'ppl-cov':>7} {'ppl-prec':>8} {'org-cov':>7} {'org-prec':>8} "
            f"{'top-cov':>7} {'top-prec':>8} {'plc-cov':>7} {'plc-prec':>8} "
            f"{'rel-cov':>7} {'rel-prec':>8} {'n-F1':>5} {'r-F1':>5} {'invented':>10} "
            f"{'org':>5} {'place':>5} {'resident':>9}")
    print(head)
    print("-" * len(head))
    for one in kept:
        rows = one.get("rows") or []
        scores = one.get("scores") or {}
        rss = (one.get("server") or {}).get("resident_bytes")
        exact = [r for r in rows if r.get("exact", True)]
        loose = [r for r in rows if not r.get("exact", True)]
        print(_line(str(one.get("label", "")), str(one.get("model", "")), exact, scores, rss))
        if scores.get("lower_bound") is not None:
            print(_line("  ~ lower bound", "", loose, scores["lower_bound"], None))
        for line in detail(scores):
            print(f"  {line}")


def detail(scores: Mapping[str, Any]) -> list[str]:
    """The lines under a run's row: topology, conformance, survival, resolution, and the
    consistency of a second reading when there was one."""
    out: list[str] = []
    shape = scores.get("topology") or {}
    if shape:
        got, want = shape.get("extracted") or {}, shape.get("gold") or {}
        out.append(f"topology: extracted {got.get('nodes', 0)} nodes, {got.get('edges', 0)} "
                   f"edges, {got.get('components', 0)} components, largest "
                   f"{_pct(got.get('largest_share', 0))} -- gold {want.get('nodes', 0)} nodes, "
                   f"{want.get('edges', 0)} edges, {want.get('components', 0)} components, "
                   f"largest {_pct(want.get('largest_share', 0))}")
    conf = scores.get("conformance") or {}
    if conf:
        r, e = conf.get("relations") or {}, conf.get("entities") or {}
        out.append(f"conformance: {r.get('in_vocabulary', 0)}/{r.get('of', 0)} relations in "
                   f"the world's vocabulary, {e.get('in_schema', 0)}/{e.get('of', 0)} entries "
                   f"in the schema; off-schema {conf.get('off_schema', 0)}")
    lived = scores.get("survival") or {}
    if lived.get("messages"):
        out.append(f"survival: {_pct(lived.get('mean'))} of each message's assertions in the "
                   f"folded graph, over {lived['messages']} messages")
    res = scores.get("resolution") or {}
    if res.get("splits") is not None:
        out.append(f"resolution: splits {res['splits']:.2f}, merges {res['merges']:.2f} "
                   f"(1.00 is perfect)")
    alike = scores.get("consistency") or {}
    if alike:
        j = lambda v: f"{v:.2f}" if v is not None else "-"  # noqa: E731
        out.append(f"consistency: nodes J={j(alike.get('nodes'))}, relations "
                   f"J={j(alike.get('relations'))} (second reading with {alike.get('with', '?')})")
    return out


# -- the subcommand -------------------------------------------------------------------------------

def add_arguments(sub: Any) -> Any:
    """Register ``extract`` on the bench's subparsers."""
    ap = sub.add_parser("extract", allow_abbrev=False,
                        help="read a world's messages with a model and score the extraction "
                             "against the truth that wrote them")
    ap.add_argument("label", help="what this run is, e.g. flash-next-extract")
    ap.add_argument("--world", required=True, metavar="DIR",
                    help="what ml-stack-world make --out wrote, or what simulate wrote "
                         "(with messages.jsonl). Without messages, a few working days are "
                         "simulated with the template writer into a temporary directory")
    ap.add_argument("--serve", action="append", default=[], metavar="MODEL",
                    help="a model to put up, read with, and take down: a name, a path or "
                         "an hf: reference. One at a time")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080",
                    help="the model reading, when nothing is served (default: %(default)s)")
    ap.add_argument("--sample", type=int, default=SAMPLE, metavar="N",
                    help="how many messages to read, arcs and chatter both kept "
                         "(default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="which N (default: %(default)s)")
    ap.add_argument("--context", type=int, default=65536, metavar="N",
                    help="total context for a --serve'd model (default: %(default)s)")
    ap.add_argument("--parallel", type=int, default=1, metavar="N",
                    help="slots for a --serve'd model (default: %(default)s -- extraction "
                         "reads one message at a time and never splits the GPU; a second "
                         "slot only halves the context each has)")
    ap.add_argument("--serve-port", type=int, default=8099)
    ap.add_argument("--n-max", type=int, default=None, metavar="N",
                    help="tokens the draft head guesses ahead each step, over the profile's "
                         "measured length. Extraction repeats what it has just read -- "
                         "names, ids, the schema's own keys -- so a longer draft may pay "
                         "here where it lost on answering; this is how that is measured")
    ap.add_argument("--per-message", type=float, default=PER_QUESTION, metavar="SECONDS",
                    help="the most one message may take before it is recorded as timed out "
                         "-- nothing extracted, the cap as its wall clock -- and the next is "
                         "read (default: %(default)s)")
    ap.add_argument("--kept", default=str(HOME / "runs.ladybug"),
                    help="where to keep the run (default: %(default)s)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"read only {SMOKE_MESSAGES} messages, to prove the whole path -- "
                         f"serve, read, fold, score, save and read the run back -- before "
                         f"spending the GPU on it")
    ap.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True,
                    help="serve the model in its measured shape from ml-stack's profiles "
                         "(build, head, cache type, thinking budget, raw flags); "
                         "--no-profile serves it bare")
    ap.add_argument("--twice", action="store_true",
                    help="read the sample a second time with the model's own card (or the "
                         "same settings again when the card names none) and report the "
                         "Jaccard of the two graphs: a model that gives a different graph "
                         "each run is a finding")
    ap.add_argument("--anyway", action="store_true",
                    help="measure even when the server is already busy")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the sampling temperature; the default is the model's card")
    ap.add_argument("--top-p", type=float, default=None, help="override top_p")
    ap.add_argument("--top-k", type=int, default=None, help="override top_k")
    ap.add_argument("--min-p", type=float, default=None, help="override min_p")
    ap.add_argument("--no-queue", action="store_true",
                    help="fail at once if another measurement holds the GPU, rather than "
                         "queue behind it")
    ap.add_argument("--detach", action="store_true",
                    help="run this in the background, owned by nobody's terminal, with its "
                         f"output in a log under {HOME / 'logs'}; status, tail -f and stop "
                         "as for run")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="do not download an hf: model before the measuring lock is taken")
    return ap


def twice(client: Any, picked: Sequence[Mapping[str, Any]], graph: Mapping[str, Any],
          scores: Mapping[str, Any], *, per_message: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """The sample read again -- with the card's sampling when the model names any, else at
    the same settings -- and how alike the two folds are. Returns ``(consistency, what the
    second reading was)``; the second reading's rows are kept under ``server.twice``."""
    card = dict(getattr(client, "card", None) or {})
    if card:
        again = type(client)(client.base_url, timeout=per_message, **card)
        how = "the card: " + " ".join(f"{k[:4]}{v}" for k, v in sorted(card.items()))
    else:
        again = client
        how = "the same settings again"
    print(f"\n  reading the sample again with {how}")
    rows, second = measure(again, picked, graph, per_message=per_message, log=print)
    alike = {**consistency(scores.get("folded") or {}, second.get("folded") or {}), "with": how}
    return alike, {"with": how, "sampling": dict(getattr(again, "sampling", {}) or {}),
                   "rows": [asdict(r) for r in rows], "scores": {k: v for k, v in second.items()
                                                                 if k != "folded"}}


def main(args: Any) -> int:
    """``ml-stack-bench extract``: sample, read, fold, score, keep, and print the table."""
    if len(args.serve) > 1:
        print("error: --serve takes one model; a comparison is one run per model",
              file=sys.stderr)
        return 2
    graph, messages, note = load_world(args.world)
    if note:
        print(note)
    if not messages:
        print(f"error: no messages in {args.world}", file=sys.stderr)
        return 2
    n = SMOKE_MESSAGES if args.smoke else args.sample
    picked = sample_messages(messages, n, seed=args.seed)
    arcs = sum(1 for m in picked if (m.get("attrs") or {}).get("arc"))
    loose = sum(1 for m in picked if not (m.get("attrs") or {}).get("asserts_exact", True))
    try:
        held = gold(graph, picked)
    except ValueError as why:
        print(f"error: {why}", file=sys.stderr)
        return 2
    meta = (graph.get("meta") or {}).get("world") or {}
    world = {"kind": meta.get("kind", ""), "size": meta.get("size", ""),
             "seed": meta.get("seed", 0), "organisation": meta.get("organisation", ""),
             "digest": _which(graph), "messages": len(messages), "where": str(args.world)}
    sample = {"n": len(picked), "seed": args.seed, "arcs": arcs, "chatter": len(picked) - arcs,
              "model_written": loose,
              "gold": {b: len(held["nodes"][b]) for b in BUCKETS} | {
                  "others": len(held["others"]), "relations": len(held["relations"])}}

    model = (str(find_model(args.serve[0])).rsplit("/", 1)[-1].removesuffix(".gguf")
             if args.serve else "")
    if not args.serve:
        model = str(footprint(args.base_url).get("model") or "").removesuffix(".gguf") or args.label
    per, source = estimate(args.kept, model, len(picked))
    print(f"{args.label}: {len(picked)} of {len(messages)} messages ({arcs} from arcs"
          + (f", {loose} model-written, scored against a lower bound" if loose else "")
          + f") over a {world['kind'] or 'small'} world; the gold holds "
          + ", ".join(f"{v} {k}" for k, v in sample["gold"].items())
          + f"; about {per:.0f} s/msg ({source}), so about {len(picked) * per / 60:.0f} min")

    from ml_stack.client import Client

    sampling = sampling_from(args)

    def read_and_keep(client: Any, reading: Sequence[Mapping[str, Any]], *, held: Mapping[str, Any],
                      twice_over: bool, n: int) -> str:
        """``reading`` through ``client``, folded, scored, kept and read back: the key."""
        rows, scores = measure(client, reading, graph, per_message=args.per_message, log=print)
        held = dict(held)
        if twice_over:
            scores["consistency"], held["twice"] = twice(client, reading, graph, scores,
                                                         per_message=args.per_message)
        key = save(args.kept, rows, label=args.label, model=model, world=world, scores=scores,
                   sample={**sample, "n": n}, held=held)
        print(f"kept as {key}")
        table(read_back(args.kept, [key]))
        return key

    if args.serve:
        from ml_stack.serve import serve

        found = find_model(args.serve[0])
        began = time.time()
        # the model's measured shape -- its build, head, cache type, thinking budget, raw
        # flags -- unless told to serve it bare: an extraction measured on mainline without
        # the head (2026-09-02) measured a different program from the one that answers
        lease: dict[str, Any] = {"port": args.serve_port, "context": args.context,
                                 "parallel": args.parallel, "timeout": 900.0,
                                 "cache_reuse": 256, "warmup": False}
        manager = None
        if getattr(args, "profile", True):
            from ml_stack.serve.profile import profile_for

            measured = profile_for(str(found))
            if measured is not None:
                shape = measured.shape(port=args.serve_port, seats=args.parallel)
                lease = {**lease, **{k: v for k, v in shape.lease().items()
                                     if k not in ("port", "context", "parallel")}}
                manager = shape.manager()
                print(f"    serving in its measured shape: {measured.said()}"
                      if hasattr(measured, "said") else "    serving in its measured shape")
        if getattr(args, "n_max", None) is not None:
            if not lease.get("draft"):
                print("    --n-max: no draft head is being served, so there is no draft "
                      "to lengthen", file=sys.stderr)
                return 2
            lease["spec_draft_max"] = int(args.n_max)
            print(f"    draft length {args.n_max} over the profile's")
        with serve(found, manager=manager, **lease) as server:
            print(f"    up in {time.time() - began:.0f}s")
            client = Client(server.base_url, timeout=args.per_message, **sampling)
            held = {**footprint(server.base_url), "sampling": dict(client.sampling),
                    "load_s": getattr(server, "load_s", None)}
            if lease.get("spec_draft_max") is not None:
                held["spec_draft_max"] = int(lease["spec_draft_max"])
            if wants_smoke(args):
                # first, on this load: a few messages through the whole path, kept and
                # read back, before the sample that costs the GPU
                few = sample_messages(messages, SMOKE_MESSAGES, seed=args.seed)
                print(f"\n  smoke: {len(few)} message(s) through the whole path first")
                key = read_and_keep(client, few, held=held, twice_over=False, n=len(few))
                smoked(read_back(args.kept, [key]), f"{args.label} smoke")
                print("  smoke: ok\n")
            read_and_keep(client, picked, held=held, twice_over=args.twice, n=len(picked))
    else:
        if wants_smoke(args):
            smoke_first(args)
        if not _idle(args.base_url, args):
            return 3
        client = Client(args.base_url, timeout=args.per_message, **sampling)
        read_and_keep(client, picked, held={**footprint(args.base_url),
                                            "sampling": dict(client.sampling)},
                      twice_over=args.twice, n=len(picked))
    return 0
