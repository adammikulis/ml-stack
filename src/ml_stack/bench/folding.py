"""What a model read out of the messages, folded into one graph and scored against the gold.

`same` is when two names are one name; `fold` gathers every extraction into one graph by
those names and `_named` reads a relation's end back off it. `score` is that graph against
the gold -- coverage and precision per bucket, relations, invented entries, conformance,
fact survival and resolution -- with `topology` the shape of both graphs and `consistency`
the Jaccard of two readings of the same messages.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.bench.truth import BUCKETS
from ml_stack.entities.spelling import close

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
    return len(xs) == len(ys) and all(close(p, q) for p, q in zip(xs, ys, strict=False))


def _first(label: Any) -> str:
    return (_norm(label).split() or [""])[0]

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
        group = clusters[kind]
        hit = next((c for c in group if any(same(name, n) for n in c["names"])), None)
        if hit is None and kind == "people" and len(_norm(name).split()) == 1:
            by_first = [c for c in group if any(_first(n) == _norm(name) for n in c["names"])]
            hit = by_first[0] if len(by_first) == 1 else None
        if hit is None and kind == "people":
            # a full name arriving after its first name alone
            alone = [c for c in group if all(len(_norm(n).split()) == 1 for n in c["names"])
                     and any(_norm(n) == _first(name) for n in c["names"])]
            hit = alone[0] if len(alone) == 1 else None
        if hit is None:
            hit = {"name": name, "names": [], "attrs": {}}
            group.append(hit)
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


def topology(folded: Mapping[str, Any], truth: Mapping[str, Any]) -> dict[str, Any]:
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
    ids = [i for b in BUCKETS for i in truth["nodes"][b]]
    return {"extracted": _components(names, edges),
            "gold": _components(ids, [(s, t) for s, _, t in truth["relations"]])}


def _buckets(folded: Mapping[str, Any], truth: Mapping[str, Any]
             ) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[int]]:
    """Per bucket the rates, which gold node each cluster named, and how many each absorbed.

    A cluster naming something the messages asserted under ``others`` -- a project, a
    department -- is dropped, neither found nor invented: the schema had no bucket for it.
    """
    by_kind: dict[str, dict[str, Any]] = {}
    where: dict[str, str] = {}                       # cluster name -> gold id
    absorbed: list[int] = []                         # per mapped cluster, gold ids it names
    for bucket in BUCKETS:
        wanted = truth["nodes"][bucket]
        found: set[str] = set()
        said = invented = 0
        for cluster in (folded.get("nodes") or {}).get(bucket) or ():
            hits = [h for n in cluster["names"]
                    if (h := _resolve(n, wanted, people=bucket == "people"))]
            hit = hits[0] if hits else ""
            if hit:
                found.add(hit)
                where[cluster["name"]] = hit
                absorbed.append(len(set(hits)))
                said += 1
            elif any(_resolve(n, truth["others"]) for n in cluster["names"]):
                continue
            else:
                said += 1
                invented += 1
        by_kind[bucket] = {**_rates(len(found), len(wanted), said), "of": len(wanted),
                           "found": len(found), "said": said, "invented": invented}
    return by_kind, where, absorbed


def _totalled(by_kind: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every bucket together, and the count and rate of invented people and organisations."""
    total = {key: sum(int(by_kind[b][key]) for b in BUCKETS)
             for key in ("of", "found", "said", "invented")}
    nodes = {**_rates(total["found"], total["of"], total["said"]), **total}
    named = sum(int(by_kind[b]["said"]) for b in ("people", "orgs"))
    made_up = sum(int(by_kind[b]["invented"]) for b in ("people", "orgs"))
    return nodes, {"count": made_up, "of": named,
                   "rate": round(made_up / named, 4) if named else 0.0}


def _relations(folded: Mapping[str, Any], truth: Mapping[str, Any],
               truth_all: Mapping[str, Mapping[str, Any]],
               where: Mapping[str, str]) -> tuple[dict[str, Any], set[int]]:
    """The extracted relations against the gold's, and which gold relations were matched.

    One matches when both ends resolve to its ends, in that direction, and the names are
    `same`. A relation with an end the messages asserted under ``others`` is dropped from
    both sides, the way an entry naming one is.
    """
    def end(name: str) -> str | None:
        """A gold id, "" for something asserted under others, None for an invented thing."""
        if name in where:
            return where[name]
        hit = _resolve(name, truth_all, people=True)
        if hit:
            return hit
        return "" if _resolve(name, truth["others"]) else None

    matched: set[int] = set()
    said_rels = right_rels = 0
    scored = [i for i, (s, _rel, t) in enumerate(truth["relations"])
              if s in truth_all and t in truth_all]
    for r in folded.get("relations") or ():
        src, dst = end(str(r["from"])), end(str(r["to"]))
        if src == "" or dst == "":
            continue
        said_rels += 1
        if src is None or dst is None:
            continue
        for i in scored:
            s, rel, t = truth["relations"][i]
            if s == src and t == dst and same(r["rel"], rel):
                matched.add(i)
                right_rels += 1
                break
    return {**_rates(len(matched), len(scored), said_rels), "of": len(scored),
            "found": len(matched), "said": said_rels,
            "invented": said_rels - right_rels}, matched


def _attrs(folded: Mapping[str, Any], truth: Mapping[str, Any],
           truth_all: Mapping[str, Mapping[str, Any]],
           where: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """Of the organisations and places the extraction put on people the gold also joins to
    one, how many were the gold's: what it said, not what it left blank."""
    labels = {i: str(n.get("label") or "") for i, n in truth_all.items()}
    has: dict[str, dict[str, list[str]]] = {"org": {}, "place": {}}
    for s, _, t in truth["relations"]:
        for key, bucket in (("org", "orgs"), ("place", "places")):
            if s in truth["nodes"]["people"] and t in truth["nodes"][bucket]:
                has[key].setdefault(s, []).append(labels[t])
    out: dict[str, dict[str, int]] = {}
    for key in ("org", "place"):
        stated = right = 0
        for cluster in (folded.get("nodes") or {}).get("people") or ():
            who = where.get(cluster["name"])
            claimed = cluster.get("attrs", {}).get(key)
            if not who or not claimed or who not in has[key]:
                continue
            stated += 1
            right += any(same(claimed, true) for true in has[key][who])
        out[key] = {"stated": stated, "right": right}
    return out


def _conformance(folded: Mapping[str, Any], vocabulary: Sequence[str]) -> dict[str, Any]:
    """The extracted relations named in the gold's vocabulary and the entries under a key
    the schema has; the rest is ``off_schema``."""
    rels_all = list(folded.get("relations") or ())
    in_vocab = sum(1 for r in rels_all if any(same(r["rel"], v) for v in vocabulary))
    on_schema = sum(len(v) for k, v in (folded.get("nodes") or {}).items() if k in BUCKETS)
    off_schema = sum(len(v) for k, v in (folded.get("nodes") or {}).items() if k not in BUCKETS)
    return {"relations": {"in_vocabulary": in_vocab, "of": len(rels_all),
                          "share": round(in_vocab / len(rels_all), 4) if rels_all else None},
            "entities": {"in_schema": on_schema, "of": on_schema + off_schema,
                         "share": round(on_schema / (on_schema + off_schema), 4)
                         if on_schema + off_schema else None},
            "off_schema": (len(rels_all) - in_vocab) + off_schema}


def _survival(per_message: Sequence[Mapping[str, Any]],
              truth_all: Mapping[str, Mapping[str, Any]], present: set[str],
              kept_rels: set[tuple[str, ...]]) -> dict[str, Any]:
    """The share of one message's own assertions still in the folded graph, averaged over
    the messages that assert anything."""
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
    return {"mean": round(sum(shares) / len(shares), 4) if shares else None,
            "messages": len(shares)}


def _resolution(where: Mapping[str, str], absorbed: Sequence[int]) -> dict[str, Any]:
    """``splits``, extracted nodes per gold node, and ``merges``, gold nodes per extracted
    node; 1.0 is perfect for both."""
    per_gold: dict[str, int] = {}
    for one in where.values():
        per_gold[one] = per_gold.get(one, 0) + 1
    return {"splits": round(sum(per_gold.values()) / len(per_gold), 4) if per_gold else None,
            "merges": round(sum(absorbed) / len(absorbed), 4) if absorbed else None}


def score(folded: Mapping[str, Any], truth: Mapping[str, Any], *,
          per_message: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The folded extraction against the gold: the rates, and what each is of.

    ``by_kind`` and ``nodes`` are coverage, precision and F1 per bucket and over all of
    them, ``relations`` the same for the triples, and ``invented`` the count and rate over
    extracted people and organisations. ``attrs`` is `_attrs`, ``topology`` is `topology`,
    ``conformance`` is `_conformance`, ``resolution`` is `_resolution`, and ``survival``
    is `_survival` over ``per_message`` -- each sampled message's ``asserts``.
    """
    truth_all: dict[str, Mapping[str, Any]] = {i: n for b in BUCKETS
                                               for i, n in truth["nodes"][b].items()}
    by_kind, where, absorbed = _buckets(folded, truth)
    nodes, invented = _totalled(by_kind)
    relations, matched = _relations(folded, truth, truth_all, where)
    kept_rels = {tuple(truth["relations"][i]) for i in matched}
    return {"nodes": nodes, "by_kind": by_kind, "relations": relations, "invented": invented,
            "attrs": _attrs(folded, truth, truth_all, where),
            "topology": topology(folded, truth),
            "conformance": _conformance(folded, list(truth.get("vocabulary") or ())),
            "survival": _survival(per_message, truth_all, set(where.values()), kept_rels),
            "resolution": _resolution(where, absorbed)}


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
