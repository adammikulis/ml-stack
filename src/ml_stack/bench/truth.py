"""The world a message came from, and the truth behind a sample of its messages.

`load_world` reads a world's graph and messages, simulating a few working days when it
holds none; `sample_messages` draws a stratified sample of them; `gold` is the union of
what those messages assert, labelled from the truth graph; `as_extraction` writes that
gold back in the schema `schema` returns, which is what a perfect extractor returns.
`BUCKETS` names the four kinds the schema has a word for.
"""

from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.world import about

# The buckets the generic schema has a word for, as `simulate.asserts_of` files them. A
# world asserts more -- departments, projects, events, under ``others`` -- and an
# extraction naming one of those is neither right nor wrong.
BUCKETS = ("people", "orgs", "topics", "places")
# How many working days to simulate when the world has no messages of its own.
DAYS = 5


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

        seed = int(about.read(where).get("seed", 0) or 0)
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
    for group in grouped.values():
        rng.shuffle(group)
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
        asserts = (m.get("attrs") or {}).get("asserts")
        if not isinstance(asserts, Mapping):
            raise ValueError(f"message {m.get('id')!r} carries no attrs.asserts; simulate the "
                             f"world again with a build that records what each message states")
        if not (m.get("attrs") or {}).get("asserts_exact", True):
            out["exact"] = False
        for bucket in BUCKETS:
            for one in asserts.get(bucket) or ():
                if str(one) in nodes:
                    out["nodes"][bucket][str(one)] = nodes[str(one)]
        for one in asserts.get("others") or ():
            if str(one) in nodes:
                out["others"][str(one)] = nodes[str(one)]
        for r in asserts.get("relations") or ():
            if len(r) == 3 and str(r[0]) in nodes and str(r[2]) in nodes:
                key = (str(r[0]), str(r[1]), str(r[2]))
                if key not in seen:
                    seen.add(key)
                    out["relations"].append(list(key))
    return out


def as_extraction(truth: Mapping[str, Any]) -> dict[str, Any]:
    """The gold written back in the schema's shape: what a perfect extractor would return,
    and what the scorer must give 100% to."""
    label = {i: str(n.get("label") or i) for b in BUCKETS for i, n in truth["nodes"][b].items()}
    label.update({i: str(n.get("label") or i) for i, n in truth["others"].items()})
    return {"people": [{"name": label[i], "role": "", "org": "", "place": ""}
                       for i in truth["nodes"]["people"]],
            "orgs": [{"name": label[i], "kind": ""} for i in truth["nodes"]["orgs"]],
            "topics": [label[i] for i in truth["nodes"]["topics"]],
            "places": [label[i] for i in truth["nodes"]["places"]],
            "relations": [{"from": label.get(s, s), "rel": r, "to": label.get(t, t)}
                          for s, r, t in truth["relations"]]}
