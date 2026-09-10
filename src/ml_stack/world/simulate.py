"""Days on which the people of an invented organisation talk to each other.

`simulate` is a clock. Each working day it picks conversations -- the arcs `world.story`
scheduled for that day, plus routine chatter along whatever relations the graph holds --
and writes each as a thread of two to eight messages in the product it belongs in, as a
stream of `Message` an emitter can export. ``mix`` is the share of threads a `writer`
writes (`world.speaking`); the rest come from `world.sentences` and need no model.

An arc's end writes a fact back into the graph as a typed edge carrying the message it was
said in. Every message says what it asserts in ``attrs["asserts"]``: the ids the writer put
into the sentence and the relations it stated, which is the gold `ml_stack.bench.extract`
scores an extraction against. Nothing here is a real person or organisation.
"""

from __future__ import annotations

import dataclasses
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ml_stack.messages import Message
from ml_stack.world import World, Writer, about
from ml_stack.world.sentences import _STATED, template_writer
from ml_stack.world.speaking import ModelWriter, model_writer
from ml_stack.world.story import (
    DOWNWARD,
    ORG_KINDS,
    OUTCOMES,
    PEER,
    PLACE_KINDS,
    PROJECT_KINDS,
    TOPIC_KINDS,
    UPWARD,
    calendar,
    facts_for,
    groups,
    people_of,
    place_of,
    slug,
)

__all__ = ["CHATTER", "asserts_of", "run", "simulate"]

# Day 0 of every simulation, a Monday, so weekdays fall out of the day number.
EPOCH = datetime(2025, 9, 1, tzinfo=timezone.utc)
WORK_START, WORK_END = 9, 18          # local hours in which anybody writes anything
SHORTEST, LONGEST = 2, 8              # messages in a thread
MEAN_LENGTH = 5.0                     # what per_day is divided by to get threads per person

# Routine conversation kinds, and the relation class each belongs to.
CHATTER = {
    "peer": ("standup", "ask", "share"),
    "line": ("checkin", "plan"),
    "group": ("share", "ask", "plan"),
    "cross": ("ask", "handoff"),
}

PROMPT = ("Reply in character to the thread about {about}, in one or two sentences, in your "
          "voice. Say only what you would say in the thread; no preamble, no name prefix.")


# -- time ----------------------------------------------------------------------------------

def _zone(world: World, person: str) -> Any:
    place = place_of(world.graph, person)
    name = str(((place or {}).get("attrs") or {}).get("timezone") or "") if place else ""
    try:
        return ZoneInfo(name) if name else timezone.utc
    except (KeyError, ValueError):
        return timezone.utc


def _work_start(day: int, zone: Any, rng: random.Random, spread_hours: float) -> float:
    """A unix time on ``day`` at the start of work in ``zone``, plus up to ``spread_hours``."""
    local_date = (EPOCH + timedelta(days=day)).date()
    start = datetime(local_date.year, local_date.month, local_date.day, WORK_START,
                     tzinfo=zone)
    return start.timestamp() + rng.uniform(0, max(0.0, spread_hours) * 3600)


def _next_in_hours(after: float, zone: Any, rng: random.Random) -> float:
    """``after``, or the next moment inside work hours in ``zone`` when it falls outside."""
    local = datetime.fromtimestamp(after, zone)
    if WORK_START <= local.hour < WORK_END and local.weekday() < 5:
        return after
    date = local.date()
    if local.hour >= WORK_END or local.weekday() >= 5:
        date += timedelta(days=1)
    while date.weekday() >= 5:
        date += timedelta(days=1)
    start = datetime(date.year, date.month, date.day, WORK_START, tzinfo=zone)
    return start.timestamp() + rng.uniform(0, 1800)


def _stamp(unix: float) -> str:
    return f"{unix:.6f}"


def _poisson(rng: random.Random, mean: float) -> int:
    # Knuth's, fine for the small means here; measured nothing, needs no numpy
    if mean <= 0:
        return 0
    limit, k, p = 2.718281828 ** -mean, 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1



# -- the graph as relations ------------------------------------------------------------------

class _Relations:
    """Who is joined to whom, by class, read once from the graph."""

    def __init__(self, graph: Mapping[str, Any]) -> None:
        self.by_id = {str(n["id"]): n for n in (graph.get("nodes") or ()) if n.get("id")}
        self.people = people_of(graph)
        self.label = {i: str(n.get("label") or i) for i, n in self.by_id.items()}
        self.groups = groups(graph)
        self.of: dict[str, list[str]] = {p: [] for p in self.people}
        for gid, held in self.groups.items():
            for p in held["people"]:
                self.of.setdefault(p, []).append(gid)
        self.peers: dict[str, list[str]] = {p: [] for p in self.people}
        self.line: dict[str, list[str]] = {p: [] for p in self.people}
        # every relation with a person at the near end that a message knows how to state
        self.stated: dict[str, list[tuple[str, str, str]]] = {}
        for edge in graph.get("edges") or ():
            a, b = str(edge.get("source") or ""), str(edge.get("target") or "")
            rel = str(edge.get("rel") or edge.get("relation") or "")
            if rel in _STATED and a in self.of and b in self.by_id and a != b:
                held = self.stated.setdefault(a, [])
                if (a, rel, b) not in held:
                    held.append((a, rel, b))
            if a not in self.of or b not in self.of:
                continue
            if rel in PEER:
                self._add(self.peers, a, b)
                self._add(self.peers, b, a)
            elif rel in UPWARD or rel in DOWNWARD:
                self._add(self.line, a, b)
                self._add(self.line, b, a)

    @staticmethod
    def _add(into: dict[str, list[str]], a: str, b: str) -> None:
        if b not in into[a]:
            into[a].append(b)

    def facts(self, who: Sequence[str]) -> list[tuple[str, str, str]]:
        """Every statable relation with one of ``who`` at the near end, in graph order.

        What a thread among these people can truthfully say about itself: who they work
        with, who they report to, what they work on, which unit they belong to, where
        they are and what they know. The order is the graph's, so a seed reproduces it.
        """
        out: list[tuple[str, str, str]] = []
        for one in who:
            out.extend(self.stated.get(str(one), ()))
        return list(dict.fromkeys(out))

    def common_group(self, a: str, b: str) -> str:
        shared = [g for g in self.of.get(a, ()) if g in self.of.get(b, ())]
        return shared[0] if shared else ""

    def strangers(self, person: str) -> list[str]:
        mine = set(self.of.get(person, ()))
        return [p for p in self.people
                if p != person and not (mine & set(self.of.get(p, ())))
                and p not in self.peers.get(person, ()) and p not in self.line.get(person, ())]


# -- the sampler -------------------------------------------------------------------------------

_EMAIL_SUBJECTS = {
    "checkin": ("Catching up on {project}", "1:1 notes", "Quick check-in"),
    "plan": ("Next steps on {project}", "Planning {project}", "This week"),
    "ask": ("Question about {project}", "A quick one on {topic}", "Can you help with {project}?"),
    "handoff": ("Handing over {project}", "Re: {project} from {group}", "Introducing {project}"),
}


def _venue(rel: _Relations, cls: str, kind: str, who: Sequence[str], facts: Mapping[str, str],
           rng: random.Random) -> tuple[str, str]:
    """Where a routine thread of that class among ``who`` lands: ``(source, channel)``."""
    me, other = who[0], who[1] if len(who) > 1 else who[0]
    if cls == "peer":
        shared = rel.common_group(me, other)
        if shared and rng.random() < 0.6:
            return ("slack", slug(rel.groups[shared]["label"]))
        return ("slack", "dm:" + ",".join(sorted((me, other))))
    if cls == "line":
        if rng.random() < 0.6:
            return ("slack", "dm:" + ",".join(sorted((me, other))))
        return ("email", rng.choice(_EMAIL_SUBJECTS[kind]).format(**facts))
    if cls == "group":
        return ("slack", slug(facts.get("group") or "general"))
    if rng.random() < 0.5:
        return ("email", rng.choice(_EMAIL_SUBJECTS[kind]).format(**facts))
    return ("teams", "chat:" + "-".join(slug(rel.label[p]) for p in who[:2]))


def _chatter(world: World, rel: _Relations, day: int, rng: random.Random,
             per_day: float) -> list[dict[str, Any]]:
    """Routine threads for one day, a Poisson-ish number per person, along real relations."""
    plans: list[dict[str, Any]] = []
    for me in world.people:
        for _ in range(_poisson(rng, per_day / MEAN_LENGTH)):
            options: list[tuple[str, int]] = []
            if rel.peers.get(me):
                options.append(("peer", 3))
            if rel.line.get(me):
                options.append(("line", 2))
            if any(len(rel.groups[g]["people"]) > 1 for g in rel.of.get(me, ())):
                options.append(("group", 3))
            if rel.strangers(me):
                options.append(("cross", 1))
            if not options:
                others = [p for p in world.people if p != me]
                if not others:
                    return plans
                options = [("cross", 1)]
            cls = rng.choices([c for c, _ in options], [w for _, w in options])[0]
            group = ""
            if cls == "peer":
                who = [me, rng.choice(rel.peers[me])]
            elif cls == "line":
                who = [me, rng.choice(rel.line[me])]
            elif cls == "group":
                group = rng.choice([g for g in rel.of[me] if len(rel.groups[g]["people"]) > 1])
                others = [p for p in rel.groups[group]["people"] if p != me]
                rng.shuffle(others)
                who = [me, *others[:rng.randint(1, 3)]]
            else:
                pool = rel.strangers(me) or [p for p in world.people if p != me]
                who = [me, rng.choice(pool)]
            kind = rng.choice(CHATTER[cls])
            facts = facts_for(world.graph, who, rng, group=group or rel.common_group(*who[:2]))
            about = {"standup": "what {first} is on today", "ask": "{first}'s question about "
                     "{project}", "share": "something {first} found about {topic}",
                     "checkin": "{first}'s check-in with {second}", "plan": "planning "
                     "{project}", "handoff": "handing {project} over"}[kind].format(**facts)
            plans.append({"kind": kind, "who": who, "about": about,
                          "where": _venue(rel, cls, kind, who, facts, rng), "facts": facts,
                          "arc": None})
    return plans


def _arc_threads(world: World, rel: _Relations, day: int, rng: random.Random,
                 held: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One thread for every arc alive on ``day``, in the arc's next venue."""
    plans: list[dict[str, Any]] = []
    for arc in held:
        start, until = int(arc.get("day", 0)), int(arc.get("until", arc.get("day", 0)))
        if not start <= day <= until:
            continue
        who = [p for p in arc.get("who") or () if p in rel.of] or list(world.people[:2])
        venues = list(arc.get("where") or ()) or [("slack", "general")]
        source, channel = venues[(day - start) % len(venues)]
        if channel == "dm":
            channel = "dm:" + ",".join(sorted(who[:2]))
        facts = _with_ids(dict(arc.get("facts") or facts_for(
            world.graph, who, rng, group=str(arc.get("group") or ""))), arc, rel)
        plans.append({"kind": str(arc.get("kind") or "arc"), "who": who,
                      "about": str(arc.get("about") or ""), "where": (str(source), str(channel)),
                      "facts": facts, "arc": arc, "last": day == until})
    return plans


def _with_ids(facts: dict[str, str], arc: Mapping[str, Any], rel: _Relations) -> dict[str, str]:
    """The arc's facts with the ``_id`` behind each name put back.

    A calendar is written to JSON with the names and not the ids, so an arc read from disk
    knows it is about "Lantern" and not that Lantern is ``project:lantern``. The arc's own
    ``subject``, ``group`` and ``to`` say most of it; the rest is the entry whose label is
    the name, which is exact because the name was read off that label.
    """
    group, to, subject = (str(arc.get(k) or "") for k in ("group", "to", "subject"))
    known = {"group": group, "group2": to if to and to != group else "",
             "project": subject if rel.by_id.get(subject, {}).get("kind") in PROJECT_KINDS else ""}
    kinds = {"project": PROJECT_KINDS, "org": ORG_KINDS, "topic": TOPIC_KINDS,
             "place": PLACE_KINDS, "group": None, "group2": None}
    for slot, wanted in kinds.items():
        if facts.get(slot + "_id"):
            continue
        held = known.get(slot, "")
        name = str(facts.get(slot) or "")
        if not held and name:
            held = next((i for i, n in rel.by_id.items() if str(n.get("label") or "") == name
                         and (wanted is None or str(n.get("kind") or "") in wanted)), "")
        facts[slot + "_id"] = held if held and str(rel.label.get(held, "")) == name else ""
    return facts


# -- outcomes ------------------------------------------------------------------------------------

def _outcome(world: World, arc: Mapping[str, Any], said_in: str, day: int) -> dict[str, Any] | None:
    """The fact an arc's end leaves in the graph, as one typed edge. None when it has none."""
    kind = str(arc.get("outcome") or "")
    if kind not in OUTCOMES:
        return None
    who = [str(p) for p in (arc.get("who") or ())]
    ids = {str(n["id"]) for n in (world.graph.get("nodes") or ())}
    if not who or who[0] not in ids:
        return None
    if kind == "now_works_with":
        if len(who) < 2:
            return None
        source, target = who[0], who[1]
    elif kind == "joined":
        source, target = who[0], str(arc.get("group") or arc.get("to") or "")
    elif kind == "moved_to":
        source, target = who[0], str(arc.get("to") or arc.get("group") or "")
    else:
        source, target = who[0], str(arc.get("subject") or "")
    if target not in ids or target == source:
        return None
    edge = {"source": source, "rel": kind, "target": target, "weight": 1, "messages": [said_in],
            "attrs": {"said_in": said_in, "day": day, "arc": str(arc.get("kind") or ""),
                      "about": str(arc.get("about") or "")}}
    world.graph.setdefault("edges", []).append(edge)
    return edge


def asserts_of(ids: Sequence[str], relations: Sequence[Sequence[str]], sender: str,
               nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """What a message asserts, bucketed by what each id is in the graph.

    ``people``, ``orgs``, ``topics`` and ``places`` by the kinds `story` recognises for
    each, everything else -- a project, a department, an event -- under ``others``, as what
    it is rather than forced into a bucket it does not belong in. The sender is always a
    person asserted. ``relations`` are ``[source, rel, target]`` as stated.
    """
    out: dict[str, Any] = {"people": [], "orgs": [], "topics": [], "places": [], "others": [],
                           "relations": [list(map(str, r)) for r in relations]}
    for one in dict.fromkeys(str(i) for i in (sender, *ids) if i):
        node = nodes.get(one)
        if node is None:
            continue
        kind = str(node.get("kind") or "")
        key = ("people" if kind == "person" else "orgs" if kind in ORG_KINDS
               else "topics" if kind in TOPIC_KINDS else "places" if kind in PLACE_KINDS
               else "others")
        out[key].append(one)
    return out


# -- the simulation --------------------------------------------------------------------------------

def _remember(store: Any, name: str, message: Message, plan: Mapping[str, Any],
              rel: _Relations, answer: Any = None) -> None:
    """One message as a turn: what the model said it was about, else the thread's facts."""
    from ml_stack.graph.thread import drew_on, remember_turn

    drew = drew_on(answer) if answer is not None else {}
    shown = list(drew.get("shown") or ())
    if message.sender not in shown:
        shown.insert(0, message.sender)
    if answer is None:
        facts = plan.get("facts") or {}
        shown.extend(str(facts[k]) for k in ("project_id", "place_id", "topic_id", "group_id")
                     if facts.get(k) in rel.by_id and str(facts[k]) not in shown)
    drew["shown"] = shown
    remember_turn(store, thread=name, role="user",
                  text=f"{rel.label.get(message.sender, message.sender)}: {message.text}",
                  drew=drew,
                  meta={"speaker": message.sender, "who": list(plan.get("who") or ()),
                        "message": message.id, "about": plan.get("about"),
                        "kind": plan.get("kind"), "day": message.attrs.get("day")})


def simulate(world: World, *, days: int, writer: Writer | None, rng: random.Random,
             mix: float = 0.1, per_day: float = 3.0, store: Any = None) -> Iterator[Message]:
    """Every message said over ``days``, in the order it was said, as `Message`.

    ``writer`` writes the ``mix`` share of threads, arcs before chatter; the rest come from
    `template_writer`, so ``mix=0.0`` (or ``writer=None``) needs no model. ``per_day`` is the
    mean messages a person sends a working day. ``store`` remembers every message as a turn
    under ``<arc>/<thread id>`` so a `model_writer` over the same store has memory. The
    calendar is ``world.calendar`` when it has one, else `story.calendar`. Outcomes are
    written into ``world.graph`` as the arcs end, so the graph handed in is the graph after.
    """
    rel = _Relations(world.graph)
    if not world.people:
        world.people = list(rel.people)
    held = list(world.calendar) or calendar(world, days, rng)
    if not world.calendar:
        world.calendar = held
    templated = template_writer(rng)
    if writer is None:
        mix = 0.0
    counter = 0
    threads_so_far = model_so_far = 0
    zones = {p: _zone(world, p) for p in world.people}

    for day in range(days):
        if day % 7 >= 5:
            continue
        plans = _arc_threads(world, rel, day, rng, held) + _chatter(world, rel, day, rng, per_day)
        # the day's share of the model, handed to the arcs first: floor(mix * threads) over
        # the whole run, and within a day the threads that will be remembered get it
        quota = int(mix * (threads_so_far + len(plans))) - model_so_far if writer else 0
        for n, plan in enumerate(plans):
            threads_so_far += 1
            by_model = n < quota
            if by_model:
                model_so_far += 1
            who = list(plan["who"])
            length = rng.randint(SHORTEST, LONGEST)
            source, channel = plan["where"]
            arc = plan.get("arc")
            arc_key = f"arc:{arc['day']}:{arc['kind']}" if arc else ""
            root_id = f"msg:{counter:06d}"
            name = f"{arc_key or 'chat'}/{root_id}"
            said: list[tuple[str, str]] = []
            truths = rel.facts(who)
            speaker = who[0]
            when = _work_start(day, zones.get(speaker, timezone.utc), rng, 8.0)
            for seq in range(length):
                if seq:
                    others = [p for p in who if p != speaker] or who
                    speaker = rng.choice(others)
                    when = _next_in_hours(when + rng.uniform(60, 2400),
                                          zones.get(speaker, timezone.utc), rng)
                persona = dict(world.personas.get(speaker) or {})
                persona.setdefault("id", speaker)
                persona.setdefault("label", rel.label.get(speaker, speaker))
                context = {"thread": root_id, "arc_key": arc_key, "kind": plan["kind"],
                           "org_kind": world.kind, "about": plan["about"],
                           "where": (source, channel), "said": list(said),
                           "facts": plan.get("facts") or {}, "seq": seq, "of": length,
                           "speaker": speaker, "others": [p for p in who if p != speaker],
                           "labels": rel.label, "arc": arc, "day": day, "truths": truths,
                           "last": bool(plan.get("last")) and seq == length - 1,
                           "outcome": (arc or {}).get("outcome") if arc else None}
                prompt = PROMPT.format(about=plan["about"])
                text, wrote = "", "template"
                if by_model:
                    text = str(writer(persona, prompt, context) or "").strip()  # type: ignore[misc]
                    wrote = "model"
                if not text or any(text == t for _, t in said):
                    text, wrote = templated(persona, prompt, context), "template"
                message_id = root_id if seq == 0 else f"msg:{counter:06d}"
                stated: list[list[str]] = []
                if wrote == "model":
                    asserted = list(getattr(writer, "last_ids", ()) or ())
                else:
                    last = getattr(templated, "last", None) or {}
                    asserted = list(last.get("ids") or ())
                    # the relation the sentence stated outright, both ends named in full
                    stated = [list(map(str, r)) for r in (last.get("relations") or ())]
                if arc and plan.get("last") and seq == length - 1:
                    # the outcome is written either way; the message asserts it only
                    # when its sentence named both ends, since a closer that says "that
                    # is the call" states nothing an extractor could read
                    edge = _outcome(world, arc, message_id, day)
                    if edge and edge["source"] in asserted and edge["target"] in asserted:
                        stated.append([edge["source"], edge["rel"], edge["target"]])
                recipients: tuple[str, ...] = ()
                if source != "slack" or channel.startswith("dm:"):
                    recipients = tuple(p for p in who if p != speaker)
                message = Message(
                    id=message_id, source=source, channel=channel, sender=speaker,
                    ts=_stamp(when), text=text, recipients=recipients,
                    thread=None if seq == 0 else root_id,
                    kind="message" if seq == 0 else "reply",
                    attrs={"kind": plan["kind"], "about": plan["about"], "arc": arc_key,
                           "day": day, "writer": "model" if by_model else "template",
                           "wrote": wrote,
                           "asserts": asserts_of(asserted, stated, speaker, rel.by_id),
                           "asserts_exact": wrote != "model"})
                counter += 1
                said.append((speaker, text))
                if store is not None:
                    _remember(store, name, message, plan, rel,
                              getattr(writer, "last", None) if wrote == "model" else None)
                yield message


def _namespace(graph: Mapping[str, Any]) -> str:
    """What one world's own quote ids are told apart by, when its graph meets another's."""
    meta = (graph.get("meta") or {}).get("world") or {}
    return f"{meta.get('kind', '')}/{meta.get('size', '')}/{meta.get('seed', '')}"


def _reconcilable(graph: Mapping[str, Any]) -> dict[str, Any]:
    """``graph`` with each node's own introduction quotes as its `provenance`, namespaced to
    this world, and their text joined into `attrs.passage` -- what `absorb` reads a node by."""
    ns = _namespace(graph)
    said = graph.get("messages") or {}
    nodes = []
    for node in graph.get("nodes") or ():
        mids = [str(m) for m in (node.get("messages") or ())]
        passage = " ".join(str((said.get(m) or {}).get("text") or "") for m in mids).strip()
        attrs = dict(node.get("attrs") or {})
        if passage:
            attrs["passage"] = passage
        nodes.append({**node, "attrs": attrs, "provenance": [f"{ns}:{m}" for m in mids]})
    return {**graph, "nodes": nodes, "messages": {f"{ns}:{m}": v for m, v in said.items()}}


def _absorbed(store: Any, graph: Mapping[str, Any], *, judge: Any = None) -> dict[str, Any]:
    """``graph`` reconciled against what ``store`` already holds -- the same concepts land on
    the nodes a previous world or run already gave them, before this one is written.

    Read for the judge's second look: this world's own quotes first, then the store's
    (an earlier world's, still there under its own namespace); the two are merged into what
    is written back, so a later reconciliation can still read both.
    """
    from ml_stack.graph.absorbing import absorb

    incoming = _reconcilable(graph)
    held = store.get_doc("messages") if hasattr(store, "get_doc") else None
    texts = dict(held) if isinstance(held, Mapping) else {}

    def sources(unit: str) -> str:
        found = texts.get(unit) or incoming["messages"].get(unit)
        return str((found or {}).get("text") or "")

    report = absorb(store, incoming, judge=judge, sources=sources)
    return {**report.graph, "messages": {**texts, **incoming["messages"]}}


def run(world_dir: str | Path, out_dir: str | Path, *, days: int, mix: float,
        model_url: str | None = None, seed: int, judge: Any = None) -> dict[str, Any]:
    """Simulate a world on disk and write what was said beside it.

    Reads ``graph.json``, ``personas.json`` and, when present, ``calendar.json`` and
    ``world.json`` from ``world_dir``; writes ``messages.jsonl``, the updated ``graph.json``
    and the ``calendar.json`` used into ``out_dir``. With ``model_url`` and ``mix > 0`` the
    model share is written by `model_writer` with memory in ``out_dir/memory.ladybug``,
    under `ml_stack.lock.only_one` on ``out_dir/simulate.lock``, and `_absorbed` reconciles
    this world's graph against what that store already holds first. ``judge`` is a
    `ml_stack.graph.judging.ModelJudge` for the close spellings a plain match cannot settle.

    Returns the counts: threads, messages, the model/template split, outcomes, and what a
    message cost in model calls.
    """
    from ml_stack.files import read_json, write_json

    world_dir, out_dir = Path(world_dir).expanduser(), Path(out_dir).expanduser()
    graph = read_json(world_dir / "graph.json", None)
    if not isinstance(graph, Mapping):
        raise FileNotFoundError(f"no graph.json in {world_dir}")
    personas = read_json(world_dir / "personas.json", {}) or {}
    said = about.read(world_dir)
    kind = str(said.get("kind") or (graph.get("meta") or {}).get("kind") or "company")
    people = [str(p) for p in (said.get("people") or list(personas) or people_of(graph))]
    world = World(graph=dict(graph), people=people, personas=dict(personas),
                  calendar=list(read_json(world_dir / "calendar.json", []) or []),
                  seed=seed, size=str(said.get("size") or "small"), kind=kind)
    rng = random.Random(seed)
    if not world.calendar:
        world.calendar = calendar(world, days, rng)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, Any] = {"days": days, "people": len(world.people),
                              "arcs": len(world.calendar), "threads": 0, "messages": 0,
                              "model_threads": 0, "template_threads": 0, "outcomes": 0,
                              "by_source": {}, "model_calls": 0,
                              "messages_per_model_call": None}
    before = len(world.graph.get("edges") or ())

    def write_all(writer: ModelWriter | None, store: Any) -> None:
        with (out_dir / "messages.jsonl").open("w", encoding="utf-8") as out:
            for message in simulate(world, days=days, writer=writer, rng=rng, mix=mix,
                                    store=store):
                out.write(json.dumps(dataclasses.asdict(message), ensure_ascii=False) + "\n")
                counts["messages"] += 1
                counts["by_source"][message.source] = counts["by_source"].get(message.source, 0) + 1
                if message.thread is None:
                    counts["threads"] += 1
                    key = "model_threads" if message.attrs.get("writer") == "model" else "template_threads"
                    counts[key] += 1
        if writer is not None:
            counts["model_calls"] = writer.calls
            if writer.calls:
                counts["messages_per_model_call"] = round(writer.messages / writer.calls, 3)

    if model_url and mix > 0:
        from ml_stack.client import Client
        from ml_stack.graph.store import GraphStore
        from ml_stack.lock import only_one

        with only_one(out_dir / "simulate.lock"):
            with GraphStore(out_dir / "memory.ladybug") as store:
                store.write(_absorbed(store, world.graph, judge=judge))
                write_all(model_writer(Client(model_url), world, store), store)
    else:
        write_all(None, None)

    counts["outcomes"] = len(world.graph.get("edges") or ()) - before
    write_json(out_dir / "graph.json", world.graph)
    write_json(out_dir / "calendar.json", world.calendar)
    return counts
