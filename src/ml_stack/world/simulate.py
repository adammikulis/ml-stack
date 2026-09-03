"""Days on which the people of an invented organisation talk to each other.

`simulate` is a clock. Each working day it picks conversations -- the arcs `world.story`
scheduled for that day, plus routine chatter along whatever relations the graph holds --
and writes each as a thread of two to eight messages in the product it belongs in: a team
channel or DM for people who work together, a 1:1 DM or an email up and down a reporting
line, email or Teams across groups. What comes out is a stream of `Message`, ready for an
emitter to write in each product's export shape.

Who says what is a `writer`, ``(persona, prompt, context) -> str``. `template_writer` needs
no model: sentences per conversation kind and organisation kind, slot-filled with names
read out of the graph, so even templated chatter is about a real project in a real place.
`model_writer` has a persona speak through `graph.ask.converse` over the subgraph it knows,
with its own system prompt, the thread so far as turns and what it said in earlier threads
of the same arc as memory -- so what it said last week is what it says this week. ``mix`` is
the share of threads the model writes; the arcs go to it first and routine chatter last,
because an arc is where consistency is noticed.

An arc's end writes a fact back into the graph -- a decision, a move, a new collaboration,
a joining -- as a typed edge carrying the message it was said in, so later conversations and
the truth agree. Nothing here is a real person or organisation.

Most messages also state one relation the graph already holds among the people talking --
who works with whom, who reports to whom, what somebody works on, which unit they belong
to -- as a plain sentence naming both ends in full, so a reader has an edge to find and
not only names. See `_STATED` and `STATE`.

Every message also says what it asserts, in ``attrs["asserts"]``: the ids of the people,
organisations, topics, places and other entries the writer put into that sentence, and the
relations it stated, as ``[source, rel, target]``. The template writer knows exactly which
slots it filled, so its record is exact (``attrs["asserts_exact"]`` is True); the model
writer's is the opening it was grounded in plus what its answer drew on, a lower bound on
what the persona may have named, and is flagged False. That record is the gold an
extraction is scored against (`ml_stack.graph.bench_extract`): nothing infers it back out
of the text.
"""

from __future__ import annotations

import dataclasses
import json
import random
import string
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ml_stack.world import Message, World
from ml_stack.world.story import (DOWNWARD, ORG_KINDS, OUTCOMES, PEER, PLACE_KINDS,
                                  TOPIC_KINDS, UPWARD, calendar, facts_for, groups, people_of,
                                  place_of, slug)

__all__ = ["CHATTER", "ModelWriter", "Writer", "asserts_of", "model_writer", "run",
           "simulate", "template_writer"]

Writer = Callable[[Mapping[str, Any], str, Mapping[str, Any]], str]

# Day 0 of every simulation, a Monday, so weekdays fall out of the day number.
EPOCH = datetime(2025, 9, 1, tzinfo=timezone.utc)
WORK_START, WORK_END = 9, 18          # local hours in which anybody writes anything
SHORTEST, LONGEST = 2, 8              # messages in a thread
MEAN_LENGTH = 5.0                     # what per_day is divided by to get threads per person
ROUNDS = 4                            # tool rounds a persona gets to look something up

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


# -- the relations a message states ------------------------------------------------------------

# Plain sentences that state one relation the graph already holds, naming both ends in
# full. Without them a templated thread is first names and pronouns -- nothing a reader
# could take a relation out of, and so nothing an extraction can be scored on: a run over
# forty messages once yielded a single gold edge, which made the relation columns of
# `ml-stack-bench extract` meaningless. The key is the relation as the world's own
# vocabulary spells it (`world.organisation`), which is what the gold carries and what
# conformance is measured against, so nothing here invents a relation name.
_STATED: dict[str, tuple[str, ...]] = {
    "works_with": ("{a} works with {b} on most of this.",
                   "{a} and {b} work together, so either of them can pick it up.",
                   "For context: {a} works with {b} week to week."),
    "now_works_with": ("{a} works with {b} now.",
                       "Since the change, {a} works with {b}.",
                       "{a} and {b} are a pair now."),
    "reports_to": ("{a} reports to {b}, so that is the line to take it up.",
                   "Worth saying that {a} reports to {b}.",
                   "{a} reports to {b} -- ask there first."),
    "advises": ("{a} advises {b}.", "{b} is advised by {a}.",
                "{a} advises {b}, so the two of them should agree it."),
    "mentors": ("{a} mentors {b}.", "{a} is mentoring {b} this year.",
                "{b} is being mentored by {a}."),
    "leads": ("{a} leads {b}.", "{a} is the one who leads {b}.",
              "{a} leads {b}, so it is their call."),
    "chairs": ("{a} chairs {b}.", "{a} is the one who chairs {b}."),
    "sits_on": ("{a} sits on {b}.", "{a} sits on {b} and can raise it there."),
    "moderates": ("{a} moderates {b}.", "{a} moderates {b}, so flag it to them."),
    "maintains": ("{a} maintains {b}.", "{a} is the one who maintains {b}."),
    "member_of": ("{a} is a member of {b}.", "{a} is in {b}.",
                  "{a} belongs to {b}, if that helps."),
    "part_of": ("{a} is part of {b}.", "{a} is in {b}.",
                "{a} belongs to {b}, if that helps."),
    "works_on": ("{a} works on {b}.", "{a} is working on {b} at the moment.",
                 "{b} is what {a} works on."),
    "contributes_to": ("{a} contributes to {b}.", "{a} is a contributor to {b}."),
    "works_at": ("{a} works at {b}.", "{a} is at {b}.",
                 "{a} works at {b}, for anyone who has not met them."),
    "based_in": ("{a} is based in {b}.", "{a} lives in {b}.",
                 "{a} is based in {b}, so mind the hours."),
    "experienced_in": ("{a} knows about {b}.", "{a} is experienced in {b}.",
                       "{a} is the one who knows {b}."),
    "attended": ("{a} attended {b}.", "{a} was at {b}."),
    "joined": ("{a} joined {b}.", "{a} is in {b} now."),
    "moved_to": ("{a} moved to {b}.", "{a} is with {b} now."),
}

# The share of messages that state one such relation. High, because the gold is only as
# large as what was said and a stratified sample of forty messages is what it is scored
# over; a message states a relation at most once, and never one already stated in its
# own thread.
STATE = 0.7

# Relations about how the organisation is put together, rather than about one person.
# A graph holds far more ``experienced_in`` than ``reports_to``, so drawing uniformly
# would leave a forty-message sample with almost no reporting lines in it; these are
# preferred `STRUCTURAL` of the time when the thread has one to state.
_SHAPE = frozenset({"works_with", "now_works_with", "reports_to", "advises", "mentors",
                    "leads", "chairs", "sits_on", "moderates", "maintains", "part_of",
                    "member_of",
                    "works_on", "contributes_to", "joined", "moved_to"})
STRUCTURAL = 0.55


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
    from ml_stack.world.story import PROJECT_KINDS

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
    elif kind in ("joined", "moved_to"):
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


# -- the template writer ------------------------------------------------------------------------

_OPENERS: dict[str, tuple[str, ...]] = {
    # company
    "launch": ("Launch check for {project}: where are we on the {place} rollout?",
               "Two weeks to the {project} launch. What is still red?",
               "{other}, can you own the launch notes for {project}?",
               "Quick launch sync on {project} before {org} sees it."),
    "incident": ("{project} is throwing errors again -- who is on it?",
                 "Incident open on {project}. {other}, are you seeing it from support?",
                 "Paging {group}: {project} is down for {org}.",
                 "Something is wrong with {project} since this morning."),
    "new_hire": ("Welcome to {group}, {first}! Grab me whenever you want a walkthrough.",
                 "Everyone say hello to {first}, starting with us in {place} this week.",
                 "{first}, your first ticket is on {project} -- ask anything.",
                 "Morning {first}, want to pair on {project} this afternoon?"),
    "escalation": ("{org} have escalated again about {project}. Who has context?",
                   "Heads up: {org} want a call about {project} today.",
                   "{other}, {org} are unhappy; can you and I take the {project} call?",
                   "Escalation from {org} in the queue -- {project}, priority one."),
    "offsite": ("Offsite in {place}: who still needs travel booked?",
                "Agenda draft for {place} is up. Comments by tomorrow please.",
                "Is anyone driving to {place} or are we all on the train?",
                "One session slot left for the {place} offsite -- topics?"),
    "quarterly_review": ("Quarterly numbers for {group} are in. {other}, first read?",
                         "Prep for the review: {project} is the headline.",
                         "Can we close the {group} review by Thursday?",
                         "Review deck for {group}: I need the {project} slide."),
    "reorg": ("Heads up: {first} is moving from {group} to {group2}.",
              "We are shifting {project} ownership to {group2}. Questions here.",
              "{other}, can you help {first} hand {project} over to {group2}?",
              "Org change: {group} and {group2} are swapping a few people."),
    "deadline_slip": ("{project} is going to miss the date. Options?",
                      "Honest status: {project} needs another two weeks.",
                      "{other}, what would it take to hold the {project} deadline?",
                      "Slip on {project} -- I want it said out loud before {org} asks."),
    # community
    "intro": ("Hi all, {first} here, based in {place}. Mostly into {topic}.",
              "New here -- I am {first}, and I mostly lurk on {topic}.",
              "Hello! {first}, {place}. Pointed here by a friend who works on {project}.",
              "Just joined. I am {first} and I have questions about {topic} already."),
    "question": ("Anyone here dealt with {topic} on {project}?",
                 "Question: what do people use for {topic}?",
                 "Stuck on {topic}. Does anyone in {place} know this?",
                 "Is there a known answer for {topic}, or do I write it up?"),
    "meetup": ("Meetup in {place} next month -- who is in?",
               "Venue for {place} is booked. Speakers wanted.",
               "Can we do the {place} meetup on a Thursday this time?",
               "{other}, would you do a short talk on {topic} at the meetup?"),
    "job_post": ("{org} are hiring for {topic} work. Happy to refer.",
                 "Posting a role at {org}; ask me anything about it.",
                 "Anyone looking? {org} need someone who knows {topic}.",
                 "{other}, this {org} role reads like your CV."),
    "recommendation": ("Who should I ask about {topic}?",
                       "Looking for a recommendation: someone good at {topic} in {place}.",
                       "Anyone rate a contractor for {project}-type work?",
                       "Need a name for {topic}. Who do people trust?"),
    "intro_between": ("{first}, meet {second}. You both care about {topic}.",
                      "Connecting {first} and {second} -- {project} overlaps.",
                      "Intro as promised: {second}, {first} is the one I mentioned.",
                      "{first} and {second}, you should compare notes on {topic}."),
    # university
    "paper_deadline": ("{project} deadline is Friday. Who has the figures?",
                       "Draft of {project} is in the shared folder. Read it today.",
                       "{other}, can you take the related work for {project}?",
                       "We need the {project} abstract by tonight."),
    "grant": ("Grant call is out. {project} fits it; who writes?",
              "Proposal for {project}: I need a budget line from {group}.",
              "{other}, would you co-PI on {project}?",
              "Deadline for the {project} proposal moved up. Regroup."),
    "seminar": ("Seminar on {topic} next week -- room and time to confirm.",
                "Who is hosting the {topic} speaker in {place}?",
                "Slides for the {topic} seminar: {other}, are yours ready?",
                "Can we move the {topic} seminar to Thursday?"),
    "defence": ("{first}'s defence is scheduled. Committee, please confirm.",
                "Reading {first}'s thesis this week; questions on chapter three.",
                "{other}, can you chair {first}'s defence?",
                "Defence prep: {first}, run the talk past us first."),
    "lab_move": ("{first} is moving to {group2} next month.",
                 "Handing {project} over as {first} moves labs.",
                 "{other}, anything {first} should take to {group2}?",
                 "Lab move: {first}'s desk in {place} frees up soon."),
    # open-source
    "release": ("Cutting {project} release this week. Blockers?",
                "Changelog for {project} is drafted. {other}, review?",
                "Release branch for {project} is open.",
                "One more fix and {project} ships."),
    "bug_fix": ("Found a bug in {project}: {topic} path returns nothing.",
                "Repro for the {project} bug is in the issue. {other}, yours?",
                "Is the {project} crash known? Hitting it in {place}.",
                "Fix for {project} is up; small diff."),
    "rfc": ("RFC on {topic} for {project} is posted. Read before Friday.",
            "Opening discussion on {topic}: I think {project} needs it.",
            "{other}, your objection on the {topic} RFC -- still stands?",
            "RFC round two for {topic}. What changed is at the top."),
    "first_pr": ("First PR from {first} on {project}! Reviewers?",
                 "Hi, {first} here -- opened my first PR against {project}.",
                 "{other}, would you review {first}'s {project} PR gently?",
                 "Welcome {first}; the {project} PR looks good so far."),
    "advisory": ("Private: a security report on {project} came in.",
                 "Advisory for {project} -- embargo until the release.",
                 "{other}, can you verify the {project} report today?",
                 "Patch for the {project} advisory is ready to review."),
    # nonprofit
    "fundraiser": ("Fundraiser for {project}: target and date to confirm.",
                   "{org} might match donations for {project}. {other}, follow up?",
                   "Venue in {place} for the {project} fundraiser?",
                   "Fundraiser copy for {project} needs a story."),
    "programme_launch": ("Launching {project} in {place}. Field team ready?",
                         "{project} launch: partners in {place} confirmed.",
                         "{other}, the {project} launch needs comms by Monday.",
                         "Checklist for the {project} launch is up."),
    "volunteer_drive": ("Volunteer drive for {place}: {first} is our first sign-up.",
                        "Welcome {first}! Shifts in {place} start next week.",
                        "{other}, can you onboard {first} for {project}?",
                        "Need ten more volunteers in {place} for {project}."),
    "board_meeting": ("Board meets Thursday. {project} is on the agenda.",
                      "Board pack: {other}, I need the {project} numbers.",
                      "Pre-read for the board on {project} is circulated.",
                      "Board asked about {project} again."),
    # routine
    "standup": ("Today: {project}, then reviews.", "On {project} all day; ping if needed.",
                "Standup from {place}: {project} first.", "Morning -- picking up {project}."),
    "ask": ("{other}, do you have a minute on {project}?", "Quick one: who owns {topic} now?",
            "Is there a doc for {project}?", "{other}, where does {topic} live these days?"),
    "share": ("Found something useful on {topic}; link in the thread.",
              "Sharing notes from {place} on {project}.",
              "{other}, this {topic} write-up is worth your time.",
              "FYI on {project}: numbers moved."),
    "checkin": ("How is {project} going, honestly?", "1:1 today -- anything on {project}?",
                "{other}, want to talk about {topic} this week?",
                "Check-in: what is blocking you on {project}?"),
    "plan": ("Let us plan {project} for next week.", "{other}, can we scope {project}?",
             "Planning: {project} milestones.", "What does {project} need from {group}?"),
    "handoff": ("Handing {project} over to you, {other}.",
                "{other}, {group} is passing {project} across -- context here.",
                "Cross-team ask: {project} needs a hand from {group2}.",
                "Introducing {project} to your side, {other}."),
}

_MIDDLES: dict[str, tuple[str, ...]] = {
    "company": ("I can take the {project} piece if {other} takes {org}.",
                "{org} will ask about this; let us have an answer first.",
                "Checked with {group} -- they are fine either way.",
                "We said the same thing last quarter about {project}.",
                "Can we keep {place} out of this round?",
                "The {project} dashboard says otherwise, for what it is worth.",
                "I will write it up after the {org} call."),
    "community": ("I had the same problem with {topic}; happy to share notes.",
                  "{other} knows more about {topic} than I do.",
                  "There is a good thread on this from the {place} folks.",
                  "Not my area, but {project} did something similar.",
                  "Happy to make an intro if that helps.",
                  "Bump -- still curious about {topic}.",
                  "Thanks all, this is exactly why I joined."),
    "university": ("The reviewers will ask about {topic}; we need a paragraph.",
                   "{other} has the data from the {place} study.",
                   "Can we cite the {project} preprint here?",
                   "The department is fine with the dates, checked today.",
                   "I will run the {topic} numbers again tonight.",
                   "Office hours clash; can we do it after the {topic} lecture?",
                   "The figure for {project} is ready."),
    "open-source": ("CI is green on {project} after the rebase.",
                    "Left comments on the diff; mostly naming.",
                    "{other} maintains that part of {project}, defer to them.",
                    "This touches the {topic} path -- needs a test.",
                    "Let us not block the release on it.",
                    "Tagged it good-first-issue for {first}.",
                    "Changelog entry added for {project}."),
    "nonprofit": ("The field team in {place} needs a week's notice.",
                  "{org} said yes in principle; paperwork next.",
                  "Can we fit this into the {project} budget?",
                  "{other}, the volunteers will ask about transport.",
                  "The board will want a number, not a story.",
                  "Comms can turn this round by Monday.",
                  "Let us keep {place} as the pilot."),
}

_GENERIC: tuple[str, ...] = (
    "Agreed, {other}.", "Works for me.", "Can we decide by end of day?",
    "I would rather we did not guess at {topic}.", "Happy to take that.",
    "Let us keep it small.", "Who else needs to know about {project}?",
    "Same as what {other} said, from my side.", "Noted; I will follow up.",
    "One more thing on {project}: dates.", "Fine by me if {group} are fine.",
    "Let me check and come back today.", "That matches what I saw in {place}.",
    "Push back if this is wrong.", "Can we take this to a call?",
)

_CLOSERS: dict[str, tuple[str, ...]] = {
    "decision": ("Decision: we go with {other}'s plan for {project}. Writing it down.",
                 "Settled -- {project} as discussed, {other} owns it.",
                 "OK, that is the call on {project}. Thanks all.",
                 "Let us lock it: {project}, {place}, next week."),
    "moved_to": ("Done: {first} is now with {group2}. Welcome across.",
                 "Move confirmed -- {first} to {group2} from Monday.",
                 "{first}'s move to {group2} is official. Thanks {group}."),
    "now_works_with": ("{first} and {second} are going to work on {project} together.",
                       "Good -- {first} and {second}, you are a pair now.",
                       "Great, so {first} and {second} take it from here."),
    "joined": ("Official: {first} is in {group}. Welcome!",
               "{first} is one of us now. Glad to have you.",
               "That is {first} fully onboarded in {group}."),
    "": ("Thanks, all.", "Sorted, then.", "Good -- talk tomorrow.",
         "Leaving it there for today.", "Cheers {other}.", "That answers it."),
}

# A touch of voice, keyed by words a persona's ``voice`` sentence tends to carry. Measured
# nothing; it exists so two people with different voices do not read identically.
_VOICES: dict[str, tuple[tuple[str, str], ...]] = {
    "terse": (("", ""), ("", " Done."), ("Short version: ", "")),
    "blunt": (("Bluntly: ", ""), ("", " No."), ("", " That is it.")),
    "warm": (("Hey -- ", ""), ("", " Thanks for bearing with me."), ("", " Appreciate it.")),
    "formal": (("For the record, ", ""), ("", " Kind regards."), ("To confirm: ", "")),
    "cheerful": (("Ooh, ", ""), ("", " Exciting!"), ("", " Love it.")),
    "careful": (("If I have this right, ", ""), ("", " Correct me if not."),
                ("Tentatively: ", "")),
    "dry": (("", " Naturally."), ("As ever, ", ""), ("", " What could go wrong.")),
}

_TAILS = (" (again)", " -- as I said", " -- repeating myself", " -- still true")


def template_writer(rng: random.Random) -> Writer:
    """A writer that needs no model.

    Sentences per conversation kind and organisation kind, filled with names read out of
    the graph -- the speaker's project, place, subject, the group and the person they are
    talking to -- so what it writes is about something true. A thread never gets the same
    sentence twice: candidates already used in the thread are skipped, and when every one
    has been, a varying tail is added, then a count, so the guarantee holds however long a
    thread runs.

    Most messages also state one relation the graph holds among the people in the thread,
    appended as a plain sentence naming both ends in full -- "Ilva Rendevaan reports to
    Osun Klaithe." -- drawn from ``context["truths"]`` and never the same one twice in a
    thread. That is what an extraction has a relation to find at all; without it a
    templated corpus asserts entities and almost no edges.

    After each call ``write.last`` is ``{"ids": [...], "relations": [[s, rel, t]]}``: the
    graph ids of exactly the slots the chosen sentence was filled with -- the project, the
    place, the person addressed -- plus both ends of the relation it stated, and that
    relation as a triple, in the world's own vocabulary. That is what the message asserts
    and what an extraction of it is scored against. A slot filled with a fallback ("the
    project") names nothing.
    """
    used: dict[str, set[str]] = {}
    told: dict[str, set[tuple[str, str, str]]] = {}

    def write(persona: Mapping[str, Any], prompt: str, context: Mapping[str, Any]) -> str:
        write.last = {"ids": [], "relations": []}  # type: ignore[attr-defined]
        thread = str(context.get("thread") or "")
        if thread not in used:
            if len(used) > 64:
                used.clear()
                told.clear()
            used[thread] = set()
            told[thread] = set()
        seen, stated = used[thread], told.setdefault(thread, set())
        kind = str(context.get("kind") or "ask")
        org_kind = str(context.get("org_kind") or "company")
        seq, of = int(context.get("seq") or 0), int(context.get("of") or 2)
        outcome = str(context.get("outcome") or "") if context.get("last") else ""
        if seq == 0:
            pool = _OPENERS.get(kind) or _OPENERS["ask"]
        elif seq == of - 1:
            pool = _CLOSERS.get(outcome) or _CLOSERS[""]
        else:
            pool = _MIDDLES.get(org_kind, ()) + _GENERIC
        slots = _slots(persona, context)
        ids = _slot_ids(persona, context)
        candidates = [(t, t.format(**slots)) for t in pool]
        rng.shuffle(candidates)
        voice = str(persona.get("voice") or "").casefold()
        flavours = [f for word, fs in _VOICES.items() if word in voice for f in fs]

        def said(template: str, sentence: str) -> str:
            seen.add(sentence)
            named, relations = _filled(template, ids), []
            fact = _fact(context, stated, rng)
            if fact:
                stated.add(fact)
                source, rel, target = fact
                sentence += " " + rng.choice(_STATED[rel]).format(
                    a=_label(context, source), b=_label(context, target))
                named += [i for i in (source, target) if i not in named]
                relations = [[source, rel, target]]
            write.last = {"ids": named, "relations": relations}  # type: ignore[attr-defined]
            return sentence

        for template, sentence in candidates:
            if flavours and rng.random() < 0.35:
                head, tail = rng.choice(flavours)
                sentence = (head + sentence[0].lower() + sentence[1:] if head else sentence) + tail
            if sentence not in seen:
                return said(template, sentence)
        template, base = candidates[0]
        for tail in _TAILS:
            if base + tail not in seen:
                return said(template, base + tail)
        n = 2
        while f"{base} ({n})" in seen:
            n += 1
        return said(template, f"{base} ({n})")

    write.last = {"ids": [], "relations": []}  # type: ignore[attr-defined]
    return write


def _label(context: Mapping[str, Any], node_id: str) -> str:
    """How a message names an entry: the label the graph gave it, else its id."""
    return str((context.get("labels") or {}).get(node_id) or node_id)


def _fact(context: Mapping[str, Any], stated: set[tuple[str, str, str]],
          rng: random.Random) -> tuple[str, str, str] | None:
    """One relation this message states, or None: a `STATE` share of them state one.

    Drawn from the truths the thread's people carry, never one the thread has already
    said, so a long thread walks through what is true about the people in it rather than
    repeating the first fact eight times.
    """
    fresh = [tuple(f) for f in (context.get("truths") or ()) if tuple(f) not in stated]
    if not fresh or rng.random() >= STATE:
        return None
    shape = [f for f in fresh if f[1] in _SHAPE]
    rest = [f for f in fresh if f[1] not in _SHAPE]
    pool = shape if shape and (not rest or rng.random() < STRUCTURAL) else rest or shape
    picked = pool[rng.randrange(len(pool))]
    return (str(picked[0]), str(picked[1]), str(picked[2]))


def _slot_ids(persona: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, str]:
    """The graph id behind each slot a template can name, "" where the slot would be
    filled with a fallback or with somebody the thread cannot tell apart."""
    facts = dict(context.get("facts") or {})
    labels = context.get("labels") or {}
    me = str(context.get("speaker") or persona.get("id") or "")
    others = [str(p) for p in (context.get("others") or ()) if str(p) != me]
    who = [p for p in (me, *others) if p]

    def by_first(name: str) -> str:
        hits = [p for p in who if str(labels.get(p, "")).split()[:1] == [name]] if name else []
        return hits[0] if len(hits) == 1 else ""

    out = {"me": me, "other": others[0] if others else ""}
    for slot in ("project", "org", "topic", "place", "group", "group2"):
        out[slot] = str(facts.get(slot + "_id") or "")
    other_first = str(labels.get(others[0], "")).split()[:1] if others else []
    out["first"] = by_first(str(facts.get("first") or (other_first[0] if other_first else "")))
    out["second"] = by_first(str(facts.get("second") or ""))
    return out


def _filled(template: str, ids: Mapping[str, str]) -> list[str]:
    """The ids of the slots ``template`` names, in the order it names them, each once."""
    out: list[str] = []
    for _, name, _, _ in string.Formatter().parse(template):
        held = ids.get(str(name or ""), "")
        if held and held not in out:
            out.append(held)
    return out


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


def _slots(persona: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, str]:
    facts = dict(context.get("facts") or {})
    labels = context.get("labels") or {}
    me = str(context.get("speaker") or "")
    others = [str(p) for p in (context.get("others") or ()) if str(p) != me]
    other = str(labels.get(others[0], others[0]) if others else "all").split()[0]
    facts.setdefault("project", "the project")
    facts.setdefault("place", "the office")
    facts.setdefault("org", "the customer")
    facts.setdefault("topic", "the usual")
    facts.setdefault("group", "the team")
    facts.setdefault("group2", "the other team")
    facts.setdefault("first", other)
    facts.setdefault("second", other)
    facts["other"] = other
    facts["me"] = str(labels.get(me, persona.get("label") or me)).split()[0] if me else "me"
    return facts


# -- the model writer ----------------------------------------------------------------------------

class _Counting:
    """A client that counts what it is asked, so a run can say what a message cost."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls = 0

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.client.chat(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


def _subgraph(graph: Mapping[str, Any], ids: Sequence[str]) -> dict[str, Any]:
    """The part of ``graph`` a persona knows: those nodes, the edges among them, their messages."""
    keep = {str(i) for i in ids}
    nodes = [n for n in (graph.get("nodes") or ()) if str(n.get("id")) in keep]
    edges = [e for e in (graph.get("edges") or ())
             if str(e.get("source")) in keep and str(e.get("target")) in keep]
    held = graph.get("messages") or {}
    wanted = {m for n in nodes for m in (n.get("messages") or ())}
    return {"nodes": nodes, "edges": edges,
            "messages": {m: held[m] for m in wanted if m in held}}


class ModelWriter:
    """A persona speaking through the tool loop over the subgraph it knows.

    Each call is one `converse`: the persona's system prompt, its memory of earlier
    threads in the same arc (read from ``store``) and the thread so far as turns, the tools
    over its subgraph, what the thread is about handed over as the ``opening`` -- its own
    entry, the project, the place, the people in the thread -- and the prompt asking for
    one or two sentences in its voice. The reply is stripped of planning and of any
    written-out ``show`` call; an empty reply is returned empty and the simulation falls
    back to a template for that message.

    The opening is there for cost as much as grounding: a persona answering from its
    turns calls no tool, and `converse` sends an answer that touched nothing back to look
    (two more calls) -- unless it was handed something first. What remains is two calls a
    message: the answer, then `converse` asking what the answer was about, and that
    second call is kept as ``last`` because its ids are exactly the ``Drew`` edges the
    thread memory wants. ``calls`` counts round trips and ``messages`` what they produced.
    """

    def __init__(self, client: Any, world: World, store: Any = None, *,
                 rounds: int = ROUNDS, memory: int = 2) -> None:
        self.client = _Counting(client)
        self.world = world
        self.store = store
        self.rounds = rounds
        self.memory = memory
        self.messages = 0
        self.last: Any = None
        # what the last message asserts, as far as is known: the opening it was grounded
        # in and every id its answer drew on -- a lower bound, since the persona may have
        # named more than it looked up
        self.last_ids: list[str] = []
        self._subgraphs: dict[tuple[str, int, int], dict[str, Any]] = {}

    @property
    def calls(self) -> int:
        return self.client.calls

    def known(self, persona: Mapping[str, Any]) -> dict[str, Any]:
        """The subgraph this persona knows, recomputed when the graph has grown."""
        graph = self.world.graph
        key = (str(persona.get("id") or ""), len(graph.get("nodes") or ()),
               len(graph.get("edges") or ()))
        if key not in self._subgraphs:
            ids = list(persona.get("knows") or [])
            if persona.get("id") and persona["id"] not in ids:
                ids.append(persona["id"])
            self._subgraphs.clear()
            self._subgraphs[key] = _subgraph(graph, ids)
        return self._subgraphs[key]

    def remembered(self, persona: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, str]]:
        """Turns from this persona's earlier threads in the same arc, oldest first."""
        arc_key = str(context.get("arc_key") or "")
        if self.store is None or not arc_key:
            return []
        from ml_stack.graph.thread import follow, threads

        me = str(persona.get("id") or "")
        current = str(context.get("thread") or "")
        names = [t["thread"] for t in threads(self.store)
                 if t["thread"].startswith(arc_key + "/") and not t["thread"].endswith("/" + current)]
        out: list[dict[str, str]] = []
        for name in sorted(names)[-self.memory:]:
            turns = follow(self.store, name, working=False)
            if not any(me in (t.meta.get("who") or ()) for t in turns):
                continue
            out.extend({"role": "assistant" if t.meta.get("speaker") == me else "user",
                        "content": t.text} for t in turns)
        return out

    def __call__(self, persona: Mapping[str, Any], prompt: str, context: Mapping[str, Any]) -> str:
        from ml_stack.graph.ask import SYSTEM, converse, spoken_show, tools_for, without_notes

        graph = self.known(persona)
        me = str(persona.get("id") or "")
        labels = context.get("labels") or {}
        turns = self.remembered(persona, context)
        turns.extend({"role": "assistant" if speaker == me else "user",
                      "content": f"{labels.get(speaker, speaker)}: {text}"}
                     for speaker, text in (context.get("said") or ()))
        known = {str(n.get("id")) for n in graph.get("nodes") or ()}
        facts = context.get("facts") or {}
        opening = [me, *(str(facts.get(k) or "") for k in ("project_id", "place_id",
                                                             "topic_id", "group_id")),
                   *(str(p) for p in context.get("others") or ())]
        opening = list(dict.fromkeys(i for i in opening if i in known))
        answer = converse(prompt, graph, self.client, turns=turns,
                          system=str(persona.get("system") or SYSTEM),
                          tools=tools_for(graph), rounds=self.rounds, opening=opening)
        self.last = answer
        from ml_stack.graph.thread import drew_on

        drawn = [i for ids in drew_on(answer).values() for i in ids]
        self.last_ids = list(dict.fromkeys(i for i in (*opening, *drawn) if i in known))
        text, _ids = spoken_show(without_notes(answer.content))
        text = " ".join(text.split())
        if text:
            self.messages += 1
        return text


def model_writer(client: Any, world: World, store: Any = None, *,
                 rounds: int = ROUNDS) -> ModelWriter:
    """`ModelWriter` over ``client`` for that world; ``store`` is where memory is read from."""
    return ModelWriter(client, world, store, rounds=rounds)


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
    from ml_stack.graph.tidy import absorb

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
    ``world.json`` (``kind``, ``size``, ``people``) from ``world_dir``; writes
    ``messages.jsonl`` (one `Message` a line), the updated ``graph.json`` and the
    ``calendar.json`` used into ``out_dir``. With ``model_url`` and ``mix > 0`` the model
    share is written by `model_writer` with memory in ``out_dir/memory.ladybug``, under
    `ml_stack.lock.only_one` on ``out_dir/simulate.lock`` so two runs never share a model.
    Before that memory is written, `_absorbed` reconciles this world's graph against
    whatever it already holds -- a second run, or a different invented world sharing the
    same store -- so the same person or organisation under a plural or a case variant lands
    on the node already there rather than doubling it. ``judge`` is a `ml_stack.graph.tidy
    .ModelJudge` for the close spellings a plain match cannot settle; without one those are
    left as new nodes and reported, the way `absorb` always leaves them.
    Returns the counts: threads, messages, the model/template split, outcomes, and what a
    message cost in model calls.
    """
    from ml_stack.files import read_json, write_json

    world_dir, out_dir = Path(world_dir).expanduser(), Path(out_dir).expanduser()
    graph = read_json(world_dir / "graph.json", None)
    if not isinstance(graph, Mapping):
        raise FileNotFoundError(f"no graph.json in {world_dir}")
    personas = read_json(world_dir / "personas.json", {}) or {}
    about = read_json(world_dir / "world.json", {}) or {}
    kind = str(about.get("kind") or (graph.get("meta") or {}).get("kind") or "company")
    people = [str(p) for p in (about.get("people") or list(personas) or people_of(graph))]
    world = World(graph=dict(graph), people=people, personas=dict(personas),
                  calendar=list(read_json(world_dir / "calendar.json", []) or []),
                  seed=seed, size=str(about.get("size") or "small"), kind=kind)
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
