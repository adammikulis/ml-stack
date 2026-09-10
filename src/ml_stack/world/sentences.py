"""The sentences a message is made of when no model writes it.

An opener per conversation kind, middles per organisation kind, closers per outcome, a
touch of voice, and a plain sentence for every relation the graph can hold. `template_writer`
fills the slots with names read out of the graph -- the speaker's project, place, subject,
group and the person addressed -- and records the ids of exactly what it named, so a message
written without a model still asserts something an extraction can be scored against.
"""

from __future__ import annotations

import random
import string
from collections.abc import Mapping
from typing import Any

from ml_stack.world import Writer

__all__ = ["STATE", "STRUCTURAL", "template_writer"]


# -- the relations a message states ------------------------------------------------------------

# Plain sentences that state one relation the graph already holds, naming both ends in
# full. The key is the relation as `world.kinds` spells it, which is what the gold carries.
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

# The share of messages that state one such relation; at most one a message, and never
# one already stated in its own thread.
STATE = 0.7

# Relations about how the organisation is put together, rather than about one person.
# These are preferred `STRUCTURAL` of the time when the thread has one to state.
_SHAPE = frozenset({"works_with", "now_works_with", "reports_to", "advises", "mentors",
                    "leads", "chairs", "sits_on", "moderates", "maintains", "part_of",
                    "member_of",
                    "works_on", "contributes_to", "joined", "moved_to"})
STRUCTURAL = 0.55


# -- what a message is made of -------------------------------------------------------------------

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
    "decision": ("Decision: we go with {first}'s plan for {project}. Writing it down.",
                 "Settled -- {project} as discussed, {first} owns it.",
                 "{first} made the call on {project}. Thanks all.",
                 "Let us lock it: {first} takes {project}, {place}, next week."),
    "moved_to": ("Done: {first} is now with {group2}. Welcome across.",
                 "Move confirmed -- {first} to {group2} from Monday.",
                 "{first}'s move to {group2} is official. Thanks {group}."),
    "now_works_with": ("{first} and {second} are going to work on {project} together.",
                       "Good -- {first} and {second}, you are a pair now.",
                       "Great, so {first} and {second} take it from here."),
    "joined": ("Official: {first} is in {group}. Welcome!",
               "{first} is one of us in {group} now. Glad to have you.",
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

    Sentences per conversation kind and organisation kind, filled with names read out of the
    graph, so what it writes is about something true. A thread never gets the same sentence
    twice: one already used is skipped, then a varying tail is added, then a count. A `STATE`
    share also state one relation the graph holds among the people in the thread, as a plain
    sentence naming both ends in full and never the same one twice in a thread.

    After each call ``write.last`` is ``{"ids": [...], "relations": [[s, rel, t]]}``: the
    graph ids of exactly the slots the sentence was filled with, plus both ends of the
    relation it stated. A slot filled with a fallback ("the project") names nothing.
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
                if head and not template.startswith("{"):
                    sentence = head + sentence[0].lower() + sentence[1:]
                sentence += tail
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
