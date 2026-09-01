"""Reasons for the people of an invented organisation to talk to each other.

Chatter with nothing behind it reads as noise: the same greetings, threads that go nowhere.
What makes a real workspace legible is that most conversations are *about* something with a
start and an end -- a launch, an outage, a new arrival -- and that the same people keep
coming back to it. `calendar` lays such arcs over a run of days, picks who is in each from
the groups and relations the graph holds, and says where the talking happens; `world.simulate`
does the talking.

What there is to talk about depends on what the organisation is. A company has launches and
incidents; a community has introductions and questions that get answered; a university has
deadlines and defences; an open-source project has releases and RFCs; a nonprofit has
fundraisers and board meetings. `ARCS` holds one table per kind.

Nothing here assumes a schema beyond ``nodes`` and ``edges``. A "group" is any node that is
not a person and has people joined to it -- a department, a lab, a channel, a chapter -- and
groups are named by words in their labels, so an incident finds "engineering" and "support"
whatever the graph calls their kinds. Every name in an arc is read out of the graph, which
invented it.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["ARCS", "Arc", "KINDS", "OUTCOMES", "calendar", "facts_for", "groups", "kind_of",
           "people_of", "place_of", "slug"]

# The organised groups this knows how to schedule a life for.
KINDS = ("company", "community", "university", "open-source", "nonprofit")

# The facts an arc's end can write back into the graph, as edge relations. Small and typed,
# so a reader of the graph -- and the next conversation -- can tell them apart from chatter.
OUTCOMES = ("decision", "moved_to", "now_works_with", "joined")

# Node kinds that stand for the things arcs are about. A graph may call them anything; these
# are the spellings tried, in order, before falling back to "whatever people are joined to".
GROUP_KINDS = ("department", "team", "group", "lab", "channel", "chapter", "committee",
               "board", "working_group", "programme", "program", "faculty", "guild", "squad")
PROJECT_KINDS = ("project", "repo", "repository", "product", "paper", "grant", "event",
                 "course", "release", "campaign", "opportunity")
PLACE_KINDS = ("place", "city", "office", "location", "campus")
ORG_KINDS = ("org", "organisation", "organization", "customer", "company", "sponsor",
             "funder", "partner")
TOPIC_KINDS = ("topic", "skill", "subject", "interest")

# Relations that mean "these two do things together", "one answers to the other", and "this
# person is in that group". A community has none of the second; the sampler only ever uses
# what it finds.
PEER = frozenset({"works_with", "now_works_with", "collaborates_with", "pairs_with",
                  "co_authors", "reviews_for", "co_maintains"})
UPWARD = frozenset({"reports_to", "advised_by", "supervised_by", "mentored_by", "managed_by"})
DOWNWARD = frozenset({"manages", "advises", "supervises", "mentors", "leads"})
MEMBERSHIP = frozenset({"member_of", "part_of", "in", "belongs_to", "works_in", "moderates",
                        "leads", "maintains", "studies_in", "volunteers_with", "sits_on",
                        "chairs", "runs", "joined", "moved_to", "works_on", "contributes_to"})


@dataclass(frozen=True)
class Arc:
    """One kind of thing worth talking about, and who it pulls in.

    ``groups`` names the groups that take part, each as the words that would appear in
    such a group's label (any of them matches); ``size`` bounds how many people; ``where``
    is the venues a thread of it may land in, as ``(source, template)``; ``about`` and the
    templates fill from `facts_for`. ``outcome`` is what its end writes to the graph, or
    None. ``newcomer`` puts the least-joined person in the graph first in ``who``.
    """

    kind: str
    groups: tuple[tuple[str, ...], ...]
    size: tuple[int, int]
    where: tuple[tuple[str, str], ...]
    about: str
    outcome: str | None = "decision"
    span: tuple[int, int] = (1, 3)
    newcomer: bool = False


_ENG = ("engineering", "platform", "infra", "backend", "dev")
_SUPPORT = ("support", "success", "operations", "ops", "helpdesk")
_PRODUCT = ("product", "design")
_MARKETING = ("marketing", "comms", "communications", "growth", "brand")
_SALES = ("sales", "revenue", "partnerships", "account")
_LEADERS = ("leadership", "exec", "management", "directors", "board")
_MODS = ("moderators", "admins", "organisers", "organizers", "stewards", "core")
_MAINT = ("maintainers", "core", "committers")
_DEV = ("development", "fundraising", "advancement", "donors")
_PROGRAMMES = ("programmes", "programs", "services", "outreach", "field")
_VOLUNTEERS = ("volunteers", "community")
_LAB = ("lab", "group", "research")
_DEPT = ("department", "faculty", "school")
_ANY = ()

ARCS: dict[str, tuple[Arc, ...]] = {
    "company": (
        Arc("launch", (_PRODUCT, _MARKETING, _SALES), (4, 7),
            (("slack", "launch-{project_slug}"), ("email", "Launch plan: {project}"),
             ("teams", "{project_slug}-launch")),
            "the launch of {project}", span=(3, 5)),
        Arc("incident", (_ENG, _SUPPORT), (3, 6),
            (("slack", "incidents"), ("teams", "{project_slug}-war-room"),
             ("email", "Incident report: {project}")),
            "the outage in {project}", span=(1, 2)),
        Arc("new_hire", (_ANY,), (3, 5),
            (("slack", "{group_slug}"), ("slack", "dm"), ("email", "Welcome, {first}")),
            "{first}'s first week in {group}", outcome="joined", span=(4, 5), newcomer=True),
        Arc("escalation", (_SUPPORT, _SALES), (3, 5),
            (("email", "Escalation from {org}"), ("teams", "{org_slug}-escalation"),
             ("slack", "customers")),
            "the escalation from {org} about {project}", span=(2, 3)),
        Arc("offsite", (_ANY,), (5, 8),
            (("slack", "offsite-{place_slug}"), ("email", "Offsite in {place}")),
            "the offsite in {place}", span=(2, 3)),
        Arc("quarterly_review", (_LEADERS, _ENG, _SALES), (3, 6),
            (("email", "Quarterly review: {group}"), ("teams", "quarterly-review")),
            "the quarterly review of {group}", span=(1, 2)),
        Arc("reorg", (_ENG, _PRODUCT), (4, 6),
            (("email", "Changes to {group} and {group2}"), ("slack", "announcements")),
            "{first} moving from {group} to {group2}", outcome="moved_to", span=(2, 3)),
        Arc("deadline_slip", (_ENG, _PRODUCT), (3, 5),
            (("slack", "{project_slug}"), ("email", "{project}: revised dates")),
            "the slipped deadline on {project}", span=(2, 4)),
    ),
    "community": (
        Arc("intro", (_MODS,), (2, 4),
            (("slack", "introductions"), ("slack", "dm")),
            "{first} introducing themselves", outcome="joined", span=(1, 2), newcomer=True),
        Arc("question", (_ANY,), (3, 5),
            (("slack", "help"), ("slack", "{topic_slug}")),
            "{first}'s question about {topic}", span=(1, 2)),
        Arc("meetup", (_MODS, _ANY), (4, 8),
            (("slack", "events"), ("email", "Meetup in {place}")),
            "the meetup in {place}", span=(2, 4)),
        Arc("job_post", (_ANY,), (2, 4),
            (("slack", "jobs"), ("email", "Re: role at {org}")),
            "the role at {org}", outcome="now_works_with", span=(1, 3)),
        Arc("recommendation", (_ANY,), (3, 5),
            (("slack", "recommendations"), ("slack", "{topic_slug}")),
            "who to ask about {topic}", span=(1, 2)),
        Arc("intro_between", (_ANY,), (3, 3),
            (("email", "Intro: {first} and {second}"), ("slack", "dm")),
            "{first} and {second} being introduced", outcome="now_works_with", span=(1, 2)),
    ),
    "university": (
        Arc("paper_deadline", (_LAB,), (3, 5),
            (("slack", "{group_slug}"), ("email", "{project}: submission")),
            "the submission of {project}", span=(3, 5)),
        Arc("grant", (_LAB, _DEPT), (3, 5),
            (("email", "Grant proposal: {project}"), ("teams", "{project_slug}-proposal")),
            "the grant proposal for {project}", span=(2, 4)),
        Arc("seminar", (_DEPT, _LAB), (4, 8),
            (("email", "Seminar: {topic}"), ("slack", "seminars")),
            "the seminar on {topic}", span=(1, 2)),
        Arc("defence", (_LAB, _DEPT), (4, 6),
            (("email", "Thesis defence: {first}"), ("teams", "{first_slug}-defence")),
            "{first}'s thesis defence", span=(1, 3)),
        Arc("lab_move", (_LAB, _LAB), (3, 5),
            (("email", "{first} moving to {group2}"), ("slack", "{group_slug}")),
            "{first} moving from {group} to {group2}", outcome="moved_to", span=(2, 3)),
    ),
    "open-source": (
        Arc("release", (_MAINT,), (3, 6),
            (("slack", "releases"), ("email", "[{project}] release")),
            "the next release of {project}", span=(2, 4)),
        Arc("bug_fix", (_ANY, _MAINT), (2, 4),
            (("slack", "bugs"), ("teams", "{project_slug}-triage")),
            "the bug {first} found in {project}", span=(1, 3)),
        Arc("rfc", (_MAINT, _ANY), (4, 7),
            (("email", "RFC: {topic} in {project}"), ("slack", "design")),
            "the RFC on {topic}", span=(3, 5)),
        Arc("first_pr", (_MAINT,), (2, 3),
            (("slack", "contributors"), ("slack", "dm")),
            "{first}'s first pull request to {project}", outcome="joined", span=(1, 3),
            newcomer=True),
        Arc("advisory", (_MAINT,), (2, 4),
            (("email", "[security] {project}"), ("teams", "{project_slug}-security")),
            "the security advisory for {project}", span=(1, 2)),
    ),
    "nonprofit": (
        Arc("fundraiser", (_DEV, _MARKETING), (3, 6),
            (("slack", "fundraising"), ("email", "Fundraiser: {project}")),
            "the fundraiser for {project}", span=(3, 5)),
        Arc("programme_launch", (_PROGRAMMES, _MARKETING), (3, 6),
            (("slack", "{project_slug}"), ("email", "Launching {project} in {place}")),
            "launching {project} in {place}", span=(2, 4)),
        Arc("volunteer_drive", (_VOLUNTEERS, _PROGRAMMES), (3, 6),
            (("slack", "volunteers"), ("email", "Volunteers for {place}")),
            "{first} joining the volunteers in {place}", outcome="joined", span=(2, 3),
            newcomer=True),
        Arc("board_meeting", (_LEADERS,), (3, 6),
            (("email", "Board meeting: {project}"), ("teams", "board")),
            "the board's view of {project}", span=(1, 2)),
    ),
}


# -- reading the graph ---------------------------------------------------------------------

def slug(text: str) -> str:
    """``text`` as a channel or chat name: lower case, words joined by hyphens."""
    out = re.sub(r"[^a-z0-9]+", "-", str(text or "").casefold()).strip("-")
    return out[:40] or "general"


def _nodes(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(n["id"]): n for n in (graph.get("nodes") or ()) if n.get("id")}


def _rel(edge: Mapping[str, Any]) -> str:
    return str(edge.get("rel") or edge.get("relation") or "")


def kind_of(node: Mapping[str, Any] | None) -> str:
    """A node's kind, as written or as its ``type`` attribute says."""
    if node is None:
        return ""
    return str(node.get("kind") or (node.get("attrs") or {}).get("type") or "")


def people_of(graph: Mapping[str, Any]) -> list[str]:
    """Every person id, in graph order."""
    return [i for i, n in _nodes(graph).items() if kind_of(n) == "person"]


def groups(graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every group and who is in it: ``{id: {"label", "kind", "people": [ids]}}``.

    A group is a node of a group-like kind with people joined to it by any relation. When
    the graph has none of those kinds, any non-person node with two or more people joined
    to it stands in -- a project, a channel, a place -- so an unfamiliar schema still yields
    somebody to put in a room together.
    """
    by_id = _nodes(graph)
    joined: dict[str, list[str]] = {}
    for edge in graph.get("edges") or ():
        a, b = str(edge.get("source") or ""), str(edge.get("target") or "")
        if a not in by_id or b not in by_id:
            continue
        for person, other in ((a, b), (b, a)):
            if kind_of(by_id[person]) == "person" and kind_of(by_id[other]) != "person":
                members = joined.setdefault(other, [])
                if person not in members:
                    members.append(person)

    def held(ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        return {i: {"label": str(by_id[i].get("label") or i), "kind": kind_of(by_id[i]),
                    "people": list(joined[i])} for i in ids}

    proper = [i for i in joined if kind_of(by_id[i]) in GROUP_KINDS]
    if proper:
        return held(proper)
    return held(i for i, members in joined.items()
                if len(members) >= 2 and kind_of(by_id[i]) not in (*PLACE_KINDS, *TOPIC_KINDS))


def _joined_to(graph: Mapping[str, Any], person: str, kinds: Sequence[str]) -> list[str]:
    """Ids of those kinds joined to ``person`` by any edge, either way round."""
    by_id = _nodes(graph)
    out: list[str] = []
    for edge in graph.get("edges") or ():
        a, b = str(edge.get("source") or ""), str(edge.get("target") or "")
        other = b if a == person else a if b == person else ""
        if other and other in by_id and kind_of(by_id[other]) in kinds and other not in out:
            out.append(other)
    return out


def place_of(graph: Mapping[str, Any], person: str) -> Mapping[str, Any] | None:
    """The place node a person is joined to -- directly, or through their organisation."""
    by_id = _nodes(graph)
    direct = _joined_to(graph, person, PLACE_KINDS)
    if direct:
        return by_id[direct[0]]
    for org in _joined_to(graph, person, ORG_KINDS):
        through = _joined_to(graph, org, PLACE_KINDS)
        if through:
            return by_id[through[0]]
    return None


def _of_kinds(graph: Mapping[str, Any], kinds: Sequence[str]) -> list[str]:
    return [i for i, n in _nodes(graph).items() if kind_of(n) in kinds]


def facts_for(graph: Mapping[str, Any], who: Sequence[str], rng: random.Random, *,
              group: str = "", group2: str = "") -> dict[str, str]:
    """The names a conversation among ``who`` can be grounded in, read out of the graph.

    Prefers what the first of ``who`` is joined to -- their project, their place, their
    subject -- and falls back to anything of that kind in the graph, and then to a neutral
    word, so a template always fills and fills with something true wherever it can. Also
    carries each name's ``_id`` and ``_slug`` so outcomes and channel names can use them.
    """
    by_id = _nodes(graph)
    label = lambda i: str(by_id[i].get("label") or i) if i in by_id else ""  # noqa: E731
    lead = who[0] if who else ""
    out: dict[str, str] = {}

    def choose(name: str, kinds: Sequence[str], fallback: str) -> None:
        own = _joined_to(graph, lead, kinds) if lead else []
        pool = own or _of_kinds(graph, kinds)
        picked = rng.choice(pool) if pool else ""
        out[name] = label(picked) or fallback
        out[name + "_id"] = picked
        out[name + "_slug"] = slug(out[name])

    choose("project", PROJECT_KINDS, "the project")
    choose("org", ORG_KINDS, "the customer")
    choose("topic", TOPIC_KINDS, "the usual")
    place = place_of(graph, lead) if lead else None
    if place is None:
        anywhere = _of_kinds(graph, PLACE_KINDS)
        place = by_id[rng.choice(anywhere)] if anywhere else None
    out["place"] = str(place.get("label") or "the office") if place else "the office"
    out["place_id"] = str(place.get("id") or "") if place else ""
    out["place_slug"] = slug(out["place"])
    for name, gid in (("group", group), ("group2", group2)):
        out[name] = label(gid) or ("the team" if name == "group" else "the other team")
        out[name + "_id"] = gid
        out[name + "_slug"] = slug(out[name])
    names = [label(p) for p in who]
    out["first"] = (names[0].split() or ["somebody"])[0] if names else "somebody"
    out["first_slug"] = slug(out["first"])
    out["second"] = (names[1].split() or ["somebody"])[0] if len(names) > 1 else "somebody"
    out["names"] = ", ".join(n for n in names if n)
    return out


# -- scheduling -----------------------------------------------------------------------------

def _match_group(held: Mapping[str, Mapping[str, Any]], words: Sequence[str],
                 rng: random.Random, taken: Iterable[str] = ()) -> str:
    """A group whose label carries one of ``words``; any group when none does or words is empty."""
    avoid = set(taken)
    free = [g for g in held if g not in avoid] or list(held)
    if not free:
        return ""
    if words:
        hits = [g for g in free if any(w in held[g]["label"].casefold() for w in words)]
        if hits:
            return rng.choice(hits)
    return rng.choice(free)


def _least_joined(graph: Mapping[str, Any], people: Sequence[str], rng: random.Random) -> str:
    """The person the graph knows least about: fewest edges, then fewest mentions."""
    degree: dict[str, int] = {p: 0 for p in people}
    for edge in graph.get("edges") or ():
        for end in (str(edge.get("source") or ""), str(edge.get("target") or "")):
            if end in degree:
                degree[end] += 1
    by_id = _nodes(graph)
    order = sorted(people, key=lambda p: (degree[p], int(by_id[p].get("mentions") or 0)))
    low = [p for p in order if degree[p] == degree[order[0]]]
    return rng.choice(low)


def _weekdays(days: int) -> list[int]:
    # day 0 is a Monday; nobody launches on a Saturday
    return [d for d in range(days) if d % 7 < 5]


def calendar(world: Any, days: int, rng: random.Random) -> list[dict[str, Any]]:
    """Arcs over ``days`` for that world, in day order, reproducible from ``rng``.

    Each is ``{"day", "until", "kind", "who", "about", "where", "outcome", "subject", "to",
    "group"}``: the people are drawn from the groups the arc names (an incident is
    engineering and support; a launch is product, marketing and sales), the venues from the
    arc's templates filled with the graph's own names, and ``subject`` / ``to`` are the node
    ids the outcome will be written against. Roughly one arc every four working days, one
    more per twenty people, and never on a weekend.
    """
    kind = str(getattr(world, "kind", "") or "company")
    if kind not in ARCS:
        raise ValueError(f"no arcs for a {kind!r}; known kinds: {', '.join(ARCS)}")
    graph = world.graph
    people = list(getattr(world, "people", None) or people_of(graph))
    if not people:
        return []
    held = groups(graph)
    working = _weekdays(days)
    if not working:
        return []
    wanted = max(1, round(len(working) / 4) + len(people) // 20)

    order = list(ARCS[kind])
    rng.shuffle(order)
    out: list[dict[str, Any]] = []
    for n in range(wanted):
        arc = order[n % len(order)]
        chosen: list[str] = []
        for words in arc.groups:
            gid = _match_group(held, words, rng, taken=chosen)
            if gid and gid not in chosen:
                chosen.append(gid)
        pool: list[str] = []
        for gid in chosen:
            pool.extend(p for p in held[gid]["people"] if p not in pool and p in people)
        low, high = arc.size
        size = rng.randint(low, high)
        who: list[str] = []
        if arc.newcomer:
            who.append(_least_joined(graph, people, rng))
        rest = [p for p in pool if p not in who]
        rng.shuffle(rest)
        who.extend(rest[:max(0, size - len(who))])
        if len(who) < low:
            spare = [p for p in people if p not in who]
            rng.shuffle(spare)
            who.extend(spare[:low - len(who)])
        group = chosen[0] if chosen else ""
        group2 = chosen[1] if len(chosen) > 1 else ""
        facts = facts_for(graph, who, rng, group=group, group2=group2)
        start = rng.choice(working)
        until = min(start + rng.randint(*arc.span) - 1, days - 1)
        where = [(source, template.format(**facts)) for source, template in arc.where]
        subject = facts["project_id"] or group or (who[1] if len(who) > 1 else who[0])
        out.append({"day": start, "until": until, "kind": arc.kind, "who": who,
                    "about": arc.about.format(**facts), "where": where,
                    "outcome": arc.outcome, "subject": subject, "to": group2 or group,
                    "group": group, "facts": {k: v for k, v in facts.items()
                                             if not k.endswith("_id")}})
    out.sort(key=lambda a: (a["day"], a["kind"]))
    return out
