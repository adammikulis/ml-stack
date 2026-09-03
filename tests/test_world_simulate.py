"""Days on which the people of an invented organisation talk to each other.

Every person, place and organisation is invented; the model is a stand-in that records what
it was handed and answers in words. Nothing reads a real graph or talks to a server.
"""

from __future__ import annotations

import dataclasses
import json
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from ml_stack.graph.ask import TIGHT_SYSTEM_SENTENCE
from ml_stack.world import Message, World
from ml_stack.world.simulate import (LONGEST, SHORTEST, WORK_END, WORK_START, model_writer, run,
                                     simulate, template_writer)
from ml_stack.world.story import ARCS, OUTCOMES, calendar, groups


def spoken_as(persona: str) -> str:
    """A persona's own system prompt as `converse` sends it -- tight, which is the asking:
    the sentence about naming only what a tool returned rides along, so a person in an
    invented world does not invent a name on top of it."""
    return persona + " " + TIGHT_SYSTEM_SENTENCE

PEOPLE = {
    "person:ada": ("Ada Lovelace", "eng", "place:turin", "terse and dry"),
    "person:bea": ("Bea Marlow", "eng", "place:turin", "warm"),
    "person:hedy": ("Hedy Marchetti", "eng", "place:turin", "careful"),
    "person:charles": ("Charles Babbage", "eng", "place:dunmore", "formal"),
    "person:oskar": ("Oskar Trent", "eng", "place:dunmore", "blunt"),
    "person:milo": ("Milo Fenwick", "eng", "place:dunmore", "cheerful"),
    "person:mary": ("Mary Somerville", "support", "place:turin", "warm"),
    "person:otto": ("Otto Vance", "support", "place:turin", "terse"),
    "person:nell": ("Nell Ashgrove", "support", "place:dunmore", "formal"),
    "person:bram": ("Bram Ostley", "support", "place:dunmore", "dry"),
    "person:iris": ("Iris Bellweather", "support", "place:dunmore", "cheerful"),
    "person:pell": ("Pell Grantham", "support", "place:turin", "careful"),
}
LEADS = {"eng": "person:ada", "support": "person:mary"}


def tiny_world(kind: str = "company") -> World:
    nodes = [{"id": "dept:eng", "kind": "department", "label": "Engineering", "mentions": 6,
              "attrs": {}, "messages": []},
             {"id": "dept:support", "kind": "department", "label": "Customer Support",
              "mentions": 6, "attrs": {}, "messages": []},
             {"id": "project:lantern", "kind": "project", "label": "Lantern", "mentions": 9,
              "attrs": {}, "messages": []},
             {"id": "place:turin", "kind": "place", "label": "Turin", "mentions": 6,
              "attrs": {"timezone": "Europe/Rome"}, "messages": []},
             {"id": "place:dunmore", "kind": "place", "label": "Dunmore", "mentions": 6,
              "attrs": {"timezone": "America/New_York"}, "messages": []},
             {"id": "org:pellard", "kind": "org", "label": "Pellard Foundry", "mentions": 2,
              "attrs": {"type": "customer"}, "messages": []},
             {"id": "topic:robotics", "kind": "topic", "label": "robotics", "mentions": 4,
              "attrs": {}, "messages": []}]
    edges = []
    for pid, (label, dept, place, _voice) in PEOPLE.items():
        nodes.append({"id": pid, "kind": "person", "label": label, "mentions": 3,
                      "attrs": {"role": "engineer" if dept == "eng" else "support"},
                      "messages": []})
        edges.append({"source": pid, "rel": "member_of", "target": f"dept:{dept}"})
        edges.append({"source": pid, "rel": "based_in", "target": place})
        if pid != LEADS[dept]:
            edges.append({"source": pid, "rel": "reports_to", "target": LEADS[dept]})
    team = {dept: [p for p, row in PEOPLE.items() if row[1] == dept] for dept in LEADS}
    for members in team.values():
        for a, b in zip(members, members[1:]):
            edges.append({"source": a, "rel": "works_with", "target": b})
    for pid in team["eng"]:
        edges.append({"source": pid, "rel": "works_on", "target": "project:lantern"})
    edges.append({"source": "person:ada", "rel": "interested_in", "target": "topic:robotics"})
    edges.append({"source": "person:mary", "rel": "interested_in", "target": "topic:robotics"})
    graph = {"nodes": nodes, "edges": edges, "messages": {}}
    people = list(PEOPLE)
    personas = {pid: {"voice": voice, "system": f"You are {label}. You write {voice}.",
                      "knows": [p for p, row in PEOPLE.items() if row[1] == dept]
                      + [f"dept:{dept}", "project:lantern", place]}
                for pid, (label, dept, place, voice) in PEOPLE.items()}
    return World(graph=graph, people=people, personas=personas, kind=kind)


def all_of(world: World, **kw) -> list[Message]:
    return list(simulate(world, rng=random.Random(kw.pop("seed", 3)), **kw))


def by_thread(messages: list[Message]) -> dict[str, list[Message]]:
    out: dict[str, list[Message]] = {}
    for m in messages:
        out.setdefault(m.thread or m.id, []).append(m)
    return out


# -- the calendar ------------------------------------------------------------------------------

def test_the_calendar_is_the_same_twice_and_every_arc_names_people_the_graph_holds():
    world = tiny_world()
    once = calendar(world, 20, random.Random(7))
    again = calendar(world, 20, random.Random(7))
    assert once == again and once
    known = set(world.people)
    for arc in once:
        assert arc["kind"] in {a.kind for a in ARCS["company"]}
        assert arc["who"] and set(arc["who"]) <= known
        assert 0 <= arc["day"] <= arc["until"] < 20 and arc["day"] % 7 < 5
        assert arc["about"] and arc["where"]
        assert all(source in ("slack", "email", "teams") for source, _ in arc["where"])
    assert calendar(world, 20, random.Random(8)) != once


def test_an_incident_pulls_in_engineering_and_support_by_the_groups_labels():
    world = tiny_world()
    held = groups(world.graph)
    assert set(held) == {"dept:eng", "dept:support"}
    assert len(held["dept:eng"]["people"]) == 6
    incidents = [a for r in range(40) for a in calendar(world, 12, random.Random(r))
                 if a["kind"] == "incident"]
    assert incidents
    depts = {PEOPLE[p][1] for arc in incidents for p in arc["who"]}
    assert depts == {"eng", "support"}
    assert any("Lantern" in a["about"] for a in incidents)


@pytest.mark.parametrize("kind", ["company", "community", "university", "open-source",
                                  "nonprofit"])
def test_each_kind_of_organisation_gets_arcs_that_suit_it(kind):
    world = tiny_world(kind)
    got = {a["kind"] for r in range(12) for a in calendar(world, 15, random.Random(r))}
    assert got and got <= {a.kind for a in ARCS[kind]}
    other = {a.kind for k, arcs in ARCS.items() if k != kind for a in arcs}
    assert not (got & other - {a.kind for a in ARCS[kind]})


def test_a_kind_of_organisation_nobody_wrote_arcs_for_is_refused_by_name():
    with pytest.raises(ValueError, match="circus"):
        calendar(tiny_world("circus"), 5, random.Random(0))


# -- the simulation ----------------------------------------------------------------------------

def test_templated_days_produce_threads_between_real_people_in_work_hours():
    world = tiny_world()
    messages = all_of(world, days=10, writer=None, mix=0.0)
    assert messages
    known = set(world.people)
    zones = {pid: ZoneInfo({"place:turin": "Europe/Rome",
                            "place:dunmore": "America/New_York"}[row[2]])
             for pid, row in PEOPLE.items()}
    for m in messages:
        assert m.sender in known and set(m.recipients) <= known
        assert m.source in ("slack", "email", "teams")
        assert len(m.ts.split(".")[1]) == 6
        local = datetime.fromtimestamp(float(m.ts), zones[m.sender])
        assert WORK_START <= local.hour < WORK_END, (m.sender, local)
        assert local.weekday() < 5
        assert m.attrs["writer"] == "template" and m.text.strip()
    for root, thread in by_thread(messages).items():
        assert SHORTEST <= len(thread) <= LONGEST
        assert thread[0].id == root and thread[0].kind == "message" and thread[0].thread is None
        assert all(t.kind == "reply" and t.thread == root for t in thread[1:])
        stamps = [float(t.ts) for t in thread]
        assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)
        texts = [t.text for t in thread]
        assert len(set(texts)) == len(texts), texts
        # a DM or an email says who it went to; a channel does not need to
        for t in thread:
            if t.source != "slack" or t.channel.startswith("dm:"):
                assert t.recipients and t.sender not in t.recipients
    assert len(by_thread(messages)) == sum(1 for m in messages if m.thread is None)


def test_the_same_seed_says_the_same_things():
    assert all_of(tiny_world(), days=6, writer=None) == all_of(tiny_world(), days=6, writer=None)


def test_templated_chatter_is_grounded_in_the_graphs_own_names():
    messages = all_of(tiny_world(), days=8, writer=None)
    said = " ".join(m.text for m in messages)
    assert "Lantern" in said
    assert "Turin" in said or "Dunmore" in said
    assert "Pellard Foundry" in said or "robotics" in said


def test_routine_chatter_goes_where_the_relation_says():
    messages = all_of(tiny_world(), days=10, writer=None, per_day=6.0)
    chatter = [m for m in messages if m.thread is None and not m.attrs["arc"]]
    assert chatter
    where = {(m.attrs["kind"], m.source) for m in chatter}
    # peers in a team channel or DM; a reporting line in a DM or email; across groups, email or Teams
    assert any(k in ("standup", "share") and s == "slack" for k, s in where)
    assert any(k in ("checkin", "plan") and s in ("slack", "email") for k, s in where)
    channels = {m.channel for m in chatter if m.source == "slack"}
    assert "engineering" in channels or "customer-support" in channels
    assert any(c.startswith("dm:") for c in channels)


def test_no_writer_and_no_chatter_is_a_quiet_world():
    world = tiny_world()
    world.calendar = []
    messages = all_of(world, days=3, writer=None, per_day=0.0)
    arcs = {m.attrs["arc"] for m in messages}
    assert all(arcs) and messages    # only the calendar spoke


# -- outcomes ------------------------------------------------------------------------------------

def test_an_arcs_end_writes_a_typed_edge_that_names_the_message_it_was_said_in():
    world = tiny_world()
    world.calendar = [
        {"day": 0, "until": 1, "kind": "new_hire", "who": ["person:pell", "person:otto"],
         "about": "Pell's first week", "where": [("slack", "customer-support")],
         "outcome": "joined", "group": "dept:support", "to": "dept:support",
         "subject": "project:lantern"},
        {"day": 0, "until": 0, "kind": "reorg", "who": ["person:milo", "person:ada"],
         "about": "Milo moving", "where": [("email", "Changes")], "outcome": "moved_to",
         "group": "dept:eng", "to": "dept:support", "subject": "project:lantern"},
        {"day": 1, "until": 1, "kind": "launch", "who": ["person:bea", "person:hedy"],
         "about": "the launch of Lantern", "where": [("teams", "lantern-launch")],
         "outcome": "decision", "group": "dept:eng", "to": "dept:eng",
         "subject": "project:lantern"},
    ]
    before = len(world.graph["edges"])
    messages = all_of(world, days=2, writer=None, per_day=0.0)
    ids = {m.id for m in messages}
    new = world.graph["edges"][before:]
    assert [e["rel"] for e in new] == ["moved_to", "joined", "decision"]
    for edge in new:
        assert edge["rel"] in OUTCOMES
        assert edge["attrs"]["said_in"] in ids and edge["messages"] == [edge["attrs"]["said_in"]]
    assert {(e["source"], e["target"]) for e in new} == {
        ("person:milo", "dept:support"), ("person:pell", "dept:support"),
        ("person:bea", "project:lantern")}
    # the last day's thread closes the arc, and the closing message is the one named
    last = {m.id: m for m in messages}[new[1]["attrs"]["said_in"]]
    assert last.attrs["day"] == 1 and last.attrs["arc"] == "arc:0:new_hire"
    # and the truth shows up in what people say: the closer names the group joined
    assert "Customer Support" in last.text or "Pell" in last.text


# -- the model writer --------------------------------------------------------------------------

@dataclasses.dataclass
class Reply:
    content: str = ""
    tool_calls: list | None = None
    thinking: str | None = None
    finish_reason: str | None = None


class SpeakingModel:
    """Answers in words that differ every time, and says what it was about when asked."""

    def __init__(self):
        self.seen: list[list[dict]] = []
        self.answers = 0

    def chat(self, messages, tools=None, **_):
        self.seen.append([dict(m) for m in messages])
        offered = [str((t.get("function") or {}).get("name")) for t in (tools or [])]
        if offered == ["show"]:
            return Reply(tool_calls=[{"id": "c1", "function": {
                "name": "show", "arguments": json.dumps({"ids": ["project:lantern"]})}}])
        self.answers += 1
        return Reply(content=f"We need to check. Lantern is on track from my side, take {self.answers}.")


def test_a_persona_speaks_over_the_subgraph_it_knows_with_its_own_system_prompt():
    world = tiny_world()
    model = SpeakingModel()
    writer = model_writer(model, world)
    persona = dict(world.personas["person:ada"], id="person:ada", label="Ada Lovelace")
    context = {"thread": "msg:000000", "arc_key": "", "kind": "incident", "org_kind": "company",
               "about": "the outage in Lantern", "said": [("person:mary", "It is down.")],
               "labels": {"person:mary": "Mary Somerville"}, "speaker": "person:ada",
               "others": ["person:mary"], "facts": {"project_id": "project:lantern"}}
    text = writer(persona, "Reply in character.", context)
    # planning is cut off the front; what is left is the message
    assert text == "Lantern is on track from my side, take 1."
    first = model.seen[0]
    assert first[0] == {"role": "system",
                        "content": spoken_as(world.personas["person:ada"]["system"])}
    assert {"role": "user", "content": "Mary Somerville: It is down."} in first
    assert first[-1]["content"] == "Reply in character."
    # what the thread is about was read out first, from the subgraph Ada knows
    handed = [m["content"] for m in first if "A search turned up" in m.get("content", "")][0]
    assert "Ada Lovelace (person)" in handed and "works_on Lantern" in handed
    assert "Bea Marlow" in handed                # engineering is in Ada's subgraph
    assert "Mary Somerville" not in handed       # support is not
    assert "robotics" not in handed              # nor an edge to an entry she does not know
    # two calls a message: the answer, and what it was about -- kept for the memory. The
    # asking is tight, and what a message is about survives it because the thread's own
    # subject was handed over at the start: handed over counts as known, and a guess does not.
    assert writer.calls == 2 and writer.messages == 1
    assert writer.last.show == ["project:lantern"]
    assert [t["function"]["name"] for t in model.seen[1] if False] == []


def test_what_a_persona_said_last_time_in_an_arc_is_a_turn_when_it_speaks_again(tmp_path):
    pytest.importorskip("ladybug")
    from ml_stack.graph.store import GraphStore
    from ml_stack.graph.thread import follow, threads

    world = tiny_world()
    world.calendar = [{"day": 0, "until": 1, "kind": "incident",
                       "who": ["person:ada", "person:bea"], "about": "the outage in Lantern",
                       "where": [("slack", "incidents")], "outcome": "decision",
                       "group": "dept:eng", "to": "dept:eng", "subject": "project:lantern"}]
    model = SpeakingModel()
    with GraphStore(tmp_path / "memory.ladybug") as store:
        store.write(world.graph)
        writer = model_writer(model, world, store)
        messages = list(simulate(world, days=2, writer=writer, rng=random.Random(1), mix=1.0,
                                 per_day=0.0, store=store))
        names = [t["thread"] for t in threads(store)]
        assert len(names) == 2 and all(n.startswith("arc:0:incident/") for n in names)
        day0, day1 = by_thread(messages).values()
        assert all(m.attrs["wrote"] == "model" for m in day0 + day1)
        # every message became a turn, joined to its speaker
        kept = follow(store, f"arc:0:incident/{day0[0].id}")
        assert [t.meta["message"] for t in kept] == [m.id for m in day0]
        # joined to the speaker and to what the model said the message was about
        assert all(t.drew["shown"][0] == t.meta["speaker"] for t in kept)
        assert all("project:lantern" in t.drew["shown"] for t in kept)
        assert all("found" in t.drew for t in kept)      # the opening it was handed
        # two calls a message, so the first call of day 1 is at 2 * len(day0): it carries
        # the whole of day 0 as turns before the prompt, this speaker's own words as assistant
        assert len(model.seen) == 2 * len(messages)
        first_of_day1 = model.seen[2 * len(day0)]
        speaker = day1[0].sender
        assert first_of_day1[0]["content"] == spoken_as(world.personas[speaker]["system"])
        turns = first_of_day1[1:1 + len(day0)]
        assert [m["content"].split(": ", 1)[1] for m in turns] == [m.text for m in day0]
        assert [m["role"] for m in turns] == [
            "assistant" if m.sender == speaker else "user" for m in day0]
        assert any(m.sender == speaker for m in day0)   # so a memory of its own words exists
    assert world.graph["edges"][-1]["rel"] == "decision"


def test_the_model_share_of_threads_is_the_mix_asked_for_and_the_arcs_come_first():
    world = tiny_world()
    spoken: list[str] = []

    def spy(persona, prompt, context):
        spoken.append(context["thread"])
        return f"{persona['label']} says so, message {len(spoken)}."

    messages = all_of(world, days=10, writer=spy, mix=0.5, per_day=4.0)
    roots = [m for m in messages if m.thread is None]
    by_model = [m for m in roots if m.attrs["writer"] == "model"]
    assert len(by_model) == len(roots) // 2
    assert {m.attrs["wrote"] for m in messages if m.attrs["writer"] == "model"} == {"model"}
    # on any day, the arcs' threads go to the model before the chatter does
    for day in {m.attrs["day"] for m in roots}:
        today = [m for m in roots if m.attrs["day"] == day]
        arcs = [m for m in today if m.attrs["arc"]]
        chatter = [m for m in today if not m.attrs["arc"]]
        if arcs and any(m.attrs["writer"] == "model" for m in chatter):
            assert all(m.attrs["writer"] == "model" for m in arcs)
    assert not [m for m in all_of(world, days=5, writer=spy, mix=0.0)
                if m.attrs["writer"] == "model"]


def test_an_empty_model_reply_falls_back_to_a_template_rather_than_an_empty_message():
    messages = all_of(tiny_world(), days=3, writer=lambda *_: "", mix=1.0)
    assert messages and all(m.text.strip() for m in messages)
    assert {m.attrs["wrote"] for m in messages} == {"template"}
    assert {m.attrs["writer"] for m in messages} == {"model"}


# -- run() -------------------------------------------------------------------------------------

def test_run_reads_a_world_from_disk_and_writes_the_messages_and_the_graph_after(tmp_path):
    world = tiny_world("community")
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "graph.json").write_text(json.dumps(world.graph))
    (tmp_path / "in" / "personas.json").write_text(json.dumps(world.personas))
    (tmp_path / "in" / "world.json").write_text(json.dumps({"kind": "community"}))
    counts = run(tmp_path / "in", tmp_path / "out", days=10, mix=0.0, seed=5)
    lines = (tmp_path / "out" / "messages.jsonl").read_text().splitlines()
    rows = [Message(**json.loads(line)) for line in lines]
    assert counts["messages"] == len(rows) > 0
    assert counts["threads"] == sum(1 for r in rows if r.thread is None)
    assert counts["template_threads"] == counts["threads"] and counts["model_threads"] == 0
    assert counts["model_calls"] == 0 and counts["messages_per_model_call"] is None
    assert counts["people"] == 12 and counts["days"] == 10
    assert sum(counts["by_source"].values()) == counts["messages"]
    after = json.loads((tmp_path / "out" / "graph.json").read_text())
    grown = len(after["edges"]) - len(world.graph["edges"])
    assert counts["outcomes"] == grown >= 0
    held = json.loads((tmp_path / "out" / "calendar.json").read_text())
    assert counts["arcs"] == len(held) > 0
    assert {a["kind"] for a in held} <= {a.kind for a in ARCS["community"]}
    assert not (tmp_path / "out" / "simulate.lock").exists()   # no model, no lock
    # rows carry a valid Message shape all the way through
    assert all(r.source in ("slack", "email", "teams") and r.sender in world.people for r in rows)


def test_run_without_a_graph_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="graph.json"):
        run(tmp_path, tmp_path / "out", days=1, mix=0.0, seed=0)


def test_the_template_writer_never_says_the_same_thing_twice_in_a_thread():
    writer = template_writer(random.Random(0))
    persona = {"id": "person:ada", "label": "Ada Lovelace", "voice": "terse"}
    said = set()
    for seq in range(40):
        context = {"thread": "t", "kind": "standup", "org_kind": "company", "seq": seq,
                   "of": 40, "speaker": "person:ada", "others": ["person:bea"],
                   "labels": {"person:bea": "Bea Marlow"},
                   "facts": {"project": "Lantern", "place": "Turin"}}
        text = writer(persona, "", context)
        assert text not in said
        said.add(text)


def test_run_with_a_model_takes_the_lock_keeps_memory_beside_the_output_and_prices_it(
        tmp_path, monkeypatch):
    pytest.importorskip("ladybug")
    import ml_stack.client

    world = tiny_world()
    world.calendar = [{"day": 0, "until": 0, "kind": "incident",
                       "who": ["person:ada", "person:bea"], "about": "the outage in Lantern",
                       "where": [("slack", "incidents")], "outcome": "decision",
                       "group": "dept:eng", "to": "dept:eng", "subject": "project:lantern"}]
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "graph.json").write_text(json.dumps(world.graph))
    (tmp_path / "in" / "personas.json").write_text(json.dumps(world.personas))
    (tmp_path / "in" / "calendar.json").write_text(json.dumps(world.calendar))
    made: list[str] = []

    def fake_client(url, **_):
        made.append(url)
        return SpeakingModel()

    monkeypatch.setattr(ml_stack.client, "Client", fake_client)
    counts = run(tmp_path / "in", tmp_path / "out", days=1, mix=1.0,
                 model_url="http://127.0.0.1:8080", seed=2)
    assert made == ["http://127.0.0.1:8080"]
    assert counts["model_threads"] == counts["threads"] >= 1 and counts["template_threads"] == 0
    # two calls a message, so half a message per call; the lock let go, the memory kept
    assert counts["model_calls"] == 2 * counts["messages"]
    assert counts["messages_per_model_call"] == 0.5
    assert counts["outcomes"] == 1
    assert (tmp_path / "out" / "memory.ladybug").exists()
    assert (tmp_path / "out" / "simulate.lock").read_text() == ""
    rows = [json.loads(line) for line in (tmp_path / "out" / "messages.jsonl").read_text().splitlines()]
    assert all(r["attrs"]["wrote"] == "model" for r in rows)


# -- what a message asserts -------------------------------------------------------------------

def _norm(text: str) -> str:
    import re

    return re.sub(r"[^\w]+|_+", " ", text.casefold()).strip()


def _named_in(text: str, node: dict) -> bool:
    """Whether ``text`` writes the node's label as whole words -- a person also by first name."""
    held = " " + _norm(text) + " "
    label = _norm(node["label"])
    return f" {label} " in held or (node["kind"] == "person"
                                    and f" {label.split()[0]} " in held)


def test_a_template_written_message_asserts_exactly_the_ids_its_sentence_names():
    world = tiny_world()
    nodes = {n["id"]: n for n in world.graph["nodes"]}
    messages = all_of(world, days=10, writer=None, per_day=3.0)
    assert messages
    for m in messages:
        held = m.attrs["asserts"]
        assert m.attrs["asserts_exact"] is True
        assert set(held) == {"people", "orgs", "topics", "places", "others", "relations"}
        ids = [i for k in ("people", "orgs", "topics", "places", "others") for i in held[k]]
        assert len(ids) == len(set(ids))
        assert m.sender in held["people"], "the sender counts as a person"
        for i in ids:
            assert i in nodes
            if i != m.sender:
                assert _named_in(m.text, nodes[i]), (i, m.text)
        # and every entry the sentence names is asserted -- nothing the writer filled is lost
        for i, node in nodes.items():
            if _named_in(m.text, node):
                assert i in ids, (i, m.text)
        # each bucket holds what its name says
        assert all(nodes[i]["kind"] == "person" for i in held["people"])
        assert all(nodes[i]["kind"] == "org" for i in held["orgs"])
        assert all(nodes[i]["kind"] == "topic" for i in held["topics"])
        assert all(nodes[i]["kind"] == "place" for i in held["places"])
        assert all(nodes[i]["kind"] in ("project", "department") for i in held["others"])
        # a relation is either an arc's outcome or one the truth graph already holds, and
        # both its ends are named in the sentence it was stated in
        vocabulary = {str(e.get("rel")) for e in world.graph["edges"]} | set(OUTCOMES)
        for source, rel, target in held["relations"]:
            assert source in ids and target in ids and rel in vocabulary
            if rel not in OUTCOMES:
                assert any(e["source"] == source and e.get("rel") == rel
                           and e["target"] == target for e in world.graph["edges"])


def test_the_union_of_asserts_over_a_run_is_the_set_of_entries_its_messages_mention():
    world = tiny_world()
    nodes = {n["id"]: n for n in world.graph["nodes"]}
    messages = all_of(world, days=10, writer=None, per_day=3.0)
    asserted = {i for m in messages for k in ("people", "orgs", "topics", "places", "others")
                for i in m.attrs["asserts"][k]}
    mentioned = {m.sender for m in messages} | {
        i for m in messages for i, node in nodes.items() if _named_in(m.text, node)}
    assert asserted == mentioned
    assert "project:lantern" in asserted and len(asserted & set(PEOPLE)) > 1


def test_an_outcome_is_asserted_by_the_closing_message_only_when_it_names_both_ends():
    world = tiny_world()
    world.calendar = [
        {"day": 0, "until": 0, "kind": "reorg", "who": ["person:milo", "person:ada"],
         "about": "Milo moving", "where": [("email", "Changes")], "outcome": "moved_to",
         "group": "dept:eng", "to": "dept:support", "subject": "project:lantern",
         "facts": {"first": "Milo", "group": "Engineering", "group2": "Customer Support",
                   "project": "Lantern"}},
    ]
    found = False
    for seed in range(6):
        world.graph["edges"] = [e for e in world.graph["edges"] if e.get("rel") != "moved_to"]
        messages = all_of(world, days=1, writer=None, per_day=0.0, seed=seed)
        last = messages[-1]
        edge = next(e for e in world.graph["edges"] if e.get("rel") == "moved_to")
        assert edge["attrs"]["said_in"] == last.id
        # a message may also state a relation the graph already holds; the outcome is the
        # one relation this test is about
        stated = [r for r in last.attrs["asserts"]["relations"] if r[1] in OUTCOMES]
        said = last.text.casefold()             # a voice flavour may lower-case the opener
        if "customer support" in said and "milo" in said:
            assert stated == [["person:milo", "moved_to", "dept:support"]]
            assert "dept:support" in last.attrs["asserts"]["others"]
            found = True
        else:
            assert stated == []
    assert found, "every moved_to closer names the person and the group they moved to"


def test_a_model_written_message_asserts_its_grounding_as_a_lower_bound():
    world = tiny_world()
    world.calendar = [{"day": 0, "until": 0, "kind": "incident",
                       "who": ["person:ada", "person:bea"], "about": "the outage in Lantern",
                       "where": [("slack", "incidents")], "outcome": "decision",
                       "group": "dept:eng", "subject": "project:lantern"}]
    messages = all_of(world, days=1, writer=model_writer(SpeakingModel(), world), mix=1.0,
                      per_day=0.0)
    spoken = [m for m in messages if m.attrs["wrote"] == "model"]
    assert spoken
    for m in spoken:
        assert m.attrs["asserts_exact"] is False
        held = m.attrs["asserts"]
        assert m.sender in held["people"]
        assert "project:lantern" in held["others"]        # the opening, and what show drew on


def test_asserts_round_trip_through_messages_jsonl_and_the_scraper_rows(tmp_path):
    from ml_stack.sources import rows as scraper
    from ml_stack.world.emit import rows

    world = tiny_world()
    messages = all_of(world, days=2, writer=None, per_day=2.0)
    line = json.dumps(dataclasses.asdict(messages[0]))
    back = Message(**json.loads(line))
    assert back.attrs["asserts"] == messages[0].attrs["asserts"]
    assert back.attrs["asserts_exact"] is True
    people = {pid: {"label": row[0]} for pid, row in PEOPLE.items()}
    written = rows(messages, people)
    assert written and all("asserts" in r and "asserts_exact" in r for r in written)
    by_ts = {m.ts: m for m in messages if m.source == "slack"}
    assert all(r["asserts"] == by_ts[r["ts"]].attrs["asserts"] for r in written)
    # the scraper-row reader ignores the key, as it ignores anything it does not know
    read = scraper.read(written, people)
    assert read and all("asserts" not in m.attrs for m in read)
