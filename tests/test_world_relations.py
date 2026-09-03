"""The relations an invented world says out loud, and the gold they leave behind.

A templated corpus used to be first names and pronouns: an extraction run over forty of its
messages had one relation to find, which made the relation columns of ``ml-stack-bench
extract`` meaningless. Now most messages state one relation the graph already holds --
who works with whom, who reports to whom, what somebody works on, which unit they belong
to -- naming both ends in full, and carry it as gold.

Every person, place and organisation here is invented by `world.organisation`; nothing
reads a real graph, a scrape or a served model, and the corpus is written into ``tmp_path``.
"""

from __future__ import annotations

import json

import pytest

from ml_stack.files import write_json
from ml_stack.world.organisation import make
from ml_stack.world.questions import questions
from ml_stack.world.simulate import _SHAPE, _STATED, run
from ml_stack.world.story import OUTCOMES

SEED = 7
DAYS = 10
SAMPLE = 40
# what a small world of each kind is built out of, and so what its messages can state
SHAPED = {"community": {"part_of", "works_with"},
          "company": {"part_of", "works_with", "reports_to", "works_on"},
          "university": {"part_of", "works_with", "advises", "works_on"},
          "open-source": {"part_of", "works_with", "contributes_to"},
          "nonprofit": {"part_of", "works_with", "reports_to", "works_on"}}


def talked(where, kind: str = "community", days: int = DAYS):
    """A small invented world of ``kind`` that talked for ``days``, as (graph, messages)."""
    world = make(kind, "small", SEED)
    write_json(where / "graph.json", world.graph)
    write_json(where / "personas.json", world.personas)
    write_json(where / "calendar.json", world.calendar)
    write_json(where / "world.json", {"kind": world.kind, "size": world.size, "seed": SEED,
                                      "people": world.people})
    out = where / "talk"
    run(where, out, days=days, mix=0.0, seed=SEED)
    graph = json.loads((out / "graph.json").read_text(encoding="utf-8"))
    messages = [json.loads(line) for line in
                (out / "messages.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return graph, messages


@pytest.fixture(scope="module")
def community(tmp_path_factory):
    return talked(tmp_path_factory.mktemp("community"))


def stated(messages):
    return [(m, tuple(map(str, r))) for m in messages
            for r in (m["attrs"]["asserts"]["relations"] or ())]


# -- how much gold a sample carries --------------------------------------------------------

def test_a_stratified_sample_of_forty_messages_asserts_twenty_relations_or_more(community):
    """The bench reads forty messages; the gold it scores relations against is only what
    those forty said, so the corpus has to say enough of it in forty."""
    from ml_stack.graph.bench.extract import gold, sample_messages

    graph, messages = community
    picked = sample_messages(messages, SAMPLE, seed=SEED)
    assert len(picked) == SAMPLE
    held = gold(graph, picked)
    assert held["exact"] is True
    assert len(held["relations"]) >= 20, held["relations"]
    # and they are relations about how the community is put together, not one repeated fact
    kinds = {rel for _s, rel, _t in held["relations"]}
    assert len(kinds) >= 3
    assert kinds & _SHAPE


@pytest.mark.parametrize("kind", sorted(SHAPED))
def test_every_kind_of_world_states_the_relations_it_is_built_out_of(tmp_path, kind):
    _graph, messages = talked(tmp_path, kind, days=4)
    said = {rel for _m, (_s, rel, _t) in stated(messages)}
    assert SHAPED[kind] <= said, (kind, sorted(said))


# -- and whether the gold is true ----------------------------------------------------------

def test_every_asserted_relation_is_an_edge_the_truth_graph_holds(community):
    """Nothing is inferred back out of the text: an asserted relation is a row of the
    graph the world was made from, spelled in the world's own vocabulary."""
    graph, messages = community
    edges = {(str(e["source"]), str(e.get("rel")), str(e["target"])) for e in graph["edges"]}
    vocabulary = {str(e.get("rel")) for e in graph["edges"]}
    seen = stated(messages)
    assert len(seen) > 100
    for message, triple in seen:
        assert triple[1] in vocabulary, (triple, message["id"])
        assert triple in edges, (triple, message["id"])


def test_the_sentence_a_relation_was_stated_in_names_both_ends(community):
    """"Who works with whom" is only gold if a reader could have read it: both labels are
    in the message, in full, as the graph spells them."""
    graph, messages = community
    labels = {str(n["id"]): str(n.get("label") or "") for n in graph["nodes"]}
    # an arc's outcome is asserted by a closer that named both ends its own way -- a first
    # name, a group's slug -- and is not one of the sentences this is about
    outcomes = {(str(e["source"]), str(e.get("rel")), str(e["target"])) for e in graph["edges"]
                if (e.get("attrs") or {}).get("said_in")}
    checked = 0
    for message, (source, rel, target) in stated(messages):
        if (source, rel, target) in outcomes and rel in OUTCOMES:
            continue
        text = message["text"].casefold()
        assert labels[source].casefold() in text, (source, message["text"])
        assert labels[target].casefold() in text, (target, message["text"])
        assert rel in _STATED
        checked += 1
    assert checked > 100


def test_a_message_states_a_relation_and_still_says_something_of_its_own(community):
    """The fact is appended to what the speaker was going to say, not instead of it."""
    _graph, messages = community
    with_a_fact = [m for m, _ in stated(messages)]
    assert len(with_a_fact) / len(messages) > 0.5
    assert any(m["attrs"]["asserts"]["relations"] == [] for m in messages), "not every one"
    for message in with_a_fact[:50]:
        assert len(message["text"].split(".")) >= 2, message["text"]


# -- and whether the same relations can be asked about --------------------------------------

@pytest.mark.parametrize("kind", ["community", "company"])
def test_a_relation_question_is_answered_by_the_relation_the_graph_holds(kind):
    """"Who does X report to?" and "Who works with Y?" are drawn from the same relations
    the messages state, and their answers are the graph's own."""
    world = make(kind, "small", SEED)
    nodes = {str(n["id"]): n for n in world.graph["nodes"]}
    label = {str(n["id"]): str(n.get("label") or "") for n in world.graph["nodes"]}
    out: dict[str, dict[str, list[str]]] = {}
    inc: dict[str, dict[str, list[str]]] = {}
    for e in world.graph["edges"]:
        out.setdefault(str(e["rel"]), {}).setdefault(str(e["source"]), []).append(str(e["target"]))
        inc.setdefault(str(e["rel"]), {}).setdefault(str(e["target"]), []).append(str(e["source"]))
    asked = questions(world, 400)
    by_person = {label[i]: i for i in nodes if nodes[i]["kind"] == "person"}

    lines = works = 0
    for q in asked:
        for i in q["expect"]:
            assert i in nodes, q
        if q["q"].startswith("Who does ") and q["q"].endswith(" report to?"):
            who = by_person[q["q"][len("Who does "):-len(" report to?")]]
            assert q["kind"] == "person"
            assert sorted(q["expect"]) == sorted(set(out.get("reports_to", {}).get(who, ())))
            lines += 1
        elif q["q"].startswith("Who works with ") and " who knows" not in q["q"]:
            who = by_person[q["q"][len("Who works with "):-1]]
            assert q["kind"] == "person"
            beside = set(out.get("works_with", {}).get(who, ())) | set(
                inc.get("works_with", {}).get(who, ()))
            assert sorted(q["expect"]) == sorted(beside)
            works += 1
    assert works, "a world where people work together is asked who works with whom"
    assert bool(lines) is ("reports_to" in out), "asked exactly where the world has a line"


def test_a_path_question_is_a_chain_of_relations_the_graph_holds():
    """The answer to "how is A connected to C" is a run of entries the graph joins, so a
    reader who took the relations out of the messages could have followed it."""
    world = make("company", "small", SEED)
    joined: set[frozenset[str]] = {frozenset((str(e["source"]), str(e["target"])))
                                   for e in world.graph["edges"]}
    paths = [q for q in questions(world, 400) if q["kind"] == "path"]
    assert paths
    for q in paths:
        assert len(q["expect"]) >= 3
        for one in q["expect"]:
            assert any(frozenset((one, other)) in joined for other in q["expect"] if other != one), q
