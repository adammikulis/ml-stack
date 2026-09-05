"""`ml_stack.graph.review.Queue`: proposals listed, accepted into a real store, refused and
undone. Everything is invented and lives in tmp_path."""

from __future__ import annotations

import pytest

from ml_stack.files import read_json, write_json
from ml_stack.graph.review import Queue, as_change, listed

pytest.importorskip("ladybug")

from ml_stack.graph.store import GraphStore  # noqa: E402

GRAPH = {
    "meta": {"model": "m"},
    "nodes": [{"id": "person:iris", "kind": "person", "label": "Iris Bellweather",
               "mentions": 3, "attrs": {"member": True, "location": "Dunmore"}, "messages": ["m1"]},
              {"id": "person:i b", "kind": "person", "label": "I. Bellweather",
               "mentions": 1, "attrs": {}, "messages": ["m2"]},
              {"id": "topic:looms", "kind": "topic", "label": "looms", "mentions": 2,
               "attrs": {}, "messages": ["m1"]},
              {"id": "topic:compilers", "kind": "topic", "label": "compilers",
               "mentions": 2, "attrs": {}, "messages": ["m1", "m2"]}],
    "edges": [{"source": "person:iris", "target": "topic:looms", "rel": "interested_in",
               "weight": 2, "messages": ["m1"]},
              {"source": "person:iris", "target": "topic:compilers",
               "rel": "interested_in", "weight": 1, "messages": ["m1"]},
              {"source": "person:i b", "target": "topic:compilers", "rel": "mentions",
               "weight": 1, "messages": ["m2"]}],
    "messages": {"m1": {"text": "hello"}, "m2": {"text": "again"}},
}

EDITS = [{"op": "set_attr", "target": "person:iris", "name": "location", "value": "Ontario"},
         {"op": "rename", "target": "topic:looms", "value": "weaving"},
         {"op": "remove_edge", "target": "person:iris", "other": "topic:compilers",
          "name": "interested_in"},
         {"op": "merge_nodes", "target": "person:iris", "other": "person:i b"},
         {"op": "remove_relation", "target": "nobody|knows|nothing"}]

KEY = "2026-01-02T00:00:00Z|fix me"


def proposal(edits=EDITS, status="proposed"):
    return {KEY: {"at": "2026-01-02T00:00:00Z", "text": "fix me", "status": status,
                  "edits": list(edits)}}


def stored(tmp_path):
    path = tmp_path / "graph.ladybug"
    with GraphStore(path) as store:
        store.write(GRAPH)
    return path


def test_listed_carries_the_key_as_id_and_the_place_as_index():
    rows = listed({"b|2": {"text": "second"}, "a|1": {"text": "first"}})
    assert [(r["id"], r["index"], r["text"]) for r in rows] == [("a|1", 1, "first"),
                                                                ("b|2", 2, "second")]


def test_as_change_speaks_the_stores_vocabulary():
    assert as_change({"op": "remove", "target": "topic:x"}).op == "remove_node"
    assert as_change({"op": "set_attr", "target": "a", "name": "n", "value": "v"}).op \
        == "set_attribute"
    edge = as_change({"op": "remove_relation", "target": "a|knows|b"})
    assert (edge.op, edge.target, edge.name, edge.other) == ("remove_edge", "a", "knows", "b")
    assert edge.reason == "accepted in review"
    broken = as_change({"op": "remove_relation", "target": "a|b"})
    assert not broken.sound and "source|relation|target" in broken.problems[0]


def test_an_accepted_proposal_lands_in_the_store_read_back_on_a_fresh_handle(tmp_path):
    store = stored(tmp_path)
    write_json(tmp_path / "proposals.json", proposal())
    exported: list = []
    queue = Queue(tmp_path / "proposals.json", store=store,
                  exported=lambda: exported.append(True) and None)

    problems = queue.act(KEY, "accept")

    assert len(problems) == 1 and problems[0].startswith("remove_edge nobody"), problems
    assert read_json(tmp_path / "proposals.json", {})[KEY]["status"] == "accepted"
    assert exported == [True]
    with GraphStore(store, read_only=True) as held:
        graph = held.read()
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert by_id["person:iris"]["attrs"]["location"] == "Ontario"
    assert by_id["topic:looms"]["label"] == "weaving"
    assert "person:i b" not in by_id, "folded into Iris"
    assert ("person:iris", "interested_in", "topic:compilers") not in {
        (e["source"], e["rel"], e["target"]) for e in graph["edges"]}


def test_refusing_marks_the_proposal_and_touches_nothing_else(tmp_path):
    store = stored(tmp_path)
    write_json(tmp_path / "proposals.json", proposal())
    queue = Queue(tmp_path / "proposals.json", store=store)
    assert queue.act(KEY, "refuse") == []
    assert read_json(tmp_path / "proposals.json", {})[KEY]["status"] == "refused"
    with GraphStore(store, read_only=True) as held:
        assert {n["id"] for n in held.read()["nodes"]} == {n["id"] for n in GRAPH["nodes"]}


def test_a_journal_is_written_first_and_undo_takes_it_back(tmp_path):
    """The journal is what a rebuild applies over fresh extractions; an edit the store
    would not take is still in it, and undo edits only the journal."""
    def fold(edit, held):
        if edit["op"] == "remove_relation":
            return "remove_relation is not something this journal can say"
        held.setdefault("taken", []).append(edit["op"])
        return None

    def unfold(edit, held):
        held["taken"] = [op for op in held.get("taken", []) if op != edit["op"]]

    write_json(tmp_path / "proposals.json", proposal())
    queue = Queue(tmp_path / "proposals.json", store=tmp_path / "missing.ladybug",
                  journal=(tmp_path / "journal.json", fold, unfold))
    problems = queue.act(KEY, "accept")
    assert problems == ["remove_relation is not something this journal can say"]
    assert read_json(tmp_path / "journal.json", {}) == {
        "taken": ["set_attr", "rename", "remove_edge", "merge_nodes"]}
    assert read_json(tmp_path / "proposals.json", {})[KEY]["status"] == "accepted"

    assert queue.act(KEY, "undo") == []
    assert read_json(tmp_path / "journal.json", {}) == {"taken": []}
    assert read_json(tmp_path / "proposals.json", {})[KEY]["status"] == "proposed"


def test_a_store_that_is_not_there_yet_is_left_alone(tmp_path):
    said: list = []
    write_json(tmp_path / "proposals.json", proposal())
    queue = Queue(tmp_path / "proposals.json", store=tmp_path / "later.ladybug", log=said.append)
    assert queue.act(KEY, "accept") == []
    assert not (tmp_path / "later.ladybug").exists()
    assert said == ["store: not there yet, so the edit waits for the rebuild"]


def test_an_unknown_key_or_action_is_refused_before_anything_is_written(tmp_path):
    write_json(tmp_path / "proposals.json", proposal())
    queue = Queue(tmp_path / "proposals.json")
    with pytest.raises(KeyError):
        queue.act("nope", "accept")
    with pytest.raises(ValueError, match="shred"):
        queue.act(KEY, "shred")
    assert read_json(tmp_path / "proposals.json", {})[KEY]["status"] == "proposed"
