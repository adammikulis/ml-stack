"""Applying proposed changes to a real store, together or not at all.

Every test applies against a real store on disk and, where it matters, closes it and opens
it again on a fresh handle.
"""

import dataclasses
import json

import pytest

from ml_stack.graph.propose import apply, proposing
from ml_stack.graph.store import GraphStore

pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")

GRAPH = {
    "nodes": [
        {"id": "person:ada", "kind": "person", "label": "Ada Lovelace", "mentions": 4,
         "attrs": {"role": "analyst"}},
        {"id": "person:bea", "kind": "person", "label": "Bea Marlow", "mentions": 2, "attrs": {}},
        {"id": "topic:compilers", "kind": "topic", "label": "compilers", "mentions": 3, "attrs": {}},
    ],
    "edges": [
        {"source": "person:ada", "target": "topic:compilers", "rel": "interested_in", "weight": 3},
    ],
}


def call(tool, /, **args):
    return {"function": {"name": tool, "arguments": json.dumps(args)}}


def a_store_holding(path, graph=GRAPH):
    with GraphStore(path) as store:
        store.write(graph)
    return GraphStore(path)


def proposals(store, *calls):
    _, gather = proposing(store.read())
    return gather(list(calls))


def test_sound_changes_land_and_survive_reopening(tmp_path):
    path = tmp_path / "g"
    with a_store_holding(path) as store:
        changes = proposals(
            store,
            call("add_node", label="Quenlow Robotics", kind="org", reason="named as an employer"),
            call("rename", id="person:ada", label="Ada of Turin", reason="how she signs"),
            call("set_attribute", id="person:bea", name="role", value="engineer",
                 reason="she said so"),
            call("remove_edge", from_id="person:ada", to_id="topic:compilers",
                 rel="interested_in", reason="she moved on"))
        assert all(c.sound for c in changes)
        out = apply(store, changes)
    assert [c.op for c in out["applied"]] == ["add_node", "rename", "set_attribute",
                                              "remove_edge"]
    assert out["skipped"] == []
    with GraphStore(path) as reopened:
        back = {n["id"]: n for n in reopened.nodes()}
        assert back["org:quenlow-robotics"]["label"] == "Quenlow Robotics"
        assert back["org:quenlow-robotics"]["kind"] == "org"
        assert back["person:ada"]["label"] == "Ada of Turin"
        assert back["person:bea"]["attrs"] == {"role": "engineer"}
        assert reopened.edges() == []


def test_an_unsound_change_is_skipped_and_the_rest_still_land(tmp_path):
    path = tmp_path / "g"
    with a_store_holding(path) as store:
        changes = proposals(
            store,
            call("remove_node", id="person:nobody", reason="asked"),
            call("rename", id="person:bea", label="Bea of Turin", reason="how she signs"))
        out = apply(store, changes)
    assert [c.op for c in out["applied"]] == ["rename"]
    assert [c.op for c in out["skipped"]] == ["remove_node"]
    assert "is not in the graph" in out["skipped"][0].problems[0]
    with GraphStore(path) as reopened:
        assert next(n for n in reopened.nodes()
                    if n["id"] == "person:bea")["label"] == "Bea of Turin"


def test_a_change_sound_when_proposed_but_stale_at_apply_is_skipped(tmp_path):
    path = tmp_path / "g"
    with a_store_holding(path) as store:
        changes = proposals(
            store,
            call("remove_edge", from_id="person:ada", to_id="topic:compilers",
                 rel="interested_in", reason="stale by the time it is applied"))
        assert changes[0].sound
        store.remove_edge("person:ada", "topic:compilers", "interested_in")
        out = apply(store, changes)
    assert out["applied"] == []
    assert "not joined that way" in out["skipped"][0].problems[0]


def test_apply_is_all_or_nothing_when_a_change_fails(tmp_path):
    path = tmp_path / "g"
    with a_store_holding(path) as store:
        changes = proposals(
            store,
            call("rename", id="person:ada", label="Ada of Turin", reason="how she signs"),
            call("set_attribute", id="person:bea", name="role", value="engineer",
                 reason="she said so"))

        def wobbly(*_args, **_kwargs):
            raise RuntimeError("the store went away")

        store.set_attribute = wobbly
        with pytest.raises(RuntimeError, match="went away"):
            apply(store, changes)
    with GraphStore(path) as reopened:
        back = {n["id"]: n for n in reopened.nodes()}
        assert back["person:ada"]["label"] == "Ada Lovelace"
        assert back["person:bea"]["attrs"] == {}


def test_applying_changes_does_not_alter_the_change_objects(tmp_path):
    path = tmp_path / "g"
    with a_store_holding(path) as store:
        changes = proposals(
            store,
            call("rename", id="person:ada", label="Ada of Turin", reason="how she signs"),
            call("remove_node", id="person:nobody", reason="asked"))
        before = [dataclasses.replace(c) for c in changes]
        store.drop(["person:ada"])
        apply(store, changes)
    assert changes == before


def test_an_added_node_gets_an_id_the_graph_can_address(tmp_path):
    path = tmp_path / "g"
    with a_store_holding(path) as store:
        changes = proposals(
            store,
            call("add_node", label="Quenlow Robotics", kind="org", reason="named as an employer"),
            call("add_edge", from_id="person:ada", to_id="org:quenlow-robotics",
                 rel="works_at", reason="she said so"))
        assert not changes[1].sound
        out = apply(store, changes)
    assert [c.op for c in out["applied"]] == ["add_node", "add_edge"]
    assert out["skipped"] == []
    with GraphStore(path) as reopened:
        joined = {(e["source"], e["rel"], e["target"]) for e in reopened.edges()}
    assert ("person:ada", "works_at", "org:quenlow-robotics") in joined
