"""Relational Board migration, same-authority replica exchange and access isolation."""

from copy import deepcopy

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import Denied
from ml_stack.workspace.board_graph import MEMORY, BoardGraph
from ml_stack.workspace.chain import ChainLog


def test_legacy_board_messages_and_cursors_migrate_once(tmp_path):
    base = tmp_path / "legacy"
    base.mkdir()
    boards = ChainLog(base / "boards.jsonl")
    boards.append({"kind": "board", "op": "create", "name": "#project", "by": "writer",
                   "open": False, "project": "project-key", "title": "Project"})
    bus = ChainLog(base / "bus.jsonl")
    root = bus.append({"kind": "msg", "from": "writer", "to": "#project", "body": "root"})
    bus.append({"kind": "msg", "from": "reader", "to": "#project", "body": "reply", "thread": root["seq"]})
    before = (base / "bus.jsonl").read_bytes()
    graph = BoardGraph(base)
    assert graph.state()[0]["#project"]["members"] == {"writer"}
    with graph.opened() as store:
        assert len(store.nodes("board-message")) == 2
        assert len(store.edges("REPLIES_TO")) == 1
        assert len(store.edges("FOR_PROJECT")) == 1
    assert (base / "bus.jsonl").read_bytes() == before
    assert graph.state()[0]["#project"]["project"] == "project-key"


@pytest.fixture
def replicas(monkeypatch, tmp_path):
    left = Kit(clean_env(monkeypatch, tmp_path))
    left.agent("reader")
    left.ws.board.create(left.owner, "#shared", private=True)
    left.ws.board.add(left.owner, "#shared", "reader")
    right = Kit(tmp_path / "replica")
    right.agent("reader")
    right.ws.board.create(right.owner, "#shared", private=True)
    with left.ws.board.store.log.graph.opened() as a:
        workspace = a.get_doc("board-scope")["workspace"]
    with GraphStore(right.base / "coordination.db", buffer_pool_size=MEMORY) as coordination:
        for node in coordination.nodes("workspace"):
            coordination.drop([node["id"]], force=True)
        coordination.upsert_node({"id": workspace, "kind": "workspace"})
    # Build the second replica under the same authoritative identity before any Board writes.
    old = right.base / "board.db"
    old.rename(right.base / "discarded-test-board.db")
    right.ws.board.create(right.owner, "#shared", private=True)
    return left, right


def test_same_authority_messages_deduplicate_and_replies_keep_stable_edges(replicas):
    left, right = replicas
    root = left.ws.send(left.owner, "#shared", "note", "origin root")
    left.ws.send(left.owner, "#shared", "note", "origin reply", reply_to=root["seq"])
    right.ws.send(right.owner, "reader", "status", "local sequence one")
    payload = left.ws.board.export_graph(left.owner)
    assert right.ws.board.combine_graph(right.owner, payload) == 2
    assert right.ws.board.combine_graph(right.owner, payload) == 0
    rows = right.ws.bus.log.rows()
    assert rows[-1]["thread"] == rows[-2]["seq"] != root["seq"]
    assert rows[-1]["thread_id"] == rows[-2]["id"]
    assert right.ws.board.store.state()[0]["#shared"]["members"] == {"owner"}
    assert right.ws.bus.log.verify().ok


def test_foreign_scope_immutable_collision_and_agent_access_are_refused(replicas):
    left, right = replicas
    left.ws.send(left.owner, "reader", "status", "kept")
    payload = left.ws.board.export_graph(left.owner)
    foreign = deepcopy(payload)
    foreign["workspace"] = "workspace:" + "f" * 32
    with pytest.raises(ValueError, match="another workspace"):
        right.ws.board.combine_graph(right.owner, foreign)
    assert right.ws.board.combine_graph(right.owner, payload) == 1
    collision = deepcopy(payload)
    collision["events"][0]["row"]["body"] = "changed"
    with pytest.raises(ValueError):
        right.ws.board.combine_graph(right.owner, collision)
    worker = right.agent("worker")
    with pytest.raises(Denied):
        right.ws.board.combine_graph(worker, payload)
    with pytest.raises(Denied):
        right.ws.board.export_graph(worker)
    assert [row["body"] for row in right.ws.bus.log.rows()] == ["kept"]
