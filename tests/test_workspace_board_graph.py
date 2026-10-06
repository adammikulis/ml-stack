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
    boards.append(
        {
            "kind": "board",
            "op": "create",
            "name": "#project",
            "by": "writer",
            "open": False,
            "project": "project-key",
            "title": "Project",
        }
    )
    bus = ChainLog(base / "bus.jsonl")
    root = bus.append({"kind": "msg", "from": "writer", "to": "#project", "body": "root"})
    bus.append(
        {"kind": "msg", "from": "reader", "to": "#project", "body": "reply", "thread": root["seq"]}
    )
    graph = BoardGraph(base)
    assert graph.state()[0]["#project"]["members"] == {"writer"}
    with graph.opened() as store:
        assert len(store.nodes("board-message")) == 2
        assert len(store.edges("REPLIES_TO")) == 1
        assert len(store.edges("FOR_PROJECT")) == 1
    assert not (base / "bus.jsonl").exists()
    assert graph.state()[0]["#project"]["project"] == "project-key"


@pytest.fixture
def replicas(monkeypatch, tmp_path):
    left = Kit(clean_env(monkeypatch, tmp_path))
    left.agent("reader")
    left.ws.board.create(left.owner, "#shared", private=True)
    left.ws.board.add(left.owner, "#shared", "reader")
    right = Kit(tmp_path / "replica")
    right.agent("reader")
    with left.ws.board.store.log.graph.opened() as a:
        workspace = a.get_doc("board-scope")["workspace"]
    with GraphStore(right.base / "coordination.db", buffer_pool_size=MEMORY) as coordination:
        for node in coordination.nodes("workspace"):
            coordination.drop([node["id"]], force=True)
        coordination.upsert_node({"id": workspace, "kind": "workspace"})
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


def test_nested_replies_project_immediate_parent_on_divergent_replica(replicas):
    left, right = replicas
    root = left.ws.send(left.owner, "#shared", "note", "root")
    parent = left.ws.send(left.owner, "#shared", "note", "parent", reply_to=root["seq"])
    left.ws.send(left.owner, "#shared", "note", "child", reply_to=parent["seq"])
    right.ws.send(right.owner, "reader", "status", "local")
    right.ws.board.combine_graph(right.owner, left.ws.board.export_graph(left.owner))
    rows = right.ws.bus.log.rows()
    assert rows[-1]["reply_to"] == rows[-2]["seq"]
    assert rows[-1]["thread"] == rows[-3]["seq"]
    assert rows[-1]["reply_id"] == rows[-2]["id"]
    with right.ws.board.store.log.graph.opened() as graph:
        edge = next(
            edge
            for edge in graph.edges("REPLIES_TO")
            if edge["source"] == rows[-1]["id"] + ":message"
        )
        assert edge["target"] == rows[-2]["id"] + ":message"


def test_pruning_erases_event_payload_and_preserves_origin_continuation(replicas):
    left, right = replicas
    first = left.ws.send(left.owner, "reader", "status", "secret-expired-payload")
    left.ws.send(left.owner, "reader", "status", "kept")
    assert left.ws.bus.log.prune_prefix(lambda row: row["seq"] == first["seq"]) == 1
    assert [row["body"] for row in left.ws.bus.log.rows()] == ["kept"]
    with left.ws.bus.log.graph.opened() as graph:
        assert all(
            node.get("row", {}).get("body") != "secret-expired-payload"
            for node in graph.nodes("board-event")
        )
    left.ws.send(left.owner, "reader", "status", "continued")
    assert left.ws.bus.log.verify().ok
    with pytest.raises(ValueError, match="checkpoint"):
        right.ws.board.combine_graph(right.owner, left.ws.board.export_graph(left.owner))


@pytest.mark.redteam
def test_prefix_deletion_and_envelope_rewiring_are_detected(replicas):
    left, _ = replicas
    left.ws.send(left.owner, "reader", "status", "kept")
    row = left.ws.bus.log.rows()[-1]
    with GraphStore(left.base / "board.db", buffer_pool_size=MEMORY) as graph:
        event = next(node for node in graph.nodes("board-event") if node["id"] == row["id"])
        graph.upsert_node({**event, "thread_id": "rewired"})
    from ml_stack.workspace.chain import ChainBroken

    with pytest.raises(ChainBroken, match="envelope"):
        left.ws.bus.log.rows()


@pytest.mark.redteam
def test_exchange_rejects_cross_audience_replies_and_schema_poison(replicas):
    left, right = replicas
    left.ws.send(left.owner, "#general", "note", "public")
    left.ws.send(left.owner, "#shared", "note", "private")
    payload = left.ws.board.export_graph(left.owner)
    from ml_stack.workspace.chain import _digest

    event = payload["events"][-1]
    event["thread_id"] = payload["events"][0]["id"]
    event["row"]["thread_id"] = event["thread_id"]
    event["row"]["hash"] = _digest(event["row"]["prev"], event["row"])
    with pytest.raises(ValueError, match="audience"):
        right.ws.board.combine_graph(right.owner, payload)
    assert right.ws.bus.log.rows() == []
    payload = left.ws.board.export_graph(left.owner)
    del payload["events"][0]["row"]["body"]
    with pytest.raises(ValueError, match="row"):
        right.ws.board.combine_graph(right.owner, payload)


@pytest.mark.redteam
def test_exchange_rejects_fabricated_origin_prefix_and_node_overwrite(replicas):
    left, right = replicas
    left.ws.send(left.owner, "reader", "status", "one")
    left.ws.send(left.owner, "reader", "status", "two")
    payload = left.ws.board.export_graph(left.owner)
    payload["events"] = payload["events"][1:]
    with pytest.raises(ValueError, match="chain"):
        right.ws.board.combine_graph(right.owner, payload)
    payload = left.ws.board.export_graph(left.owner)
    payload["events"][0]["id"] = payload["workspace"] + ":board:" + "0" * 64
    with pytest.raises(ValueError, match="identity"):
        right.ws.board.combine_graph(right.owner, payload)


def test_legacy_migration_retries_erasure_after_committed_import(monkeypatch, tmp_path):
    base = tmp_path / "legacy-retry"
    base.mkdir()
    ChainLog(base / "bus.jsonl").append(
        {"kind": "msg", "from": "writer", "to": "reader", "body": "kept"}
    )
    original = BoardGraph._erase_legacy

    def interrupted(*args):
        raise OSError("injected interruption")

    monkeypatch.setattr(BoardGraph, "_erase_legacy", interrupted)
    with pytest.raises(OSError, match="interruption"):
        BoardGraph(base).state()
    assert (base / "bus.jsonl").exists()
    monkeypatch.setattr(BoardGraph, "_erase_legacy", original)
    BoardGraph(base).state()
    assert not (base / "bus.jsonl").exists()
    with BoardGraph(base).opened() as graph:
        assert len(graph.nodes("board-message")) == 1


def test_surviving_replies_keep_stable_expired_ids_without_remote_sequences(replicas):
    left, right = replicas
    root = left.ws.send(left.owner, "#shared", "note", "expired")
    left.ws.send(left.owner, "#shared", "note", "retained", reply_to=root["seq"])
    right.ws.board.combine_graph(right.owner, left.ws.board.export_graph(left.owner))
    assert left.ws.bus.log.prune_prefix(lambda row: row["body"] == "expired") == 1
    assert right.ws.bus.log.prune_prefix(lambda row: row["body"] == "expired") == 1
    row = right.ws.bus.log.rows()[0]
    assert row["thread_id"] and row["reply_id"]
    assert row["reply_to"] == row["thread"] == root["seq"]
    assert right.ws.board.combine_graph(right.owner, left.ws.board.export_graph(left.owner)) == 0


@pytest.mark.redteam
@pytest.mark.parametrize("tamper", ["seal", "projection"])
def test_missing_seal_or_modified_projection_refuses_reads(replicas, tamper):
    left, _ = replicas
    left.ws.send(left.owner, "reader", "status", "kept")
    with GraphStore(left.base / "board.db", buffer_pool_size=MEMORY) as graph:
        if tamper == "seal":
            graph.query("MATCH (d:Doc {key:$key}) DELETE d", {"key": "event-evidence:bus"})
        else:
            graph.put_doc("projection:bus", {"seq": 999})
    from ml_stack.workspace.chain import ChainBroken

    with pytest.raises(ChainBroken, match="integrity"):
        left.ws.bus.log.rows()


@pytest.mark.redteam
@pytest.mark.parametrize("expiry", ["bad", {}, True, -1, float("inf")])
def test_exchange_rejects_invalid_expiry(replicas, expiry):
    left, right = replicas
    left.ws.send(left.owner, "reader", "status", "kept")
    payload = left.ws.board.export_graph(left.owner)
    row = payload["events"][0]["row"]
    row["expires"] = expiry
    from ml_stack.workspace.chain import _digest

    row["hash"] = _digest(row["prev"], row)
    with pytest.raises(ValueError):
        right.ws.board.combine_graph(right.owner, payload)
    assert right.ws.bus.log.rows() == []


def test_reply_to_surviving_message_keeps_expired_root_thread(replicas):
    left, _ = replicas
    root = left.ws.send(left.owner, "#shared", "note", "expired")
    parent = left.ws.send(left.owner, "#shared", "note", "parent", reply_to=root["seq"])
    left.ws.bus.log.prune_prefix(lambda row: row["body"] == "expired")
    left.ws.send(left.owner, "#shared", "note", "child", reply_to=parent["seq"])
    rows = left.ws.bus.log.rows()
    assert rows[-1]["thread_id"] == rows[-2]["thread_id"]
    assert rows[-1]["thread"] == root["seq"]
    assert rows[-1]["reply_to"] == parent["seq"]
