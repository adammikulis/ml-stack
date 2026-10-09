"""Stores written before the board names were made plain read the same through the new names."""

from __future__ import annotations

import json

from ml_stack import board_names
from ml_stack.graph.store import GraphStore
from ml_stack.memory import vault
from ml_stack.reputation.sealed import SealedGraph
from ml_stack.workspace.claims import Claims

AUTHORITY = "a" * 32
OLD_OWNER = f"canonical:{AUTHORITY}:codex-fixture"
NEW_OWNER = f"board:{AUTHORITY}:codex-fixture"


def old_backlog(path):
    with GraphStore(path) as graph:
        graph.upsert_node({"id": "issue-dispatch:1", "kind": "issue-dispatch", "label": "one"})
        graph.upsert_node({"id": "task-1", "kind": "canonical-task-ref", "label": "Fix it",
                           "attrs": {"workspace": "w"}})
        graph.upsert_edge({"source": "issue-dispatch:1", "target": "task-1", "rel": "canonical-task"})


def snapshot(path):
    with GraphStore(path) as graph:
        return graph.nodes(), graph.edges()


def test_an_old_backlog_store_reads_the_same_through_the_new_names(tmp_path):
    path = tmp_path / "issue-backlog.db"
    old_backlog(path)
    graph = board_names.opened(path)
    try:
        assert [node["id"] for node in graph.nodes("task-ref")] == ["task-1"]
        assert graph.nodes("task-ref")[0]["label"] == "Fix it"
        assert graph.nodes("canonical-task-ref") == []
        assert [(e["source"], e["target"]) for e in graph.edges("task")] == [("issue-dispatch:1", "task-1")]
        assert graph.edges("canonical-task") == []
    finally:
        graph.close()


def test_an_old_sealed_evidence_store_and_claims_file_read_through_the_new_names(tmp_path):
    keys = vault.PassphraseKeys(lambda _: "isolated names key")
    sealed = SealedGraph(tmp_path / "rep" / "graph.enc", keys=keys)
    sealed.edit(lambda g: g.upsert_node({
        "id": "work:1", "kind": "work_evidence", "label": "t",
        "attrs": {"source": "canonical-taskboard", "nested": [{"source": "canonical-taskboard"}]}}))
    board_names.marker(sealed.path).unlink()
    sealed.close()
    reopened = SealedGraph(tmp_path / "rep" / "graph.enc", keys=keys)
    attrs = reopened.graph().nodes("work_evidence")[0]["attrs"]
    assert attrs == {"source": "taskboard", "nested": [{"source": "taskboard"}]}
    reopened.close()
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({"version": 1, "claims": {
        "branch:x": {"kind": "branch", "key": "x", "owner": OLD_OWNER, "pid": 0, "since": 1.0,
                     "expires": 9e12, "note": ""}}}))
    store = Claims(tmp_path, 60.0)
    assert store._load()["branch:x"]["owner"] == NEW_OWNER
    assert json.loads(path.read_text())["claims"]["branch:x"]["owner"] == NEW_OWNER


def test_the_migration_is_idempotent(tmp_path):
    path = tmp_path / "issue-backlog.db"
    old_backlog(path)
    graph = board_names.opened(path)
    graph.close()
    first = snapshot(path)
    assert board_names.marker(path).read_text() == str(board_names.VERSION)
    with GraphStore(path) as graph:
        assert board_names.rewrite_graph(graph) == 0
    board_names.opened(path).close()
    assert snapshot(path) == first
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps({"version": 1, "claims": {"a:b": {"owner": OLD_OWNER}}}))
    assert board_names.migrate_claims(claims) == 1
    assert board_names.migrate_claims(claims) == 0
    assert board_names.rewritten(board_names.rewritten(OLD_OWNER)) == NEW_OWNER


def test_a_store_with_no_old_names_is_untouched(tmp_path):
    path = tmp_path / "issue-backlog.db"
    with GraphStore(path) as graph:
        graph.upsert_node({"id": "task-1", "kind": "task-ref", "label": "Fix it",
                           "attrs": {"source": "taskboard", "owner": NEW_OWNER}})
        graph.upsert_node({"id": "n", "kind": "note", "label": "the canonical: form of canonical-task"})
        graph.upsert_edge({"source": "task-1", "target": "n", "rel": "task"})
    before = snapshot(path)
    board_names.opened(path).close()
    assert snapshot(path) == before
    claims = tmp_path / "claims.json"
    text = json.dumps({"version": 1, "claims": {"a:b": {"owner": NEW_OWNER, "note": "canonical:x"}}})
    claims.write_text(text)
    assert board_names.migrate_claims(claims) == 0
    assert claims.read_text() == text
