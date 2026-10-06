"""Authenticated local project enrollment recovery."""

import pytest

from ml_stack.workspace import guide, project, tokens
from ml_stack.workspace.identity import Denied

pytest_plugins = ("test_workspace_quickstart",)


@pytest.mark.parametrize("board_exists", [False, True])
def test_agent_connect_completes_initial_project_membership(base, ws, tmp_path, board_exists):
    found = project.describe(str(tmp_path))
    token = ws.registry.bootstrap_agent("worker", found,
                                        (ws.limits.mints_per_identity, ws.limits.agents_live))
    tokens.store(base, "worker", token)
    ws.registry.record_model("worker", "test-model", "test-harness", "claimed")
    if board_exists:
        ws.board.store.project_board(found)
    for _ in range(2):
        assert guide.agent_connect(ws, "worker", found)["state"] == "connected"
    boards, _ = ws.board.store.state()
    selected = [row for row in boards.values() if row["project"] == found["key"]]
    assert len(selected) == 1 and selected[0]["members"] == {"worker"}
    joins = [row for row in ws.board.store.log.rows()
             if row.get("op") == "join" and row.get("who") == "worker"]
    assert len(joins) == 1
    assert tokens.load(base, "worker") == token
    assert ws.registry.info("worker")["model_state"] == "claimed"


def test_agent_connect_preserves_explicit_project_leave(base, ws, tmp_path):
    found = project.describe(str(tmp_path))
    guide.agent_connect(ws, "worker", found)
    token = tokens.load(base, "worker")
    board = ws.board.store.project_board(found)
    ws.board.leave(token, board)
    with pytest.raises(Denied, match="no membership"):
        guide.agent_connect(ws, "worker", found)
    assert "worker" not in ws.board.store.state()[0][board]["members"]
    assert tokens.load(base, "worker") == token


@pytest.mark.parametrize("authority", ["invited", "device"])
def test_agent_connect_never_repairs_foreign_membership(base, ws, tmp_path, authority):
    found = project.describe(str(tmp_path))
    token = ws.registry.bootstrap_agent("worker", found,
                                        (ws.limits.mints_per_identity, ws.limits.agents_live))
    tokens.store(base, "worker", token)
    agents = ws.registry._load()
    if authority == "invited":
        agents["worker"]["minted_by"] = "project-host"
    else:
        agents["worker"]["session_device"] = "paired-device-fingerprint"
    ws.registry._save(agents)
    before = ws.board.store.log.rows()
    with pytest.raises(Denied, match="no membership"):
        guide.agent_connect(ws, "worker", found)
    assert ws.board.store.log.rows() == before


