"""Person-authorized Coding route schemas and native subprocess lifecycle."""
from __future__ import annotations

import time

import pytest
from coding_kit import SESSION, fixture_worker
from test_fleet_ui import Serving
from workspace_kit import Kit

from ml_stack.fleet.conversations import Conversations
from ml_stack.workspace import coding_turns


@pytest.fixture
def coding_api(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_WORKSPACE_HOME", str(tmp_path / "workspace"))
    kit = Kit(tmp_path / "workspace")
    monkeypatch.setattr(coding_turns, "worker", fixture_worker)
    server = Serving(tmp_path)
    server.ui.settings.setup_done = True
    server.ui.conversations = Conversations(tmp_path / "chats")
    folder = tmp_path / "project"
    folder.mkdir()
    made = server.ui.conversations.start(model="test-model", settings={"mode": "coding", "project": str(folder), "role": "read-only"})
    try:
        yield server, made, kit
    finally:
        if hasattr(server.ui, "coding_turns"):
            for turn in server.ui.coding_turns.turns.values():
                if turn.state in ("starting", "running", "cancelling"):
                    turn.cancel()
        server.close()


def post(server, cid, operation, body):
    return server.call(f"/ui/coding/{cid}/{operation}", method="POST", body=body,
                       headers={"Origin": f"http://127.0.0.1:{server.port}", "Sec-Fetch-Site": "same-origin"})


def finished(server, cid):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status, result, _ = server.call(f"/ui/coding/{cid}/status")
        assert status == 200
        if result["state"] in ("completed", "failed", "cancelled"):
            return result
        time.sleep(.02)
    pytest.fail("coding turn did not finish")


def test_native_turns_resume_and_revoke_only_their_agent(coding_api):
    server, conversation, kit = coding_api
    assert post(server, conversation.id, "start", {"message": "inspect the brakes"})[0] == 202
    assert finished(server, conversation.id)["state"] == "completed"
    assert post(server, conversation.id, "start", {"message": "inspect the sensors"})[0] == 202
    final = finished(server, conversation.id)
    assert final["state"] == "completed"
    assert final["session"] == SESSION
    saved = server.ui.conversations.get(conversation.id)
    assert [message.content for message in saved.messages] == ["inspect the brakes", "Inspected: inspect the brakes",
                                                              "inspect the sensors", "Inspected: inspect the sensors"]
    assert kit.ws.registry.role_of("owner") == "human"
    assert not kit.ws.registry.role_of(f"chat-{conversation.id}")
    assert not (kit.base / "tokens" / f"chat-{conversation.id}").exists()


def test_cancel_stops_native_process_and_the_next_turn_can_start(coding_api):
    server, conversation, _kit = coding_api
    assert post(server, conversation.id, "start", {"message": "wait until cancelled"})[0] == 202
    deadline = time.monotonic() + 10
    while server.ui.coding_turns.status(conversation.id)["state"] != "running":
        assert time.monotonic() < deadline
        time.sleep(.02)
    assert post(server, conversation.id, "start", {"message": "second"})[0] == 400
    began = time.monotonic()
    assert post(server, conversation.id, "cancel", {})[0] == 200
    assert finished(server, conversation.id)["state"] == "cancelled"
    assert time.monotonic() - began < 5
    assert post(server, conversation.id, "start", {"message": "continue inspection"})[0] == 202
    assert finished(server, conversation.id)["state"] == "completed"


@pytest.mark.parametrize("body", [{"message": []}, {"message": ""}, {"message": "hello", "role": "admin"},
                                  {"message": "hello", "parent": "owner"}, [], {"message": "hello", "token": "secret"}])
def test_client_cannot_supply_identity_or_unsupported_fields(coding_api, body):
    server, conversation, _kit = coding_api
    assert post(server, conversation.id, "start", body)[0] == 400
    assert server.ui.conversations.get(conversation.id).messages == []


def test_coding_start_requires_person_page_origin_and_ui_header(coding_api):
    server, conversation, _kit = coding_api
    path = f"/ui/coding/{conversation.id}/start"
    body = {"message": "hello"}
    assert server.call(path, method="POST", body=body)[0] == 403
    assert server.call(path, method="POST", body=body, headers={"Origin": "https://evil.invalid"})[0] == 403
    assert server.call(path, method="POST", body=body, ui_header=False,
                       headers={"Origin": f"http://127.0.0.1:{server.port}"})[0] == 403
    assert server.ui.conversations.get(conversation.id).messages == []


def test_coding_extension_is_declared_in_installed_metadata():
    from importlib.metadata import entry_points
    entries = [entry for entry in entry_points(group="ml_stack.ui_routes") if entry.name == "coding"]
    assert len(entries) == 1
    assert entries[0].value == "ml_stack.workspace.coding_routes:route"
    assert entries[0].load().__module__ == "ml_stack.workspace.coding_routes"


def test_launcher_refusal_is_failed_instead_of_a_completed_empty_turn(coding_api):
    server, conversation, kit = coding_api
    assert post(server, conversation.id, "start", {"message": "startup rejected"})[0] == 202
    status = finished(server, conversation.id)
    assert status["state"] == "failed"
    assert "status 2" in status["error"]
    assert not kit.ws.registry.role_of(f"chat-{conversation.id}")
    assert len(server.ui.conversations.get(conversation.id).messages) == 1
