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
    while server.ui.coding_turns.status(conversation.id)["session"] != SESSION:
        assert time.monotonic() < deadline
        time.sleep(.02)
    assert post(server, conversation.id, "start", {"message": "second"})[0] == 400
    began = time.monotonic()
    assert post(server, conversation.id, "cancel", {})[0] == 200
    assert finished(server, conversation.id)["state"] == "cancelled"
    assert time.monotonic() - began < 5
    assert post(server, conversation.id, "start", {"message": "continue inspection"})[0] == 202
    resumed = finished(server, conversation.id)
    assert resumed["state"] == "completed"
    assert resumed["session"] == SESSION


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


def test_joined_coding_session_is_required_before_starting_a_worker(coding_api, monkeypatch):
    from ml_stack.fleet import routes

    server, conversation, _kit = coding_api
    monkeypatch.setattr(routes, "in_cluster", lambda _: True)
    assert post(server, conversation.id, "start", {"message": "unsigned"})[0] == 401
    assert server.ui.conversations.get(conversation.id).messages == []
    cookie = server.ui.sessions.cookie_header(server.ui.sessions.open("person"))
    status, _result, _headers = server.call(f"/ui/coding/{conversation.id}/start", method="POST",
        body={"message": "signed"}, cookie=cookie,
        headers={"Origin": f"http://127.0.0.1:{server.port}", "Sec-Fetch-Site": "same-origin"})
    assert status == 202


def test_catalogue_uses_real_modelinfo_paths_and_coding_profile(monkeypatch, tmp_path):
    from ml_stack.hub.discover import ModelInfo
    from ml_stack.workspace import coding_routes, localmodel

    installed = ModelInfo(id="canonical", name="Qwen3.8-27B-Q4_K_XL.gguf", path=tmp_path / "model.gguf",
                          format="gguf", size_bytes=1234, source="local")
    monkeypatch.setattr(coding_routes.hub, "discover", lambda **kwargs: [installed])
    calls = []
    def choose(**kwargs):
        calls.append(kwargs)
        return localmodel.Pick(ref=installed.id, name=installed.name)
    monkeypatch.setattr(coding_routes.localmodel, "choose", choose)
    result = coding_routes.catalogue()
    assert result["models"] == [{"name": installed.name, "ref": str(installed.path)}]
    assert result["default_model"] == str(installed.path)
    selection = calls[0]["selection"]
    assert selection.coding is True and selection.search is False
    assert selection.context == coding_routes.LIMIT
    assert calls[0]["installed"] == [installed]


def test_launch_adapter_forwards_exact_context_and_mtp_head(monkeypatch):
    from ml_stack import coding

    calls = []
    monkeypatch.setitem(coding.HARNESSES, "codex", lambda argv, **options: calls.append(argv) or 0)
    assert coding.launch_coding_agent("/models/qwen.gguf", "read-only", "/project", context=262144,
                                     draft="/models/mtp-qwen.gguf") == 0
    assert calls[0][calls[0].index("--ctx") + 1] == "262144"
    assert calls[0][calls[0].index("--draft") + 1] == "/models/mtp-qwen.gguf"
