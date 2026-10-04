"""Saved conversation settings through the guarded daemon API."""
from __future__ import annotations

import pytest
from test_fleet_ui import Serving

from ml_stack.fleet.conversations import Conversations


@pytest.fixture
def conversation_api(tmp_path):
    served = Serving(tmp_path)
    served.ui.settings.setup_done = True
    served.ui.conversations = Conversations(tmp_path / "chats")
    try:
        yield served
    finally:
        served.close()


def test_settings_update_preserves_title_and_messages(conversation_api):
    served = conversation_api
    status, made, _ = served.call("/ui/conversations", method="POST", body={"title": "Code review", "model": "small",
                                 "settings": {"mode": "coding", "project": "/tmp/example", "effort": "low"}})
    assert status == 201
    served.ui.conversations.append(made["id"], "user", "Review this code")
    status, updated, _ = served.call(f"/ui/conversations/{made['id']}", method="POST",
                                     body={"model": "large", "settings": {"temperature": .4}})
    assert status == 200
    assert updated["title"] == "Code review"
    assert updated["model"] == "large"
    assert updated["settings"]["project"] == "/tmp/example"
    assert updated["settings"]["temperature"] == .4
    assert updated["messages"][0]["content"] == "Review this code"


@pytest.mark.parametrize("body", [{"title": "Changed", "settings": {"temperature": True}},
                                  {"settings": {"mode": "unknown"}}, {"settings": []},
                                  {"model": 42}, {"model": None}, {"settings": None}, {"unknown": "field"}, []])
def test_invalid_updates_return_400_without_changing_the_chat(conversation_api, body):
    served = conversation_api
    made = served.ui.conversations.start(model="small", title="Kept")
    status, response, _ = served.call(f"/ui/conversations/{made.id}", method="POST", body=body)
    assert status == 400
    assert response["error"]
    kept = served.ui.conversations.get(made.id)
    assert kept.title == "Kept"
    assert kept.model == "small"
    assert kept.settings == made.settings


def test_conversation_mutations_require_ui_authority(conversation_api):
    served = conversation_api
    made = served.ui.conversations.start(title="Protected")
    for method, payload in (("POST", {"title": "Changed"}), ("DELETE", None)):
        path = f"/ui/conversations/{made.id}"
        assert served.call(path, method=method, body=payload, ui_header=False)[0] == 403
        assert served.call(path, method=method, body=payload,
                           headers={"Origin": "https://foreign.invalid"})[0] == 403
    assert served.ui.conversations.get(made.id).title == "Protected"
